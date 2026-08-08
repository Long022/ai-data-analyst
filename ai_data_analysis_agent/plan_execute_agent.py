"""
Plan-and-Execute Agent — 强制硬约束的多步骤推理

流程:
  1. Planner  → LLM 先生成 JSON 步骤计划
  2. Executor → 确定性循环执行，上一步结果代入下一步
  3. Synthesizer → LLM 综合所有中间结果给出最终答案

外部接口和 agent.py 的 create_agent 完全一致:
  from plan_execute_agent import create_agent
  agent, db_path, df = create_agent(api_key, temp_path)
"""

import json
import re
import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from agent import _build_components
from prompts import get_table_schema


# ========== Prompts ==========

PLANNER_SYSTEM = """你是数据分析规划器。根据表结构和用户问题，生成 JSON 格式的执行步骤。

## 表结构
{schema}

## 可用工具
- sql_db_schema: 查看表结构（参数: table_name 字符串）
- sql_db_query: 执行 SQL 查询（参数: SQL 语句字符串，只允许 SELECT）
- sql_db_query_checker: 检查 SQL 语法（参数: SQL 语句字符串）
- python_repl: 执行 Python 代码（参数: Python 代码字符串，可用变量 df, pd, np）

## 规则
1. 简单问题 1-2 步，复杂多步骤问题 3-5 步
2. 第 1 步通常用 sql_db_schema 确认列名
3. 如果步骤 N+1 依赖步骤 N 的结果，用 <<step_N>> 作为占位符（执行时自动替换）
4. SQL 只允许 SELECT
5. 只输出 JSON 数组，不要 ```json 代码块，不要任何其他文字

## 输出格式
[{{"step": 1, "description": "查看表结构确认列名", "tool": "sql_db_schema", "input": "uploaded_data"}},
 {{"step": 2, "description": "找出数量最多的 3 个类别", "tool": "sql_db_query", "input": "SELECT category, COUNT(*) as cnt FROM uploaded_data GROUP BY category ORDER BY cnt DESC LIMIT 3"}},
 {{"step": 3, "description": "分析这 3 个类别的平均金额", "tool": "sql_db_query", "input": "SELECT category, AVG(amount) FROM uploaded_data WHERE category IN (<<step_2>>) GROUP BY category"}}]"""

SYNTHESIS_SYSTEM = """你是数据分析师。根据以下执行结果，回答用户的原始问题。

## 用户问题
{question}

## 执行步骤与结果
{execution_log}

用中文简洁回答，附带关键数字。如果某个步骤出错，说明可能原因并给出建议。"""


# ========== PlanExecuteAgent ==========

class PlanExecuteAgent:
    """Plan-and-Execute Agent，拥有和 LangGraph CompiledStateGraph 相同的 .invoke() / .stream() 接口"""

    def __init__(self, llm, tool_map, db_path, df):
        self.llm = llm
        self.tool_map = tool_map
        self.db_path = db_path
        self.df = df

    # ---- 内部 ----

    def _call_tool(self, name: str, input_str: str) -> str:
        """调用单个工具，返回字符串结果"""
        if name not in self.tool_map:
            return f"错误: 未知工具 '{name}'"
        tool = self.tool_map[name]
        try:
            if name == "python_repl":
                return str(tool.invoke({"query": input_str}))
            else:
                return str(tool.invoke(input_str))
        except Exception as e:
            return f"工具执行错误: {type(e).__name__}: {e}"

    def _resolve_placeholders(self, text: str, results: dict) -> str:
        """替换 <<step_N>> 占位符为实际执行结果"""
        def _replace(m):
            key = m.group(1).strip()
            return results.get(key, f"(未找到步骤 {key} 的结果)")

        # 只取结果中的关键数据行（去掉表头描述），避免 token 爆炸
        resolved_results = {}
        for k, v in results.items():
            resolved_results[k] = v[:500] if len(str(v)) > 500 else str(v)

        # 替换 <<step_N>> 格式的占位符
        text = re.sub(r'<<(.+?)>>', _replace, text)

        # 也支持 <<N>> 简写
        for k, v in resolved_results.items():
            # 提取 step_3_data 中的关键值
            text = text.replace(f"<<{k}>>", v)

        return text

    def _generate_plan(self, question: str):
        """调用 LLM 生成执行计划，返回步骤列表"""
        schema = get_table_schema(self.df)
        prompt = PLANNER_SYSTEM.format(schema=schema)
        messages = [SystemMessage(content=prompt), HumanMessage(content=question)]
        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # 容错：如果 LLM 输出嵌在 markdown 代码块中，提取 JSON
        for pattern in [r'```(?:json)?\s*\n?(.*?)\n?```', r'\[.*\]']:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                raw = match.group(1) if match.lastindex else match.group(0)
                break

        try:
            plan = json.loads(raw)
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass

        # JSON 解析失败：回退为单步骤，直接用 LLM 回答
        return [{"step": 1, "description": "直接回答", "tool": "done", "input": "fallback"}]

    def _synthesize(self, question: str, plan: list, results: dict) -> str:
        """综合执行结果，生成最终答案"""
        lines = []
        for step in plan:
            s = step["step"]
            desc = step.get("description", "")
            result = results.get(f"step_{s}", "(无结果)")
            lines.append(f"步骤{s}: {desc}\n结果: {result}")
        execution_log = "\n\n".join(lines)

        prompt = SYNTHESIS_SYSTEM.format(question=question, execution_log=execution_log)
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content.strip()

    # ---- 公开接口 ----

    def stream(self, input_dict: dict):
        """
        流式执行 Plan-and-Execute。
        每次 yield 的格式和 CompiledStateGraph.stream(mode="values") 一致:
          {"messages": [...]}
        """

        # 提取用户问题（输入格式兼容单条消息和消息列表）
        msgs = input_dict.get("messages", [])
        if not msgs:
            yield {"messages": []}
            return

        question = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else ""
        if hasattr(msgs[-1], 'content'):
            question = msgs[-1].content
        if not question:
            yield {"messages": [AIMessage(content="（空问题）")]}
            return

        accumulator = list(msgs)  # 逐步累加消息

        # ---- Phase 1: 生成计划 ----
        try:
            plan = self._generate_plan(question)
        except Exception as e:
            accumulator.append(AIMessage(content=f"计划生成失败: {e}"))
            yield {"messages": list(accumulator)}
            return

        # 如果回退到 done step，直接用 LLM 回答
        if len(plan) == 1 and plan[0].get("tool") == "done":
            try:
                answer = self._synthesize(question, plan, {})
            except Exception:
                answer = "无法处理此问题，请简化后重试。"
            accumulator.append(AIMessage(content=answer))
            yield {"messages": list(accumulator)}
            return

        # ---- Phase 2: 逐步执行 ----
        results = {}

        for i, step in enumerate(plan):
            step_num = step.get("step", i + 1)
            desc = step.get("description", f"步骤 {step_num}")
            tool_name = step.get("tool", "done")
            tool_input = step.get("input", "")

            # 替换占位符
            resolved_input = self._resolve_placeholders(tool_input, results)

            # 发送 tool_call 消息（UI 显示状态和代码）
            if tool_name != "done":
                tc_msg = AIMessage(content="", tool_calls=[{
                    "name": tool_name,
                    "args": ({"query": resolved_input} if tool_name == "python_repl"
                             else {"statement": resolved_input}),
                    "id": f"plan_call_{step_num}",
                }])
                accumulator.append(tc_msg)
                yield {"messages": list(accumulator)}

                # 执行工具
                output = self._call_tool(tool_name, resolved_input)
                results[f"step_{step_num}"] = output

                # 发送 ToolMessage
                t_msg = ToolMessage(content=str(output)[:4000], tool_call_id=f"plan_call_{step_num}")
                accumulator.append(t_msg)
                yield {"messages": list(accumulator)}

        # ---- Phase 3: 综合答案 ----
        try:
            answer = self._synthesize(question, plan, results)
        except Exception:
            # Synthesizer 出错：手动拼接结果
            parts = []
            for s in plan:
                step_key = f"step_{s['step']}"
                parts.append(f"**步骤 {s['step']}: {s.get('description', '')}**\n{results.get(step_key, '')}")
            answer = "\n\n".join(parts)

        accumulator.append(AIMessage(content=answer))
        yield {"messages": list(accumulator)}

    def invoke(self, input_dict: dict):
        """非流式调用，返回最终 messages"""
        last = None
        for event in self.stream(input_dict):
            last = event
        return last if last else {"messages": [AIMessage(content="（无结果）")]}


# ========== 工厂函数 ==========

def create_agent(api_key: str, temp_path: str):
    """
    创建 Plan-and-Execute Agent（硬约束版）

    参数和返回值与 agent.py 的 create_agent 完全一致:
      参数: api_key, temp_path
      返回: (agent, db_path, df)

    agent 对象有 .invoke() 和 .stream() 方法，
    可以直接替代 agent.py 版本的 agent。
    """

    llm, tools, tool_map, db_path, df, _system_message = _build_components(api_key, temp_path)

    agent = PlanExecuteAgent(llm=llm, tool_map=tool_map, db_path=db_path, df=df)

    return agent, db_path, df
