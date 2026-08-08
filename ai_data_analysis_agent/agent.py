"""
agent.py - LangChain SQL Agent 模块
负责：创建 SQL Agent、管理数据库连接、网络连接管理
"""

import os
import tempfile
import sqlite3
import pandas as pd
import httpx
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.tools import StructuredTool
from config import DEFAULT_MODEL, DEFAULT_BASE_URL, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES
from prompts import build_system_message


# ========== 网络连接管理 ==========

_http_client = None


def clear_proxy_env():
    """清除代理环境变量 + NO_PROXY=* 绕过 Windows 系统代理"""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"





def get_http_client():
    """获取持久化的 httpx 客户端（连接池 + 内置重试）"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            transport=httpx.HTTPTransport(retries=3),
        )
    return _http_client



# ========== DeepSeek LLM ==========

class DeepSeekLLM(ChatOpenAI):
    """DeepSeek V4 Pro 自定义 LLM，继承 ChatOpenAI，预设所有默认值"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client=None,
        http_async_client=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            model_kwargs={
                "reasoning_effort": "high",
                "extra_body": {
                    "thinking": {"type": "enabled"},
                },
            },
            **kwargs,
        )


# ========== 自定义 Python REPL ==========

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "range": range, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}


def make_python_repl(df):
    """创建受限 Python 执行工具，只开放 pandas/numpy + 安全内置函数"""
    import numpy as np

    def execute(query: str) -> str:
        """执行 Python 代码进行数据分析。可用变量: df(DataFrame), pd(pandas), np(numpy)。返回执行结果。"""
        namespace = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "df": df}
        try:
            result = eval(query, namespace, {})
            if hasattr(result, "to_string"):
                return result.to_string()
            return str(result)
        except SyntaxError:
            try:
                exec(query, namespace, {})
                return "执行完毕"
            except Exception as e:
                return f"错误: {type(e).__name__}: {e}"
        except Exception as e:
            return f"错误: {type(e).__name__}: {e}"

    return StructuredTool.from_function(
        func=execute,
        name="python_repl",
        description="执行 Python 代码进行数据分析（统计检验、相关性、数据变换等）。可用变量: df(DataFrame), pd(pandas), np(numpy)。返回执行结果。",
    )


# ========== Agent 创建 ==========

def _build_components(api_key: str, temp_path: str):
    """构建 Agent 所需的核心组件（LLM、工具、数据库、DataFrame）。
    供 create_agent 和 plan_execute_agent 共用。"""

    clear_proxy_env()
    df = pd.read_csv(temp_path)

    # 创建 SQLite 数据库
    db_path = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
    conn = sqlite3.connect(db_path)
    df.to_sql("uploaded_data", conn, index=False, if_exists="replace")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM uploaded_data")
    db_row_count = cursor.fetchone()[0]
    conn.close()

    if len(df) != db_row_count:
        raise RuntimeError(
            f"数据写入异常：DataFrame 有 {len(df)} 行，但 SQLite 只有 {db_row_count} 行"
        )

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    http_client = get_http_client()

    llm = DeepSeekLLM(api_key=api_key, http_client=http_client)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    tools.append(make_python_repl(df))

    system_message = build_system_message(df)

    # 工具名 → 工具对象映射
    tool_map = {}
    for t in tools:
        tool_map[t.name] = t

    return llm, tools, tool_map, db_path, df, system_message


def create_agent(api_key: str, temp_path: str):
    """
    创建 LangChain SQL Agent

    参数:
        api_key: DeepSeek API Key
        temp_path: CSV 文件的临时路径

    返回:
        agent: CompiledStateGraph
        db_path: SQLite 数据库文件路径
        df: 原始 DataFrame
    """

    llm, tools, tool_map, db_path, df, system_message = _build_components(api_key, temp_path)

    agent = create_langchain_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,
    )

    return agent, db_path, df
