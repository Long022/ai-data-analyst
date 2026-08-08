"""
Prompt 模板 — 基于 DataFrame 动态生成 system message 和 few-shot 示例
"""

import pandas as pd


def get_table_schema(df: pd.DataFrame) -> str:
    """从 DataFrame 生成表结构描述"""
    lines = ["表名：uploaded_data", "列名和类型："]
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "int64":
            type_name = "整数"
        elif dtype == "float64":
            type_name = "浮点数"
        elif dtype == "datetime64[ns]":
            type_name = "日期时间"
        else:
            type_name = "文本"
        lines.append(f"  - {col} ({type_name})")
    return "\n".join(lines)


def generate_fewshot_examples(df: pd.DataFrame) -> str:
    """根据 DataFrame 的真实列名生成 few-shot SQL 示例"""
    columns = df.columns.tolist()
    numeric_cols = [c for c in columns if df[c].dtype in ("int64", "float64")]
    text_cols = [c for c in columns if df[c].dtype == "object"]

    examples = []
    examples.append('Q: "总共有多少条数据?"\nSQL: SELECT COUNT(*) FROM uploaded_data')

    if numeric_cols:
        num_col = numeric_cols[0]
        examples.append(f'Q: "{num_col} 的平均值?"\nSQL: SELECT AVG({num_col}) FROM uploaded_data')

    if text_cols:
        text_col = text_cols[0]
        examples.append(f'Q: "每个 {text_col} 的数量"\nSQL: SELECT {text_col}, COUNT(*) as cnt FROM uploaded_data GROUP BY {text_col} ORDER BY cnt DESC')

    examples.append('Q: "前 10 行"\nSQL: SELECT * FROM uploaded_data LIMIT 10')
    return "\n".join(examples)


def build_system_message(df: pd.DataFrame) -> str:
    """基于 DataFrame 构建 system message"""
    schema = get_table_schema(df)
    examples = generate_fewshot_examples(df)

    return f"""你是数据分析师。可用工具：SQL 查询 + Python REPL（pandas 已加载为 df）。

## 表结构
{schema}

## 规则
- 先用 sql_db_schema 查看表结构，禁止编造列名
- 简单查询/聚合用 SQL；统计分析/相关性/复杂计算用 Python REPL
- SQL 只允许 SELECT，禁止修改操作
- 用中文简洁回答，附带关键数字
- 用户要求"仪表盘"/"概览"/"数据画像"时，不要生成 markdown 表格或 ASCII 图表。直接告知："页面顶部已展示完整的图形化仪表盘（类别分布图、数值直方图、相关性矩阵），可以直接查看。您有具体分析问题我可以帮您深挖。"

## 多步骤问题（重要）
遇到包含"先…再…"、"然后"、"之后"等连接词或明显分两步以上的复杂问题时：
1. 先简要说明计划（一句话），再逐步执行
2. 每一步的结果（如查询出的 ID 列表）要直接代入下一步的参数中
3. 所有步骤完成后，给出综合结论，而非仅最后一步的结果
4. 简单问题不要过度拆分，直接回答即可

## 错误处理
SQL 执行出错时，不要只说"出错了"。必须：
1. 分析错误信息，定位具体原因（列名不存在？语法错误？类型不匹配？）
2. 如果列名不存在，用 sql_db_schema 查看真实列名，找出最相似的列
3. 向用户说明："列名 'xxx' 不存在，您是不是指 'yyy'？"然后自动用正确列名重试
4. 最多重试 2 次，仍失败则建议用户检查数据

## SQL 示例
{examples}

## 错误修正示例
Q: "sales 的平均值?"
→ 执行 SELECT AVG(sales) FROM uploaded_data → 报错: no such column: sales
→ 调用 sql_db_schema 查看列名 → 发现表中只有 total_sales
→ 回答: "列名 'sales' 不存在，表中实际列是 'total_sales'，我用它来计算。"
→ 执行 SELECT AVG(total_sales) FROM uploaded_data → 得到结果

## 多步骤示例
Q: "先找出数量最多的 3 个类别，再看它们的平均金额"
→ 计划: 先查 Top3 类别名，再对每个类别算平均金额
→ 步骤1: SELECT category, COUNT(*) as cnt FROM uploaded_data GROUP BY category ORDER BY cnt DESC LIMIT 3
→ 得到: ("电子产品", "食品", "服装")
→ 步骤2: SELECT category, AVG(amount) as avg_amount FROM uploaded_data WHERE category IN ('电子产品','食品','服装') GROUP BY category
→ 结论: "数量 Top3 是电子产品(1200条)/食品(980条)/服装(750条)。平均金额分别为 ¥850/¥45/¥320，电子产品量价双高。"

## Python 示例
Q: "数值列的相关性?" → Python: df.corr(numeric_only=True)
Q: "A 列和 B 列的 t 检验?" → Python: from scipy import stats; stats.ttest_ind(df['A'].dropna(), df['B'].dropna())
Q: "A 列的基本统计?" → Python: df['A'].describe()
"""
