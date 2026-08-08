import re
import tempfile
import csv
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv
load_dotenv()
from agent import create_agent, clear_proxy_env
from config import MAX_FILE_SIZE_MB, MAX_ROWS
try:
    from pdf_report import generate_pdf_report
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

THEME_COLORS = px.colors.qualitative.Plotly
_THEME_CSS = """
<style>
    /* ================================================================
       0. 全局
       ================================================================ */
    html, body, [class*="css"] {
        font-family: "Inter", "Microsoft YaHei", "PingFang SC", -apple-system, sans-serif;
    }

    /* ================================================================
       1. 侧边栏
       ================================================================ */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 8px;
        transition: all 0.2s;
    }

    /* ================================================================
       2. Tab 栏
       ================================================================ */
    button[data-baseweb="tab"] {
        font-weight: 500;
        font-size: 14px;
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
        transition: all 0.2s;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
    button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #E2E8F0;
    }

    /* ================================================================
       3. 聊天气泡
       ================================================================ */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 85%;
        animation: fadeInUp 0.3s ease;
    }
    /* 用户消息 — 蓝色气泡靠右 */
    [data-testid="stChatMessage"][aria-label*="user" i],
    div[data-testid="stChatMessage"]:has(.stChatMessage[data-testid="stChatMessage"]):nth-of-type(odd) {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    /* 助手消息 — 灰色气泡靠左 */
    [data-testid="stChatMessage"]:not([aria-label*="user" i]) {
        background-color: #F1F5F9;
        border-bottom-left-radius: 4px;
    }
    /* 聊天气泡内文字 */
    [data-testid="stChatMessage"] p {
        margin: 0;
        line-height: 1.6;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ================================================================
       4. 指标卡
       ================================================================ */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.25s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-color: #2563EB;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #2563EB;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #64748B;
        font-weight: 500;
    }

    /* ================================================================
       5. 数据表格
       ================================================================ */
    /* 表头 */
    [data-testid="stDataFrame"] thead th,
    .stDataFrame thead th {
        background-color: #F1F5F9 !important;
        color: #1E293B !important;
        font-weight: 600;
        font-size: 13px;
        border-bottom: 2px solid #2563EB !important;
    }
    /* 隔行变色 */
    [data-testid="stDataFrame"] tbody tr:nth-child(even),
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    /* hover 高亮 */
    [data-testid="stDataFrame"] tbody tr:hover,
    .stDataFrame tbody tr:hover {
        background-color: #EFF6FF !important;
    }
    /* 单元格 */
    [data-testid="stDataFrame"] td,
    .stDataFrame td {
        font-size: 13px;
        padding: 6px 10px;
    }

    /* ================================================================
       6. 滚动条
       ================================================================ */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }

    /* ================================================================
       7. 加载动画 — 骨架屏脉冲
       ================================================================ */
    .stSpinner > div {
        border-color: #2563EB transparent transparent transparent !important;
    }
    /* skeleton pulse for empty placeholders */
    .skeleton-pulse {
        animation: skeletonPulse 1.5s ease-in-out infinite;
        background: linear-gradient(90deg, #E2E8F0 25%, #F1F5F9 50%, #E2E8F0 75%);
        background-size: 200% 100%;
        border-radius: 8px;
    }
    @keyframes skeletonPulse {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ================================================================
       8. 代码块 — 终端风格
       ================================================================ */
    pre, code {
        font-family: "Cascadia Code", "Fira Code", "JetBrains Mono", "Consolas", monospace;
    }
    .stCodeBlock pre,
    div[data-testid="stCodeBlock"] pre,
    .stCode pre {
        background-color: #0D1117 !important;          /* GitHub dark */
        color: #E6EDF3 !important;
        border-radius: 10px !important;
        padding: 16px !important;
        border: 1px solid #30363D;
        font-size: 13px;
        line-height: 1.6;
        overflow-x: auto;
    }
    /* 行内 code */
    code:not(pre code) {
        background-color: #F1F5F9;
        color: #2563EB;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }

    /* ================================================================
       9. Expander / 卡片
       ================================================================ */
    .stExpander {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.2s;
    }

    /* ================================================================
       10. 按钮
       ================================================================ */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* ================================================================
       11. 输入框
       ================================================================ */
    input[type="text"], textarea {
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
    }
    input[type="text"]:focus, textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }

    /* ================================================================
       12. 页脚隐藏
       ================================================================ */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ================================================================
       13. 标题栏渐变
       ================================================================ */
    h1 {
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }

    /* ================================================================
       14. Radio / Checkbox → 滑动开关
       ================================================================ */
    [data-baseweb="radio"] label,
    [data-baseweb="checkbox"] label {
        cursor: pointer;
        padding: 6px 12px;
        border-radius: 20px;
        transition: all 0.2s;
    }
    [data-baseweb="radio"] label:hover,
    [data-baseweb="checkbox"] label:hover {
        background-color: #EFF6FF;
    }

    /* ================================================================
       15. 空状态
       ================================================================ */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #64748B;
    }
    .empty-state .icon {
        font-size: 64px;
        margin-bottom: 16px;
        opacity: 0.6;
    }
    .empty-state .title {
        font-size: 20px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 8px;
    }
    .empty-state .desc {
        font-size: 14px;
        color: #94A3B8;
    }
</style>
"""


# ========== 数据预处理 ==========


def preprocess_and_save(file):
    """预处理上传文件或本地文件路径，返回 (temp_path, df)"""
    try:
        # 支持两种输入：Streamlit UploadedFile 或本地文件路径字符串
        if isinstance(file, str):
            file_size_mb = os.path.getsize(file) / (1024 * 1024)
            file_name = os.path.basename(file)
            if file_name.endswith(".csv"):
                df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
            else:
                st.error("不支持的文件格式，请上传 CSV 或 Excel 文件。")
                return None, None
        else:
            file_size_mb = file.size / (1024 * 1024)
            file_name = file.name
            if file_name.endswith(".csv"):
                df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
            else:
                st.error("不支持的文件格式，请上传 CSV 或 Excel 文件。")
                return None, None

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"文件过大 ({file_size_mb:.1f} MB)，超过限制 ({MAX_FILE_SIZE_MB} MB)。")
            return None, None

        if len(df) > MAX_ROWS:
            st.warning(f"数据行数过多 ({len(df):,} 行)，已自动采样前 {MAX_ROWS:,} 行进行分析。")
            df = df.head(MAX_ROWS)

        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df
    except Exception as e:
        st.error(f"文件处理错误: {e}")
        return None, None


# ========== 示例查询 ==========

def generate_example_queries(df: pd.DataFrame) -> list[str]:
    queries = ["总共有多少条数据？"]
    numeric_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64")]
    text_cols = [c for c in df.columns if df[c].dtype == "object"]

    if text_cols:
        queries.append(f"统计每个 {text_cols[0]} 的数量")
    if numeric_cols:
        queries.append(f"{numeric_cols[0]} 最高的前 5 条数据")
        queries.append(f"{numeric_cols[0]} 的平均值是多少？")
        if len(numeric_cols) >= 2:
            queries.append(f"{numeric_cols[0]} 和 {numeric_cols[1]} 的相关性如何？")
    if text_cols:
        queries.append(f"{text_cols[0]} 中包含最多数据的是哪个？")
    queries.append("显示前 10 行数据")
    return queries[:6]


# ========== 可视化 ==========

CHART_KEYWORDS = ["画图", "图表", "可视化", "柱状图", "饼图", "折线图", "散点图", "直方图",
                  "chart", "graph", "plot", "bar", "pie", "line", "scatter", "histogram",
                  "占比", "比例", "趋势", "分布", "变化", "走势", "相关性"]


def has_chart_request(question: str) -> bool:
    return any(kw in question.lower() for kw in CHART_KEYWORDS)


def infer_chart_config(question: str, df: pd.DataFrame):
    q = question.lower()
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64")]

    if any(kw in q for kw in ["饼图", "pie", "占比", "比例"]):
        chart_type = "pie"
    elif any(kw in q for kw in ["折线", "趋势", "line", "变化", "走势"]):
        chart_type = "line"
    elif any(kw in q for kw in ["分布", "直方图", "histogram"]):
        chart_type = "histogram"
    elif any(kw in q for kw in ["散点", "scatter", "相关性"]):
        chart_type = "scatter"
    else:
        chart_type = "bar"

    x_col = text_cols[0] if text_cols else df.columns[0]
    for col in df.columns:
        if col.lower() in q:
            x_col = col
            break

    y_col = None
    if chart_type in ("line", "scatter", "histogram") and num_cols:
        y_col = num_cols[0]
        for col in num_cols:
            if col.lower() in q:
                y_col = col
                break

    return chart_type, x_col, y_col


def generate_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None):
    fig, ax = plt.subplots(figsize=(7, 4))

    if chart_type == "bar":
        data = df.groupby(x_col).size().nlargest(20)
        data.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{x_col} 分布（柱状图）")
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "pie":
        data = df.groupby(x_col).size().nlargest(10)
        ax.pie(data.values, labels=data.index, autopct="%1.1f%%")
        ax.set_title(f"{x_col} 占比（饼图）")

    elif chart_type == "line":
        if y_col:
            df_sorted = df.sort_values(x_col)
            ax.plot(df_sorted[x_col], df_sorted[y_col], marker="o", color="steelblue")
            ax.set_title(f"{y_col} 随 {x_col} 变化趋势（折线图）")
            ax.tick_params(axis="x", rotation=45)
        else:
            st.warning("折线图需要选择 Y 轴")

    elif chart_type == "scatter":
        if y_col:
            ax.scatter(df[x_col], df[y_col], alpha=0.5, color="steelblue")
            ax.set_title(f"{x_col} vs {y_col} 散点图")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

    elif chart_type == "histogram":
        if y_col and df[y_col].dtype in ("int64", "float64"):
            ax.hist(df[y_col].dropna(), bins=20, color="steelblue", edgecolor="white")
            ax.set_title(f"{y_col} 分布直方图")

    plt.tight_layout()
    return fig


# ========== 临时文件清理 ==========

def cleanup_temp_files():
    for key in ["temp_path", "db_path"]:
        path = st.session_state.get(key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
            st.session_state[key] = None


# ========== 智能错误分类 ==========

def classify_error(error: Exception) -> str:
    error_str = str(error)
    error_type = type(error).__name__

    if "ConnectTimeout" in error_type or "ConnectError" in error_type or "ConnectionError" in error_type:
        return "无法连接到 DeepSeek API，请检查网络连接"
    if "ReadTimeout" in error_type or "ReadError" in error_type:
        return "API 响应超时，请重试或检查网络稳定性"
    if "401" in error_str or "AuthenticationError" in error_type or "invalid" in error_str.lower():
        return "API Key 无效，请检查并重新输入"
    if "429" in error_str or "RateLimitError" in error_type:
        return "请求过于频繁（API 限流），请稍后重试"
    if re.search(r'(?:status|error|http|code).*?\b(500|502|503)\b|\b(500|502|503)\b.*?(?:status|error|http|server)',
                 error_str, re.IGNORECASE):
        return "DeepSeek 服务器异常，请稍后重试"
    return error_str


# ========== 渲染组件 ==========

def render_data_overview(df: pd.DataFrame):
    """4 个指标卡：行数、列数、缺失值、内存"""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总行数", f"{len(df):,}")
    col2.metric("总列数", len(df.columns))
    col3.metric("缺失值", f"{df.isnull().sum().sum():,}")
    col4.metric("内存占用", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")


def render_auto_dashboard(df: pd.DataFrame):
    """仪表盘 expander：3 层（类别分布 → 数值直方图 → 详细信息）"""
    num_df = df.select_dtypes(include=["int64", "float64"])
    txt_df = df.select_dtypes(include=["object"])
    _df = df.sample(min(len(df), 5000), random_state=42) if len(df) > 5000 else df
    _colors = THEME_COLORS

    with st.expander("仪表盘", expanded=True):
        # ===== 第一层：类别分布 =====
        if not txt_df.empty:
            st.subheader("类别分布")
            for col in txt_df.columns[:3]:
                vc = _df[col].value_counts().nlargest(15)
                left, right = st.columns([3, 2])
                with left:
                    fig = px.bar(
                        x=vc.index, y=vc.values, color_discrete_sequence=[_colors[0]],
                        title=f"{col} — 各类别数量", labels={"x": col, "y": "数量"})
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key=f"dash_bar_{col}")
                with right:
                    pie_data = vc.nlargest(10)
                    fig = px.pie(
                        names=pie_data.index, values=pie_data.values,
                        title=f"{col} — 占比", color_discrete_sequence=_colors)
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True, key=f"dash_pie_{col}")

        # ===== 第二层：数值分布 =====
        if not num_df.empty:
            st.subheader("数值分布")
            num_cols = num_df.columns.tolist()
            for i in range(0, len(num_cols), 3):
                batch = num_cols[i:i + 3]
                row_cols = st.columns(len(batch))
                for j, col_name in enumerate(batch):
                    with row_cols[j]:
                        fig = px.histogram(
                            _df, x=col_name, nbins=30,
                            color_discrete_sequence=[_colors[j % len(_colors)]],
                            title=f"{col_name} 分布", labels={col_name: col_name})
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True, key=f"dash_hist_{col_name}")

        # ===== 第三层：详细信息 =====
        st.subheader("详细信息")
        detail_left, detail_right = st.columns(2)

        with detail_left:
            if not num_df.empty:
                st.caption("数值统计")
                st.dataframe(num_df.describe(), use_container_width=True)
            if not txt_df.empty:
                st.caption("类别概览")
                cat_data = []
                for col in txt_df.columns[:10]:
                    n_unique = df[col].nunique()
                    top_val = df[col].value_counts().index[0] if n_unique > 0 else "-"
                    top_pct = (df[col].value_counts().iloc[0] / len(df) * 100) if n_unique > 0 else 0
                    cat_data.append({
                        "列名": col, "唯一值": n_unique,
                        "最高频值": str(top_val)[:30], "占比": f"{top_pct:.1f}%"
                    })
                st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

        with detail_right:
            if len(num_df.columns) >= 2:
                corr = _df[num_df.columns].corr()
                fig = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    range_color=[-1, 1], title="相关性矩阵")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="dash_corr")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                fig = px.bar(
                    x=missing.index, y=missing.values, color_discrete_sequence=["coral"],
                    title="缺失值统计", labels={"x": "列名", "y": "缺失数量"})
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True, key="dash_missing")


def render_column_details(df: pd.DataFrame):
    """列详细信息 expander"""
    with st.expander("列详细信息"):
        info = pd.DataFrame({
            "列名": df.columns,
            "类型": df.dtypes.values,
            "非空值": df.count().values,
            "空值": df.isnull().sum().values,
        })
        st.dataframe(info, use_container_width=True)


# ========== 主入口 ==========

def main():
    clear_proxy_env()
    st.set_page_config(page_title="AI 数据分析助手", page_icon=":bar_chart:", layout="wide")
    st.session_state._css_injected = True
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.title("AI 数据分析助手 (DeepSeek)")

    # ---- 环境变量 ----
    env_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    env_data_file = os.getenv("DATA_FILE", "")

    # ---- session state 初始化 ----
    defaults = {
        "agent": None,
        "df": None,
        "temp_path": None,
        "db_path": None,
        "messages": [],
        "data_loaded": False,
        "auto_file_loaded": False,
        "_response_cache": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ===== Sidebar（全局：API + 文件上传）=====
    with st.sidebar:
        st.header("API 配置")
        api_key = st.text_input(
            "DeepSeek API Key:", type="password",
            value=env_api_key if env_api_key else "",
        )
        if api_key:
            st.session_state.api_key = api_key
            st.success("API Key 已保存")
        else:
            st.warning("请先输入 DeepSeek API Key")

        st.divider()

        # 如果设置了 DATA_FILE 环境变量，自动加载
        if env_data_file and not st.session_state.auto_file_loaded:
            if os.path.isfile(env_data_file):
                with st.spinner(f"自动加载: {os.path.basename(env_data_file)}"):
                    temp_path, df = preprocess_and_save(env_data_file)
                    if temp_path is not None and df is not None:
                        st.session_state.temp_path = temp_path
                        st.session_state.df = df
                        st.session_state.auto_file_loaded = True
                        st.success(f"已加载: {os.path.basename(env_data_file)} ({len(df):,} 行)")
                    else:
                        st.error(f"无法加载: {env_data_file}")
            else:
                st.warning(f"文件不存在: {env_data_file}")

        st.header("数据上传")
        uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])

    # ===== 数据来源解析 =====
    data_source = None
    data_source_name = None

    if uploaded_file is not None:
        data_source = uploaded_file
        data_source_name = uploaded_file.name
    elif st.session_state.auto_file_loaded:
        data_source = st.session_state.temp_path
        data_source_name = os.path.basename(env_data_file)

    # ===== Agent 初始化 =====
    if data_source is not None and st.session_state.get("api_key"):
        if st.session_state.get("_last_file_name") != data_source_name:
            cleanup_temp_files()
            st.session_state.agent = None
            st.session_state.data_loaded = False
            st.session_state.messages = []
            st.session_state._response_cache = {}
            st.session_state._last_file_name = data_source_name

        if not st.session_state.data_loaded:
            if isinstance(data_source, str):
                temp_path = data_source
                df = st.session_state.df
            else:
                temp_path, df = preprocess_and_save(data_source)
                if temp_path is not None and df is not None:
                    st.session_state.temp_path = temp_path
                    st.session_state.df = df

            if temp_path is not None and df is not None:
                with st.spinner("正在初始化 AI 助手..."):
                    try:
                        agent, db_path, _ = create_agent(
                            st.session_state.api_key, temp_path
                        )
                        st.session_state.agent = agent
                        st.session_state.db_path = db_path
                        st.session_state.data_loaded = True
                    except Exception as e:
                        st.error(classify_error(e))
                        with st.expander("查看原始错误"):
                            st.exception(e)
                        st.stop()

    elif data_source is not None and not st.session_state.get("api_key"):
        st.markdown('<div class="empty-state"><div class="icon">🔑</div><div class="title">需要 API Key</div><div class="desc">请在侧边栏输入 DeepSeek API Key 开始分析</div></div>', unsafe_allow_html=True)
    elif data_source is None:
        st.markdown('<div class="empty-state"><div class="icon">📊</div><div class="title">欢迎使用 AI 数据分析助手</div><div class="desc">上传 CSV 或 Excel 文件，用自然语言探索你的数据</div></div>', unsafe_allow_html=True)

    # ===== 主区域：4 个 Tab =====
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3, tab4 = st.tabs(["对话分析", "数据画像", "SQL 编辑器", "导出报告"])

        # ---- Tab 1: 对话分析 ----
        with tab1:
            # 示例查询
            example_queries = generate_example_queries(df)
            st.caption("快速提问：")
            cols = st.columns(len(example_queries))
            for i, query in enumerate(example_queries):
                with cols[i]:
                    if st.button(query, key=f"example_{i}", use_container_width=True):
                        st.session_state._pending_query = query

            # 清空对话
            col_clear, _ = st.columns([1, 5])
            with col_clear:
                if st.session_state.messages and st.button("清空对话"):
                    st.session_state.messages = []
                    st.session_state._response_cache = {}
                    st.rerun()

            st.divider()

            # 消息历史
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 输入
            user_input = st.chat_input("输入你的数据分析问题...")

            pending = st.session_state.pop("_pending_query", None)
            if pending:
                user_input = pending

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                cache_key = f"{st.session_state.get('temp_path', '')}|{user_input}"

                history_messages = []
                for m in st.session_state.messages[-12:]:
                    history_messages.append({"role": m["role"], "content": m["content"]})
                history_messages.append({"role": "user", "content": f"数据问题：{user_input}"})

                streamed_code = []
                with st.chat_message("assistant"):
                    if cache_key in st.session_state._response_cache:
                        result = st.session_state._response_cache[cache_key]
                        st.caption("缓存命中")
                        st.markdown(result)
                    else:
                        response_placeholder = st.empty()
                        status_text = st.empty()
                        result = ""

                        try:
                            for event in st.session_state.agent.stream(
                                {"messages": history_messages},
                                stream_mode="values"
                            ):
                                messages = event.get("messages", [])
                                last_msg = messages[-1] if messages else None
                                if last_msg is None:
                                    continue

                                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                    for tc in last_msg.tool_calls:
                                        name = tc.get('name', '')
                                        args = tc.get('args', {})
                                        if 'schema' in name:
                                            status_text.text("查看表结构...")
                                        elif 'query' in name or 'sql' in name.lower():
                                            status_text.text("执行查询...")
                                            sql = args.get('statement', args.get('query', ''))
                                            if sql and sql not in [c for c, _ in streamed_code]:
                                                streamed_code.append((sql, "sql"))
                                        elif 'checker' in name:
                                            status_text.text("验证 SQL...")
                                        elif 'python' in name.lower() or 'repl' in name.lower():
                                            status_text.text("执行 Python...")
                                            code = args.get('query', args.get('command', ''))
                                            if code and code not in [c for c, _ in streamed_code]:
                                                streamed_code.append((code, "python"))
                                        else:
                                            status_text.text(f"调用工具: {name}...")

                                if hasattr(last_msg, 'content') and last_msg.content:
                                    result = last_msg.content
                                    response_placeholder.markdown(result + "▌")

                            status_text.empty()
                            response_placeholder.markdown(result)

                            for code, lang in streamed_code:
                                label = "执行的 SQL" if lang == "sql" else "执行的 Python"
                                with st.expander(label):
                                    st.code(code, language=lang)

                            st.session_state._response_cache[cache_key] = result

                        except Exception as e:
                            raw_error = str(e)
                            friendly = classify_error(e)
                            result = f"{friendly}\n\n<details><summary>原始错误</summary>\n\n```\n{type(e).__name__}: {raw_error}\n```\n</details>"
                            response_placeholder.markdown(result)

                st.session_state.messages.append({"role": "assistant", "content": result})

                # 自动图表
                chart_triggered = has_chart_request(user_input) or has_chart_request(result)
                if not chart_triggered and streamed_code:
                    for code, lang in streamed_code:
                        if lang == "sql" and ("GROUP BY" in code.upper() or "COUNT" in code.upper()):
                            chart_triggered = True
                            break

                if chart_triggered:
                    chart_type, x_col, y_col = infer_chart_config(user_input + " " + result, df)
                    try:
                        for code, lang in streamed_code:
                            if lang == "sql":
                                import sqlite3
                                conn = sqlite3.connect(st.session_state.db_path)
                                sql_df = pd.read_sql_query(code, conn)
                                conn.close()
                                if len(sql_df.columns) >= 2 and len(sql_df) <= 100:
                                    fig, ax = plt.subplots(figsize=(7, 4))
                                    sql_df.set_index(sql_df.columns[0]).plot(
                                        kind="barh", ax=ax, color="steelblue", legend=False
                                    )
                                    ax.set_title(f"查询结果（{sql_df.columns[0]}）")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    st.session_state.setdefault("_charts", []).append((fig, "自动图表"))
                                    break
                        else:
                            fig = generate_chart(df, chart_type, x_col, y_col)
                            st.pyplot(fig)
                            plt.close(fig)
                            st.session_state.setdefault("_charts", []).append((fig, "自动图表"))
                    except Exception as e:
                        st.caption(f"图表生成失败: {e}")

        # ---- Tab 2: 数据画像 ----
        with tab2:
            st.subheader("数据预览")
            show_full = st.checkbox("显示完整数据", value=False)
            st.dataframe(df if show_full else df.head(10), use_container_width=True)
            render_data_overview(df)
            render_auto_dashboard(df)
            render_column_details(df)

        # ---- Tab 3: SQL 编辑器 ----
        with tab3:
            with st.expander("表结构参考", expanded=False):
                schema_lines = [f"- **{col}** ({'数值' if df[col].dtype in ('int64','float64') else '文本'})" for col in df.columns]
                st.markdown("\n".join(schema_lines) + f"\n\n表名: `uploaded_data` | 行数: {len(df):,}")
            sql = st.text_area(
                "SQL 查询",
                value="SELECT * FROM uploaded_data LIMIT 10",
                height=140,
                key="sql_editor",
                placeholder="SELECT ... FROM uploaded_data ...",
            )
            col1, _col2 = st.columns([1, 5])
            with col1:
                run = st.button("执行查询", use_container_width=True)
            if run and sql.strip():
                import sqlite3
                try:
                    conn = sqlite3.connect(st.session_state.db_path)
                    result_df = pd.read_sql_query(sql, conn)
                    conn.close()
                    st.caption(f"返回 {len(result_df):,} 行")
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        label="导出结果 CSV",
                        data=result_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="sql_result.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"SQL 执行错误: {e}")

        # ---- Tab 4: 导出报告 ----
        with tab4:
            st.subheader("导出报告")

            # PDF 报告
            if _PDF_AVAILABLE:
                left, right = st.columns(2)
                with left:
                    if st.button("生成 PDF 报告", use_container_width=True):
                        with st.spinner("生成 PDF..."):
                            pdf_bytes = generate_pdf_report(
                                df=df,
                                messages=st.session_state.messages,
                                charts=st.session_state.get("_charts", []),
                            )
                            st.session_state._pdf_bytes = pdf_bytes
                with right:
                    if st.session_state.get("_pdf_bytes"):
                        st.download_button(
                            label="下载 PDF 报告",
                            data=st.session_state._pdf_bytes,
                            file_name="analysis_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
            else:
                st.caption("PDF 导出需要 fpdf2 库: pip install fpdf2")

            st.divider()

            # CSV 导出
            st.subheader("导出数据")
            csv_type = st.radio("导出内容", ["原始数据", "对话记录"], horizontal=True)
            if csv_type == "原始数据":
                csv_data = df.to_csv(index=False).encode("utf-8-sig")
                csv_name = "data_export.csv"
            else:
                csv_lines = ["role,content"]
                for m in st.session_state.messages:
                    csv_lines.append(f'{m["role"]},{m["content"][:500].replace(chr(10), " ")}')
                csv_data = "\n".join(csv_lines).encode("utf-8-sig")
                csv_name = "chat_history.csv"
            st.download_button(
                label="下载 CSV",
                data=csv_data,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
            )

            st.divider()

            # 对话记录
            st.subheader("对话记录")
            if st.session_state.messages:
                for i, m in enumerate(st.session_state.messages):
                    role_label = "用户" if m["role"] == "user" else "助手"
                    with st.expander(f"#{i + 1} {role_label}: {m['content'][:40]}…"):
                        st.markdown(m["content"])
            else:
                st.caption("暂无对话记录")


if __name__ == "__main__":
    main()
