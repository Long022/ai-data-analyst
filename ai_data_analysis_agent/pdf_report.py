"""
PDF 报告生成器 — 将数据分析结果导出为 PDF 文件
"""

import io
import os
import re
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF


# 查找系统中支持中文的字体
def _find_cjk_font():
    """查找系统可用的中日韩字体"""
    candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",       # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",      # SimHei
        "C:/Windows/Fonts/simsun.ttc",      # SimSun
        "C:/Windows/Fonts/simkai.ttf",      # KaiTi
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


_CJK_FONT_PATH = _find_cjk_font()


class AnalysisReport(FPDF):
    """自定义 PDF 报告，支持中文"""

    def __init__(self, title: str = "数据分析报告"):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        # 注册中文字体
        if _CJK_FONT_PATH:
            self.add_font("CJK", "", _CJK_FONT_PATH, uni=True)
            self.add_font("CJK", "B", _CJK_FONT_PATH, uni=True)
            self.font_name = "CJK"
        else:
            self.font_name = "Helvetica"

    def _font(self, style="", size=10):
        self.set_font(self.font_name, style, size)

    def header(self):
        if self.page_no() == 1:
            return  # skip header on title page
        self._font("B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, self.title, align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self._font("", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_title(self, text: str):
        self._font("B", 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, text, align="L")
        self.ln(15)

    def add_section(self, title: str):
        self._font("B", 12)
        self.set_text_color(50, 50, 150)
        self.cell(0, 10, title, align="L")
        self.ln(10)

    def add_metric(self, label: str, value: str):
        self._font("", 10)
        self.set_text_color(50, 50, 50)
        self.cell(50, 7, f"{label}:")
        self._font("B", 10)
        self.cell(0, 7, str(value))
        self.ln(6)

    def add_table(self, df: pd.DataFrame, col_widths: list = None):
        """渲染 DataFrame 为表格"""
        if df.empty:
            self._font("", 9)
            self.cell(0, 7, "(No data)")
            self.ln(7)
            return

        if col_widths is None:
            available = self.w - self.l_margin - self.r_margin
            col_widths = [available / len(df.columns)] * len(df.columns)

        # Header
        self._font("B", 8)
        self.set_fill_color(230, 230, 240)
        for i, col in enumerate(df.columns):
            self.cell(col_widths[i], 7, str(col)[:25], border=1, fill=True)
        self.ln()

        # Data rows (max 30)
        self._font("", 7)
        for _, row in df.head(30).iterrows():
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val)[:30], border=1)
            self.ln()

        self.ln(5)

    def add_markdown_table(self, header: list, rows: list):
        """渲染从 markdown 文本解析出的表格"""
        available = self.w - self.l_margin - self.r_margin
        col_w = available / len(header) if header else available
        self._font("B", 7)
        self.set_fill_color(230, 230, 240)
        for cell in header:
            self.cell(col_w, 6, str(cell)[:20], border=1, fill=True)
        self.ln()
        self._font("", 7)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_w if i < len(header) else col_w, 5, str(cell)[:25], border=1)
            self.ln()
        self.ln(3)

    def _render_markdown_segments(self, text: str):
        """解析文本中的 markdown 表格和普通段落，混合渲染"""
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
                # 检测 markdown 表格: 需要表头行 + 分隔行 + 至少一行数据
                header_line = line
                sep_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if sep_line.startswith("|") and "---" in sep_line:
                    header = [h.strip() for h in header_line.split("|")[1:-1]]
                    data_rows = []
                    j = i + 2
                    while j < len(lines) and lines[j].strip().startswith("|"):
                        data_rows.append([c.strip() for c in lines[j].strip().split("|")[1:-1]])
                        j += 1
                    if header and data_rows:
                        self.add_markdown_table(header, data_rows)
                    i = j
                    continue
            # 普通文本行: 累积连续的非表格行一起输出
            if line:
                # 清理 markdown 格式标记
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
                clean = re.sub(r'[*_]{1,2}(.+?)[*_]{1,2}', r'\1', clean)
                clean = re.sub(r'`(.+?)`', r'\1', clean)
                clean = re.sub(r'#{1,6}\s*', '', clean)
                self._font("", 8)
                self.set_text_color(50, 50, 50)
                self.multi_cell(0, 5, clean)
            i += 1

    def add_qa_pair(self, q: str, a: str):
        self._font("B", 9)
        self.set_text_color(30, 100, 30)
        self.cell(0, 6, f"Q: {q[:120]}")
        self.ln(7)
        if "|" in a and "---" in a:
            self._render_markdown_segments(a)
        else:
            self._font("", 8)
            self.set_text_color(50, 50, 50)
            answer_text = a[:3000] + ("..." if len(a) > 3000 else "")
            self.multi_cell(0, 5, answer_text)
        self.ln(5)

    def add_chart_image(self, fig, caption: str = ""):
        """将 matplotlib figure 嵌入 PDF"""
        if fig is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, format="png", dpi=100, bbox_inches="tight")
            tmp_path = tmp.name

        img_w = self.w - self.l_margin - self.r_margin - 10
        img_h = img_w * 0.55
        if self.get_y() + img_h > self.h - 30:
            self.add_page()
        self.image(tmp_path, x=self.l_margin + 5, w=img_w, h=img_h)
        if caption:
            self.ln(3)
            self._font("", 8)
            self.cell(0, 5, caption, align="C")
        self.ln(8)
        os.unlink(tmp_path)


def generate_pdf_report(
    df: pd.DataFrame,
    messages: list,
    charts: list = None,
    output_path: str = None,
) -> bytes:
    """
    生成 PDF 数据分析报告

    参数:
        df: 数据 DataFrame
        messages: 对话消息列表 [{"role": "user"/"assistant", "content": "..."}]
        charts: [(fig, caption), ...] matplotlib 图表列表
        output_path: 输出文件路径（可选），不指定则返回 bytes

    返回:
        PDF 文件的 bytes 内容
    """

    pdf = AnalysisReport("数据分析报告")
    pdf.alias_nb_pages()
    pdf.add_page()

    # ---- 标题 ----
    pdf.add_title("数据分析报告")
    pdf._font("", 9)
    pdf.cell(0, 6, f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.ln(12)

    # ---- 数据摘要 ----
    pdf.add_section("1. 数据摘要")
    pdf.add_metric("行数", f"{len(df):,}")
    pdf.add_metric("列数", str(len(df.columns)))
    pdf.add_metric("缺失值", f"{df.isnull().sum().sum():,}")
    pdf.add_metric("内存", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    pdf.ln(5)

    # 列类型表
    col_info = pd.DataFrame({
        "列名": df.columns,
        "类型": df.dtypes.values.astype(str),
        "非空值": df.count().values,
        "空值": df.isnull().sum().values,
    })
    pdf.add_table(col_info)

    # ---- 数值统计 ----
    num_df = df.select_dtypes(include=["int64", "float64"])
    if not num_df.empty:
        pdf.add_section("2. 数值统计")
        pdf.add_table(num_df.describe().round(2))

    # ---- 类别概览 ----
    txt_df = df.select_dtypes(include=["object"])
    if not txt_df.empty:
        pdf.add_section("3. 类别概览")
        cat_rows = []
        for col in txt_df.columns:
            n_unique = df[col].nunique()
            top_val = df[col].value_counts().index[0] if n_unique > 0 else "-"
            cat_rows.append({"列名": col, "唯一值": n_unique, "最高频值": str(top_val)[:30]})
        pdf.add_table(pd.DataFrame(cat_rows))

    # ---- 图表 ----
    if charts:
        pdf.add_section("4. 图表")
        for fig, caption in charts[:6]:
            pdf.add_chart_image(fig, caption)

    # ---- 问答 ----
    qa_pairs = []
    for i, m in enumerate(messages):
        if m["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            qa_pairs.append((m["content"], messages[i + 1]["content"]))

    if qa_pairs:
        pdf.add_section("5. 问答记录")
        for q, a in qa_pairs[-10:]:
            pdf.add_qa_pair(q, a)

    # 输出
    if output_path:
        pdf.output(output_path)
        with open(output_path, "rb") as f:
            return f.read()
    else:
        return bytes(pdf.output(dest="S"))
