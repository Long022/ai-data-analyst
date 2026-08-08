# AI 数据分析助手 (LangChain + DeepSeek)

基于 LangChain SQL Agent 和 DeepSeek 的 AI 数据分析工具，通过自然语言查询即可分析 CSV/Excel 数据。

## 功能

- **文件上传** — 支持 CSV 和 Excel 文件，自动检测数据类型和编码
- **自然语言查询** — 用中文直接提问，Agent 自动生成 SQL 并执行
- **多轮对话** — 支持追问和上下文记忆
- **数据预览** — 显示数据类型、缺失值统计、内存占用
- **图表可视化** — 柱状图、饼图、折线图、散点图、直方图
- **示例查询** — 根据数据列名自动生成快速提问按钮
- **结果导出** — 查询结果可下载为 CSV

## 运行

```bash
pip install -r requirements.txt
streamlit run ai_data_analyst.py
```

1. 在侧边栏输入 DeepSeek API Key
2. 上传 CSV 或 Excel 文件
3. 用自然语言提问

## 依赖

- streamlit — Web UI
- langchain-community + langchain-openai — SQL Agent
- pandas + numpy — 数据处理
- matplotlib — 图表
- openpyxl — Excel 读取
- openai — DeepSeek API 兼容接口
