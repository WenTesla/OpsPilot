# OpsPilot — 智能运维 Agent（LangGraph）

> 告警来了，它照着你的 SOP 带你排障。

告警诊断 / 根因分析 / 处置建议的 Agent，知识来源为用户上传的 **PDF / Markdown** 运维文档，检索走 RAG。

```
docs/       RAG 模块设计文档（v0.2）
frontend/   类 ChatGPT 前端（单文件，零依赖）
backend/    FastAPI + LangGraph + RAG 服务（同时托管前端）
```

## 快速开始

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

无需任何 API Key 即可跑通全链路（内置 Mock 模型 + 本地向量索引）：上传一份 SOP（`backend/smoke_test.py` 里有样例内容），然后问「CPU 使用率超过 90% 怎么排查」。

配置 `OPENAI_API_KEY`（+ `pip install langchain-openai`）后切换为真实大模型；安装 `sentence-transformers` 后自动使用 BGE-M3 语义向量。

## 当前状态（v0.1 可运行版本）

- ✅ PDF / MD 上传、解析、切分、入库、删除
- ✅ 混合检索（稠密向量 + BM25 + RRF）+ 重排 + 引用溯源（文件名 · 页码/章节）
- ✅ LangGraph 主图：agent ⇄ ToolNode（RAG 作为工具）
- ✅ 类 ChatGPT 前端：流式对话、Markdown、引用角标、知识库管理
- ⏳ 评估体系、Milvus/ES 接入、异步任务队列、告警接入与执行器

详见 `docs/RAG模块设计文档.md` 与 `backend/README.md`。
