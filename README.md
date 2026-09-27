# OpsPilot — 智能运维 Agent（LangGraph）

> 告警来了，它照着你的 SOP 带你排障。

告警诊断 / 根因分析 / 处置建议的 Agent，知识来源为用户上传的 **PDF / Markdown** 运维文档，检索走 RAG。

```
docs/       RAG 模块设计文档（v0.2）
frontend/   类 ChatGPT 前端（单文件，零依赖）
backend/    FastAPI + LangGraph + RAG 服务（同时托管前端）
examples/   示例运维文档（SOP / 复盘），可直接拖进页面上传体验
```

## 快速开始

```bash
cd backend
uv venv --python 3.13
uv pip install -r requirements.txt langchain-openai

cp .env.example .env     # 按需填写大模型密钥等，见下方「环境变量」一节

uv run python -m uvicorn app.main:app
# 浏览器打开 http://127.0.0.1:8000
```

无需任何 API Key 即可跑通全链路（内置 Mock 模型 + 本地向量索引）。启动后把 `examples/` 下的示例 SOP 拖进页面上传（入口在左侧「知识库」区块内的「＋ 上传文档」），然后问「下单接口大量 504，第一步该查什么？」。`backend/smoke_test.py` 里也有一段可直接跑的样例内容。

### 环境变量

所有配置集中在 **`backend/.env`**（启动时由 `app/config.py` 自动加载，不用再敲一长串命令行参数）：

```bash
cd backend && cp .env.example .env
```

**优先级：命令行显式设置 > `.env` > 代码默认值**。所以临时换模型调试不必改文件：

```bash
LLM_MODEL=deepseek-reasoner uv run python -m uvicorn app.main:app
```

常用项：

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | 大模型。留空则用内置 Mock 模型，全链路照样能跑 |
| `EMBED_BACKEND` | `hash`（零依赖，BM25 兜底）/ `local`（BGE-M3 语义向量）/ `openai` |
| `RAG_MIN_COVER` | 最低相关性门槛，低于此值返回空而非硬塞上下文 |
| `RAG_TOP_K` / `RAG_CANDIDATE_K` | 返回条数 / 每路候选数 |
| `HOST` / `PORT` | 监听地址 |
| `OPSPILOT_DATA_DIR` | 索引目录，默认 `backend/data` |

> **⚠️ DeepSeek 必读**：DeepSeek **不提供 `/embeddings` 接口**。而 `EMBED_BACKEND=auto` 时一旦检测到 `OPENAI_API_KEY` 就会去调 `/embeddings` 并直接 404。所以用 DeepSeek 时 `.env` 里要把 `EMBED_BACKEND` 写成 `hash`（靠 BM25 走关键词召回）或 `local`（本地 BGE-M3，不走 DeepSeek）。想要语义召回，`local` 最省事。

> **⚠️ `local`（BGE-M3）模式**：模型约 2.2GB，已下载到 `backend/models/bge-m3`，`EMBED_MODEL` 直接指向该本地目录，**加载不再走 HuggingFace cache**。若需要在别的机器重新下载：
> ```bash
> HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 uv run python -c \
>   "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')"
> ```
> 两个变量都不能少：`HF_ENDPOINT` 绕开直连 502，`HF_HUB_DISABLE_XET=1` 绕开镜像站不支持 xet 分块传输导致的 CAS 401。
> CPU 上加载约 13s、单次检索 250~500ms、入库约 4~5s/份，可接受。

> **切换 EMBED_BACKEND 后无需手动重传**：hash 是 384 维、BGE-M3 是 1024 维，维度不符的旧向量会被引擎判为失效，服务启动时自动用 `data/files/` 下的原文件重建索引（日志里有 `[rag] ✓ xxx：N chunks`）。

**密钥安全**：`backend/.env` 已在 `.gitignore` 中忽略，只有模板 `.env.example` 入库。想彻底跳过 `.env` 加载：`OPSPILOT_SKIP_DOTENV=1`。

## 当前状态（v0.1 可运行版本）

- ✅ PDF / MD 上传、解析、切分、入库、删除
- ✅ 混合检索（稠密向量 + BM25 + RRF）+ 重排 + 引用溯源（文件名 · 页码/章节）
- ✅ LangGraph 主图：agent ⇄ ToolNode（RAG 作为工具）
- ✅ 类 ChatGPT 前端：流式对话、Markdown、引用角标、知识库管理
- ⏳ 评估体系、Milvus/ES 接入、异步任务队列、告警接入与执行器

详见 `docs/RAG模块设计文档.md` 与 `backend/README.md`。
