# OpsPilot 后端（LangGraph + RAG）v0.1 可运行版本

FastAPI 服务，同时托管前端静态页面。启动后浏览器访问 **http://127.0.0.1:8000** 即可使用完整应用。

## 启动

```bash
# 1. 安装依赖
uv venv --python 3.13
uv pip install -r requirements.txt langchain-openai

# 2. 生成本地配置（按需填写，之后会被自动加载）
cp .env.example .env

# 3. 启动（在 backend/ 目录下）
uv run python -m uvicorn app.main:app

# 4. 打开 http://127.0.0.1:8000
```

### 环境变量

配置统一放 **`backend/.env`**，启动时由 `app/config.py` 自动加载（查找顺序 `backend/.env` → 仓库根 `.env`）。`.env` 已被 Git 忽略，`.env.example` 是模板。

**优先级：命令行显式设置 > `.env` > 代码默认值**，所以临时调试不必改文件：

```bash
LLM_MODEL=deepseek-reasoner uv run python -m uvicorn app.main:app
```

设 `OPSPILOT_SKIP_DOTENV=1` 可完全关闭 `.env` 加载（回退到纯环境变量模式）。

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 空 | 配了就走真实大模型（需 `pip install langchain-openai`），不配用内置 Mock 模型 |
| `OPENAI_BASE_URL` | 空 | OpenAI 兼容端点（如 DeepSeek/通义/本地 vLLM） |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `LLM_TEMPERATURE` | `0.2` | 排障场景建议保持低温 |
| `EMBED_BACKEND` | `auto` | `auto`/`local`(BGE-M3)/`openai`/`hash`。**用 DeepSeek 必须显式设 `hash`**，它没有 `/embeddings` 接口 |
| `EMBED_MODEL` | `BAAI/bge-m3` | `EMBED_BACKEND=local` 时使用的句向量模型 |
| `RAG_RERANK` | `0` | 设为 `1` 且装了 sentence-transformers 时启用 CrossEncoder 重排 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排模型 |
| `RAG_TOP_K` | `5` | 返回条数 |
| `RAG_CANDIDATE_K` | `20` | 向量路 / BM25 路各自的候选数 |
| `RAG_MIN_COVER` | `0.12` | 最低「IDF 加权查询覆盖率」，低于此值视为无关、不返回（防凑数） |
| `CHUNK_TARGET_CHARS` | `700` | 切分时段落累积字符上限 |
| `CHUNK_OVERLAP_CHARS` | `80` | 相邻 chunk 尾部重叠字符数 |
| `MAX_UPLOAD_MB` | `50` | 单文件上传大小上限 |
| `OPSPILOT_DATA_DIR` | `backend/data` | 索引与文件存放目录 |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 监听地址（`python app.main` 方式启动时生效） |

## 已实现的能力

| 文档章节 | 实现 | 位置 |
|---|---|---|
| §3 知识接入（PDF/MD 上传） | ✅ 上传接口 + 幂等 + 失败状态 | `app/main.py`、`app/rag/parse.py` |
| §4 切分 | ✅ MD 按标题层级、PDF 按页，chunk 携带章节路径/页码 | `app/rag/parse.py` |
| §5.1 检索流程 | ✅ 稠密 + BM25 + RRF 融合 + 重排 + 上下文组装 | `app/rag/store.py`、`pipeline.py` |
| §5.3 混合检索 | ✅（BM25 为内置实现，非 ES） | `store.py::_bm25_rank` |
| §5.4 元数据过滤 | ✅ service / env / doc_type / doc_id | `store.py::_filter_idx` |
| §5.5 Rerank | ⚠️ 默认用「融合分 + IDF 加权覆盖率」兜底；配 `RAG_RERANK=1` 走 CrossEncoder | `store.py::_rescore` |
| §6 LangGraph 集成 | ✅ StateGraph + ToolNode，RAG 作为 Tool | `app/agent/graph.py` |
| §7 评估 | ❌ 未实现（M3） | — |
| 附录 A 文档管理 API | ✅ | `app/main.py` |

## 目录结构

```
backend/
  app/
    config.py            配置（环境变量）
    main.py              FastAPI 路由 + SSE + 静态托管
    rag/
      parse.py           PDF/MD 解析 + 结构感知切分
      embed.py           Embedding 适配器（BGE-M3 / OpenAI / 本地哈希兜底）
      store.py           向量检索 + BM25 + RRF + 重排 + 磁盘持久化
      pipeline.py        入库/检索/上下文组装，引用收集（ContextVar）
    agent/
      llm.py             模型工厂 + MockChatModel（无密钥可用）
      graph.py           LangGraph 主图：agent ⇄ tools
    storage/docs.py      文档元数据（JSON 持久化）
  smoke_test.py          离线链路自测（解析→入库→检索→Agent）
  api_test.py            接口自测（上传→列表→SSE 对话）
```

## 验证

```bash
uv run python smoke_test.py     # 不需启动服务（链路自测）
uv run python api_test.py       # 需先启动 uvicorn（接口自测）
uv run python eval_rag.py       # RAG 效果评测（检索质量 / 引用溯源 / 是否真的依据文档作答）
```

评测跑在独立临时索引目录里，不会污染正式知识库；`.env` 配了 `OPENAI_API_KEY` 会额外跑生成层与拒答层，不配也能跑前三层。

> 想用 Mock 模型跑评测（不消耗 token、验证纯检索能力）：`OPSPILOT_SKIP_DOTENV=1 uv run python eval_rag.py`

评测用例在 `eval_cases.json`，报告输出到 `eval_report.md`。

| 验证层 | 检查什么 | 目前结果 |
|---|---|---|
| 检索层 | Recall@5 / MRR，该命中的章节有没有进候选 | 100% / 0.917 |
| 溯源层 | 引用里的章节是否真实存在于原文（防幻觉锚点） | 0 条不通过 |
| **生成层** | **哨兵测试**：塞一条编造规定（CPU 阈值一律 42%、必须执行 `sentinel-ack`），看模型是否照着复述。能复述 = 确实依据文档作答，而非通用知识 | 通过 |
| 拒答层 | 库里没有的问题是否说明「未覆盖」而不硬编 | 通过（生成层兜底） |

配了 `OPENAI_API_KEY` 会额外跑生成层与拒答层；不配也能跑前三层。

## 已知限制（对应文档里的 M2~M4）

1. **向量库是本地 numpy + JSON**，未接 Milvus/ES；数据量上万 chunk 后需替换（接口已隔离，替换 `ChunkStore` 即可）
2. **未配 API Key 时用 Mock 模型**，只做检索结果摘要式汇总，不会做真正的推理；配 Key 后即自动切换
3. **Embedding 默认降级为本地哈希向量**（非语义），主要靠 BM25 兜底；装 `sentence-transformers` 后自动用 BGE-M3，效果显著提升
4. 无权限体系、无异步任务队列（上传解析为同步执行，大 PDF 会阻塞请求）
5. 评估体系（RAGAS / 评测集）尚未实现 —— 现用自研的 `eval_rag.py`（见上方「验证」）
6. **分词过朴素导致相关性门槛区分度差**：`tokenize` 是「英文按词 + 中文单字/二字组」，实测相关查询的覆盖率最低 0.145、无关查询最高 0.144，几乎贴在一起。`RAG_MIN_COVER` 只能做粗过滤，真正的拒答靠 LLM 兜底。换 jieba 分词或接入 BGE-M3 语义向量后才能有效拉开差距
