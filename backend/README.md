# OpsPilot 后端（LangGraph + RAG）v0.1 可运行版本

FastAPI 服务，同时托管前端静态页面。启动后浏览器访问 **http://127.0.0.1:8000** 即可使用完整应用。

## 启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动（在 backend/ 目录下）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3. 打开 http://127.0.0.1:8000
```

可选环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 空 | 配了就走真实大模型（需 `pip install langchain-openai`），不配用内置 Mock 模型 |
| `OPENAI_BASE_URL` | 空 | OpenAI 兼容端点（如 DeepSeek/通义/本地 vLLM） |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名 |
| `EMBED_BACKEND` | `auto` | `auto`/`local`(BGE-M3)/`openai`/`hash` |
| `RAG_RERANK` | `0` | 设为 `1` 且装了 sentence-transformers 时启用 CrossEncoder 重排 |
| `RAG_TOP_K` | `5` | 返回条数 |
| `OPSPILOT_DATA_DIR` | `backend/data` | 索引与文件存放目录 |

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
python smoke_test.py     # 不需启动服务
python api_test.py       # 需先启动 uvicorn
```

## 已知限制（对应文档里的 M2~M4）

1. **向量库是本地 numpy + JSON**，未接 Milvus/ES；数据量上万 chunk 后需替换（接口已隔离，替换 `ChunkStore` 即可）
2. **未配 API Key 时用 Mock 模型**，只做检索结果摘要式汇总，不会做真正的推理；配 Key 后即自动切换
3. **Embedding 默认降级为本地哈希向量**（非语义），主要靠 BM25 兜底；装 `sentence-transformers` 后自动用 BGE-M3，效果显著提升
4. 无权限体系、无异步任务队列（上传解析为同步执行，大 PDF 会阻塞请求）
5. 评估体系（RAGAS / 评测集）尚未实现
