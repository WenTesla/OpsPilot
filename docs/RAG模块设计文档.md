# 智能运维 Agent（LangGraph）— RAG 模块设计文档

> 版本：v0.2（需求变更：知识源仅支持 PDF / Markdown 上传）
> 状态：设计评审中
> 范围：本档仅覆盖 RAG（检索增强生成）子系统的设计，不含告警接入、执行器、审批流等其他模块
> 变更记录：v0.2 移除 Confluence/Wiki 爬取、工单 CDC、告警字典等接入方式，知识源统一收敛为**用户上传的 PDF / MD 文件**；新增文档管理 API 契约（附录 A）与前端集成说明（第 9 节）

---

## 1. 背景与目标

### 1.1 项目背景

智能运维（AIOps）Agent 基于 LangGraph 构建，核心能力包括：告警诊断、根因分析、处置建议与自动化执行。Agent 的推理依赖企业私有的运维知识：SOP 手册、Runbook、复盘报告、产品文档等。

LLM 自身参数中不包含这些知识，且知识持续更新。因此需要一个 RAG 子系统，负责**知识的 upload（上传）→ ingest（摄入）→ index（索引）→ retrieve（检索）→ augment（增强）**全链路。

**知识接入方式（v0.2 收敛）**：用户通过 Web 前端（类 ChatGPT 交互界面）上传 **PDF 或 Markdown** 文件，系统解析入库。不支持其他格式与自动抓取。

### 1.2 设计目标

| 目标 | 说明 | 衡量方式 |
|---|---|---|
| 准确性 | 检索内容与问题高度相关，减少幻觉 | 检索命中率 Recall@5 ≥ 0.85；答案忠实度 ≥ 0.9 |
| 时效性 | 文件上传完成到可检索 ≤ 2 分钟（解析+嵌入耗时） | 上传任务状态监控 |
| 可溯源 | 每条生成结论都能引用来源文档（含页码/章节） | 答案附 citation，抽样核验 |
| 可运维 | 索引可重建、可灰度、可回滚；文档可管理（增删查） | 全量重建时间 ≤ 30 分钟；支持索引版本管理 |
| 低延迟 | 检索链路 P95 ≤ 800ms（不含 LLM 生成） | 链路 tracing |

### 1.3 非目标（Non-Goals）

- **不支持** PDF / Markdown 以外的格式（Word/Confluence/HTML/工单 CDC 等均不在范围内）；
- 不负责 Agent 的任务编排与状态机（属 LangGraph 主图范畴，本档仅定义接口）；
- 不负责实时指标/日志的查询计算（属监控数据面，通过 Tool 直连 Prometheus/日志平台，不走 RAG）；
- 不做多租户知识隔离（单团队场景），仅保留 `permission` 字段设计以备扩展。

---

## 2. 整体架构

### 2.1 RAG 在 Agent 中的位置

```
                         ┌────────────────────────────────────────────┐
                         │              LangGraph Agent               │
                         │                                            │
   告警/用户请求 ──────▶  │  ┌────────┐   ┌──────────┐   ┌─────────┐  │
                         │  │ 理解/路由 │──▶│ 规划 Planner│──▶│ 工具执行  │  │
                         │  └────────┘   └──────────┘   └────┬────┘  │
                         │                                    │       │
                         │              ┌─────────────────────┼─────┐ │
                         │              │        Tools        │     │ │
                         │              │  ┌───────┐ ┌────────▼───┐ │ │
                         │              │  │监控查询│ │  RAG 检索   │ │ │
                         │              │  │SQL/CMDB│ │(本档范围)   │ │ │
                         │              │  └───────┘ └────────────┘ │ │
                         │              └──────────────────────────┘ │
                         └────────────────────────────────────────────┘
                                    ▲                    │
                     SSE/WebSocket  │                    ▼
              ┌─────────────────────┴───────┐   ┌───────────────────────────┐
              │   Web 前端（类 ChatGPT UI）  │   │     离线/近线数据面（本档）   │
              │   对话 + 文件上传(PDF/MD)    │──▶│ 上传 API → 解析 → 清洗 →    │
              │   知识库管理（文档列表/删除）  │   │ 切分 → 嵌入 → 双写索引       │
              └─────────────────────────────┘   │ (Milvus + ES)             │
                                                └───────────────────────────┘
```

要点：

- RAG 对 Agent 暴露为**一个（或一组）Tool**，由主图通过 ReAct/规划节点调用；
- 用户通过前端上传 PDF/MD 文件，走**文档管理 API**（见附录 A）进入数据面管道；
- 数据面（ingest→index）与查询面（retrieve）解耦：查询面同步调用，数据面异步任务化（大 PDF 解析嵌入耗时长，不能阻塞上传请求）。

### 2.2 技术选型

| 组件 | 选型 | 备选 | 理由 |
|---|---|---|---|
| 编排框架 | LangGraph | — | 项目约束 |
| 向量库 | **Milvus**（自建） | Qdrant / PGVector | 支持标量过滤 + 稠密/稀疏混合检索 |
| 关键词检索 | **Elasticsearch** | Milvus BM25 | BM25 成熟、支持复杂过滤；若未部署 ES 可降级 Milvus 内置 BM25 |
| Embedding | **BGE-M3** | text-embedding-3 / GTE | 单模型同时输出 dense+sparse，中文效果好，可私有化部署 |
| Rerank | **bge-reranker-v2-m3** | Cohere Rerank | 中文跨语检索强；私有化 |
| PDF 解析 | **Unstructured**（hi_res 模式） | pdfplumber / PyMuPDF | 保留章节结构、表格转 Markdown；扫描件走 OCR 兜底 |
| MD 解析 | markdown-it + 自研清洗 | — | MD 为结构化文本，解析成本极低 |
| 异步任务 | Celery + Redis | arq / RQ | 上传后的解析嵌入任务异步化，可重试、可查进度 |
| 评估 | RAGAS + 自建评测集 | — | 社区标准 |

---

## 3. 知识接入（仅 PDF / MD 上传）

### 3.1 知识源

| 项目 | 说明 |
|---|---|
| 支持格式 | `.pdf`、`.md` / `.markdown` |
| 接入方式 | 前端上传（拖拽 / 选择文件），调用 `POST /api/documents` |
| 单文件限制 | ≤ 50MB；页数 ≤ 500 页 |
| 元数据 | 上传时可选填：标题、类别（`sop` / `postmortem` / `doc` / `other`）、关联服务、适用环境 |
| 拒绝项 | 加密 PDF（无法解析）、扫描件无文字层且 OCR 失败、空文档 |

### 3.2 Ingestion 管道（异步任务）

```
前端上传 PDF/MD
   │  POST /api/documents（立即返回 task_id，状态 processing）
   ▼
① Parser（PDF: Unstructured 保留章节/表格；MD: 直接读文本 + front-matter 解析）
   ▼
② Normalizer（统一为内部 Doc 格式，记录页码/章节定位）
   ▼
③ Cleaner（去噪、脱敏）
   ▼
④ Chunker（按格式路由的切分策略，见第 4 节）
   ▼
⑤ Embedder（BGE-M3 批量向量化）
   ▼
⑥ Indexer（双写 Milvus + ES，带 index_version）
   ▼
状态更新为 ready（前端轮询 / SSE 推送，知识库列表可见）
```

统一的内部文档格式：

```json
{
  "doc_id": "hash(file_content)",
  "filename": "MySQL故障处理手册.pdf",
  "source": "upload",
  "format": "pdf | md",
  "locator": "页码或 heading 锚点（citation 跳转用）",
  "doc_type": "sop | postmortem | doc | other",
  "title": "用户填写或取自文档首标题",
  "content": "解析后全文",
  "service": ["order-service", "mysql"],
  "env": ["prod", "staging"],
  "uploaded_by": "user_id",
  "updated_at": 1727000000,
  "ingest_version": "2026-09-26T10:00:00Z"
}
```

**关键机制：**

1. **幂等**：以文件内容 hash 为 `doc_id`；重复上传同内容文件直接返回已存在，不重嵌入；
2. **索引版本**：全量重建生成新 `index_version`，写入新 collection（如 `rag_chunks_v7`），别名（alias）原子切换，灰度与秒级回滚；
3. **删除**：用户在前端知识库列表删除文档 → 同步删除 ES/Milvus 中对应 chunk；
4. **失败处理**：解析/嵌入任务失败自动重试 3 次，仍失败则状态置 `failed` 并记录原因，前端可见并可重新上传。

### 3.3 数据清洗

- PDF：去除页眉页脚、水印文字、目录页（Unstructured 结构标签 + 启发式规则）；
- MD：去除 HTML 注释、折叠区块（`<details>`）展开为正文；
- **近似去重**：SimHash 判定与已有文档重复，提示用户确认覆盖或跳过；
- **低质过滤**：解析后正文 < 100 字的文档标记为「疑似空文档」，不入索引；
- 敏感信息脱敏：IP、token、密码、手机号（正则 + NER 兜底），在 Normalizer 阶段统一完成，**索引中不存在明文敏感信息**。

---

## 4. 切分（Chunking）策略

按文件**格式**与**类别**路由：

| 格式/类别 | 切分策略 | 说明 |
|---|---|---|
| Markdown | **结构感知**：按标题层级（`#`~`####`）切分，section 过长（>512 token）再按段落二切 | 保留 Heading Path（如「MySQL 故障处理 > 主从延迟 > 排查步骤」）存入 chunk 元数据，检索时拼回上下文；MD 代码块整体保留不切断 |
| PDF（SOP/手册类） | 结构感知：按解析出的章节层级切分 | Unstructured 输出 section 结构；每个 chunk 记录**页码范围**，citation 可定位到页 |
| PDF（复盘报告类） | 按章节（时间线/根因/改进项）切分 | 时间线整体保留，避免割裂因果 |
| 兜底 | 递归字符切分，chunk 512 token，overlap 64 | 无结构的纯文本 PDF |

**通用参数：**

- 目标 chunk 大小：**256~512 token**（中文），overlap 10%~15%；
- 每个 chunk 携带完整元数据：`doc_id / filename / locator(页码或标题锚) / title / heading_path / doc_type / service / env / updated_at / index_version`；
- **Contextual Retrieval（上下文增强）**：为每个 chunk 用轻量 LLM 生成 1~2 句「该片段在全文中的位置说明」，前置拼接后再嵌入。离线一次性成本，可显著提升检索命中率（M4 再开启，以自测评测为准）。

---

## 5. 检索策略（核心）

### 5.1 检索流程总览

```
用户/Agent 查询
   │
   ▼
① Query 理解（实体识别 → 元数据过滤条件；Agent 场景下 query 已由主 LLM 改写）
   │
   ▼
② 混合检索（并行）
   ├── 稠密向量检索（Milvus, top 20, 带元数据过滤）
   └── BM25 关键词检索（ES, top 20, 同过滤条件）
   │
   ▼
③ 融合去重（RRF, Reciprocal Rank Fusion, k=60）
   │
   ▼
④ Rerank（bge-reranker-v2-m3, top 20 → top 5）
   │
   ▼
⑤ 上下文组装（按 doc_type 分组、citation 标注、token 预算裁剪）
   │
   ▼
返回 Agent：{contexts[], citations[], scores[]}
```

### 5.2 查询理解

运维查询常含：告警名（`CPU使用率超过90%`）、服务名、指标名、错误码。两种手段：

1. **实体识别（轻量）**：正则 + 词典抽取服务名、告警级别、环境，转为**元数据过滤条件**而非全文匹配；
2. **查询改写**：Agent 场景下由主 LLM 在调用 tool 时生成自包含查询（LangGraph 的 tool call 天然携带改写后的 query，无需单独改写模型）。

HyDE 在纯上传文档（无工单 Q/A 对）场景下收益有限，**暂不引入**，留作后续实验项。

### 5.3 混合检索与融合

- **为什么混合**：告警名、错误码、配置项名称等**精确术语**靠 BM25 命中；自然语言描述的故障现象靠向量语义命中。纯任一方案在运维语料上都有明显盲区；
- **RRF 融合**：`score(d) = Σ 1/(k + rank_i(d))`，k=60，无需归一化两路分数。

### 5.4 元数据过滤（运维场景的关键）

向量检索不是裸搜，**先过滤再检索**（Milvus 支持带过滤的 ANN）：

```python
filter_expr = (
    f'service in ["order-service"] && '      # 来自查询实体识别
    f'env in ["prod"] && '
    f'index_version == "{CURRENT_ALIAS}"'
)
```

无法识别实体时不强加过滤（避免零召回），改为检索后软加权：命中 `service` 的结果分数 ×1.2。

### 5.5 Rerank

- Rerank 是**性价比最高的一环**：top20→top5，延迟约 100~200ms（GPU）；
- Rerank 输入 = query + chunk（含 heading_path 前缀），输出重排分数；
- 阈值兜底：重排后最高分 < 0.3 时返回「知识库无高置信结果」，**让 Agent 显式走降级路径**（转人工/仅用 LLM 知识并声明无引用），而不是硬塞低质上下文。

### 5.6 上下文组装

- Token 预算：默认 4K（可按主模型配置）；
- 多样性：同一 `doc_id` 最多取 2 个 chunk，保证信息面；
- 排序：按 rerank 分数降序，sop > postmortem > doc > other（doc_type 权重）；
- 输出结构化，每条 context 附 `[1][2]...` 编号与来源（文件名 + 页码/章节），供 LLM 引用并由前端渲染为可点击 citation。

---

## 6. 与 LangGraph 的集成

### 6.1 定位

RAG 检索封装为 **Tool**，注入主图的工具节点。Agent 侧无须感知 Milvus/ES 的存在。

### 6.2 State 定义（主图相关片段）

```python
from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class OpsPilotState(TypedDict):
    messages: Annotated[list, add_messages]        # 对话/工具消息
    alert: dict                                    # 触发告警的上下文
    retrieved: list[dict]                          # RAG 结果缓存（供后续节点复用）
    diagnosis: str | None                          # 诊断结论
```

### 6.3 RAG Tool 封装

```python
from langchain_core.tools import tool

@tool
def search_ops_knowledge(query: str, top_k: int = 5) -> str:
    """检索运维知识库（用户上传的 SOP、手册、复盘报告等 PDF/MD 文档）。

    当需要故障处置步骤、历史相似案例、根因分析参考时调用。
    返回带来源引用（文件名+页码/章节）的知识片段。
    """
    # 不做元数据过滤：LLM 推断的 service/env 实体不可靠，硬过滤会直接把候选清空导致零召回
    result = rag_pipeline.retrieve(query=query, top_k=top_k)
    if not result.hits:
        return "知识库中未找到高置信相关内容。建议：转人工或基于告警指标进一步排查。"
    return rag_pipeline.format_context(result)   # 含 [n] 引用与来源定位
```

### 6.4 主图接入示例

```python
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

tools = [search_ops_knowledge, query_prometheus, query_cmdb]
tool_node = ToolNode(tools)

def should_continue(state: OpsPilotState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

graph = StateGraph(OpsPilotState)
graph.add_node("agent", llm_with_tools_node)   # 绑定 tools 的 LLM 节点
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile(checkpointer=...)          # 生产建议接持久化 checkpointer
```

### 6.5 设计取舍说明

| 决策 | 选择 | 理由 |
|---|---|---|
| RAG 作为 Tool vs 固定前置节点 | **Tool** | 诊断类问题才需要检索；闲聊/纯指标查询跳过，省延迟与成本 |
| 检索结果进 messages vs State | **两者** | messages 给 LLM 消费；`state.retrieved` 留档供评估与后续节点复用，避免重复检索 |
| 单一检索 Tool vs 按 doc_type 拆多 Tool | **先单一 + doc_type 参数** | 拆分会让 LLM 选择负担大；若评测发现误传率高再拆 |

---

## 7. 评估方案

### 7.1 评测集建设

- 从已上传文档对应的真实运维问题中整理 **200+ 条**，人工标注：问题、期望命中的文档 chunk、期望答案要点；
- 覆盖分布：SOP 查询 50% / 复盘报告 30% / 无答案（应拒答）20%。

### 7.2 指标体系

| 层 | 指标 | 工具 | 目标 |
|---|---|---|---|
| 检索 | Recall@5 / MRR | 自建脚本 | ≥0.80 / ≥0.70 |
| 检索 | 拒答准确率（无答案时正确返回空） | 自建 | ≥0.80 |
| 生成 | 忠实度 Faithfulness / 答案相关性 | RAGAS | ≥0.90 / ≥0.85 |
| 工程 | 端到端检索 P95 延迟 | Tracing | ≤800ms |
| 业务 | Agent 采用 RAG 结果后的处置建议采纳率 | 上线后人工标注 | 持续观测 |

### 7.3 回归机制

- 评测集进 Git，任何检索策略变更（切分参数/过滤逻辑/rerank 阈值）必须跑回归，报告对比；
- 线上**采样记录**每次检索的 query、hits、最终答案是否被引用，周报自动生成 badcase 清单回流评测集。

---

## 8. 工程化与运维

- **延迟预算分解**（P95 目标 800ms）：向量检索 ≤150ms + BM25 ≤150ms（并行取 max）+ rerank ≤300ms + 组装 ≤50ms + 网络/序列化余量；
- **缓存**：query 归一化后的 embedding 结果 LRU 缓存；热门告警类型的检索结果短 TTL（5min）缓存；
- **可观测**：每次检索生成 trace（含过滤条件、两路召回数、融合后数量、rerank 分数分布），接现有 APM；上传任务有独立 task 状态日志；
- **容量**：按纯上传模式预估初版 10 万 chunk 以内，Milvus 单副本即可；
- **安全**：向量库与 ES 内网部署；上传文件落对象存储，服务器端二次校验扩展名与 MIME，防伪造上传。

---

## 9. 前端集成（类 ChatGPT UI）

前端为独立的 Web 应用（详见 `frontend/`，含可运行 Demo），与 RAG 相关的交互面：

| 功能 | 前端行为 | 对接接口 |
|---|---|---|
| 文件上传 | 输入框附件按钮 / 拖拽，仅接受 `.pdf` `.md` | `POST /api/documents` |
| 知识库管理 | 侧栏「知识库」列表：文件名、类别、状态（processing/ready/failed）、删除 | `GET/DELETE /api/documents` |
| 引用展示 | 回答中 `[1][2]` 渲染为可点击角标，悬浮显示「文件名 · 页码/章节」 | citation 元数据来自 RAG 检索结果 |
| 对话 | 流式输出（SSE），与 LangGraph 后端 `/api/chat` 对接 | LangGraph 侧以 SSE 暴露 token 流 |

Demo 版前端内置 Mock 模式（无后端时可独立运行演示），通过修改配置切换到真实后端地址。

---

## 10. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| M1 | 上传 API + PDF/MD 解析入库 + 纯向量检索 + Tool 接入主图 + 前端上传与对话 Mock | 前端上传→入库→「查 SOP→给步骤」闭环 |
| M2 | 混合检索 + Rerank + 元数据过滤 + citation 定位到页码/章节 | 评测集 Recall@5 ≥ 0.80 |
| M3 | 评估体系 + 索引版本化/灰度 + 缓存与 tracing + 前端接真实后端 | 回归流水线上线，SSE 流式对话可用 |
| M4 | Contextual Retrieval、拒答优化、文档管理完善（重复检测/覆盖确认） | 全部设计目标达成 |

---

## 附录 A：文档管理 API 契约（前端对接用）

```
POST /api/documents
  multipart/form-data: file(.pdf|.md)
  → 202 {"doc_id": "...", "task_id": "...", "status": "processing"}

GET /api/documents
  → 200 {"items": [{"doc_id", "filename", "format", "status",
                     "chunk_count", "updated_at"}]}

GET /api/documents/{doc_id}
  → 200 文档详情（含解析状态、失败原因）

DELETE /api/documents/{doc_id}
  → 204（同步删除索引中对应 chunk）
```

状态机：`processing → ready | failed`（failed 可查看原因，支持重新上传）。

## 附录 B：术语表

| 术语 | 说明 |
|---|---|
| RRF | Reciprocal Rank Fusion，多路检索结果按排名倒数融合 |
| Rerank | 用交叉编码器对候选粗排结果精排 |
| Contextual Retrieval | 为 chunk 生成全文上下文说明后再嵌入的技术 |
| SSE | Server-Sent Events，服务端向浏览器推送流式响应 |
