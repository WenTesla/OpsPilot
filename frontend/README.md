# OpsPilot 前端（类 ChatGPT UI）

单文件、零依赖的对话前端（`index.html`），双击即可在浏览器中运行。

## 功能

- **对话**：流式输出（打字机效果）、Markdown 渲染（标题/表格/代码块/引用角标）、代码块复制、停止生成
- **会话管理**：多会话、自动标题、删除，localStorage 持久化
- **知识库（RAG 入口）**：
  - 上传入口在侧栏「知识库」区块内的「＋ 上传文档」（支持点击与拖放），仅接受 `.pdf` / `.md` / `.markdown`
  - 也可把文件拖到页面任意位置上传，拖入时上传区会高亮
  - 侧栏折叠时可用顶栏 📚 按钮展开侧栏并定位到知识库
  - 侧栏知识库列表：格式图标、解析状态（解析中/已入库/失败）、删除
  - 回答中的 `[1][2]` 渲染为引用角标，底部展示「文件名 · 页码/章节」来源
- **其他**：明暗主题切换、侧栏折叠、移动端适配

## 两种运行模式

| 模式 | 说明 |
|---|---|
| **Mock（默认）** | `index.html` 顶部 `API_BASE = ""` 时为 Mock 模式，无后端即可完整演示：上传假入库、内置运维问答（试试「CPU 使用率超过 90%」「MySQL 主从延迟」） |
| **真实后端** | 把 `API_BASE` 改为后端地址（如 `http://localhost:8000`），调用下列接口 |

## 对接的后端接口契约

详见 `../docs/RAG模块设计文档.md` 附录 A：

```
POST   /api/documents          上传 PDF/MD（multipart），返回 {doc_id, task_id, status}
GET    /api/documents          知识库列表
GET    /api/documents/{id}     文档详情（轮询解析状态）
DELETE /api/documents/{id}     删除文档（同步删索引）

POST   /api/chat               SSE 流式对话
       请求体: {conversation_id, messages:[{role, content}]}
       SSE 帧: data: {"type":"token","content":"..."}
               data: {"type":"citations","items":[{"n":1,"source":"文件名","locator":"第3页"}]}
               data: [DONE]
```

LangGraph 侧将 graph 输出以 SSE 暴露即可（`astream_events` / 自定义 streaming），citation 数据来自 RAG Tool 的检索结果（`rag_pipeline.format_context` 的结构化返回）。
