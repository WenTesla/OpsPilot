"""OpsPilot 后端：FastAPI + LangGraph + RAG。

接口（与 docs/RAG模块设计文档.md 附录 A 一致）：
    GET    /api/health
    POST   /api/documents          上传 PDF / MD
    GET    /api/documents          知识库列表
    GET    /api/documents/{doc_id} 文档详情（含解析状态）
    DELETE /api/documents/{doc_id} 删除文档（同步删索引）
    POST   /api/chat               SSE 流式对话
前端静态资源挂在 / 下，启动后浏览器访问 http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from . import config
from .agent.graph import OpsPilot
from .agent.llm import MockChatModel, get_chat_model
from .mcp.client import load_tools
from .mcp.tools import CURRENT_TOOL_EVENTS
from .rag.pipeline import CURRENT_CITATIONS, RAGPipeline
from .storage import docs as doc_store

pipeline = RAGPipeline()
MODEL = get_chat_model()

# MCP 工具在 lifespan 里初始化 —— 不能在模块层 await：
# uvicorn --reload 时 app 的 import 发生在已运行的事件循环内部，
# 模块层调用 asyncio.run() 会抛 "cannot be called from a running event loop"。
MCP_TOOLS: list = []
MCP_STATUS: dict = {"enabled": False, "servers": {}, "tools": 0, "reason": "初始化中"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时挂载 MCP 活数据工具；失败则保持纯 RAG（等同 v0.1）。"""
    global MCP_TOOLS, MCP_STATUS, agent
    MCP_TOOLS, MCP_STATUS = await load_tools()
    if MCP_STATUS["enabled"]:
        print(f"[mcp] ✓ 已挂载 {MCP_STATUS['tools']} 个活数据工具："
              f"{', '.join(t.name for t in MCP_TOOLS)}", flush=True)
    else:
        print(f"[mcp] × 活数据工具未启用（{MCP_STATUS.get('reason') or '未知原因'}），行为等同 v0.1", flush=True)
    agent = OpsPilot(pipeline, extra_tools=MCP_TOOLS)
    yield


app = FastAPI(title="OpsPilot API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def _rebuild_index_if_needed() -> None:
    """切换 EMBED_BACKEND 后旧向量维度不匹配会失效（见 store.load 的校验），这里用
    `files/` 下的原文件自动重建，避免「界面上文档都在，检索却永远返回空」。

    典型触发场景：BGE-M3(1024 维) 切到 OpenAI(1536 维)，或反之。
    """
    docs = doc_store.list_docs()
    if not docs:
        return
    indexed = {r["doc_id"] for r in pipeline.store.records}
    missing = [d for d in docs if d.get("status") == "ready" and d["doc_id"] not in indexed]
    if not missing:
        return

    print(f"[rag] {len(missing)} 份文档缺少向量索引，开始重建…", flush=True)
    for d in missing:
        src = Path(d.get("path") or "")
        if not src.is_file():
            doc_store.update_doc(d["doc_id"], status="failed", error="索引已失效且原文件缺失，请重新上传")
            print(f"[rag] × {d['filename']}：原文件缺失（{src}）", flush=True)
            continue
        try:
            t0 = time.time()
            n = pipeline.ingest_file(
                str(src), d["filename"], d.get("format", "md"), d["doc_id"],
            )
            doc_store.update_doc(d["doc_id"], status="ready", chunk_count=n,
                                 cost_ms=int((time.time() - t0) * 1000))
            print(f"[rag] ✓ {d['filename']}：{n} chunks", flush=True)
        except Exception as e:
            doc_store.update_doc(d["doc_id"], status="failed", error=str(e)[:300])
            print(f"[rag] × {d['filename']}：{e}", flush=True)


_rebuild_index_if_needed()


# 默认值（纯知识库），lifespan 启动后按需重建为「RAG + MCP 工具」版本。
# 这样即便 lifespan 阶段出错，Agent 也不会是未定义的悬空引用。
agent = OpsPilot(pipeline)


# ============================ 健康检查 ============================
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "mock-ops-model" if isinstance(MODEL, MockChatModel) else config.LLM_MODEL,
        "rag": pipeline.stats(),
        "docs": len(doc_store.list_docs()),
        "mcp": MCP_STATUS,
    }


# ============================ 文档管理 ============================
@app.post("/api/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_EXT:
        raise HTTPException(400, f"不支持的格式：{suffix or '未知'}（仅 PDF / MD）")

    content = await file.read()
    if len(content) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"文件超过 {config.MAX_UPLOAD_MB}MB")

    doc_id = hashlib.md5(content).hexdigest()[:16]
    if doc_store.get_doc(doc_id):
        return JSONResponse({"doc_id": doc_id, "status": "ready", "duplicate": True}, status_code=200)

    fmt = "pdf" if suffix == ".pdf" else "md"
    saved = config.FILES_DIR / f"{doc_id}_{file.filename}"
    with open(saved, "wb") as f:
        f.write(content)

    doc = doc_store.add_doc({
        "doc_id": doc_id, "filename": file.filename, "format": fmt,
        "status": "processing", "chunk_count": 0, "size": len(content),
        "path": str(saved),
    })

    t0 = time.time()
    try:
        n = pipeline.ingest_file(
            str(saved), file.filename, fmt, doc_id,
        )
        doc_store.update_doc(doc_id, status="ready", chunk_count=n, cost_ms=int((time.time() - t0) * 1000))
    except Exception as e:
        doc_store.update_doc(doc_id, status="failed", error=str(e)[:300])
        raise HTTPException(500, f"解析失败：{e}")

    return {"doc_id": doc_id, "status": "ready", "chunk_count": n, "cost_ms": int((time.time() - t0) * 1000)}


@app.get("/api/documents")
async def list_documents():
    return {"items": doc_store.list_docs()}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    d = doc_store.get_doc(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    return d


@app.delete("/api/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str):
    d = doc_store.remove_doc(doc_id)
    if not d:
        raise HTTPException(404, "文档不存在")
    pipeline.delete_doc(doc_id)
    try:
        Path(d.get("path", "")).unlink(missing_ok=True)
    except Exception:
        pass
    return None


# ============================ 对话（SSE） ============================
class ChatRequest(BaseModel):
    conversation_id: str = "default"
    messages: list[dict]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _chunk_text(msg) -> str:
    """从 LangGraph 消息流中取出可展示的文本增量。"""
    if not isinstance(msg, AIMessageChunk):
        return ""
    if getattr(msg, "tool_calls", None):
        return ""
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_text = ""
    for m in reversed(req.messages):
        if m.get("role") == "user":
            user_text = (m.get("content") or "").strip()
            break
    if not user_text:
        raise HTTPException(400, "缺少用户消息")

    async def gen():
        box: list[dict] = []
        tool_box: list[dict] = []
        tool_sent = 0
        token = CURRENT_CITATIONS.set(box)
        tool_token = CURRENT_TOOL_EVENTS.set(tool_box)
        streamed = ""
        try:
            yield _sse({"type": "status", "content": "正在分析…"})
            async for mode, payload in agent.stream(req.conversation_id, user_text):
                if mode == "messages":
                    msg = payload[0] if isinstance(payload, tuple) else payload
                    text = _chunk_text(msg)
                    if text:
                        streamed += text
                        # 一次性拿到整段（如 Mock 模型）时也切分成小块，保证前端打字机效果
                        for piece in (_split(text) if len(text) > 40 else [text]):
                            if piece:
                                yield _sse({"type": "token", "content": piece})
                                await asyncio.sleep(0.012)
                elif mode == "updates":
                    # 工具节点跑完后，把这一轮新增的工具事件推给前端
                    has_new = len(tool_box) > tool_sent
                    while tool_sent < len(tool_box):
                        yield _sse(tool_box[tool_sent])
                        tool_sent += 1
                    if has_new and isinstance(payload, dict) and "tools" in payload:
                        yield _sse({"type": "status", "content": "已完成工具调用，正在生成结论…"})
        except Exception as e:
            yield _sse({"type": "error", "content": f"生成失败：{e}"})
        finally:
            final = ""
            try:
                st = await agent.final_state(req.conversation_id)
                msgs = st.values.get("messages", [])
                last = msgs[-1] if msgs else None
                if last is not None and not getattr(last, "tool_calls", None):
                    final = last.content if isinstance(last.content, str) else ""
            except Exception:
                pass

            # 未拿到流式 token（例如 Mock 模型一次性返回）时，按字切分模拟打字机效果
            rest = final[len(streamed):] if final.startswith(streamed) else final
            for piece in _split(rest):
                if piece:
                    yield _sse({"type": "token", "content": piece})
                await asyncio.sleep(0.012)

            citations = sorted(box, key=lambda c: c.get("n", 0))
            yield _sse({"type": "citations", "items": citations})
            yield _sse({"type": "tools", "items": tool_box})
            yield _sse({"type": "done"})
            yield "data: [DONE]\n\n"
            CURRENT_CITATIONS.reset(token)
            CURRENT_TOOL_EVENTS.reset(tool_token)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ============================ MCP 活数据工具 ============================
@app.get("/api/mcp/status")
async def mcp_status():
    """当前 MCP 工具的挂载情况。"""
    return {
        **MCP_STATUS,
        "tools": [{"name": t.name, "description": (t.description or "")[:120]} for t in MCP_TOOLS],
    }


@app.post("/api/mcp/reload")
async def mcp_reload():
    """MCP 服务比主服务晚起来时，用它免重启重连。"""
    global MCP_TOOLS, MCP_STATUS, agent
    from .mcp import client as mcp_client

    tools, status = await mcp_client.reload_tools()
    MCP_TOOLS, MCP_STATUS = tools, status
    # 用新工具集重建 Agent（会话历史在 checkpointer 里，thread_id 不变故不影响）
    agent = OpsPilot(pipeline, extra_tools=MCP_TOOLS)
    if status["enabled"]:
        print(f"[mcp] ✓ 重新挂载 {status['tools']} 个工具", flush=True)
    else:
        print(f"[mcp] × 重连失败：{status.get('reason')}", flush=True)
    return {"ok": status["enabled"], "status": status}


def _split(text: str, size: int = 6):
    return [text[i:i + size] for i in range(0, len(text), size)]


# ============================ 静态前端 ============================
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
