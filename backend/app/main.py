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
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel

from . import config
from .agent.graph import OpsPilot
from .agent.llm import MockChatModel, get_chat_model
from .rag.pipeline import CURRENT_CITATIONS, RAGPipeline
from .storage import docs as doc_store

app = FastAPI(title="OpsPilot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

pipeline = RAGPipeline()
agent = OpsPilot(pipeline)
MODEL = get_chat_model()


# ============================ 健康检查 ============================
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "mock-ops-model" if isinstance(MODEL, MockChatModel) else config.LLM_MODEL,
        "rag": pipeline.stats(),
        "docs": len(doc_store.list_docs()),
    }


# ============================ 文档管理 ============================
@app.post("/api/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("other"),
    service: str = Form(""),
    env: str = Form(""),
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
        "doc_type": doc_type if doc_type in config.CHUNK_DOC_TYPES else "other",
        "service": [s.strip() for s in service.split(",") if s.strip()],
        "env": [s.strip() for s in env.split(",") if s.strip()],
        "status": "processing", "chunk_count": 0, "size": len(content),
        "path": str(saved),
    })

    t0 = time.time()
    try:
        n = pipeline.ingest_file(
            str(saved), file.filename, fmt, doc_id,
            doc_type=doc["doc_type"], service=doc["service"], env=doc["env"],
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
        token = CURRENT_CITATIONS.set(box)
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
                    if isinstance(payload, dict) and "tools" in payload:
                        yield _sse({"type": "status", "content": "已检索知识库，正在生成结论…"})
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
            yield _sse({"type": "done"})
            yield "data: [DONE]\n\n"
            CURRENT_CITATIONS.reset(token)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _split(text: str, size: int = 6):
    return [text[i:i + size] for i in range(0, len(text), size)]


# ============================ 静态前端 ============================
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
