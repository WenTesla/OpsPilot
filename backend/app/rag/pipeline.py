"""RAG 管道：入库（ingest）与检索（retrieve）。

对外接口与文档第 3 / 5 节一致：
    ingest_file(...) -> chunk 数量
    retrieve(query, filters, top_k) -> SearchResult
    format_context(result) -> (给 LLM 的上下文文本, 给前端的 citations)
"""
from __future__ import annotations

import time
from contextvars import ContextVar

from .. import config
from .embed import Embedder
from .parse import chunk_blocks, parse_file, Chunk
from .store import ChunkStore, SearchResult

# 单次请求内收集引用（并发安全）
CURRENT_CITATIONS: ContextVar[list | None] = ContextVar("opsagent_citations", default=None)


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = ChunkStore(self.embedder, self.embedder.dim)

    # ---------------- 入库 ----------------
    def ingest_file(
        self,
        path: str,
        filename: str,
        fmt: str,
        doc_id: str,
        doc_type: str = "other",
        service: list[str] | None = None,
        env: list[str] | None = None,
    ) -> int:
        # 幂等：同 doc_id 重新入库前先清掉旧 chunk
        self.store.delete_by_doc(doc_id)
        blocks = parse_file(path, fmt)
        chunks: list[Chunk] = chunk_blocks(
            blocks, doc_id=doc_id, filename=filename, doc_type=doc_type,
            service=service or [], env=env or [],
            target=config.CHUNK_TARGET_CHARS, overlap=config.CHUNK_OVERLAP_CHARS,
        )
        return self.store.add(chunks, [c.text for c in chunks])

    def delete_doc(self, doc_id: str) -> int:
        return self.store.delete_by_doc(doc_id)

    # ---------------- 检索 ----------------
    def retrieve(self, query: str, filters: dict | None = None, top_k: int | None = None) -> SearchResult:
        t0 = time.time()
        hits = self.store.search(query, filters=filters or {}, top_k=top_k or config.TOP_K)
        return SearchResult(hits=hits, took_ms=int((time.time() - t0) * 1000))

    # ---------------- 上下文组装 ----------------
    @staticmethod
    def format_context(result: SearchResult) -> tuple[str, list[dict]]:
        if not result.hits:
            return "知识库中未找到高置信相关内容。", []
        parts, citations = [], []
        for i, h in enumerate(result.hits, start=1):
            loc = h.heading_path or h.locator or ""
            head = f"[{i}] 《{h.filename}》 {loc}".strip()
            parts.append(f"{head}\n{h.text}")
            citations.append({"n": i, "source": h.filename, "locator": loc, "score": round(h.score, 4)})
        header = f"（以下是从知识库检索到的 {len(result.hits)} 条资料，引用格式为 [编号]）\n\n"
        return header + "\n\n---\n\n".join(parts), citations

    def stats(self) -> dict:
        return {
            "embedding": self.embedder.describe,
            "rerank": config.USE_RERANK,
            **self.store.stats(),
        }


def collect_citations(citations: list[dict]) -> None:
    """Tool 调用时把引用写入当前请求上下文，供 SSE 返回给前端。"""
    box = CURRENT_CITATIONS.get()
    if box is not None:
        for c in citations:
            if c not in box:
                box.append(c)
