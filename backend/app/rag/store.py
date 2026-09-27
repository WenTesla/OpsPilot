"""本地检索存储：稠密向量（余弦）+ BM25（稀疏）+ RRF 融合 + 可选 Rerank。

设计上对应文档第 5 节的检索流程；生产环境把本类替换为 Milvus + ES 即可，
对外接口（add / delete / search）保持不变。
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, field, is_dataclass

import numpy as np

from .. import config
from .embed import tokenize


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    filename: str
    locator: str
    heading_path: str
    doc_id: str


class ChunkStore:
    def __init__(self, embedder, dim: int):
        self.embedder = embedder
        self.dim = dim
        self.records: list[dict] = []                       # chunk 元数据 + text
        self.vecs = np.zeros((0, dim), dtype="float32")
        self._bm25: dict[str, dict] = {"df": {}, "len": [], "avg": 0.0, "n": 0}
        self._lock = threading.RLock()
        self.load()

    # ---------------- 写入 ----------------
    def add(self, chunks, texts: list[str]) -> int:
        with self._lock:
            if not chunks:
                return 0
            vecs = self.embedder.embed(texts)
            self.records.extend(_asdict(c) for c in chunks)
            self.vecs = np.vstack([self.vecs, vecs]) if self.vecs.size else vecs
            self._rebuild_bm25()
            self.save()
            return len(chunks)

    def delete_by_doc(self, doc_id: str) -> int:
        with self._lock:
            keep_idx = [i for i, r in enumerate(self.records) if r["doc_id"] != doc_id]
            removed = len(self.records) - len(keep_idx)
            if removed:
                self.records = [self.records[i] for i in keep_idx]
                self.vecs = self.vecs[keep_idx] if self.vecs.size else self.vecs
                self._rebuild_bm25()
                self.save()
            return removed

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or config.TOP_K
        with self._lock:
            if not self.records:
                return []
            cand = list(range(len(self.records)))

            dense_rank = self._dense_rank(query, cand)
            sparse_rank = self._bm25_rank(query, cand)
            fused = _rrf([dense_rank, sparse_rank], k=config.RRF_K)
            ranked = [i for i, _ in fused][: config.CANDIDATE_K]

            scores, covers = self._rescore(query, ranked, fused)
            # 相关性门槛：过滤与问题几乎无关的候选，避免「凑够 Top-K」污染上下文
            keep = [j for j in range(len(ranked)) if covers[j] >= config.MIN_COVER]
            if not keep:
                return []
            order = sorted(keep, key=lambda j: scores[j], reverse=True)

            # 多样性：同文档最多 2 个 chunk，且去掉重复文本
            hits: list[Hit] = []
            per_doc: dict[str, int] = {}
            seen: set[str] = set()
            for j in order:
                r = self.records[ranked[j]]
                key = r["text"][:80]
                if key in seen:
                    continue
                if per_doc.get(r["doc_id"], 0) >= 2:
                    continue
                seen.add(key)
                per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
                hits.append(Hit(
                    chunk_id=r["chunk_id"], text=r["text"], score=float(scores[j]),
                    filename=r.get("filename", ""), locator=r.get("locator", ""),
                    heading_path=r.get("heading_path", ""), doc_id=r["doc_id"],
                ))
                if len(hits) >= top_k:
                    break
            return hits

    # ---- 各路召回 ----
    def _dense_rank(self, query: str, cand: list[int]) -> list[int]:
        """稠密向量召回（BGE-M3 / OpenAI 语义向量）。

        向量已 L2 归一化，点积即余弦。阈值 0.05 只用于挡掉完全无关的块，
        真正的截断靠 CANDIDATE_K；相关性把关在 _rescore 的 IDF 覆盖率 + MIN_COVER。
        """
        qv = self.embedder.embed([query])[0]
        mat = self.vecs[cand]
        sims = mat @ qv
        order = np.argsort(-sims)
        return [cand[i] for i in order if sims[i] > 0.05][: config.CANDIDATE_K]

    def _bm25_rank(self, query: str, cand: list[int]) -> list[int]:
        q = set(tokenize(query))
        if not q:
            return []
        df, avg, n = self._bm25["df"], self._bm25["avg"], self._bm25["n"]
        scored = []
        for i in cand:
            tf_map: dict[str, int] = {}
            for t in self._token_cache(i):
                tf_map[t] = tf_map.get(t, 0) + 1
            dl = len(self._token_cache(i)) or 1
            s = 0.0
            for t in q:
                if t not in tf_map:
                    continue
                idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                tf = tf_map[t]
                s += idf * (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / (avg or 1)))
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored[: config.CANDIDATE_K]]

    def _token_cache(self, i: int) -> list[str]:
        r = self.records[i]
        if "_tok" not in r:
            r["_tok"] = tokenize(r["text"])
        return r["_tok"]

    def _rebuild_bm25(self) -> None:
        df: dict[str, int] = {}
        lens = []
        for r in self.records:
            toks = tokenize(r["text"])
            r["_tok"] = toks
            lens.append(len(toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._bm25 = {"df": df, "len": lens, "avg": (sum(lens) / len(lens)) if lens else 0.0, "n": len(lens)}

    def _rescore(self, query: str, ranked: list[int], fused: list[tuple[int, float]]) -> tuple[list[float], list[float]]:
        """重排：配置了 CrossEncoder 用模型，否则用『融合分 + 字面覆盖率』兜底。

        返回 (scores, covers)。covers 为 IDF 加权查询覆盖率，用于相关性门槛过滤；
        使用 CrossEncoder 时覆盖率无意义，统一返回 1.0（即不做该过滤）。
        """
        fuse_map = dict(fused)
        base = [fuse_map.get(i, 0.0) for i in ranked]
        if config.USE_RERANK:
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(config.RERANK_MODEL)
                pairs = [[query, self.records[i]["text"]] for i in ranked]
                scores = model.predict(pairs)
                return [float(s) for s in scores], [1.0] * len(ranked)
            except Exception:
                pass
        # 兜底重排：融合分 + IDF 加权的查询覆盖率（避免"使用""定位"这类高频词乱入）
        df, n = self._bm25["df"], max(self._bm25["n"], 1)

        def idf(t: str) -> float:
            return math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))

        q = set(tokenize(query))
        q_norm = sum(idf(t) for t in q) or 1.0
        out, covers = [], []
        for j, i in enumerate(ranked):
            toks = set(self._token_cache(i))
            cover = sum(idf(t) for t in (q & toks)) / q_norm
            covers.append(cover)
            out.append(base[j] * 5.0 + cover * 1.0)
        return out, covers

    # ---------------- 持久化 ----------------
    def save(self) -> None:
        with open(config.CHUNKS_JSONL, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps({k: v for k, v in r.items() if k != "_tok"}, ensure_ascii=False) + "\n")
        np.save(config.VECTORS_NPY, self.vecs)
        with open(config.BM25_JSON, "w", encoding="utf-8") as f:
            json.dump(self._bm25, f, ensure_ascii=False)

    def load(self) -> None:
        if not config.CHUNKS_JSONL.exists() or not config.VECTORS_NPY.exists():
            return
        try:
            with open(config.CHUNKS_JSONL, "r", encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]
            self.vecs = np.load(config.VECTORS_NPY)
            # 条目数或向量维度对不上，说明索引来自另一个 EMBED_BACKEND（如 openai=1536 vs bge=1024），
            # 或者 data 目录被外部删过一部分 —— 直接判为失效重建，否则后续 add/delete/检索都会炸。
            if (
                self.vecs.ndim != 2
                or self.vecs.shape[0] != len(self.records)
                or self.vecs.shape[1] != self.dim
            ):
                self.records, self.vecs = [], np.zeros((0, self.dim), dtype="float32")
                return
            self._rebuild_bm25()
        except Exception:
            self.records, self.vecs = [], np.zeros((0, self.dim), dtype="float32")

    def stats(self) -> dict:
        return {"chunks": len(self.records), "dim": int(self.vecs.shape[1]) if self.vecs.ndim == 2 else 0}


def _asdict(c) -> dict:
    return asdict(c) if is_dataclass(c) else dict(c)


def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。"""
    score: dict[int, float] = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


@dataclass
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    took_ms: int = 0
