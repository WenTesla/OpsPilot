"""Embedding 适配器（三级降级，保证任何环境都能跑起来）。

优先级（EMBED_BACKEND=auto）：
  1. local  : sentence-transformers 加载 BGE-M3（效果最好，需下载模型）
  2. openai : OpenAI 兼容 /embeddings 接口（需 OPENAI_API_KEY）
  3. hash   : 纯本地哈希向量（无需任何依赖/网络，配合 BM25 使用）

说明：hash 向量不是语义向量，只表达字面相似；此模式下检索主要靠 BM25 兜底，
      语义召回会弱一些。接入 local/openai 后端后效果显著提升。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .. import config


def _resolve_model_path(name: str) -> str:
    """EMBED_MODEL 既可以是 HuggingFace repo id，也可以是本地目录。

    本地目录支持绝对路径，或相对 backend/ 的路径（如 `models/bge-m3`）——
    走本地目录可以绕开 HF cache 在 Windows 上建符号链接失败、产出 0 字节文件的问题。
    """
    p = Path(name)
    if p.is_dir():
        return str(p)
    if not p.is_absolute() and (config.BASE_DIR / p).is_dir():
        return str(config.BASE_DIR / p)
    return name            # 当 repo id 处理，由 sentence-transformers 去下载


class Embedder:
    def __init__(self, backend: str | None = None):
        self.backend = (backend or config.EMBED_BACKEND).lower()
        self.dim = 384
        self._model = None
        self._init()

    # -------- 初始化 --------
    def _init(self) -> None:
        if self.backend in ("auto", "local"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(_resolve_model_path(config.EMBED_MODEL))
                # sentence-transformers ≥5 把 get_sentence_embedding_dimension 改名了，两者都兼容
                get_dim = getattr(self._model, "get_embedding_dimension", None) \
                    or self._model.get_sentence_embedding_dimension
                self.dim = get_dim()
                self.backend = "local"
                return
            except Exception:
                if self.backend == "local":
                    raise
        if self.backend in ("auto", "openai") and config.OPENAI_API_KEY:
            self.backend = "openai"
            self.dim = 1024
            return
        self.backend = "hash"
        self.dim = 384

    @property
    def describe(self) -> str:
        return {
            "local": f"sentence-transformers:{config.EMBED_MODEL}",
            "openai": "openai-embeddings",
            "hash": "local-hash(兜底，非语义)",
        }[self.backend]

    # -------- 向量化 --------
    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        if self.backend == "local":
            v = self._model.encode(texts, normalize_embeddings=True, batch_size=16)
            return np.asarray(v, dtype="float32")
        if self.backend == "openai":
            return self._embed_openai(texts)
        return self._embed_hash(texts)

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        import httpx

        base = (config.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        r = httpx.post(
            f"{base}/embeddings",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": texts},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()["data"]
        vecs = np.asarray([d["embedding"] for d in data], dtype="float32")
        self.dim = vecs.shape[1]
        norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norm

    def _embed_hash(self, texts: list[str]) -> np.ndarray:
        """字符 bigram + 词 的哈希 TF 向量，L2 归一化。"""
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for tok in toks:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i]) + 1e-9
            out[i] /= n
        return out


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词，中文按字 + 二元组合（兼顾 BM25 与哈希向量）。"""
    text = text.lower()
    words = _TOKEN_RE.findall(text)
    toks: list[str] = []
    buf: list[str] = []
    for w in words:
        if len(w) == 1 and "\u4e00" <= w <= "\u9fff":
            buf.append(w)
        else:
            toks.append(w)
    for i in range(len(buf) - 1):
        toks.append(buf[i] + buf[i + 1])
    toks.extend(buf)
    return toks
