"""Embedding 适配器（只用语义向量，两级选择）。

优先级（EMBED_BACKEND=auto）：
  1. local  : sentence-transformers 加载 BGE-M3（默认，需本地权重或可访问 HF）
  2. openai : OpenAI 兼容 /embeddings 接口（需 OPENAI_API_KEY，且该厂商支持）

两者都不可用时直接抛错，不再静默降级到「假向量」：
非语义的哈希向量与 BM25 输入同源、无法互补，只会让检索质量变得不可预期。
"""
from __future__ import annotations

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
        self.dim = 0                       # 由 _init 按实际模型填实
        self._model = None
        self._init()

    # -------- 初始化 --------
    def _init(self) -> None:
        """后端选择顺序（auto 时）：本地语义向量 → OpenAI Embeddings。

        两者都不可用时直接抛错，不做静默降级——让「检索质量不可预期」在启动阶段
        就暴露出来，而不是留到线上变成答非所问。
        """
        local_err: Exception | None = None
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
            except Exception as e:
                if self.backend == "local":
                    raise
                local_err = e
        if self.backend in ("auto", "openai") and config.OPENAI_API_KEY:
            self.backend = "openai"
            self.dim = 1024
            return
        if self.backend == "hash":
            raise RuntimeError(
                "EMBED_BACKEND=hash 已移除：哈希向量与 BM25 输入同源、与关键词路不互补，"
                "只会让检索质量不可预期。请改用 local（BGE-M3）或 openai。"
            )
        raise RuntimeError(
            "没有可用的 Embedding 后端。二选一：\n"
            "  1) 本地语义向量：uv pip install sentence-transformers，并把 EMBED_MODEL 指向 BGE-M3 权重"
            "（HF repo id 或本地目录，如 models/bge-m3；国内网络建议 HF_ENDPOINT=https://hf-mirror.com "
            "且 HF_HUB_DISABLE_XET=1）；\n"
            "  2) 在线接口：配 OPENAI_API_KEY（注意 DeepSeek 等厂商没有 /embeddings 接口）。\n"
            f"当前 EMBED_BACKEND={self.backend}"
            + (f"，local 失败原因：{local_err!r}" if local_err else "")
        )

    @property
    def describe(self) -> str:
        return {
            "local": f"sentence-transformers:{config.EMBED_MODEL}",
            "openai": "openai-embeddings",
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
        raise RuntimeError(f"未知的 Embedding 后端：{self.backend}")

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


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词，中文按字 + 二元组合（BM25 专用）。"""
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
