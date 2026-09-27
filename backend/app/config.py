import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]          # backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"             # frontend/


def _load_env_file() -> None:
    """把 .env 载入进程环境变量。

    查找顺序：backend/.env → 仓库根 .env（先命中先生效）。
    显式设置的环境变量优先级更高（override=False），所以命令行仍可临时覆盖：
        LLM_MODEL=deepseek-reasoner uv run python -m uvicorn app.main:app
    设置 OPSPILOT_SKIP_DOTENV=1 可完全关闭 .env 加载。
    """
    if os.getenv("OPSPILOT_SKIP_DOTENV") == "1":
        return
    try:
        from dotenv import load_dotenv
    except ImportError:      # 没装 python-dotenv 时退回纯环境变量模式，不影响启动
        return
    for candidate in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_env_file()

DATA_DIR = Path(os.getenv("OPSPILOT_DATA_DIR", BASE_DIR / "data"))
FILES_DIR = DATA_DIR / "files"
(DATA_DIR).mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)

DOCS_JSON = DATA_DIR / "docs.json"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
VECTORS_NPY = DATA_DIR / "vectors.npy"
BM25_JSON = DATA_DIR / "bm25.json"

# ---------- 模型 ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto")      # auto|local|openai|hash
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
USE_RERANK = os.getenv("RAG_RERANK", "0") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ---------- 检索 ----------
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "20"))
RRF_K = 60
# 最低「IDF 加权查询覆盖率」：低于此值视为与问题无关，直接不返回（防止凑够 Top-K 硬塞上下文）
# 注意：这是粗过滤，真正的拒答由 LLM 依据 system prompt 判断
MIN_COVER = float(os.getenv("RAG_MIN_COVER", "0.12"))
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "700"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))

# ---------- 上传限制 ----------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXT = {".pdf", ".md", ".markdown"}

CHUNK_DOC_TYPES = {"sop", "postmortem", "doc", "other"}

# ---------- 服务 ----------
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
