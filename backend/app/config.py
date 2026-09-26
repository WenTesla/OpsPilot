import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]          # backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"             # frontend/
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
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "700"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))

# ---------- 上传限制 ----------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXT = {".pdf", ".md", ".markdown"}

CHUNK_DOC_TYPES = {"sop", "postmortem", "doc", "other"}
