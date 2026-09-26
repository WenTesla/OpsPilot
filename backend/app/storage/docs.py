import json
import threading
from datetime import datetime

from .. import config

_lock = threading.RLock()


def _load() -> list[dict]:
    if not config.DOCS_JSON.exists():
        return []
    try:
        with open(config.DOCS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(docs: list[dict]) -> None:
    tmp = config.DOCS_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    tmp.replace(config.DOCS_JSON)


def list_docs() -> list[dict]:
    with _lock:
        return list(reversed(_load()))


def get_doc(doc_id: str) -> dict | None:
    with _lock:
        return next((d for d in _load() if d["doc_id"] == doc_id), None)


def add_doc(doc: dict) -> dict:
    with _lock:
        docs = _load()
        doc.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
        docs.append(doc)
        _save(docs)
        return doc


def update_doc(doc_id: str, **fields) -> dict | None:
    with _lock:
        docs = _load()
        for d in docs:
            if d["doc_id"] == doc_id:
                d.update(fields)
                d["updated_at"] = datetime.now().isoformat(timespec="seconds")
                _save(docs)
                return d
        return None


def remove_doc(doc_id: str) -> dict | None:
    with _lock:
        docs = _load()
        target = next((d for d in docs if d["doc_id"] == doc_id), None)
        if target:
            _save([d for d in docs if d["doc_id"] != doc_id])
        return target
