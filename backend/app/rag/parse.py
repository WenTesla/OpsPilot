"""PDF / Markdown 解析。

统一输出 Block 列表：
    Block(text, locator, heading_path)
- PDF：按页切分，locator = "第 N 页"
- MD：按标题层级切分，locator = Heading Path
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Block:
    text: str
    locator: str                       # 页码或标题锚点（用于 citation）
    heading_path: str = ""             # 章节路径
    page: int | None = None            # PDF 页码


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_pdf(path: str) -> list[Block]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("解析 PDF 需要 pypdf：pip install pypdf") from e

    reader = PdfReader(path)
    blocks: list[Block] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _clean(text)
        if text.strip():
            blocks.append(Block(text=text, locator=f"第 {i} 页", page=i))
    if not blocks:
        raise ValueError("未能从 PDF 中提取到文字（可能是扫描件，需 OCR）")
    return blocks


def parse_md(path: str) -> list[Block]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # 去掉 front-matter
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            raw = raw[end + 4:]

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []   # (level, title)
    buf: list[str] = []
    cur_heading = ""

    def flush():
        if not buf:
            return
        text = _clean("\n".join(buf)).strip()
        if text:
            blocks.append(Block(text=text, locator=cur_heading or "正文开头", heading_path=cur_heading))
        buf.clear()

    for line in raw.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_heading = " > ".join(t for _, t in stack)
        else:
            buf.append(line)
    flush()

    if not blocks:
        raise ValueError("Markdown 文件内容为空")
    return blocks


def parse_file(path: str, fmt: str) -> list[Block]:
    if fmt == "pdf":
        return parse_pdf(path)
    return parse_md(path)


def _clean(text: str) -> str:
    """去噪：多余空行、页眉页脚式的纯数字行。"""
    lines = []
    for ln in text.split("\n"):
        s = ln.rstrip()
        if re.fullmatch(r"[\s\d./-]{0,20}", s):   # 纯页码/日期行
            continue
        lines.append(s)
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading_path: str = ""
    locator: str = ""
    filename: str = ""
    doc_type: str = "other"
    service: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    filename: str,
    doc_type: str,
    service: list[str],
    env: list[str],
    target: int,
    overlap: int,
) -> list[Chunk]:
    """把 Block 切成检索用的 Chunk（按段落累积，尽量不切断语义单元）。"""
    import hashlib

    chunks: list[Chunk] = []
    idx = 0
    for b in blocks:
        paras = [p.strip() for p in b.text.split("\n") if p.strip()]
        cur = ""
        for p in paras:
            if not cur:
                cur = p
            elif len(cur) + len(p) + 1 <= target:
                cur += "\n" + p
            else:
                chunks.append(_mk(b, cur, doc_id, filename, doc_type, service, env, idx))
                idx += 1
                # 尾部保留一点重叠
                cur = (cur[-overlap:] + "\n" + p) if overlap > 0 else p
        if cur:
            chunks.append(_mk(b, cur, doc_id, filename, doc_type, service, env, idx))
            idx += 1
    return [c for c in chunks if c.text.strip()]


def _mk(b: Block, text: str, doc_id: str, filename: str, doc_type: str,
        service: list[str], env: list[str], idx: int) -> Chunk:
    import hashlib
    cid = hashlib.md5(f"{doc_id}:{idx}:{text[:64]}".encode("utf-8")).hexdigest()[:16]
    # 把章节路径拼进 chunk 文本：标题往往是最强的检索信号（如"3. CPU 使用率过高"）
    full = f"{b.heading_path}\n{text}" if b.heading_path else text
    return Chunk(
        chunk_id=cid, doc_id=doc_id, text=full,
        heading_path=b.heading_path, locator=b.locator,
        filename=filename, doc_type=doc_type,
        service=service, env=env,
    )
