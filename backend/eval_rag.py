"""RAG 效果验证。

四层验证，从弱到强：
  1. 检索层   Recall@K / MRR —— 该命中的章节有没有进候选
  2. 溯源层   引用里的章节是否在原文里真实存在（防止幻觉锚点）
  3. 生成层   **哨兵测试**（最关键）—— 塞一条编造规定，看模型是否照着说
              能说出来 = 答案确实来自文档，而非通用知识
  4. 拒答层   库里没有的问题，应当检索为空，不硬编

用法（在 backend/ 目录下）：
    uv run python eval_rag.py         # 配置从 .env 读取
    OPSPILOT_SKIP_DOTENV=1 uv run python eval_rag.py   # 跳过 .env，用 Mock 模型只跑前三层

评测跑在独立临时索引目录，不会污染正式知识库。
不配 OPENAI_API_KEY 时跳过生成层与拒答层，其余两层照样跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# 必须在 import app.* 之前设置：config 在模块加载时读取环境变量
_EVAL_DIR = Path(os.getenv("OPSPILOT_DATA_DIR") or tempfile.mkdtemp(prefix="opspilot_eval_"))
os.environ["OPSPILOT_DATA_DIR"] = str(_EVAL_DIR)
# 不强制后端：跟随 .env（local=BGE-M3 / openai）。没有可用后端时 RAGPipeline 会直接报错，
# 评测宁可失败也不要跑在“非语义假向量”上得出误导性指标。

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.rag.pipeline import RAGPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CASES_FILE = Path(__file__).resolve().parent / "eval_cases.json"
REPORT = Path(__file__).resolve().parent / "eval_report.md"

SENTINEL_DOC = """# 内部约定-哨兵校验

## 1. 哨兵条款

本公司规定：所有 CPU 告警的统一阈值一律为 42%，
且处理的第一步必须执行 `ops-cli sentinel-ack` 命令进行登记，未登记者按 P0 处理。

## 2. 补充说明

该条款仅用于验证检索系统是否真的在读取文档内容，不属于真实运维规范。
"""

SENTINEL_QUERY = "我们公司对 CPU 告警的阈值和处理第一步是怎么规定的？"
SENTINEL_MARKERS = ["42%", "sentinel-ack", "哨兵"]
ABSTAIN_MARKERS = ["无法回答", "未覆盖", "未找到", "没有找到", "暂无", "不属于", "不在", "没有相关", "无法从"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def load_headings(text: str) -> set[str]:
    return {m.group(2).strip() for m in (_HEADING_RE.match(ln) for ln in text.split("\n")) if m}


def check_traceable(hit, source_texts: dict[str, str]) -> tuple[bool, str]:
    """引用必须能在原文中找到：文件名存在，且 heading_path 的每一级都真实存在。"""
    src = source_texts.get(hit.filename)
    if src is None:
        return False, "源文件不存在"
    heads = load_headings(src)
    path = hit.heading_path or ""
    if not path:
        return True, "PDF 无章节结构（按页定位）"
    for seg in [s.strip() for s in path.split(">") if s.strip()]:
        if seg not in heads:
            return False, f"章节「{seg}」不在原文标题中"
    return True, ""


async def ask(agent, cid: str, query: str) -> str:
    async for _ in agent.stream(cid, query):
        pass
    st = await agent.final_state(cid)
    msgs = st.values.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    return last.content if isinstance(last.content, str) else str(last.content)


def main() -> int:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    top_k = cases.get("top_k", 5)

    print(f"[环境] 索引目录 {_EVAL_DIR}")
    p = RAGPipeline()
    print(f"[环境] Embedding: {p.embedder.describe}")

    source_texts: dict[str, str] = {}
    print("\n=== 入库 examples/ ===")
    for f in sorted(EXAMPLES.glob("*.md")):
        n = p.ingest_file(str(f), f.name, "md", f"eval_{f.stem}")
        source_texts[f.name] = f.read_text(encoding="utf-8")
        print(f"  {f.name}: {n} chunks")
    print(f"  合计 {p.stats()['chunks']} chunks\n")

    # ---------- 1. 检索层 ----------
    print("=== 1. 检索层 Recall@%d ===" % top_k)
    rows, judges, mrr_sum, ok_cnt = [], [], 0.0, 0
    for c in cases["cases"]:
        res = p.retrieve(c["query"], top_k=top_k)
        exp_head = c.get("expect_head")
        hit_rank = 0
        for i, h in enumerate(res.hits, start=1):
            if h.filename != c["expect_file"]:
                continue
            if exp_head and exp_head not in h.heading_path:
                continue
            hit_rank = i
            break
        if hit_rank:
            ok_cnt += 1
            mrr_sum += 1.0 / hit_rank
        top = res.hits[0] if res.hits else None
        rows.append({
            "query": c["query"], "expect_file": c["expect_file"], "expect_head": exp_head or "-",
            "rank": hit_rank or "-", "took_ms": res.took_ms,
            "top1": f"{top.filename} > {top.heading_path}" if top else "（空）",
        })
        # 溯源校验：对命中的前 3 条都查
        for h in res.hits[:3]:
            ok, why = check_traceable(h, source_texts)
            judges.append((ok, h.filename, h.heading_path, why))
        print(f"  [{'OK ' if hit_rank else 'MISS'}] {c['query'][:34]}  → 命中第 {hit_rank or '-'} 位")

    recall = ok_cnt / len(cases["cases"])
    mrr = mrr_sum / len(cases["cases"])
    print(f"\n  Recall@{top_k} = {recall:.2%}   MRR = {mrr:.3f}   ({ok_cnt}/{len(cases['cases'])})")

    # ---------- 2. 溯源层 ----------
    bad = [j for j in judges if not j[0]]
    print(f"\n=== 2. 引用可溯源 === 校验 {len(judges)} 条，不通过 {len(bad)} 条")
    for ok, fn, hp, why in bad[:10]:
        print(f"  [FAIL] {fn} | {hp} | {why}")

    # ---------- 3. 拒答层 ----------
    agent = None
    if os.getenv("OPENAI_API_KEY"):
        from app.agent.graph import OpsPilot
        agent = OpsPilot(p)

    print("\n=== 3. 拒答测试（库里没有的问题）===")
    abstain_ok = True
    for c in cases.get("abstain", []):
        res = p.retrieve(c["query"], top_k=top_k)
        empty = not res.hits
        note = ""
        if res.hits and agent is not None:
            # 检索层没拦住时，看生成层会不会照 system prompt 说明「未覆盖」
            ans = asyncio.run(ask(agent, f"eval_abs_{abs(hash(c['query']))}", c["query"]))
            declined = any(k in ans for k in ABSTAIN_MARKERS)
            note = f"｜生成层{'正确拒答' if declined else '【可能硬编】'}\n        答案节选：{ans[:70]}…"
            empty = declined
        print(f"  [{'OK ' if empty else 'MISS'}] {c['query'][:30]} → 检索命中 {len(res.hits)} 条 {note}")
        abstain_ok = abstain_ok and empty

    # ---------- 4. 生成层（哨兵测试）----------
    sentinel_retr, sentinel_gen = None, None
    if not os.getenv("OPENAI_API_KEY"):
        print("\n=== 4. 生成层哨兵测试 === 未配置 OPENAI_API_KEY，跳过")
    else:
        print("\n=== 4. 生成层哨兵测试 ===")
        sentinel_path = _EVAL_DIR / "SOP-哨兵校验.md"
        sentinel_path.write_text(SENTINEL_DOC, encoding="utf-8")
        p.ingest_file(str(sentinel_path), "SOP-哨兵校验.md", "md", "eval_sentinel")

        res = p.retrieve(SENTINEL_QUERY, top_k=top_k)
        ctx_contains = any(m in (h.text + h.heading_path) for h in res.hits for m in SENTINEL_MARKERS)
        sentinel_retr = ctx_contains
        print(f"  [{'OK ' if ctx_contains else 'FAIL'}] 检索上下文包含哨兵内容：{ctx_contains}")

        if agent is None:
            from app.agent.graph import OpsPilot
            agent = OpsPilot(p)   # 持有同一 pipeline 实例，哨兵文档入库后仍可见
        t0 = time.time()
        answer = asyncio.run(ask(agent, "eval_sentinel_1", SENTINEL_QUERY))
        found = [m for m in SENTINEL_MARKERS if m in answer]
        sentinel_gen = bool(found)
        print(f"  [{'OK ' if found else 'FAIL'}] 模型答案复述了哨兵规定：{found or '未出现'}")
        print(f"      耗时 {int((time.time() - t0) * 1000)}ms，答案 {len(answer)} 字")
        print("      → 模型不可能靠通用知识编出 42% / sentinel-ack，出现即证明其依据文档作答")

    # ---------- 报告 ----------
    lines = [
        "# RAG 效果验证报告", "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Embedding 后端：{p.embedder.describe}",
        f"- 索引 chunk 数：{p.stats()['chunks']}",
        "", "## 结论速览", "",
        "| 验证层 | 指标 | 结果 |", "|---|---|---|",
        f"| 检索层 | Recall@{top_k} | {recall:.2%} |",
        f"| 检索层 | MRR | {mrr:.3f} |",
        f"| 溯源层 | 引用不通过数 | {len(bad)} / {len(judges)} |",
        f"| 拒答层 | 无关问题是否返回空 | {'是' if abstain_ok else '否'} |",
    ]
    if sentinel_retr is not None:
        lines.append(f"| 生成层 | 哨兵内容进入上下文 | {'是' if sentinel_retr else '否'} |")
    if sentinel_gen is not None:
        lines.append(f"| 生成层 | 模型复述哨兵规定 | {'是' if sentinel_gen else '否'} |")
    lines += ["", "## 逐条明细", "", "| # | 提问 | 期望文档 | 期望章节 | 命中位次 | Top1 实际命中 | 耗时 |",
              "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['query']} | {r['expect_file']} | {r['expect_head']} "
                     f"| {r['rank']} | {r['top1']} | {r['took_ms']}ms |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
