"""冒烟测试：解析 → 入库 → 检索 → LangGraph Agent 出答案（不启动服务）。"""
import asyncio
import tempfile
from pathlib import Path

from app.agent.graph import OpsPilot
from app.rag.pipeline import RAGPipeline, CURRENT_CITATIONS

SAMPLE = """# SOP-主机资源异常处理

## 3. CPU 使用率过高

### 3.1 快速定位
- 确认告警范围：单实例还是多实例
- 执行 top -c 按 CPU 排序，定位进程

### 3.2 常见根因
业务流量突增、死循环或正则回溯、JVM GC 频繁。先摘流量止损，再保留现场排查。

## 5. 磁盘空间不足
使用 df -h 确认挂载点，du -xh --max-depth=2 /data 定位大目录。
可安全清理：轮转日志、/tmp 临时文件、core dump。
禁止删除未备份的业务数据。
"""


def main():
    md = Path(tempfile.gettempdir()) / "sop_sample.md"
    md.write_text(SAMPLE, encoding="utf-8")

    p = RAGPipeline()
    n = p.ingest_file(str(md), "SOP-主机资源异常处理.md", "md", "doc_sample")
    print(f"[1] 入库 chunk 数: {n}")
    print(f"[2] 索引状态: {p.stats()}")

    res = p.retrieve("CPU 使用率超过 90% 怎么排查", top_k=3)
    print(f"[3] 检索命中 {len(res.hits)} 条, 耗时 {res.took_ms}ms")
    for i, h in enumerate(res.hits, 1):
        print(f"    [{i}] {h.filename} | {h.locator} | score={h.score:.4f} | {h.text[:50]}…")

    ctx, citations = RAGPipeline.format_context(res)
    print(f"[4] 上下文 {len(ctx)} 字, citations={[c['n'] for c in citations]}")

    # ---- Agent ----
    agent = OpsPilot(p)
    box = []
    tok = CURRENT_CITATIONS.set(box)

    async def run():
        async for mode, payload in agent.stream("smoke-1", "order-service 的 CPU 使用率超过 90%，怎么排查？"):
            if mode == "updates":
                print(f"[5] 节点执行: {list(payload.keys()) if isinstance(payload, dict) else payload}")
        st = await agent.final_state("smoke-1")
        last = st.values["messages"][-1]
        return last.content

    answer = asyncio.run(run())
    print("[6] 最终回答:\n" + "-" * 50)
    print(answer)
    print("-" * 50)
    print(f"[7] 收集到的引用: {box}")
    CURRENT_CITATIONS.reset(tok)


if __name__ == "__main__":
    main()
