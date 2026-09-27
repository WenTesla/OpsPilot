"""OpsPilot 日志 MCP Server（v0.2）

提供只读的日志类查询：日志检索 / 错误分布聚合 / 历史故障案例。
当前数据源为 `fixtures.py` 的 Mock 实现，接口形态对齐 CLS / ES 日志服务，
将来对接真实 API 只需替换 `fixtures.gen_log_entries` 的实现。

启动：
    uv run python mcp_servers/log_server.py
    # 或指定端口：MCP_LOG_PORT=8202 uv run python mcp_servers/log_server.py

⚠️ 全部为只读工具，禁止出现任何写操作。
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import (  # noqa: E402
    SERVICES, envelope, gen_log_entries, match_cases, unknown_service_error,
)

HOST = os.getenv("MCP_LOG_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_LOG_PORT", "8102"))

mcp = FastMCP("OpsPilot Logs", host=HOST, port=PORT, streamable_http_path="/mcp")

MAX_LIMIT = 60


def norm_pattern(msg: str) -> str:
    """日志模板归一化：数字替换为 N，便于把同类错误聚合计数。"""
    import re
    return re.sub(r"\d+", "N", msg)[:90]


@mcp.tool()
def search_logs(
    service: str,
    keyword: Optional[str] = None,
    level: str = "ERROR",
    minutes: int = 30,
    limit: int = 20,
) -> dict:
    """按服务、级别、关键词检索日志，返回样本与高频模式。
    回答「日志里报什么错」「有没有 timeout / 连接池之类的关键线索」时使用。

    Args:
        service: 服务名
        keyword: 关键词（如 timeout、连接池、504），不填表示不过滤
        level: ERROR / WARN / INFO / all，默认 ERROR
        minutes: 时间窗口（相对当前分钟数），默认 30
        limit: 返回样本条数上限，默认 20（上限 60）

    Returns:
        {"data": {total, returned, samples:[{ts,level,service,message,trace_id}], top_patterns:[{pattern,count}]}}
        top_patterns 是对日志消息归一化后的高频模板，比逐条看样本更快定位问题类型。
    """
    if service not in SERVICES:
        return envelope("log", "search_logs", {}, params={"service": service},
                        error=unknown_service_error(service))

    entries = gen_log_entries(service, int(minutes))
    if level not in ("all", "", None):
        entries = [e for e in entries if e["level"] == level.upper()]
    if keyword:
        kw = keyword.lower()
        entries = [e for e in entries if kw in e["message"].lower()]

    # 归一化：去掉数字与 ID，聚合成模式
    counter = Counter(norm_pattern(e["message"]) for e in entries)
    top = [{"pattern": p, "count": c} for p, c in counter.most_common(5)]

    cap = max(1, min(int(limit), MAX_LIMIT))
    samples = entries[:cap]

    return envelope(
        "log", "search_logs",
        {
            "service": service, "window_minutes": int(minutes), "level": level,
            "keyword": keyword, "total": len(entries), "returned": len(samples),
            "samples": samples, "top_patterns": top,
        },
        params={"service": service, "keyword": keyword, "level": level, "minutes": minutes},
        hint="total 为命中总数，samples 仅为前若干条样本；高频模式比样本更有诊断价值。"
             if entries else "该条件下没有命中日志，可放宽 level 或 minutes 后重试。",
    )


@mcp.tool()
def get_error_breakdown(service: str, minutes: int = 30) -> dict:
    """统计指定时间窗内错误日志的类型分布与趋势。
    回答「错误集中在哪一类」「相比之前是变多还是变少」时使用。

    Args:
        service: 服务名
        minutes: 时间窗口（分钟），默认 30

    Returns:
        {"data": {total, by_type:{错误类别:次数}, top_source, prev_total, trend_pct_vs_prev}}
        trend_pct_vs_prev 为与前一个等长窗口相比的变化百分比（正数为恶化）。
    """
    if service not in SERVICES:
        return envelope("log", "get_error_breakdown", {}, params={"service": service},
                        error=unknown_service_error(service))

    minutes = int(minutes)
    cur = [e for e in gen_log_entries(service, minutes) if e["level"] == "ERROR"]
    # 前一等长窗口的近似值：Mock 数据源用双倍窗口减去当前窗口得到，仅用于演示趋势方向
    prev_total = len([e for e in gen_log_entries(service, minutes * 2) if e["level"] == "ERROR"]) - len(cur)
    prev_total = max(0, prev_total)

    def bucket(msg: str) -> str:
        m = msg.lower()
        for key, name in (
            ("timeout", "Timeout/超时"), ("connection pool", "连接池耗尽"),
            ("lock", "锁等待"), ("replication", "主从复制"),
            ("oom", "OOM"), ("refused", "连接被拒"), ("exception", "异常抛出"),
            ("feign", "服务间调用失败"), ("gc", "GC 停顿"),
        ):
            if key in m:
                return name
        return "其它"

    counter = Counter(bucket(e["message"]) for e in cur)
    top_type, top_count = counter.most_common(1)[0] if counter else ("无", 0)
    trend = round((len(cur) - prev_total) / prev_total * 100, 1) if prev_total else None

    return envelope(
        "log", "get_error_breakdown",
        {
            "service": service, "window_minutes": minutes, "total": len(cur),
            "by_type": dict(counter.most_common()),
            "top_type": top_type, "top_count": top_count,
            "prev_window_total": prev_total, "trend_pct_vs_prev": trend,
        },
        params={"service": service, "minutes": minutes},
        hint=f"错误集中在「{top_type}」。" if top_count else "时间窗内没有 ERROR 日志。",
    )


@mcp.tool()
def search_incident_cases(symptom: str, service: Optional[str] = None, top_k: int = 3) -> dict:
    """检索历史故障案例：相似现象的已知根因、处置过程与 MTTR。
    回答「以前有没有出现过类似问题」「上次是怎么修的」时使用。

    Args:
        symptom: 现象描述，尽量带上关键症状词（如「下单超时 连接池耗尽」）
        service: 可选，限定服务名以提高匹配精度
        top_k: 返回条数，默认 3

    Returns:
        {"data": {count, cases:[{case_id,title,occurred_at,services,symptom,root_cause,
                                 resolution,mttr_min,match_score}]}}
        match_score=0 表示无关键词命中，返回的是兜底的最近案例，引用时需谨慎。
    """
    cases = match_cases(symptom, service, int(top_k))
    return envelope(
        "log", "search_incident_cases",
        {"count": len(cases), "cases": cases},
        params={"symptom": symptom, "service": service, "top_k": top_k},
        hint="match_score 越高匹配越好；为 0 时说明无关键词命中，不要当成已确认的同类故障引用。",
    )


if __name__ == "__main__":
    print(f"[log] MCP server on http://{HOST}:{PORT}/mcp (transport=streamable-http)")
    mcp.run(transport="streamable-http")
