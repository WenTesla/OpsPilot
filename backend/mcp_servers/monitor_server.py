"""OpsPilot 监控 MCP Server（v0.2）

提供只读的线上数据查询：服务清单 / 服务详情与拓扑 / 指标时序 / 活动告警。
当前数据源为 `fixtures.py` 的 Mock 实现，接口形态与真实 Prometheus / CMDB 对齐，
将来换成真实 SDK 只需替换取数部分，工具函数与返回信封保持不变。

启动：
    uv run python mcp_servers/monitor_server.py
    # 或指定端口：MCP_MONITOR_PORT=8201 uv run python mcp_servers/monitor_server.py

⚠️ 全部为只读工具。禁止在本文件中出现 restart / scale / rollback / exec / delete 等写操作，
写类工具属于 v0.4（需配套分级护栏 + dry-run + 审批）。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 兼容直接脚本启动与包导入两种方式
from fixtures import (  # noqa: E402
    ALERTS, METRICS, SERVICES, all_services, depended_by, envelope, gen_series,
    percentile, unknown_service_error,
)

HOST = os.getenv("MCP_MONITOR_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_MONITOR_PORT", "8101"))

# FastMCP 直接来自 mcp SDK 自带实现：少引一个第三方包，
# 也就避开了 fastmcp 与 langchain-mcp-adapters 对 mcp 版本互不兼容的坑（详见 backend/README 依赖说明）。
mcp = FastMCP("OpsPilot Monitor", host=HOST, port=PORT, streamable_http_path="/mcp")

MAX_POINTS = 60  # 硬上限：再多的浮点数对 LLM 没有价值，只会挤占上下文


@mcp.tool()
def list_services(env: str = "all") -> dict:
    """列出所有已知服务及其健康状态。回答「有哪些服务」「哪些服务有问题」时使用。

    Args:
        env: 环境过滤，prod / stage / all，默认 all

    Returns:
        {"data": {"services": [{name, env, owner, kind, version, replicas, status}]}}
    """
    env_arg = None if env in ("all", "", None) else env
    services = all_services(env_arg)
    return envelope(
        "monitor", "list_services", {"count": len(services), "services": services},
        params={"env": env},
        hint=f"共 {len(services)} 个服务；status=degraded 表示存在未恢复告警。",
    )


@mcp.tool()
def get_service_info(service: str) -> dict:
    """查询单个服务的详情：负责人、版本、副本数、上下游依赖、近期变更。
    回答「谁依赖它」「它依赖谁」「最近有没有发布/改配置」时使用——近期变更是判断根因的关键线索。

    Args:
        service: 服务名，如 order-service（可先用 list_services 查准确名称）

    Returns:
        {"data": {name, version, owner, replicas, kind, depends_on, depended_by, recent_changes}}
    """
    s = SERVICES.get(service)
    if not s:
        return envelope("monitor", "get_service_info", {}, params={"service": service},
                        error=unknown_service_error(service))

    return envelope(
        "monitor", "get_service_info",
        {
            "name": service, "env": s["env"], "kind": s["kind"], "version": s["version"],
            "owner": s["owner"], "replicas": s["replicas"],
            "depends_on": s.get("depends_on", []),
            "depended_by": depended_by(service),
            "recent_changes": s.get("recent_changes", []),
        },
        params={"service": service},
        hint="recent_changes 为空表示近期无变更；有变更时优先考虑变更引入的可能性。",
    )


@mcp.tool()
def query_metrics(
    service: str,
    metric: str = "cpu",
    minutes: int = 30,
    interval: str = "1m",
    max_points: int = 30,
) -> dict:
    """查询服务在指定时间窗内的指标时序，返回统计值与是否超阈值。
    回答「某指标现在多少」「最近有没有飙升/恶化」时使用。

    Args:
        service: 服务名
        metric: cpu / memory / latency_p99 / qps / error_rate
        minutes: 时间窗口（相对当前时间的分钟数），默认 30
        interval: 聚合间隔 1m / 5m / 1h，默认 1m
        max_points: 返回的最大数据点数，默认 30（上限 60）

    Returns:
        {"data": {metric, unit, points, stats:{avg,max,min,p95}, threshold, alert_triggered, trend, shape}}
        shape 为曲线形态（spike=突增 / leak=缓慢上升 / step=阶跃 / dip=下跌 / healthy=平稳），
        可直接用于判断故障类型。
    """
    if service not in SERVICES:
        return envelope("monitor", "query_metrics", {}, params={"service": service},
                        error=unknown_service_error(service))
    if metric not in METRICS:
        return envelope("monitor", "query_metrics", {}, params={"metric": metric},
                        error=f"未知指标：{metric}。可选：{', '.join(METRICS)}")

    meta = METRICS[metric]
    cap = max(3, min(int(max_points), MAX_POINTS))
    points, raw, shape = gen_series(service, metric, int(minutes), interval, cap=cap)
    values = [p["value"] for p in points]

    threshold = meta["threshold"]
    peak = max(raw)
    low = min(raw)
    if meta["cmp"] == "gt" and threshold is not None:
        triggered = peak > threshold
    elif threshold is not None:
        triggered = low < threshold
    else:
        triggered = False

    # 趋势：后 1/3 均值 vs 前 1/3 均值，变化超过 20% 判定为上升/下降
    k = max(1, len(raw) // 3)
    head = sum(raw[:k]) / k
    tail = sum(raw[-k:]) / k
    if head == 0:
        trend = "stable"
    elif tail > head * 1.2:
        trend = "up"
    elif tail < head * 0.8:
        trend = "down"
    else:
        trend = "stable"

    hint = {
        "spike": "曲线形态为突增：优先怀疑突发流量、下游超时放大或近期发布。",
        "leak": "曲线形态为持续上升：优先怀疑资源泄漏（内存泄漏 / 连接池未释放 / goroutine 泄漏）。",
        "step": "曲线形态为阶跃：优先怀疑变更（发布、配置调整、扩缩容）在同时间点引入。",
        "dip": "曲线形态为下跌：结合 qps 下降判断是否上游已经熔断或流量掉底。",
        "healthy": "曲线平稳，无异常形态。",
    }[shape]

    return envelope(
        "monitor", "query_metrics",
        {
            "service": service, "metric": metric, "unit": meta["unit"],
            "window_minutes": int(minutes), "interval": interval,
            "points": points,
            "stats": {
                "avg": round(sum(values) / len(values), 2),
                "max": max(values), "min": min(values),
                "p95": percentile(raw, 0.95),
            },
            "threshold": threshold,
            "alert_triggered": triggered,
            "trend": trend,
            "shape": shape,
        },
        params={"service": service, "metric": metric, "minutes": minutes, "interval": interval},
        hint=hint + (f" 当前 {'超过' if triggered else '未超过'}阈值 {threshold}。" if threshold else ""),
    )


@mcp.tool()
def list_active_alerts(service: str = "all", since_minutes: int = 60, severity: str = "all") -> dict:
    """列出最近 N 分钟内处于活动状态的告警。回答「现在有什么告警」「为什么报警」时使用。

    Args:
        service: 服务名，或 all 表示全部服务
        since_minutes: 回溯窗口（分钟），默认 60
        severity: critical / warning / all，默认 all

    Returns:
        {"data": {"count": N, "alerts": [{alert_id, service, name, severity, value, threshold,
                                          started_at, duration_min, desc}]}}
    """
    now = datetime.now()
    out = []
    for a in ALERTS:
        if a["started_min_ago"] > since_minutes:
            continue
        if service not in ("all", "", None) and a["service"] != service:
            continue
        if severity not in ("all", "", None) and a["severity"] != severity:
            continue
        started = now - timedelta(minutes=a["started_min_ago"])
        out.append({
            "alert_id": a["alert_id"], "service": a["service"], "name": a["name"],
            "severity": a["severity"], "value": a["value"], "threshold": a["threshold"],
            "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": a["started_min_ago"], "desc": a["desc"],
        })

    return envelope(
        "monitor", "list_active_alerts",
        {"count": len(out), "alerts": sorted(out, key=lambda x: -x["duration_min"])},
        params={"service": service, "since_minutes": since_minutes, "severity": severity},
        hint="按持续时间倒序返回。无告警时 count=0，不要臆测告警内容。",
    )


if __name__ == "__main__":
    print(f"[monitor] MCP server on http://{HOST}:{PORT}/mcp (transport=streamable-http)")
    mcp.run(transport="streamable-http")
