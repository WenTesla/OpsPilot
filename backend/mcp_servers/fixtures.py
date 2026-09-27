"""v0.2 Mock 数据源 —— 指标 / 日志 / 告警 / 服务拓扑 / 历史故障。

设计原则：
1. **形态要像真的**。`spike`（突增）、`leak`（缓慢泄漏）、`step`（变更阶跃）、`dip`（流量掉底）
   这几种曲线在真实故障里最常见，演示时 Agent 才能给出有意义的判断；纯随机噪声毫无排障价值。
2. **确定性**。同一 service+metric+分钟窗口多次查询结果一致（用 seed 而非裸 random），
   否则 LLM 前后两次调用对不上账，答案自相矛盾。
3. **工具层不感知 Mock**。真实接入 Prometheus / CLS 时只替换本模块的取数部分，
   `*_server.py` 里的工具函数与返回结构一行都不用改。

⚠️ 所有返回体的 `source.mock` 恒为 True，前端必须打「示例数据」标记，不得让用户误判为真实监控。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

MOCK = True

# ============================================================
# 统一返回信封
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def envelope(
    server: str,
    tool: str,
    data: dict,
    params: dict | None = None,
    hint: str = "",
    error: str | None = None,
) -> dict:
    """所有工具统一的返回结构。

    工具失败时**返回 error 文本而不是抛异常** —— 见设计文档 §4.7：
    /api/chat 是 SSE 流，异常会让整条流中断，前端表现为「回答到一半卡住」。
    """
    out: dict = {
        "source": {
            "server": server,
            "tool": tool,
            "collected_at": now_str(),
            "mock": MOCK,
        },
        "params_echo": params or {},
        "data": {} if error else data,
    }
    if hint:
        out["hint"] = hint
    if error:
        out["error"] = error
    return out


# ============================================================
# 服务拓扑
# ============================================================

SERVICES: dict[str, dict] = {
    "api-gateway": {
        "env": "prod", "owner": "基础架构组", "replicas": 6, "kind": "网关",
        "version": "v2.14.3", "depends_on": ["order-service", "user-service"],
    },
    "order-service": {
        "env": "prod", "owner": "交易组", "replicas": 8, "kind": "Java 服务",
        "version": "v3.8.1", "depends_on": ["payment-service", "inventory-service", "mysql-primary"],
        "recent_changes": [
            {"at": "-2h", "type": "deploy", "desc": "v3.8.0 → v3.8.1，订单查询引入新的二级索引"},
            {"at": "-26h", "type": "config", "desc": "连接池 maxActive 20 → 50"},
        ],
    },
    "payment-service": {
        "env": "prod", "owner": "支付组", "replicas": 6, "kind": "Go 服务",
        "version": "v1.22.0", "depends_on": ["mysql-primary", "redis-cache"],
        "recent_changes": [{"at": "-5d", "type": "deploy", "desc": "v1.21.4 → v1.22.0"}],
    },
    "inventory-service": {
        "env": "prod", "owner": "供应链组", "replicas": 4, "kind": "Python 服务",
        "version": "v0.9.7", "depends_on": ["elasticsearch"],
    },
    "user-service": {
        "env": "prod", "owner": "账号组", "replicas": 4, "kind": "Java 服务",
        "version": "v2.5.0", "depends_on": ["mysql-primary", "redis-cache"],
    },
    "mysql-primary": {
        "env": "prod", "owner": "DBA", "replicas": 1, "kind": "MySQL 8.0",
        "version": "8.0.34", "depends_on": [],
        "recent_changes": [{"at": "-3h", "type": "config", "desc": "innodb_buffer_pool_size 调大至 24G"}],
    },
    "redis-cache": {
        "env": "prod", "owner": "基础架构组", "replicas": 3, "kind": "Redis 7",
        "version": "7.2.4", "depends_on": [],
    },
    "elasticsearch": {
        "env": "prod", "owner": "基础架构组", "replicas": 3, "kind": "ES 8",
        "version": "8.11.0", "depends_on": [],
    },
}


def depended_by(service: str) -> list[str]:
    return [n for n, s in SERVICES.items() if service in s.get("depends_on", [])]


def all_services(env: str | None = None) -> list[dict]:
    items = []
    for name, s in SERVICES.items():
        if env and s["env"] != env:
            continue
        unhealthy = name in _alert_services()
        items.append({
            "name": name, "env": s["env"], "owner": s["owner"], "kind": s["kind"],
            "version": s["version"], "replicas": s["replicas"],
            "status": "degraded" if unhealthy else "healthy",
        })
    return items


def unknown_service_error(service: str) -> str:
    return f"未知服务：{service}。可用服务：{', '.join(SERVICES)}"


# ============================================================
# 指标
# ============================================================

METRICS: dict[str, dict] = {
    "cpu":         {"unit": "percent", "threshold": 80.0, "baseline": 22.0, "cmp": "gt"},
    "memory":      {"unit": "percent", "threshold": 85.0, "baseline": 45.0, "cmp": "gt"},
    "latency_p99": {"unit": "ms",      "threshold": 1000.0, "baseline": 180.0, "cmp": "gt"},
    "error_rate":  {"unit": "percent", "threshold": 1.0, "baseline": 0.15, "cmp": "gt"},
    "qps":         {"unit": "req/s",   "threshold": None, "baseline": 800.0, "cmp": "lt"},
}

# 每个服务的指标形态：healthy / spike / leak / step / dip
# 只有 order-service / mysql-primary / payment-service 有异常，其余保持健康，
# 这样提问才有「区别对待」的演示效果。
PROFILES: dict[str, dict[str, tuple[str, float]]] = {
    "order-service": {
        "cpu": ("leak", 88.0), "memory": ("leak", 79.0),
        "latency_p99": ("spike", 2600.0), "error_rate": ("spike", 7.4), "qps": ("dip", 210.0),
    },
    "mysql-primary": {
        "cpu": ("step", 72.0), "memory": ("leak", 81.0),
        "latency_p99": ("step", 420.0), "error_rate": ("healthy", 0.2), "qps": ("healthy", 2600.0),
    },
    "payment-service": {
        "cpu": ("healthy", 35.0), "memory": ("healthy", 52.0),
        "latency_p99": ("healthy", 240.0), "error_rate": ("healthy", 0.3), "qps": ("healthy", 620.0),
    },
}

DEFAULT_PROFILE: dict[str, tuple[str, float]] = {
    "cpu": ("healthy", 26.0), "memory": ("healthy", 48.0),
    "latency_p99": ("healthy", 190.0), "error_rate": ("healthy", 0.12), "qps": ("healthy", 520.0),
}


def gen_series(service: str, metric: str, minutes: int, interval: str, cap: int = 30):
    """生成时间序列。同参数必得同结果（seed 固定），否则多轮对话会对不上账。"""
    meta = METRICS[metric]
    shape, peak = PROFILES.get(service, DEFAULT_PROFILE).get(metric, ("healthy", meta["baseline"]))
    base = meta["baseline"]
    rng = random.Random(f"{service}|{metric}|{minutes}")

    step_min = interval_minutes(interval)
    total = max(2, min(minutes // step_min, 120))          # 生成上限 120 点，之后按 cap 抽样
    raw: list[float] = []
    for i in range(total):
        p = i / max(1, total - 1)
        if shape == "healthy":
            v = base + rng.uniform(-base * 0.12, base * 0.12)
        elif shape == "leak":
            v = base + (peak - base) * (p ** 1.6) + rng.uniform(-1.5, 1.5)
        elif shape == "spike":
            v = base if p < 0.45 else base + (peak - base) * ((p - 0.45) / 0.55) ** 2 + rng.uniform(-3, 3)
        elif shape == "step":
            v = base + rng.uniform(-2, 2) if p < 0.4 else peak + rng.uniform(-4, 4)
        elif shape == "dip":
            v = base if p < 0.5 else base - (base - peak) * ((p - 0.5) / 0.5) + rng.uniform(-8, 8)
        else:
            v = base
        raw.append(round(max(0.0, v), 2))

    # 点数超出 cap 时等距抽样，保留首尾
    if len(raw) > cap:
        idx = [round(i * (len(raw) - 1) / (cap - 1)) for i in range(cap)]
        picked = [raw[i] for i in idx]
    else:
        picked = raw

    start = datetime.now() - timedelta(minutes=minutes)
    span_sec = minutes * 60
    step_sec = span_sec / max(1, len(picked) - 1)
    pts = [
        {"ts": (start + timedelta(seconds=round(i * step_sec))).strftime("%H:%M"), "value": v}
        for i, v in enumerate(picked)
    ]
    return pts, [round(v, 2) for v in raw], shape


def interval_minutes(interval: str) -> int:
    s = (interval or "1m").strip().lower()
    if s.endswith("h"):
        return int(s[:-1] or 1) * 60
    return int(s[:-1] or 1) if s.endswith("m") else 1


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = min(len(xs) - 1, int(len(xs) * p))
    return xs[k]


# ============================================================
# 告警
# ============================================================

ALERTS: list[dict] = [
    {"alert_id": "AL-2041", "service": "order-service", "name": "HighErrorRate",
     "severity": "critical", "value": 7.4, "threshold": 1.0, "started_min_ago": 18,
     "desc": "HTTP 5xx 错误率 7.4%，持续超过阈值 1%，影响下单链路"},
    {"alert_id": "AL-2042", "service": "order-service", "name": "LatencyP99TooHigh",
     "severity": "warning", "value": 2620, "threshold": 1000, "started_min_ago": 22,
     "desc": "下单接口 P99 2620ms，超过 1000ms 阈值"},
    {"alert_id": "AL-2039", "service": "mysql-primary", "name": "ReplicationLag",
     "severity": "warning", "value": 34, "threshold": 10, "started_min_ago": 41,
     "desc": "主从复制延迟 34s，超过 10s 阈值"},
    {"alert_id": "AL-2038", "service": "payment-service", "name": "CpuUsageHigh",
     "severity": "warning", "value": 82.5, "threshold": 80, "started_min_ago": 65,
     "desc": "CPU 使用率 82.5%，缓慢上升中"},
]


def _alert_services() -> set[str]:
    return {a["service"] for a in ALERTS}


# ============================================================
# 日志
# ============================================================

LOG_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "order-service": [
        ("ERROR", "OrderService.createOrder timeout waiting for connection pool, active=50 idle=0"),
        ("ERROR", "FeignException: 504 Gateway Timeout calling payment-service /api/pay/confirm"),
        ("ERROR", "HikariPool-1 - Connection is not available, request timed out after 30000ms"),
        ("WARN",  "Slow query detected: SELECT * FROM t_order WHERE create_time > ? cost 2841ms"),
        ("ERROR", "OrderService.createOrder timeout waiting for connection pool, active=50 idle=0"),
        ("WARN",  "Thread pool queue size 512 approaching capacity 1024"),
        ("INFO",  "Order created successfully, orderId=SO20260927xxxxx"),
    ],
    "payment-service": [
        ("ERROR", "context deadline exceeded calling bank channel: UnionPay"),
        ("WARN",  "retrying payment confirm, attempt 2/3"),
        ("INFO",  "payment callback received, tradeNo=xxxx"),
    ],
    "mysql-primary": [
        ("ERROR", "Slave_IO_Running: Yes, Slave_SQL_Running: Yes, Seconds_Behind_Master: 34"),
        ("WARN",  "long transaction detected, trx_id=88213 running 128s"),
        ("WARN",  "innodb row lock wait time exceeded 50s"),
    ],
    "inventory-service": [
        ("WARN",  "deduct stock retry 1/3 for sku=88231"),
        ("INFO",  "stock synced to elasticsearch, rows=1204"),
    ],
    "user-service": [
        ("INFO",  "user token refreshed, uid=10086"),
        ("WARN",  "redis-cache get miss rate 23% higher than usual"),
    ],
}
GENERIC_LOGS = [
    ("WARN", "gc pause too long: 812ms"),
    ("INFO", "health check ok"),
]


def raw_logs(service: str) -> list[tuple[str, str]]:
    return LOG_TEMPLATES.get(service, GENERIC_LOGS)


def gen_log_entries(service: str, minutes: int) -> list[dict]:
    """按分钟铺开日志条目，条目数随时间窗线性增长（模拟真实日志密度）。"""
    rng = random.Random(f"log|{service}|{minutes}")
    templates = raw_logs(service)
    # 每分钟 0~3 条 ERROR，异常发生在最近 30 分钟内更密集
    entries: list[dict] = []
    now = datetime.now()
    for i in range(minutes, 0, -1):
        ts = now - timedelta(minutes=i)
        recent = i <= 30
        n_err = rng.randint(1, 3) if recent else rng.randint(0, 1)
        for _ in range(n_err):
            level, msg = rng.choice([t for t in templates if t[0] == "ERROR"] or templates)
            entries.append({
                "ts": ts.strftime("%Y-%m-%d %H:%M:%S"), "level": level, "service": service,
                "message": msg, "trace_id": hex(rng.getrandbits(48))[2:].zfill(12),
            })
        if rng.random() < 0.5:
            level, msg = rng.choice([t for t in templates if t[0] != "ERROR"] or templates)
            entries.append({
                "ts": ts.strftime("%Y-%m-%d %H:%M:%S"), "level": level, "service": service,
                "message": msg, "trace_id": hex(rng.getrandbits(48))[2:].zfill(12),
            })
    return entries


# ============================================================
# 历史故障案例
# ============================================================

INCIDENT_CASES: list[dict] = [
    {
        "case_id": "INC-2026-0812",
        "title": "订单服务大面积超时（连接池耗尽）",
        "services": ["order-service", "payment-service"],
        "symptom": "下单接口大量 504/超时，P99 飙升至 2.6s，错误率超过 7%",
        "keywords": ["超时", "504", "5xx", "错误率", "连接池", "下单", "慢", "延迟"],
        "root_cause": "v3.8.1 版本订单查询引入新二级索引后，慢查询增多占满 Hikari 连接池（maxActive=50），"
                      "新请求排队超过 30s 超时；同时 payment-service 调用被动延迟放大。",
        "resolution": "1) 先回滚至 v3.8.0 止血；2) 下线新索引、给 order 表 create_time 补联合索引；"
                      "3) 连接池 maxActive 回调至 20 并加等待超时告警。",
        "mttr_min": 37, "occurred_at": "2026-08-12",
    },
    {
        "case_id": "INC-2026-0703",
        "title": "MySQL 主从延迟导致读脏数据",
        "services": ["mysql-primary", "order-service"],
        "symptom": "主从复制延迟 34s，订单列表查不到刚创建的订单",
        "keywords": ["主从", "复制", "延迟", "lag", "mysql", "读不到", "不一致", "数据库"],
        "root_cause": "主库一个大事务批量更新 80 万行，从库单线程回放跟不上，Seconds_Behind_Master 累积到 34s。",
        "resolution": "1) 拆小批量更新；2) 从库开启并行回放（slave_parallel_workers=8）；3) 关键查询走主库或加延迟感知路由。",
        "mttr_min": 52, "occurred_at": "2026-07-03",
    },
    {
        "case_id": "INC-2026-0521",
        "title": "支付服务 CPU 缓慢打满（内存泄漏）",
        "services": ["payment-service"],
        "symptom": "CPU 从 35% 缓慢爬升到 82%，接口 RT 同步上升，重启后重现",
        "keywords": ["cpu", "内存", "泄漏", "缓慢", "上升", "重启", "占用", "负载"],
        "root_cause": "支付回调协程未正确释放，goroutine 泄漏导致 GC 压力持续升高，CPU 被 GC 占满。",
        "resolution": "1) 临时重启 + 扩容副本；2) 修复回调链路 context 未 cancel 的问题；3) 加 goroutine 数监控告警。",
        "mttr_min": 95, "occurred_at": "2026-05-21",
    },
]


def match_cases(symptom: str, service: str | None, top_k: int) -> list[dict]:
    """按关键词重合度排序（外加服务名命中加权）。

    这里刻意不用向量检索：案例库只有几条，关键词打分足够，且零依赖。
    v0.3 换成 RAG 索引时保持函数签名不变即可。
    """
    scored = []
    text = (symptom or "").lower()
    for c in INCIDENT_CASES:
        score = sum(1 for k in c["keywords"] if k in text)
        if service and service in c["services"]:
            score += 2
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    hits = scored[: max(1, top_k)]
    if not hits:  # 一个都没命中时给出最近案例兜底，避免 Agent 误判为「无历史」
        hits = [(0, c) for c in INCIDENT_CASES[: max(1, top_k)]]
    return [
        {
            "case_id": c["case_id"], "title": c["title"], "occurred_at": c["occurred_at"],
            "services": c["services"], "symptom": c["symptom"],
            "root_cause": c["root_cause"], "resolution": c["resolution"],
            "mttr_min": c["mttr_min"], "match_score": s,
        }
        for s, c in hits
    ]
