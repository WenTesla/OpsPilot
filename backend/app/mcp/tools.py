"""MCP 工具封装层。

职责：把 MCP 返回的原始 LangChain 工具，改造成「安全、可控、可观测」的版本。

三件事：
1. **只读校验** —— v0.2 的安全红线。任何名字含写操作动词的工具一律不暴露给 LLM。
2. **超时 + 异常兜底** —— 工具报错必须返回文本，不能抛异常。原因见文档 §4.7：
   /api/chat 是 SSE 流，异常会让整条流中断，前端表现为「回答到一半卡住」。
3. **可观测** —— 每次调用写进 ContextVar，由 SSE 推给前端渲染工具调用气泡。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from contextvars import ContextVar

from langchain_core.tools import BaseTool, StructuredTool

log = logging.getLogger("opspilot.mcp")

# 单次请求内的工具调用事件，供 SSE 消费（与 rag/pipeline.py 的 CURRENT_CITATIONS 同一套路）
CURRENT_TOOL_EVENTS: ContextVar[list | None] = ContextVar("opspilot_tool_events", default=None)

READ_PREFIXES = ("list_", "get_", "query_", "search_", "describe_", "fetch_", "count_")
WRITE_VERBS = (
    "restart", "scale", "rollback", "rollout", "exec", "delete", "remove", "update",
    "create", "kill", "deploy", "patch", "stop", "start", "write", "purge", "truncate",
)

MAX_DESC = 400
MAX_RESULT_CHARS = 4000


def is_readonly(name: str) -> bool:
    """名字里带写操作动词的一律不放行（误伤读工具的成本远低于误放写工具）。"""
    low = name.lower()
    bad = [v for v in WRITE_VERBS if v in low]
    if bad:
        log.warning("[mcp] 拦截疑似写操作工具 %s（命中动词：%s）", name, ", ".join(bad))
        return False
    return True


def summarize(text: str, limit: int = 160) -> str:
    """从工具返回的 JSON 里提炼**一行给人类看的数据要点**，用于前端气泡与日志。

    注意这里刻意不用 data.hint —— hint 是写给 LLM 的（"无告警时 count=0，不要臆测"），
    把它显示给用户纯属噪音。人在气泡里要看的是「几条告警、哪个服务、差多少」。
    """
    try:
        obj = json.loads(text)
    except Exception:
        return text.replace("\n", " ")[:limit]
    if not isinstance(obj, dict):
        return str(obj)[:limit]

    src = obj.get("source", {}) or {}
    tag = f"[{src.get('server', '?')}] "
    if obj.get("error"):
        return (f"✗ {obj['error']}")[:limit]

    d = obj.get("data", {}) or {}
    if not isinstance(d, dict):
        return (tag + str(d)[:limit])
    if d.get("stats"):
        s, trend = d["stats"], d.get("trend")
        return (f"{tag}{d.get('metric', '')} avg={s.get('avg')} max={s.get('max')} "
                f"p95={s.get('p95')} trend={trend}")[:limit]
    if d.get("alerts"):
        a = d["alerts"][0]
        return (f"{tag}{len(d['alerts'])} 条告警：{a['severity']}·{a['name']} "
                f"值{a['value']}/阈值{a['threshold']}")[:limit]
    if d.get("services"):
        bad = [s["name"] for s in d["services"] if s.get("status") != "healthy"]
        tail = f"，异常：{'、'.join(bad)}" if bad else ""
        return (f"{tag}{len(d['services'])} 个服务{tail}")[:limit]
    if d.get("cases"):
        c = d["cases"][0]
        return (f"{tag}命中历史案例 {c['case_id']} {c['title']}")[:limit]
    if "samples" in d:
        p = (d.get("top_patterns") or [{}])[0].get("pattern", "")
        return (f"{tag}日志 {d.get('total', 0)} 条，高频：「{p[:36]}」")[:limit]
    if d.get("by_type") is not None:
        return (f"{tag}错误 {d.get('total', 0)} 条，集中在「{d.get('top_type')}」")[:limit]
    if d.get("count") is not None:
        return (f"{tag}count={d['count']}")[:limit]
    return (tag + "字段：" + "、".join(list(d)[:5]))[:limit]


def extract_text(result: Any) -> str:
    """把工具的返回值压成纯文本。

    MCP 工具返回的是 content blocks（形如 [{"type":"text","text":"..."}]），
    直接 json.dumps 会给 LLM 喂一堆转义字符，必须先拆出真正的文本。
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "text" in result and isinstance(result["text"], str):
            return result["text"]
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False)


def _push_event(ev: dict) -> None:
    box = CURRENT_TOOL_EVENTS.get()
    if box is not None:
        box.append(ev)


def wrap_tool(tool: BaseTool, server: str, timeout_sec: float, max_chars: int = MAX_RESULT_CHARS) -> BaseTool:
    """把 MCP 原始工具包一层：超时控制 + 异常兜底 + 事件上报 + 结果截断。"""
    raw_name = tool.name

    async def _call(**kwargs):
        _push_event({"type": "tool_call", "name": raw_name, "server": server, "args": kwargs})
        try:
            result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=timeout_sec)
            text = extract_text(result)
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n…(结果已截断，原长度 {len(text)} 字符)"
            ok = True
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and obj.get("error"):
                    ok = False
            except Exception:
                pass
            _push_event({"type": "tool_result", "name": raw_name, "server": server,
                         "ok": ok, "summary": summarize(text)})
            return text
        except asyncio.TimeoutError:
            msg = f"工具 {raw_name} 调用超时（>{timeout_sec}s），请换其它方式获取信息或告知用户该数据暂不可用。"
            log.warning("[mcp] %s timeout after %.1fs", raw_name, timeout_sec)
            _push_event({"type": "tool_result", "name": raw_name, "server": server, "ok": False, "summary": msg})
            return f"错误：{msg}"
        except Exception as e:  # noqa: BLE001 —— 兜住一切，绝不让异常冒到 SSE 流里
            msg = f"工具 {raw_name} 调用失败：{type(e).__name__}: {e}"
            log.warning("[mcp] %s failed: %s", raw_name, e)
            _push_event({"type": "tool_result", "name": raw_name, "server": server, "ok": False, "summary": msg})
            return f"错误：{msg}"

    desc = (tool.description or raw_name).strip().replace("\n", " ")
    if len(desc) > MAX_DESC:
        desc = desc[:MAX_DESC].rstrip() + " …"

    # langchain-core 新旧版本参数名不一致：≤0.3 用 `coro`，1.x 改用 `coroutine`。
    # 写死任一都会在另一个版本上抛「Function and/or coroutine must be provided」。
    _params = inspect.signature(StructuredTool.from_function).parameters
    _fn_key = "coroutine" if "coroutine" in _params else "coro"

    return StructuredTool.from_function(  # type: ignore[call-arg]
        **{_fn_key: _call},
        name=raw_name,
        description=desc,
        args_schema=tool.args_schema,
        infer_schema=False,
    )


def prepare_tools(raw_tools: list[BaseTool], server: str, timeout_sec: float) -> list[BaseTool]:
    """过滤 + 包装一个 server 的全部工具。"""
    kept: list[BaseTool] = []
    for t in raw_tools:
        if not is_readonly(t.name):
            log.warning("[mcp] 丢弃非只读工具：%s（v0.2 只暴露只读工具）", t.name)
            continue
        kept.append(wrap_tool(t, server, timeout_sec))
    return kept
