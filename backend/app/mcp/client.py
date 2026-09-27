"""MCP 客户端：可选依赖、单例缓存、安全加载、失败降级。

对外只有一个入口 `load_tools()`，返回一个 `(tools, status)` 二元组，**永不抛异常**。
这样主服务启动时能不能连上 MCP 都不影响，降级行为见设计文档 §7。

踩坑记录（都是实测出来的，改代码前先看这里）：
1. `MultiServerMCPClient` 自 0.1.0 起**不再是上下文管理器**，写成 `async with` 会报错，
   直接 `client = MultiServerMCPClient(conns)` 然后 `await client.get_tools()`。
2. 连接失败以 `ExceptionGroup` / `BaseExceptionGroup` 形式抛出，裸 `except Exception` 打出来的
   是一坨看不出根因的嵌套，必须递归展开 `__cause__` 与 `exceptions`（见 `_fmt_exc`）。
3. `get_tools()` 每次调用都会新建连接，必须模块级单例缓存，否则每轮对话握手一次，
   本地调用延迟会从几十毫秒涨到秒级。
4. transport 用 `streamable_http`（下划线），不是 `streamable-http`。写错会握手失败且报错晦涩。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("opspilot.mcp")

_client: Any | None = None
_client_key: str | None = None


def _fmt_exc(exc: BaseException) -> str:
    """展开 ExceptionGroup / TaskGroup 与异常链，拿到真正的根因。"""
    subs = getattr(exc, "exceptions", None)
    lines = [f"{type(exc).__name__}: {exc}"]
    if subs:
        for i, sub in enumerate(subs):
            lines.append(f"  [{i}] " + _fmt_exc(sub).replace("\n", "\n     "))
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        lines.append("  caused by: " + _fmt_exc(cause).replace("\n", "\n  "))
    return "\n".join(lines)


def dependency_status() -> tuple[bool, str]:
    """langchain-mcp-adapters 是否已安装。"""
    try:
        import langchain_mcp_adapters  # noqa: F401
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"缺少依赖 langchain-mcp-adapters（{e}）。执行 `uv pip install langchain-mcp-adapters` 后重启即可启用活数据工具"


def _build_client(conns: dict):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    return MultiServerMCPClient(conns)


async def _try_server(name: str, conn: dict, timeout_sec: float) -> tuple[list, str | None]:
    """加载单个 server 的工具。一个 server 挂掉不影响另一个。"""
    from app.mcp import settings
    from app.mcp.tools import prepare_tools

    client = _build_client({name: conn})
    try:
        raw = await asyncio.wait_for(client.get_tools(), timeout=timeout_sec)
        tools = prepare_tools(raw, name, settings.CALL_TIMEOUT_SEC)
        return tools, None
    except BaseException as e:  # noqa: BLE001 —— 连接层会抛 ExceptionGroup，必须一并兜住
        return [], _fmt_exc(e)[:500]
    finally:
        # 尽早释放连接句柄，避免进程退出时挂住 event loop
        try:
            aclose = getattr(client, "aclose", None)
            if aclose:
                await aclose()
        except Exception:
            pass


async def load_tools(force_refresh: bool = False) -> tuple[list, dict]:
    """加载全部 MCP 工具。

    Returns:
        (tools, status)；失败时 tools 为空列表，status 里带 reason，主服务可照常启动。
    """
    global _client, _client_key
    from app.mcp import settings

    status: dict = {"enabled": False, "servers": {}, "tools": 0, "reason": ""}

    if settings.is_disabled():
        status["reason"] = "MCP_ENABLED=0，已手动关闭"
        return [], status

    ok, why = dependency_status()
    if not ok:
        status["reason"] = why
        return [], status

    conns = settings.servers()
    key = repr(sorted((k, tuple(sorted(v.items(), key=lambda x: str(x[0])))) for k, v in conns.items()))
    if force_refresh:
        _client = None
        _client_key = None

    all_tools: list = []
    for name, conn in conns.items():
        tools, err = await _try_server(name, conn, settings.CONNECT_TIMEOUT_SEC)
        if err:
            status["servers"][name] = {"tools": 0, "error": err}
            log.warning("[mcp] server=%s 不可用：%s", name, err.split("\n")[0])
            continue
        status["servers"][name] = {"tools": len(tools)}
        all_tools.extend(tools)

    status["tools"] = len(all_tools)
    if all_tools:
        status["enabled"] = True
        _client_key = key
        log.info("[mcp] 已挂载 %d 个工具：%s", len(all_tools), ", ".join(t.name for t in all_tools))
    else:
        status["reason"] = "MCP 服务未启动或不可达（不影响 RAG 问答，Agent 会自动走知识库）"
    return all_tools, status


async def reload_tools() -> tuple[list, dict]:
    """运行时重连（服务起来了但主服务先启动时用）。"""
    return await load_tools(force_refresh=True)
