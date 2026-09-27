"""MCP 配置。

全部走环境变量 + `backend/.env`，与其它配置项一致（见 app/config.py）。
设计目标：MCP 能力**可选**。没装依赖、或 MCP 服务没起来，都不应影响主链路。
"""
from __future__ import annotations

from app import config


def _b(name: str, default: str) -> str:
    return getattr(config, name, default)


ENABLED = _b("MCP_ENABLED", "auto")            # auto | 1 | 0
MONITOR_URL = _b("MCP_MONITOR_URL", "http://127.0.0.1:8101/mcp")
LOG_URL = _b("MCP_LOG_URL", "http://127.0.0.1:8102/mcp")
MONITOR_TRANSPORT = _b("MCP_MONITOR_TRANSPORT", "streamable_http")
LOG_TRANSPORT = _b("MCP_LOG_TRANSPORT", "streamable_http")
CONNECT_TIMEOUT_SEC = float(_b("MCP_CONNECT_TIMEOUT_SEC", "3"))
CALL_TIMEOUT_SEC = float(_b("MCP_TIMEOUT_SEC", "15"))
MAX_RETRIES = int(_b("MCP_MAX_RETRIES", "2"))
MAX_RESULT_CHARS = int(_b("MCP_MAX_RESULT_CHARS", "4000"))
MAX_TOOL_DESC_CHARS = int(_b("MCP_MAX_TOOL_DESC_CHARS", "400"))


def is_disabled() -> bool:
    return str(ENABLED).strip().lower() in ("0", "false", "no", "off")


def servers() -> dict[str, dict]:
    """拼 MultiServerMCPClient 需要的 connections 字典。

    键值为 langchain-mcp-adapters 0.3.x 的格式（注意 transport 用下划线 streamable_http）。
    """
    return {
        "monitor": {
            "transport": MONITOR_TRANSPORT,
            "url": MONITOR_URL,
            "timeout": CALL_TIMEOUT_SEC,
            "sse_read_timeout": CALL_TIMEOUT_SEC * 4,
        },
        "log": {
            "transport": LOG_TRANSPORT,
            "url": LOG_URL,
            "timeout": CALL_TIMEOUT_SEC,
            "sse_read_timeout": CALL_TIMEOUT_SEC * 4,
        },
    }
