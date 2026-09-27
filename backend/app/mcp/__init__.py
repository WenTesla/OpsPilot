"""MCP（Model Context Protocol）活数据工具接入层 —— v0.2。

    app/mcp/
      settings.py   配置项
      client.py     MultiServerMCPClient 封装（单例 / 安全加载 / 降级）
      tools.py      只读校验、超时兜底、调用事件上报

不装 langchain-mcp-adapters、或不启动 mcp_servers/ 时，`load_tools()` 返回空列表，
Agent 自动退回纯 RAG 行为，等价于 v0.1。
"""

from .client import load_tools, reload_tools
from .tools import CURRENT_TOOL_EVENTS

__all__ = ["load_tools", "reload_tools", "CURRENT_TOOL_EVENTS"]
