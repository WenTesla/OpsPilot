"""LangGraph 主图：agent ⇄ tools（RAG 检索工具）循环。

结构：
    START → agent ──(需要工具)──▶ tools ──▶ agent ──(无工具调用)──▶ END
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from .. import config
from ..rag.pipeline import RAGPipeline, collect_citations
from .llm import SYSTEM_PROMPT, get_chat_model


class OpsPilotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: list[dict]


def build_rag_tool(pipeline: RAGPipeline):
    @tool
    def search_ops_knowledge(query: str, top_k: int = 5) -> str:
        """检索运维知识库（用户上传的 SOP、手册、复盘报告等 PDF/MD 文档）。

        当需要故障处置步骤、历史相似案例、根因分析参考时调用。
        返回带来源引用 [编号] 的知识片段。
        """
        result = pipeline.retrieve(query, top_k=top_k)
        context, citations = RAGPipeline.format_context(result)
        collect_citations(citations)
        return context

    return search_ops_knowledge


class OpsPilot:
    """RAG 工具 + 可选的 MCP 活数据工具。

    `extra_tools` 为 None 时行为与 v0.1 完全一致；RAG 工具始终排在第一位，
    既保证「先查规范」的默认倾向，也兼容 Mock 模型固定取 tools[0] 的行为。
    """

    def __init__(self, pipeline: RAGPipeline, extra_tools: list | None = None):
        self.pipeline = pipeline
        self.tool = build_rag_tool(pipeline)
        self.tools = [self.tool] + list(extra_tools or [])
        self.model = get_chat_model().bind_tools(self.tools)
        self.graph = self._build()

    def _build(self):
        model = self.model

        async def agent_node(state: OpsPilotState):
            msgs = state["messages"]
            if not any(getattr(m, "type", "") == "system" for m in msgs):
                msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
            resp = await model.ainvoke(msgs)
            return {"messages": [resp]}

        def route(state: OpsPilotState) -> str:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return END

        g = StateGraph(OpsPilotState)
        g.add_node("agent", agent_node)
        g.add_node("tools", ToolNode(self.tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=MemorySaver())

    def thread_config(self, conversation_id: str) -> dict:
        # 活数据工具加入后一次排障常需 2~3 轮工具调用，原 12 的上限会被 GraphRecursionError 打断
        return {"configurable": {"thread_id": conversation_id}, "recursion_limit": 24}

    async def stream(self, conversation_id: str, user_text: str):
        """产出 (mode, chunk) 流，供 SSE 消费。"""
        state = {"messages": [HumanMessage(content=user_text)], "retrieved": []}
        cfg = self.thread_config(conversation_id)
        async for item in self.graph.astream(state, cfg, stream_mode=["updates", "messages"]):
            yield item

    async def final_state(self, conversation_id: str) -> dict:
        return await self.graph.aget_state(self.thread_config(conversation_id))
