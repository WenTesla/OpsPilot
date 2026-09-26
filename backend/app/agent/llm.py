"""LLM 适配器。

- 配置了 OPENAI_API_KEY 且装有 langchain-openai → 使用真实大模型（支持工具调用 + 流式）
- 否则 → 使用内置 MockChatModel：仍会调用 RAG 工具、基于检索片段生成带引用的回答，
  保证在没有任何外部依赖/密钥的情况下整条链路可跑通
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .. import config

SYSTEM_PROMPT = """你是企业内部的智能运维助手 OpsPilot。

工作原则：
1. 涉及故障处置、SOP、历史案例的问题，必须先调用 search_ops_knowledge 检索知识库，不要凭空回答；
2. 回答中引用知识库内容时，用 [编号] 标注来源，编号与检索结果一致；
3. 检索结果不足以支撑结论时，明确说明"知识库未覆盖"，并给出通用的排查思路；
4. 回答用中文，结构化呈现（现象 → 排查步骤 → 处置建议），给出可执行命令时标注风险。
"""


# ============================ Mock 模型 ============================
class MockChatModel(BaseChatModel):
    """无密钥时的本地兜底模型：会发起工具调用，并按检索结果汇总回答。"""

    tools: List[Any] = []
    model_name: str = "mock-ops-model"

    @property
    def _llm_type(self) -> str:
        return "mock-ops-model"

    def bind_tools(self, tools: list, **kwargs) -> "MockChatModel":
        clone = self.model_copy(update={"tools": list(tools)})
        return clone

    # ---- 核心生成 ----
    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        msg = self._next_message(messages)
        if getattr(msg, "tool_calls", None):
            chunks = [
                {
                    "name": tc["name"],
                    "args": json.dumps(tc.get("args", {}), ensure_ascii=False),
                    "id": tc.get("id"),
                    "index": i,
                }
                for i, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(content=msg.content))

    def _next_message(self, messages: List[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        last_human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

        # 第一轮：调用检索工具
        if not tool_msgs and self.tools:
            args = {"query": (last_human or "")[:200]}
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": self.tools[0].name,
                    "args": args,
                    "id": "call_mock_1",
                    "type": "tool_call",
                }],
            )

        # 第二轮：基于检索结果汇总
        if tool_msgs:
            return AIMessage(content=self._compose(last_human, tool_msgs))

        return AIMessage(content=self._fallback(last_human))

    # ---- 汇总逻辑 ----
    @staticmethod
    def _compose(query: str, tool_msgs: List[ToolMessage]) -> str:
        raw = "\n\n".join(m.content for m in tool_msgs)
        if "未找到高置信" in raw or "未找到" in raw[:80]:
            return (
                f"知识库中暂未检索到与「{query[:60]}」相关的内容。\n\n"
                "通用排查建议：\n"
                "1. 先确认影响面（单实例 / 多实例 / 单可用区）\n"
                "2. 对比近期变更（发布、配置、依赖升级）\n"
                "3. 保留现场后优先止损，再定位根因\n\n"
                "如需覆盖该场景，可在左侧上传对应的 SOP / 复盘文档（PDF 或 MD）。"
            )
        # 抽取各条资料的首段作为摘要
        blocks = re.split(r"\n(?=\[\d+\])", raw)
        items = [b.strip() for b in blocks if re.match(r"^\[\d+\]", b.strip())]
        lines = [f"针对「{query[:40]}」，知识库中检索到 {len(items)} 条相关资料：", ""]
        for b in items[:5]:
            head, _, body = b.partition("\n")
            head = head.strip()
            body = body.strip()
            # chunk 文本首行是章节路径，摘要里去掉与 head 重复的部分
            loc = head.split("》", 1)[-1].strip()
            if body.startswith(loc):
                body = body[len(loc):].strip()
            snippet = re.sub(r"\s+", " ", body)[:180].rstrip("-… ")
            lines.append(f"- {head}：{snippet}…")
        lines += [
            "",
            "> 说明：当前未配置 OPENAI_API_KEY，以上为**本地 Mock 模型**基于检索片段的摘要式汇总；"
            "配置大模型后将生成完整的推理过程与处置步骤。",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fallback(query: str) -> str:
        return f"（本地 Mock 模式）已收到你的问题：{query[:80]}"


# ============================ 模型工厂 ============================
def get_chat_model():
    if config.OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
                streaming=True,
            )
        except ImportError:
            print("[warn] 检测到 OPENAI_API_KEY 但未安装 langchain-openai，回退到 Mock 模型")
    return MockChatModel()


def model_label(model) -> str:
    return getattr(model, "model_name", None) or config.LLM_MODEL if not isinstance(model, MockChatModel) else "mock-ops-model"
