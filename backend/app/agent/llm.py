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

你手上有两类证据，按需取用，不要偏废：

A. 活数据工具（query_metrics / list_active_alerts / search_logs / get_service_info /
   list_services / get_error_breakdown / search_incident_cases）
   用于回答「现在/最近发生了什么」「某个指标是多少」「根因可能是啥」。
   涉及现状的结论必须建立在工具返回值之上，禁止编造数值。

B. 知识库工具（search_ops_knowledge）
   用于回答「该怎么处置」「流程是什么」，必须带上 [编号] 引用。

典型排障顺序：先用活数据确认现象 → 再查知识库取处置步骤 → 给出现象 + 依据 + 步骤的完整结论。
工具返回 error 时换其它工具，或如实告知该数据当前不可用，不要假装查到了。
当返回结果中 source.mock 为 true 时，结论里要说明「数据来源为演示数据，非真实监控」。
"""


# Mock 模型在无 API Key 时的「意图 → 工具」路由。
# 存在的理由：真实模型靠 function-calling 自选工具，Mock 没有这个能力，
# 若不加路由，Mock 只会调一次 RAG 工具就汇总，v0.2 的多工具编排在演示环境里就成了摆设。
KEYWORD_ROUTES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("告警", "报警", "alert", "响没响", "有没有报警"), ("list_active_alerts",)),
    (("日志", "报错", "堆栈", "exception", "错误"),
     ("get_error_breakdown", "search_logs")),
    (("cpu", "内存", "负载", "延迟", "qps", "错误率", "指标", "多少", "现在怎么样"),
     ("query_metrics",)),
    (("依赖", "拓扑", "谁调用", "上游", "下游", "变更", "发布", "版本"),
     ("get_service_info",)),
    (("之前", "上次", "历史", "复盘", "以前", "有没有过"), ("search_incident_cases",)),
    (("有哪些服务", "服务列表", "哪些服务"), ("list_services",)),
]

RAG_TOOL_NAME = "search_ops_knowledge"

# 拿到活数据后再查知识库时，给原始问题补的检索词。
# 不补的话，「CPU 使用率怎么样」这种问法检索不到《SOP-主机资源异常处理》——
# 文档里写的是「load 飙高 / cpu 打满」，字面上对不上用户的口语。
MCP_QUERY_HINTS = {
    "query_metrics": "资源使用率过高 打满 排查 处置",
    "list_active_alerts": "告警 响应流程 SOP",
    "search_logs": "错误日志 报错 排查",
    "get_error_breakdown": "错误率 5xx 异常 排查",
    "search_incident_cases": "历史故障 复盘 根因",
    "get_service_info": "服务依赖 变更回滚 处置",
    "list_services": "服务巡检 健康检查",
}

# Mock 路由里可识别的服务名（与 mcp_servers/fixtures.py 的服务清单保持一致）
SERVICE_HINTS = [
    "order-service", "payment-service", "inventory-service",
    "user-service", "mysql-primary", "redis-cache", "elasticsearch", "api-gateway",
]


# ============================ Mock 模型 ============================
class MockChatModel(BaseChatModel):
    """无密钥时的本地兜底模型：会发起工具调用，并按工具结果汇总回答。

    支持多轮工具调用（v0.2）：先按关键词挑一个活数据工具，拿到结果后再去查知识库，
    最后把「现场数据」与「SOP 依据」拼成一份完整结论 —— 与真实模型的行为保持一致。
    """

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

    def _tool_by_name(self, name: str):
        return next((t for t in self.tools if getattr(t, "name", None) == name), None)

    def _pick_tools(self, query: str) -> list[str]:
        """按关键词挑工具。返回空表示走 v0.1 的默认行为（只调 RAG）。"""
        q = (query or "").lower()
        for keys, tools in KEYWORD_ROUTES:
            if any(k.lower() in q for k in keys):
                picked = [t for t in tools if self._tool_by_name(t)]
                if picked:
                    return picked[:1]          # 一次只挑一个，避免第二轮缺少上下文就乱调
        return []

    def _next_message(self, messages: List[BaseMessage]) -> AIMessage:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        last_human = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        used = [getattr(m, "name", "") or "" for m in tool_msgs]

        # 第一轮：按意图挑活数据工具，挑不到就退回 v0.1 的「只查知识库」
        if not tool_msgs and self.tools:
            picked = self._pick_tools(last_human)
            if picked:
                return self._call(picked[0], self._mcp_args(picked[0], last_human))
            args = {"query": (last_human or "")[:200]}
            return self._call(self._rag_tool_name(), args)

        # 第二轮：已经拿到活数据 → 再去知识库要处置步骤
        if tool_msgs and not any(u == RAG_TOOL_NAME for u in used) and self._tool_by_name(RAG_TOOL_NAME):
            last_tool = used[-1]
            hint = MCP_QUERY_HINTS.get(last_tool, "")
            svc = next((s for s in SERVICE_HINTS if s in last_human), "")
            return self._call(RAG_TOOL_NAME, {"query": f"{last_human} {svc} {hint}".strip()[:200]})

        # 第三轮：汇总
        if tool_msgs:
            return AIMessage(content=self._compose(last_human, tool_msgs))

        return AIMessage(content=self._fallback(last_human))

    def _rag_tool_name(self) -> str:
        return self._tool_by_name(RAG_TOOL_NAME).name if self._tool_by_name(RAG_TOOL_NAME) else (self.tools[0].name if self.tools else "")

    def _mcp_args(self, tool: str, query: str) -> dict:
        """给 Mock 路由出来的工具拼最小可用参数（真实模型不需要这一步）。"""
        svc = next((s for s in SERVICE_HINTS if s in query), None)
        q = query.lower()
        if tool == "query_metrics":
            metric = "cpu"
            for kw, m in (("内存", "memory"), ("延迟", "latency_p99"), ("错误率", "error_rate"),
                          ("qps", "qps"), ("流量", "qps"), ("cpu", "cpu")):
                if kw in q:
                    metric = m
                    break
            args = {"metric": metric}
            if svc:
                args["service"] = svc
            return args
        if tool in ("list_active_alerts", "get_service_info", "search_logs", "get_error_breakdown"):
            if tool in ("list_active_alerts", "list_services") and not svc:
                return {}
            return {"service": svc} if svc else {"service": SERVICE_HINTS[0]}
        if tool == "search_incident_cases":
            return {"symptom": query[:120]}
        if tool == "list_services":
            return {}
        return {"query": query[:200]}

    @staticmethod
    def _call(name: str, args: dict) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_mock_{abs(hash((name, str(args)))) % 10**6}",
                         "type": "tool_call"}],
        )

    # ---- 汇总逻辑 ----
    @staticmethod
    def _fmt_observation(name: str, content: str) -> str:
        """把 MCP 工具返回的 JSON 压成 1~2 行要点。

        刻意不把 JSON 原样贴给用户：那是给 LLM 看的，人看不懂也不需要看。
        """
        try:
            obj = json.loads(content)
        except Exception:
            return f"- {name}：`{content[:160]}`"
        if not isinstance(obj, dict):
            return f"- {name}：{str(obj)[:160]}"

        src = obj.get("source", {}) or {}
        tag = f"{src.get('server', '?')}.{src.get('tool', name)}"
        mock_note = "（演示数据）" if src.get("mock") else ""
        if obj.get("error"):
            return f"- ✗ {tag} 调用失败：{obj['error']}"

        d = obj.get("data", {}) or {}
        if not isinstance(d, dict):
            return f"- {tag}{mock_note}：{str(d)[:160]}"
        # 注意：hint 是工具写给 LLM 的（"无告警时不要臆测"），对人没用，
        # 所以放在最后兜底，优先展示数据要点。
        if d.get("stats"):
            s = d["stats"]
            return (f"- {tag}{mock_note}：{d.get('metric', '')} "
                    f"avg={s.get('avg')} max={s.get('max')} p95={s.get('p95')} trend={d.get('trend')}")
        if d.get("alerts"):
            items = "；".join(f"{a['severity']}·{a['name']}({a['service']}) 值{a['value']}/阈值{a['threshold']}"
                             for a in d["alerts"][:3])
            return f"- {tag}{mock_note}：共 {d.get('count', len(d['alerts']))} 条告警 — {items}"
        if d.get("cases"):
            items = "；".join(f"{c['case_id']} {c['title']}（MTTR {c['mttr_min']}min）" for c in d["cases"][:3])
            return f"- {tag}{mock_note}：历史相似案例 — {items}"
        if d.get("services"):
            names = "、".join(s["name"] for s in d["services"][:8])
            bad = [s["name"] for s in d["services"] if s.get("status") != "healthy"]
            return f"- {tag}{mock_note}：{names}" + (f"；异常服务：{'、'.join(bad)}" if bad else "")
        if "samples" in d:
            top = d.get("top_patterns") or []
            head = top[0]["pattern"] if top else "无"
            return f"- {tag}{mock_note}：命中 {d.get('total', 0)} 条，最高频模式「{head}」"
        if d.get("by_type") is not None:
            return f"- {tag}{mock_note}：错误 {d.get('total', 0)} 条，集中类型 {list(d['by_type'])[:3]}"
        if obj.get("hint"):
            return f"- {tag}{mock_note}：{obj['hint']}"
        return f"- {tag}{mock_note}：{json.dumps(d, ensure_ascii=False)[:200]}"

    @staticmethod
    def _compose(query: str, tool_msgs: List[ToolMessage]) -> str:
        observations: list[str] = []
        rag_raw = ""
        for m in tool_msgs:
            name = getattr(m, "name", "") or ""
            if name == RAG_TOOL_NAME:
                rag_raw += str(m.content) + "\n"
            else:
                observations.append(MockChatModel._fmt_observation(name, str(m.content)))

        no_kb = ("未找到高置信" in rag_raw) or ("未找到" in rag_raw[:80])

        # 场景一：只有活数据，没有可用知识（第二轮按规定还会去查一次知识库，这里兜住异常情况）
        if observations and no_kb:
            return (
                "线上数据已取到：\n\n" + "\n".join(observations) + "\n\n"
                f"知识库中暂未检索到与「{query[:60]}」直接相关的处置规范。\n\n"
                "通用排查建议：\n"
                "1. 先确认影响面（单实例 / 多实例 / 单可用区）\n"
                "2. 对比近期变更（发布、配置、依赖升级）\n"
                "3. 保留现场后优先止损，再定位根因\n\n"
                "如需覆盖该场景，可在左侧上传对应的 SOP / 复盘文档（PDF 或 MD）。"
            )

        # 场景二：v0.1 的纯知识库路径（没有活数据时输出格式保持原样）
        if not observations:
            raw = rag_raw
            if no_kb:
                return (
                    f"知识库中暂未检索到与「{query[:60]}」相关的内容。\n\n"
                    "通用排查建议：\n"
                    "1. 先确认影响面（单实例 / 多实例 / 单可用区）\n"
                    "2. 对比近期变更（发布、配置、依赖升级）\n"
                    "3. 保留现场后优先止损，再定位根因\n\n"
                    "如需覆盖该场景，可在左侧上传对应的 SOP / 复盘文档（PDF 或 MD）。"
                )
            blocks = re.split(r"\n(?=\[\d+\])", raw)
            items = [b.strip() for b in blocks if re.match(r"^\[\d+\]", b.strip())]
            lines = [f"针对「{query[:40]}」，知识库中检索到 {len(items)} 条相关资料：", ""]
            for b in items[:5]:
                head, _, body = b.partition("\n")
                head, body = head.strip(), body.strip()
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

        # 场景三：活数据 + 知识库（v0.2 的目标路径）
        blocks = re.split(r"\n(?=\[\d+\])", rag_raw)
        items = [b.strip() for b in blocks if re.match(r"^\[\d+\]", b.strip())]
        lines = [f"针对「{query[:40]}」，先看现场再查规范：", "", "**一、现场数据（来源：MCP 活数据工具）**"]
        lines += observations
        lines += ["", "**二、知识库处置依据**"]
        if items:
            for b in items[:5]:
                head, _, body = b.partition("\n")
                head, body = head.strip(), body.strip()
                loc = head.split("》", 1)[-1].strip()
                if body.startswith(loc):
                    body = body[len(loc):].strip()
                snippet = re.sub(r"\s+", " ", body)[:180].rstrip("-… ")
                lines.append(f"- {head}：{snippet}…")
        else:
            lines.append("- （本轮未命中知识库条目）")
        lines += [
            "",
            "> 说明：当前未配置 OPENAI_API_KEY，以上为**本地 Mock 模型**的拼接式汇总；"
            "配置大模型后会给出完整推理与可执行步骤。演示数据源的采集值并非真实监控。",
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
