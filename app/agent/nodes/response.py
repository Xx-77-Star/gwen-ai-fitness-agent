from app.agent.nodes.memory_prompt import build_memory_prompt
from app.agent.nodes.tool_protocol import build_tool_protocol_messages
from app.agent.state import AgentState
from app.llm.client import LLMClient
from app.rag.service import KnowledgeRetrievalService
from app.schemas.user import UserProfileResponse

_SYSTEM_PROMPT = """你是 Gwen，一位温暖、专业、行动导向的 AI Fitness Coach。
只输出给用户看的最终回答。
我可以根据你的训练目标提供基础饮食建议，帮助你更好地支持训练效果。
我不是营养师；饮食建议基于通用健身知识，不替代专业医疗建议。
说话像一位温暖、专业、行动导向的教练：简洁、具体、可执行，不要机器报告式罗列系统限制。
绝不输出内部思考、Agent 工作流程、节点分析、工具选择过程、参数检查、缺失字段分析、日志或系统状态。
不要出现“我需要调用工具”“当前缺少信息”“无法继续执行”“请求参数”等内部术语。
信息不足时，不解释原因，自然地询问最多 2 个最关键的用户问题。
工具结果是事实依据：只使用实际 tool_results，不编造训练、天气、健康或工具数据；
有多个工具结果时，自然综合它们。
检索知识是优先参考依据，不得编造知识；引用知识时保留 source 信息。
引用 source 仅用于生成回答，不在用户回答中展示技术来源标签或内部上下文。
回答使用简体中文。涉及疼痛、伤病或疾病时，温和提醒用户寻求医生或专业医疗人员帮助。
"""


async def response_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
) -> dict[str, object]:
    """Generate only the user-facing final response."""
    messages = _short_term_messages(state)
    tool_results = list(state.get("tool_results", []))
    if state.get("answer_draft") is not None:
        draft = _user_safe_draft(str(state["answer_draft"]))
        draft_message = {"role": "assistant", "content": draft}
        return {
            "response": draft,
            "messages": [*messages, draft_message],
            "conversation_history": [*messages, draft_message],
            "stop_reason": "completed",
        }

    metadata = state.get("metadata", {})
    tool_calls = list(metadata.get("_previous_tool_calls", state.get("tool_calls", [])))
    tool_messages = build_tool_protocol_messages(tool_calls, tool_results) if tool_results else []
    generated = await llm_client.complete(
        system_prompt=build_system_prompt(state.get("profile"), state),
        messages=[*messages, *tool_messages],
    )
    response = _user_safe_draft(str(generated))
    assistant_turn = {"role": "assistant", "content": response}
    return {
        "response": response,
        "messages": [*messages, assistant_turn],
        "conversation_history": [*messages, assistant_turn],
        "stop_reason": "completed",
    }


def _short_term_messages(state: AgentState) -> list[dict[str, object]]:
    """Read canonical short-term memory with compatibility for older callers."""
    messages = list(state.get("messages", []))
    if messages:
        return messages
    return list(state.get("conversation_history", []))


def build_system_prompt(
    profile: UserProfileResponse | None,
    state: AgentState | None = None,
) -> str:
    """Add safe user profile, Memory, and retrieved knowledge to the final prompt."""
    sections = [_SYSTEM_PROMPT]
    if profile is not None:
        sections.append(
            "当前用户画像：\n"
            f"- 用户昵称：{profile.nickname}\n"
            f"- 年龄：{profile.age}\n"
            f"- 健身等级：{profile.fitness_level}\n"
            f"- 目标：{profile.goal}\n"
            f"- 每周训练频次：{profile.training_frequency} 次"
        )
    if state is not None:
        sections.append(build_memory_prompt(state))
        knowledge_context = KnowledgeRetrievalService.format_context(state.get("rag_context", []))
        if knowledge_context:
            sections.append(knowledge_context)
    return "\n\n".join(sections)


def _user_safe_draft(value: str) -> str:
    """Remove obvious reasoning artifacts before exposing a response to users."""
    forbidden = (
        "我需要调用",
        "需要调用工具",
        "tool_decision",
        "tool_execution",
        "agent node",
        "agent节点",
        "节点分析",
        "决策日志",
        "参数检查",
        "缺失参数",
        "缺少参数",
        "当前缺少",
        "信息不足",
        "数据不足",
        "请求参数",
        "系统限制",
        "无法继续执行",
    )
    for phrase in ("目前数据不足以判断长期趋势，", "目前数据不足以判断长期趋势"):
        value = value.replace(phrase, "")
    cleaned_lines = []
    for line in value.splitlines():
        normalized = line.strip()
        if normalized and any(term.lower() in normalized.lower() for term in forbidden):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or "为了帮你安排更合适的训练，我想先了解一下你的训练目标（增肌/减脂/提升体能）。"
