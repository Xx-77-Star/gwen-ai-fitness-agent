from typing import Any

from pydantic import ValidationError

from app.agent.nodes.memory_prompt import build_memory_prompt
from app.agent.nodes.tool_protocol import build_tool_protocol_messages
from app.agent.state import AgentMessage, AgentState, ToolCall
from app.llm.client import LLMClient
from app.llm.types import LLMResponse
from app.schemas.user import UserProfileResponse
from app.tools.registry import ToolRegistry

_SYSTEM_PROMPT = """你是 FitLife AI 的 Tool Decision Node。
你的职责是根据用户问题、已有上下文、可用工具和已执行的工具结果决定下一步：
- 如果不需要工具，直接返回回答草稿；
- 如果用户询问当前日期、时间、今天/昨天/明天、本周训练安排或训练连续天数，
  优先调用 get_current_time；
- 如果用户询问训练打卡历史或需要根据最近训练肌群安排训练，调用 get_workout_history_context；
- 如果需要查询用户训练记录、训练清单或训练统计，调用合适的只读训练工具；
- 如果用户问题包含多个相互独立、可以同时完成的任务，可以在同一轮返回多个 tool_calls；
- 多工具调用必须保留任务需要的工具和合理执行顺序，不要为了凑数量重复调用相同工具；
- 如果上一轮工具结果已经足够，不要再调用工具。
工具参数只允许工具 schema 定义的字段，绝对不能提供 user_id。
不要伪造工具结果，不要声称工具已经执行。
如果天气问题没有坐标，优先使用长期记忆 weather_location/city。
没有城市时先调用 geocode_city，再调用 get_current_weather。
回答使用简体中文。
"""


async def tool_decision_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
    tool_registry: ToolRegistry,
) -> dict[str, Any]:
    """Ask the LLM to choose between a tool call and a direct answer draft."""
    available_tools = tool_registry.schema_payloads()
    messages = _build_messages(state)
    try:
        response = await llm_client.complete(
            system_prompt=build_decision_prompt(state.get("profile"), state),
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )
    except Exception:
        return _decision_error("llm_decision_failed", "Tool decision request failed")

    if not isinstance(response, LLMResponse):
        return _decision_error("invalid_llm_response", "Tool decision response is invalid")

    if not response.tool_calls:
        return {
            "tool_calls": [],
            "answer_draft": response.content,
            "stop_reason": "completed",
        }

    validated_calls: list[ToolCall] = []
    for call in response.tool_calls:
        if not call.call_id or not call.name:
            return _decision_error("invalid_tool_call", "Tool call is invalid")
        if call.name not in set(tool_registry.names()):
            return _decision_error("unknown_tool", "Requested tool is not registered")
        if "user_id" in call.arguments:
            return _decision_error(
                "forbidden_argument", "user_id cannot be supplied in tool arguments"
            )
        definition = tool_registry.get(call.name)
        if definition is None:
            return _decision_error("unknown_tool", "Requested tool is not registered")
        try:
            definition.input_model.model_validate(call.arguments)
        except ValidationError:
            return _decision_error("invalid_arguments", "Tool arguments are invalid")
        validated_calls.append(
            ToolCall(
                call_id=call.call_id,
                tool_name=call.name,
                arguments=dict(call.arguments),
            )
        )

    return {
        "tool_calls": validated_calls,
        "answer_draft": response.content,
        "stop_reason": "tool_call",
    }


def build_decision_prompt(
    profile: UserProfileResponse | None,
    state: AgentState | None = None,
) -> str:
    """Include safe profile and retrieved Memory context for tool decisions."""
    sections = [_SYSTEM_PROMPT]
    if profile is not None:
        sections.append(
            "当前用户画像（仅供判断，不得放入工具参数）：\n"
            f"- 用户昵称：{profile.nickname}\n"
            f"- 健身等级：{profile.fitness_level}\n"
            f"- 目标：{profile.goal}"
        )
    if state is not None:
        sections.append(build_memory_prompt(state))
    return "\n\n".join(sections)


def _build_messages(state: AgentState) -> list[AgentMessage]:
    message = state.get("input", "").strip()
    if not message:
        raise ValueError("Tool decision requires a user message")

    history = _short_term_messages(state)
    context: list[AgentMessage] = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in history
        if turn.get("role") in {"user", "assistant"}
    ]
    if not any(turn.get("role") == "user" and turn.get("content") == message for turn in history):
        context.append({"role": "user", "content": message})

    metadata = state.get("metadata", {})
    tool_calls = list(metadata.get("_previous_tool_calls", state.get("tool_calls", [])))
    tool_results = list(state.get("tool_results", []))
    if tool_calls:
        context.extend(build_tool_protocol_messages(tool_calls, tool_results))
    return context


def _short_term_messages(state: AgentState) -> list[AgentMessage]:
    """Read the canonical short-term message list with a legacy fallback."""
    messages = list(state.get("messages", []))
    if messages:
        return messages
    return list(state.get("conversation_history", []))


def _decision_error(error_code: str, safe_message: str) -> dict[str, Any]:
    return {
        "tool_calls": [],
        "answer_draft": None,
        "stop_reason": "error",
        "metadata": {
            "tool_decision_error": {
                "code": error_code,
                "message": safe_message,
            }
        },
    }




