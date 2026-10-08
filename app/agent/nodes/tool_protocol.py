import json
from collections.abc import Iterable, Sequence
from typing import Any

from app.agent.state import ToolCall, ToolResult


def build_assistant_tool_call_message(calls: Sequence[ToolCall]) -> dict[str, Any]:
    """Build one OpenAI-compatible assistant message for a tool-call round."""
    tool_calls = []
    for call in calls:
        call_id = call.get("call_id", "")
        name = call.get("tool_name", "")
        arguments = call.get("arguments", {})
        if not call_id or not name:
            raise ValueError("assistant tool call requires call_id and tool_name")
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }


def build_tool_result_messages(
    calls: Sequence[ToolCall],
    results: Iterable[ToolResult],
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool messages correlated by tool_call_id."""
    result_list = list(results)
    results_by_call_id = {
        result.get("tool_call_id") or result.get("call_id"): result for result in result_list
    }
    messages: list[dict[str, Any]] = []
    for call in calls:
        result = results_by_call_id.get(call.get("call_id"))
        if result is not None:
            messages.append(
                build_tool_result_message(result, fallback_name=call.get("tool_name", ""))
            )
    matched_ids = {result.get("tool_call_id") or result.get("call_id") for result in result_list}
    for result in result_list:
        if result.get("call_id") not in matched_ids:
            continue
    return messages


def build_tool_result_message(
    result: ToolResult,
    *,
    fallback_name: str = "",
) -> dict[str, Any]:
    """Build one OpenAI-compatible tool result message."""
    call_id = result.get("tool_call_id") or result.get("call_id")
    if not call_id:
        raise ValueError("tool result requires tool_call_id")
    content = result.get("result") if result.get("status") == "success" else result.get("error")
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": result.get("tool_name", fallback_name),
        "content": _json_content(content),
    }


def build_tool_protocol_messages(
    calls: Sequence[ToolCall],
    results: Iterable[ToolResult],
) -> list[dict[str, Any]]:
    """Build the assistant tool-call message followed by tool result messages."""
    if not calls:
        return []
    return [
        build_assistant_tool_call_message(calls),
        *build_tool_result_messages(calls, results),
    ]


def _json_content(content: Any) -> str:
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)
