from typing import Any

from app.agent.nodes.response import response_node
from app.agent.nodes.tool_protocol import (
    build_assistant_tool_call_message,
    build_tool_protocol_messages,
    build_tool_result_message,
)


class RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def complete(self, *, messages, **kwargs):
        self.messages.append(messages)
        return "训练汇总：3 次，180 分钟。"


def tool_call() -> dict[str, Any]:
    return {
        "call_id": "call-1",
        "tool_name": "get_training_summary",
        "arguments": {"start_date": "2026-10-01", "end_date": "2026-10-07"},
    }


def tool_result() -> dict[str, Any]:
    return {
        "call_id": "call-1",
        "tool_call_id": "call-1",
        "tool_name": "get_training_summary",
        "status": "success",
        "result": {"record_count": 3, "total_duration_minutes": 180},
    }


async def test_response_node_converts_real_tool_results_to_llm_protocol() -> None:
    llm = RecordingLLM()

    result = await response_node(
        {
            "conversation_history": [],
            "tool_results": [tool_result()],
            "metadata": {"_previous_tool_calls": [tool_call()]},
            "profile": None,
        },
        llm_client=llm,
    )

    assert result["response"] == "训练汇总：3 次，180 分钟。"
    messages = llm.messages[0]
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call-1"
    assert '"record_count":3' in messages[1]["content"]


def test_assistant_tool_call_message_matches_openai_protocol() -> None:
    message = build_assistant_tool_call_message([tool_call()])

    assert message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "get_training_summary",
                    "arguments": '{"start_date":"2026-10-01","end_date":"2026-10-07"}',
                },
            }
        ],
    }


def test_tool_result_message_contains_correlation_id() -> None:
    message = build_tool_result_message(tool_result(), fallback_name="fallback")

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call-1"
    assert message["name"] == "get_training_summary"
    assert '"total_duration_minutes":180' in message["content"]


def test_tool_protocol_builds_complete_round() -> None:
    messages = build_tool_protocol_messages([tool_call()], [tool_result()])

    assert len(messages) == 2
    assert messages[0]["tool_calls"][0]["id"] == messages[1]["tool_call_id"]
