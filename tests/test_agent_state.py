from typing import Any

from app.agent.state import (
    AgentMessage,
    AgentState,
    StopReason,
    ToolCall,
    ToolResult,
    append_list,
    replace_list,
)


def test_tool_calling_lifecycle_fields_are_present() -> None:
    state: AgentState = {
        "input": "查询最近训练",
        "tool_calls": [
            ToolCall(
                call_id="call-1",
                tool_name="get_last_training_record",
                arguments={},
            )
        ],
        "tool_results": [
            ToolResult(
                call_id="call-1",
                tool_call_id="call-1",
                tool_name="get_last_training_record",
                status="success",
                result={"record": None},
            )
        ],
        "tool_rounds": 1,
        "tool_call_count": 1,
        "answer_draft": "根据最近训练记录...",
        "stop_reason": "completed",
        "messages": [
            {"role": "user", "content": "查询最近训练"},
            {"role": "assistant", "content": None, "tool_calls": []},
        ],
    }

    assert state["tool_calls"][0]["call_id"] == "call-1"
    assert state["tool_results"][0]["tool_call_id"] == "call-1"
    assert state["tool_rounds"] == 1
    assert state["tool_call_count"] == 1
    assert state["answer_draft"] == "根据最近训练记录..."
    assert state["stop_reason"] == "completed"
    assert state["messages"][0]["role"] == "user"


def test_tool_calling_lifecycle_preserves_existing_fields() -> None:
    state: AgentState = {
        "input": "今天怎么训练？",
        "user_id": "user-001",
        "profile": None,
        "profile_loaded": False,
        "fitness_level": "unknown",
        "conversation_history": [],
        "rag_context": [],
        "tool_calls": [],
        "tool_results": [],
        "tool_rounds": 0,
        "tool_call_count": 0,
        "answer_draft": None,
        "stop_reason": "completed",
        "messages": [],
        "intent": "training",
        "response": "安排一次全身力量训练。",
        "metadata": {"phase": "d1"},
    }

    assert state["input"] == "今天怎么训练？"
    assert state["conversation_history"] == []
    assert state["rag_context"] == []
    assert state["tool_results"] == []
    assert state["response"] == "安排一次全身力量训练。"


def test_tool_calling_state_reducers_accumulate_tool_events() -> None:
    assert append_list([1], [2, 3]) == [1, 2, 3]
    assert replace_list([{"role": "user"}], [{"role": "assistant"}]) == [{"role": "assistant"}]
    assert replace_list([{"role": "user"}], []) == []


def test_tool_result_type_supports_lifecycle_correlation() -> None:
    result: ToolResult = {
        "call_id": "call-1",
        "tool_call_id": "call-1",
        "tool_name": "list_training_records",
        "status": "error",
        "result": None,
        "error": "invalid_arguments",
    }

    assert result["call_id"] == "call-1"
    assert result["tool_call_id"] == "call-1"
    assert result["status"] == "error"
    assert result["error"] == "invalid_arguments"


def test_message_type_supports_assistant_tool_call_and_tool_reply() -> None:
    call: ToolCall = {
        "call_id": "call-1",
        "tool_name": "get_training_summary",
        "arguments": {},
    }
    assistant: AgentMessage = {"role": "assistant", "content": None, "tool_calls": [call]}
    tool_reply: AgentMessage = {
        "role": "tool",
        "content": {"status": "success", "data": {"record_count": 1}},
        "tool_call_id": "call-1",
    }

    assert assistant["tool_calls"] == [call]
    assert tool_reply["tool_call_id"] == "call-1"


def test_stop_reason_values_cover_lifecycle_terminal_states() -> None:
    reasons: tuple[StopReason, ...] = (
        "completed",
        "tool_call",
        "tool_limit",
        "error",
        "cancelled",
        "unknown",
    )
    expected = {
        "completed",
        "tool_call",
        "tool_limit",
        "error",
        "cancelled",
        "unknown",
    }

    assert all(reason in expected for reason in reasons)


def test_existing_tool_result_shape_remains_accepted() -> None:
    result: ToolResult = {
        "tool_name": "legacy_tool",
        "status": "success",
        "result": {"ok": True},
    }
    any_result: Any = result

    assert any_result["tool_name"] == "legacy_tool"
