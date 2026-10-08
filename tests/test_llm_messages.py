from app.llm.messages import AssistantMessage, SystemMessage, ToolResultMessage, UserMessage
from app.llm.types import LLMToolCall


def test_assistant_message_supports_tool_calls_without_content() -> None:
    message = AssistantMessage(
        tool_calls=[
            LLMToolCall(call_id="call-1", name="read_training", arguments={"limit": 3})
        ]
    )

    assert message.content is None
    assert message.tool_calls[0].call_id == "call-1"
    assert message.tool_calls[0].arguments == {"limit": 3}


def test_assistant_message_requires_content_or_tool_calls() -> None:
    try:
        AssistantMessage(content=None, tool_calls=[])
    except ValueError as exc:
        assert "content or tool_calls" in str(exc)
    else:
        raise AssertionError("empty assistant message should fail validation")


def test_tool_result_message_preserves_call_correlation_and_structured_data() -> None:
    message = ToolResultMessage(
        tool_call_id="call-1",
        name="read_training",
        content={"status": "success", "data": {"count": 1}},
    )

    assert message.role == "tool"
    assert message.tool_call_id == "call-1"
    assert message.name == "read_training"
    assert message.content == {"status": "success", "data": {"count": 1}}


def test_user_and_system_messages_are_strongly_typed() -> None:
    system = SystemMessage(content="FitLife system prompt")
    user = UserMessage(content="查询最近训练")

    assert system.role == "system"
    assert user.role == "user"
