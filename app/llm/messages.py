from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.types import LLMToolCall

__all__ = [
    "AssistantMessage",
    "LLMMessage",
    "SystemMessage",
    "ToolResultMessage",
    "UserMessage",
]


class LLMMessage(BaseModel):
    """Base transport-neutral message used by the LLM gateway."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]


class SystemMessage(LLMMessage):
    """System instructions supplied outside the conversation turns."""

    role: Literal["system"] = "system"
    content: str = Field(min_length=1)


class UserMessage(LLMMessage):
    """One user conversation turn."""

    role: Literal["user"] = "user"
    content: str = Field(min_length=1)


class AssistantMessage(LLMMessage):
    """Assistant text and/or tool calls returned by the LLM.

    The content can be null only when the message contains tool calls.
    """

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_call_turn(self) -> "AssistantMessage":
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        return self


class ToolResultMessage(LLMMessage):
    """Result of one tool call sent back to the LLM.

    ``content`` is JSON-compatible application data. The OpenAI-compatible
    client serializes non-string values as JSON text.
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str = Field(min_length=1)
    content: Any
    name: str | None = None


MessageRole = Literal["system", "user", "assistant", "tool"]
LLMMessageInput = (
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage | dict[str, Any]
)
