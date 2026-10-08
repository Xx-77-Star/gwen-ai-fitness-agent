from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "unknown"]


class LLMToolCall(BaseModel):
    """Provider-neutral tool invocation emitted by an LLM."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMToolDefinition(BaseModel):
    """OpenAI-compatible function tool definition sent to an LLM."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: dict[str, Any]


class LLMUsage(BaseModel):
    """Token usage reported by an OpenAI-compatible provider."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(BaseModel):
    """Normalized response for text generation and tool calling."""

    model_config = ConfigDict(extra="allow")

    content: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "unknown"
    usage: LLMUsage | None = None
    raw: dict[str, Any] | None = None

    @property
    def requires_tool_execution(self) -> bool:
        """Whether the assistant requested at least one tool call."""
        return bool(self.tool_calls)
