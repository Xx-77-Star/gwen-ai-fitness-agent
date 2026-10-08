from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """Provider-neutral request to execute one registered tool."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Provider-neutral result for one tool call."""

    call_id: str
    tool_name: str
    status: Literal["success", "error", "denied"]
    data: dict[str, Any] | None = None
    error_code: str | None = None
    safe_error_message: str | None = None


ToolHandler = Callable[[dict[str, Any], str, Any], dict[str, Any]]


class ToolDefinition(BaseModel):
    """Fixed tool metadata and handler.

    Runtime dependencies are not serialized here: handlers receive the caller's
    repository at execution time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=500)
    input_model: type[BaseModel]
    handler: ToolHandler