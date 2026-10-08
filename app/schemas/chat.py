from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Input contract for the chat endpoint."""

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class ChatResponse(BaseModel):
    """Output contract returned by the Agent workflow."""

    response: str
    conversation_id: str
    trace: dict[str, Any] = Field(default_factory=dict)