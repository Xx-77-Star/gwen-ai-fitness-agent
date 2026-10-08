"""Provider-neutral memory models for short-term and long-term memory."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryRole = Literal["user", "assistant", "system"]


class MemoryTurn(BaseModel):
    """One normalized short-term conversation turn."""

    model_config = ConfigDict(extra="forbid")

    role: MemoryRole
    content: str = Field(min_length=1, max_length=8000)


class UserMemory(BaseModel):
    """One durable key/value fact stored for one user."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    user_id: str = Field(min_length=1, max_length=64)
    memory_key: str = Field(min_length=1, max_length=100)
    memory_value: str = Field(min_length=1, max_length=4000)
    created_at: datetime
    updated_at: datetime


class ConversationTurnRecord(BaseModel):
    """One persisted conversation turn used for cross-request short-term Memory."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: int
    conversation_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=64)
    role: MemoryRole
    content: str = Field(min_length=1, max_length=8000)
    sequence: int = Field(ge=1)
    created_at: datetime


class MemoryContext(BaseModel):
    """Normalized memory payload shared with Agent nodes."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    short_term: list[MemoryTurn] = Field(default_factory=list)
    long_term: list[UserMemory] = Field(default_factory=list)


class UserMemoryCreate(BaseModel):
    """Input for creating or upserting one user memory item."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64)
    memory_key: str = Field(min_length=1, max_length=100)
    memory_value: str = Field(min_length=1, max_length=4000)
