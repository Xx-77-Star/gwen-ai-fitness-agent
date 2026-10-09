"""Schemas for the Gwen Memory Center management API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MemoryKey = Literal[
    "fitness_goal",
    "training_preference",
    "experience_level",
    "weather_location",
    "city",
]


class MemoryItemResponse(BaseModel):
    """One long-term memory item shown in the Gwen Memory Center."""

    model_config = ConfigDict(from_attributes=True)

    memory_key: MemoryKey
    memory_value: str
    created_at: datetime
    updated_at: datetime


class MemoryCenterResponse(BaseModel):
    """All long-term memory currently remembered for one user."""

    user_id: str = Field(min_length=1, max_length=64)
    memories: list[MemoryItemResponse] = Field(default_factory=list)
    count: int = Field(ge=0, default=0)
    last_updated_at: datetime | None = None


class MemoryUpdateRequest(BaseModel):
    """Request to replace one long-term memory value."""

    model_config = ConfigDict(extra="forbid")

    memory_value: str = Field(min_length=1, max_length=4000)


class MemoryUpdateResponse(BaseModel):
    """One long-term memory item after a successful update."""

    user_id: str
    memory_key: MemoryKey
    memory_value: str
    created_at: datetime
    updated_at: datetime


class MemoryResetResponse(BaseModel):
    """Result of clearing every long-term memory item for a user."""

    user_id: str
    reset_count: int = Field(ge=0)
