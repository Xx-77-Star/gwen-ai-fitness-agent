"""Stable Memory contracts for Agent integrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.memory.models import MemoryContext, UserMemory


class MemoryServiceInterface(Protocol):
    """Interface used by Graph nodes for memory access."""

    def retrieve(self, user_id: str, *, short_term: Sequence[Mapping[str, Any]]) -> MemoryContext:
        """Return normalized short-term and long-term user memory."""
        ...

    def remember(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        """Persist one long-term user memory item."""
        ...
