"""Memory service coordinating short-term conversation state and user facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.memory.models import MemoryContext, MemoryTurn, UserMemory


class MemoryRepository(Protocol):
    """Repository contract for long-term memory operations."""

    def list_by_user(self, user_id: str) -> list[UserMemory]:
        ...

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        ...


class MemoryService:
    """Service boundary between Agent nodes and memory repositories."""

    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def retrieve(self, user_id: str, *, short_term: Sequence[Mapping[str, Any]]) -> MemoryContext:
        normalized_turns = self.normalize_short_term(short_term)
        long_term = self._repository.list_by_user(user_id) if user_id else []
        return MemoryContext(
            user_id=user_id,
            short_term=normalized_turns,
            long_term=long_term,
        )

    def remember(
        self,
        user_id: str,
        memory_key: str,
        memory_value: str,
    ) -> UserMemory:
        return self._repository.upsert(user_id, memory_key, memory_value)

    @staticmethod
    def normalize_short_term(turns: Sequence[Mapping[str, Any]]) -> list[MemoryTurn]:
        normalized: list[MemoryTurn] = []
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role not in {"user", "assistant", "system"} or not isinstance(content, str):
                continue
            normalized.append(MemoryTurn(role=role, content=content))
        return normalized
