"""Offline Memory service and retrieval tests."""

from __future__ import annotations

from datetime import UTC, datetime

from app.memory.models import UserMemory
from app.memory.service import MemoryService


class FakeMemoryRepository:
    def __init__(self) -> None:
        self.items = {
            "user-a": [
                UserMemory(
                    id=1,
                    user_id="user-a",
                    memory_key="preferred_training",
                    memory_value="力量训练",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ]
        }

    def list_by_user(self, user_id: str) -> list[UserMemory]:
        return list(self.items.get(user_id, []))

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        item = UserMemory(
            id=len(self.items.get(user_id, [])) + 1,
            user_id=user_id,
            memory_key=memory_key,
            memory_value=memory_value,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.items.setdefault(user_id, []).append(item)
        return item


def test_user_memory_can_be_read_and_remembered() -> None:
    service = MemoryService(FakeMemoryRepository())

    context = service.retrieve(
        "user-a",
        short_term=[{"role": "user", "content": "我想继续练力量"}],
    )
    service.remember("user-a", "training_frequency", "每周四次")
    updated = service.retrieve("user-a", short_term=[])

    assert context.user_id == "user-a"
    assert context.long_term[0].memory_key == "preferred_training"
    assert context.short_term[0].role == "user"
    assert any(item.memory_key == "training_frequency" for item in updated.long_term)


def test_memory_is_isolated_between_users_and_empty_memory_is_safe() -> None:
    repository = FakeMemoryRepository()
    service = MemoryService(repository)

    user_a = service.retrieve("user-a", short_term=[])
    user_b = service.retrieve("user-b", short_term=[])
    empty = service.retrieve("user-c", short_term=[])

    assert [item.memory_key for item in user_a.long_term] == ["preferred_training"]
    assert user_b.long_term == []
    assert empty.long_term == []
    assert empty.short_term == []


def test_memory_retrieval_uses_canonical_short_term_messages() -> None:
    service = MemoryService(FakeMemoryRepository())

    context = service.retrieve(
        "user-a",
        short_term=[
            {"role": "user", "content": "第一轮"},
            {"role": "assistant", "content": "回答"},
            {"role": "tool", "content": "不应进入短期记忆"},
        ],
    )

    assert [turn.content for turn in context.short_term] == ["第一轮", "回答"]
