"""Tests for stable Memory extraction and post-response persistence."""

from __future__ import annotations

from typing import Any

from app.agent.nodes.memory_update import memory_update_node
from app.memory.extractor import RuleBasedMemoryExtractor
from app.memory.models import UserMemory


class FakeMemoryService:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, str]] = []

    def retrieve(self, user_id: str, *, short_term):
        raise AssertionError("Memory Update should not retrieve memory")

    def remember(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        self.calls.append((user_id, memory_key, memory_value))
        if self.fail:
            raise RuntimeError("database unavailable")
        return UserMemory(
            id=len(self.calls),
            user_id=user_id,
            memory_key=memory_key,
            memory_value=memory_value,
            created_at="2026-10-08T10:00:00Z",
            updated_at="2026-10-08T10:00:00Z",
        )


def state(
    message: str,
    response: str = "我会继续帮助你。",
    *,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "user_id": "user-001",
        "input": message,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ],
        "conversation_history": [],
        "response": response,
        "tool_results": tool_results or [],
        "metadata": {"existing": "value"},
    }


def test_explicit_fitness_goal_is_written_to_long_term_memory() -> None:
    service = FakeMemoryService()

    result = memory_update_node(state("我的目标是增肌"), memory_service=service)

    assert service.calls == [("user-001", "fitness_goal", "增肌")]
    assert result["memory_extractions"] == [{"key": "fitness_goal", "value": "增肌"}]
    assert result["metadata"]["memory_update"]["written_keys"] == ["fitness_goal"]


def test_training_preference_is_written_to_long_term_memory() -> None:
    service = FakeMemoryService()

    result = memory_update_node(
        state("我不喜欢跑步，更喜欢力量训练"),
        memory_service=service,
    )

    assert service.calls == [
        ("user-001", "training_preference", "不喜欢跑步，更喜欢力量训练")
    ]
    assert result["metadata"]["memory_update"]["written_keys"] == ["training_preference"]


def test_temporary_or_ordinary_chat_does_not_write_memory() -> None:
    service = FakeMemoryService()

    result = memory_update_node(state("今天天气不错"), memory_service=service)

    assert service.calls == []
    assert result["memory_extractions"] == []
    assert result["metadata"]["memory_update"]["status"] == "skipped"


def test_memory_update_failure_does_not_change_response_or_raise() -> None:
    service = FakeMemoryService(fail=True)
    input_state = state("我的目标是增肌")

    result = memory_update_node(input_state, memory_service=service)

    assert "response" not in result
    assert result["metadata"]["memory_update"]["status"] == "error"
    assert result["metadata"]["memory_update"]["error"]["type"] == "RuntimeError"
    assert result["metadata"]["existing"] == "value"


def test_extractor_does_not_store_weather_or_one_off_training_plans() -> None:
    conversation = [
        {"role": "user", "content": "今天天气不错，今天练腿"},
        {"role": "assistant", "content": "今天适合训练。"},
    ]
    extraction = RuleBasedMemoryExtractor().extract(
        conversation,
        tool_results=[{"status": "success", "result": {"temperature": 20}}],
    )

    assert extraction.memories == []
