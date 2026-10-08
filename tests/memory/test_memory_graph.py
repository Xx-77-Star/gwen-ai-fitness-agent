"""Graph integration tests for the Memory Retrieval Node."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.graph import create_agent_graph
from app.agent.nodes.memory_retrieval import memory_retrieve_node
from app.memory.models import UserMemory
from app.memory.service import MemoryService
from tests.fakes import FakeLLMClient, FakeProfileRepository, FakeTrainingRepository


class FakeMemoryRepository:
    def __init__(self, items: dict[str, list[UserMemory]] | None = None) -> None:
        self.items = items or {}
        self.calls: list[str] = []

    def list_by_user(self, user_id: str) -> list[UserMemory]:
        self.calls.append(user_id)
        return list(self.items.get(user_id, []))

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        raise AssertionError("retrieval must not write memory")


def memory_item(user_id: str, key: str, value: str) -> UserMemory:
    return UserMemory(
        id=1,
        user_id=user_id,
        memory_key=key,
        memory_value=value,
        created_at="2026-10-07T10:00:00Z",
        updated_at="2026-10-07T10:00:00Z",
    )


def graph_state(message: str = "你好") -> dict[str, Any]:
    return {
        "user_id": "user-001",
        "input": message,
        "conversation_history": [],
        "messages": [{"role": "user", "content": "上次说过我想练力量"}],
        "tool_calls": [],
        "tool_results": [],
        "tool_messages": [],
        "tool_rounds": 0,
        "tool_call_count": 0,
        "answer_draft": None,
        "stop_reason": "completed",
        "metadata": {},
    }


def test_memory_retrieval_node_reads_user_memory_without_touching_database_directly() -> None:
    repository = FakeMemoryRepository(
        {"user-001": [memory_item("user-001", "preferred_training", "力量")]}
    )

    result = memory_retrieve_node(
        graph_state(),
        memory_service=MemoryService(repository),
    )

    assert result["memory_loaded"] is True
    assert result["user_memories"][0]["memory_value"] == "力量"
    assert result["memory_turns"] == [{"role": "user", "content": "上次说过我想练力量"}]
    assert repository.calls == ["user-001"]


@pytest.mark.asyncio
async def test_memory_isolation_and_empty_memory_preserve_normal_graph_execution() -> None:
    repository = FakeMemoryRepository({"user-a": [memory_item("user-a", "goal", "增肌")]})
    service = MemoryService(repository)
    graph = create_agent_graph(
        FakeLLMClient(),
        FakeProfileRepository(),
        training_repository=FakeTrainingRepository(),
        memory_service=service,
    )

    first = graph_state("今天应该怎么训练？")
    first["user_id"] = "user-a"
    second = graph_state("今天应该怎么训练？")
    second["user_id"] = "user-b"
    first_result = await graph.ainvoke(first)
    second_result = await graph.ainvoke(second)

    assert first_result["user_memories"][0]["memory_key"] == "goal"
    assert second_result["user_memories"] == []
    assert repository.calls == ["user-a", "user-b"]


@pytest.mark.asyncio
async def test_memory_does_not_change_existing_tool_calling_flow() -> None:
    repository = FakeMemoryRepository()
    graph = create_agent_graph(
        FakeLLMClient(),
        FakeProfileRepository(),
        training_repository=FakeTrainingRepository(),
        memory_service=MemoryService(repository),
    )

    result = await graph.ainvoke(graph_state("查询训练汇总"))

    assert result["tool_results"] == []
    assert result["response"]
    assert repository.calls == ["user-001"]
