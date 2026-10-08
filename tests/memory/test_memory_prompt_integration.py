"""Integration tests proving retrieved Memory reaches Agent prompts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.nodes.memory_retrieval import memory_retrieve_node
from app.agent.nodes.response import response_node
from app.agent.nodes.tool_decision import tool_decision_node
from app.llm.types import LLMResponse, LLMToolCall
from app.memory.models import MemoryContext, MemoryTurn, UserMemory
from app.memory.service import MemoryService
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class TrainingSummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: str = Field()
    end_date: str = Field()


class FakeMemoryRepository:
    def list_by_user(self, user_id: str) -> list[UserMemory]:
        return [
            UserMemory(
                id=1,
                user_id=user_id,
                memory_key="training_preference",
                memory_value="偏好力量训练，不喜欢长时间有氧",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        ]

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        raise AssertionError("Memory Read Integration must not write memory")


class RecordingLLM:
    def __init__(self, response: LLMResponse | str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_training_summary",
            description="获取训练汇总。",
            input_model=TrainingSummaryInput,
            handler=lambda args, user_id, repository: {"record_count": 1},
        )
    )
    return registry


def retrieved_state() -> dict[str, Any]:
    retrieved = memory_retrieve_node(
        {
            "user_id": "user-001",
            "messages": [{"role": "user", "content": "根据我的偏好安排训练"}],
        },
        memory_service=MemoryService(FakeMemoryRepository()),
    )
    return {
        "user_id": "user-001",
        "input": "根据我的偏好安排训练",
        "messages": [{"role": "user", "content": "根据我的偏好安排训练"}],
        "conversation_history": [],
        **retrieved,
    }


def test_memory_retrieve_node_exposes_short_and_long_term_context() -> None:
    result = retrieved_state()

    assert isinstance(result["memory_context"], MemoryContext)
    assert result["memory_loaded"] is True
    assert result["memory_turns"] == [{"role": "user", "content": "根据我的偏好安排训练"}]
    assert result["user_memories"][0]["memory_key"] == "training_preference"


async def test_memory_is_injected_into_tool_decision_prompt() -> None:
    llm = RecordingLLM(
        LLMResponse(
            tool_calls=[
                LLMToolCall(
                    call_id="call-1",
                    name="get_training_summary",
                    arguments={"start_date": "2026-10-01", "end_date": "2026-10-08"},
                )
            ],
            finish_reason="tool_calls",
        )
    )

    result = await tool_decision_node(
        retrieved_state(),
        llm_client=llm,
        tool_registry=tool_registry(),
    )

    prompt = llm.calls[0]["system_prompt"]
    assert "用户长期记忆" in prompt
    assert "training_preference" in prompt
    assert "偏好力量训练，不喜欢长时间有氧" in prompt
    assert "用户短期记忆" in prompt
    assert "根据我的偏好安排训练" in prompt
    assert result["tool_calls"][0]["tool_name"] == "get_training_summary"


async def test_memory_is_injected_into_response_prompt() -> None:
    state = retrieved_state()
    state["answer_draft"] = None
    llm = RecordingLLM("结合你的训练偏好，建议安排力量训练。")

    result = await response_node(state, llm_client=llm)

    prompt = llm.calls[0]["system_prompt"]
    assert "用户长期记忆" in prompt
    assert "training_preference" in prompt
    assert "偏好力量训练，不喜欢长时间有氧" in prompt
    assert "用户短期记忆" in prompt
    assert "根据我的偏好安排训练" in prompt
    assert result["response"] == "结合你的训练偏好，建议安排力量训练。"


async def test_user_memories_snapshot_is_injected_when_memory_context_is_unavailable() -> None:
    state: dict[str, Any] = {
        "input": "按我的目标安排训练",
        "messages": [],
        "conversation_history": [],
        "memory_context": None,
        "user_memories": [
            {
                "id": 2,
                "user_id": "user-001",
                "memory_key": "fitness_goal",
                "memory_value": "增肌",
                "created_at": "2026-10-08T10:00:00Z",
                "updated_at": "2026-10-08T10:00:00Z",
            }
        ],
        "answer_draft": None,
        "tool_results": [],
    }
    llm = RecordingLLM("围绕增肌目标安排训练。")

    await response_node(state, llm_client=llm)

    prompt = llm.calls[0]["system_prompt"]
    assert "fitness_goal: 增肌" in prompt


async def test_duplicate_memory_facts_are_rendered_once() -> None:
    now = datetime.now(UTC)
    memory = UserMemory(
        id=1,
        user_id="user-001",
        memory_key="experience_level",
        memory_value="intermediate",
        created_at=now,
        updated_at=now,
    )
    context = MemoryContext(
        user_id="user-001",
        short_term=[MemoryTurn(role="user", content="你好")],
        long_term=[memory],
    )
    llm = RecordingLLM("你好。")
    state: dict[str, Any] = {
        "input": "你好",
        "messages": [{"role": "user", "content": "你好"}],
        "conversation_history": [],
        "memory_context": context,
        "user_memories": [memory.model_dump()],
        "answer_draft": None,
        "tool_results": [],
    }

    await response_node(state, llm_client=llm)

    prompt = llm.calls[0]["system_prompt"]
    assert prompt.count("experience_level: intermediate") == 1
