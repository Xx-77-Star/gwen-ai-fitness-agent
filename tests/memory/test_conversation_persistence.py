"""Cross-request conversation persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.nodes.conversation_persistence import conversation_persist_node
from app.agent.nodes.memory_retrieval import memory_retrieve_node
from app.database.models import Base, ConversationTurn
from app.memory.conversation_repository import SQLConversationRepository
from app.memory.models import ConversationTurnRecord, UserMemory
from app.memory.service import MemoryService


class FakeMemoryRepository:
    def list_by_user(self, user_id: str) -> list[UserMemory]:
        return []

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        raise AssertionError("conversation persistence must not write long-term memory")


class FailingConversationRepository:
    def append_turn(self, *args, **kwargs):
        raise RuntimeError("database unavailable")


def conversation_state(
    conversation_id: str = "conv-1",
    message: str = "你好",
    response: str = "你好，有什么可以帮助你？",
) -> dict[str, Any]:
    return {
        "user_id": "user-001",
        "conversation_id": conversation_id,
        "input": message,
        "response": response,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ],
        "conversation_history": [],
    }


def repository(session: Session) -> SQLConversationRepository:
    return SQLConversationRepository(session)


def test_same_conversation_recovers_recent_history() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    repo = repository(session)

    repo.append_turn("conv-1", "user-001", "user", "第一轮")
    repo.append_turn("conv-1", "user-001", "assistant", "第一轮回答")

    turns = repo.list_recent_turns("conv-1", "user-001", limit=10)

    assert [turn.content for turn in turns] == ["第一轮", "第一轮回答"]
    session.close()
    engine.dispose()


def test_conversation_windows_are_isolated_and_limited() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    repo = repository(session)

    for index in range(4):
        repo.append_turn("conv-1", "user-001", "user", f"消息{index}")
    repo.append_turn("conv-2", "user-001", "user", "另一个会话")

    conv1 = repo.list_recent_turns("conv-1", "user-001", limit=2)
    conv2 = repo.list_recent_turns("conv-2", "user-001", limit=10)

    assert [turn.content for turn in conv1] == ["消息2", "消息3"]
    assert [turn.content for turn in conv2] == ["另一个会话"]
    session.close()
    engine.dispose()


def test_memory_retrieve_prefers_persisted_conversation_window() -> None:
    class FakeConversationRepository:
        def list_recent_turns(self, conversation_id, user_id, *, limit):
            assert (conversation_id, user_id, limit) == ("conv-1", "user-001", 2)
            return [
                ConversationTurnRecord(
                    id=1,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="user",
                    content="上次我说过的话",
                    sequence=1,
                    created_at=datetime.now(UTC),
                )
            ]

    result = memory_retrieve_node(
        {
            "user_id": "user-001",
            "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "本轮消息"}],
        },
        memory_service=MemoryService(FakeMemoryRepository()),
        conversation_repository=FakeConversationRepository(),
        conversation_window=2,
    )

    assert result["memory_turns"] == [{"role": "user", "content": "上次我说过的话"}]


def test_conversation_persist_saves_user_and_assistant_turns() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    repo = repository(session)

    result = conversation_persist_node(
        conversation_state(), conversation_repository=repo
    )

    rows = session.scalars(select(ConversationTurn).order_by(ConversationTurn.sequence)).all()
    assert result["metadata"]["conversation_persistence"]["status"] == "success"
    assert result["metadata"]["conversation_persistence"]["saved_turns"] == 2
    assert [row.role for row in rows] == ["user", "assistant"]
    session.close()
    engine.dispose()


def test_conversation_persist_failure_does_not_change_response() -> None:
    state = conversation_state()
    result = conversation_persist_node(
        state,
        conversation_repository=FailingConversationRepository(),
    )

    assert "response" not in result
    assert result["metadata"]["conversation_persistence"]["status"] == "error"
    assert result["metadata"]["conversation_persistence"]["error"]["type"] == "RuntimeError"
