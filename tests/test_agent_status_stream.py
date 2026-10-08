"""Live Agent execution-status stream tests (observability only)."""

from collections.abc import Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.graph import create_agent_graph
from app.database.models import Base
from app.database.session import get_db
from app.main import app
from tests.fakes import FakeLLMClient, FakeProfileRepository


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'agent-status-test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[AsyncClient]:
    def override_get_db() -> Iterator[Session]:
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    fake_graph = create_agent_graph(FakeLLMClient(), FakeProfileRepository())
    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(
        "app.api.routes.chat.get_agent_graph_with_repository",
        lambda _: fake_graph,
    )
    try:
        yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    finally:
        app.dependency_overrides.clear()


async def _events(client: AsyncClient, conversation_id: str, user_id: str) -> list[dict]:
    response = await client.get(
        f"/chat/{conversation_id}/events",
        params={"user_id": user_id},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames: list[dict] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        import json

        frames.append(json.loads(line[len("data: "):]))
    return frames


@pytest.mark.asyncio
async def test_agent_status_stream_carries_workflow_stage_events(client: AsyncClient) -> None:
    """A real graph run must stream node lifecycle events for the status card."""
    response = await client.post(
        "/chat",
        json={
            "user_id": "user-status",
            "message": "帮我制定今天的训练计划",
            "conversation_id": "run-status-1",
        },
    )
    assert response.status_code == 200

    frames = await _events(client, "run-status-1", "user-status")
    names = [frame["node_name"] for frame in frames]

    for stage in ("memory_retrieve", "intent", "rag_retrieve", "response"):
        assert stage in names, f"{stage} missing from status stream"
        starts = [f for f in frames if f["node_name"] == stage and f["phase"] == "start"]
        ends = [f for f in frames if f["node_name"] == stage and f["phase"] == "end"]
        assert starts and ends, f"{stage} must stream both start and end"

    # never leak reasoning / prompts / tool payloads to the user-facing stream
    assert all(set(frame) == {"node_name", "phase", "error"} for frame in frames)


@pytest.mark.asyncio
async def test_agent_status_stream_rejects_other_users(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json={
            "user_id": "user-owner",
            "message": "今天练什么",
            "conversation_id": "run-status-2",
        },
    )
    assert response.status_code == 200

    other = await client.get(
        "/chat/run-status-2/events",
        params={"user_id": "someone-else"},
    )
    assert other.status_code == 403
    assert other.json()["detail"] == "conversation does not belong to user"
