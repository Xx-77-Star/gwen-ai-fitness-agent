from collections.abc import Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.graph import create_agent_graph
from app.database.models import Base
from app.database.session import get_db
from app.llm.errors import LLMError
from app.main import app
from tests.fakes import FakeLLMClient, FakeProfileRepository


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'chat-test.db'}",
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
        transport = ASGITransport(app=app)
        yield AsyncClient(transport=transport, base_url="http://test")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_supports_user_id_without_profile(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json={"user_id": "unknown-user", "message": "今天应该怎么训练？"},
    )

    assert response.status_code == 200
    assert response.json()["response"]


@pytest.mark.asyncio
async def test_chat_rejects_missing_user_id(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": "今天应该怎么训练？"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "user_id"


@pytest.mark.asyncio
async def test_chat_rejects_llm_failure_without_json_parsing_crash(client: AsyncClient) -> None:
    class FakeGraph:
        async def ainvoke(self, state, config):
            raise LLMError("<!DOCTYPE html><html>proxy error</html>")

    monkey = pytest.MonkeyPatch()
    monkey.setattr("app.api.routes.chat.get_agent_graph_with_repository", lambda _: FakeGraph())
    try:
        response = await client.post(
            "/chat", json={"user_id": "unknown-user", "message": "今天应该怎么训练？"}
        )
    finally:
        monkey.undo()

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "Upstream LLM service is unavailable"
