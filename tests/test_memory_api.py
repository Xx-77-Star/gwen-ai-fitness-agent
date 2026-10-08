"""Tests for the Gwen Memory Center management API."""

from collections.abc import Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base, UserMemory
from app.database.session import get_db
from app.main import app

PROFILE_PAYLOAD = {
    "user_id": "user-001",
    "nickname": "Alex",
    "age": 30,
    "gender": "female",
    "height": 165,
    "weight": 58,
    "fitness_level": "beginner",
    "goal": "提升力量并改善体能",
    "training_frequency": 3,
    "diet_preference": "均衡",
    "lifestyle": "久坐办公",
}


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'memory-test.db'}",
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
def client(session_factory: sessionmaker[Session]) -> Iterator[AsyncClient]:
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

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    finally:
        app.dependency_overrides.clear()


async def _create_profile(client: AsyncClient) -> None:
    response = await client.post("/profile", json=PROFILE_PAYLOAD)
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_memory_center_is_empty_until_memory_is_saved(client: AsyncClient) -> None:
    response = await client.get("/memory/user-001")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "user-001",
        "memories": [],
        "count": 0,
        "last_updated_at": None,
    }


@pytest.mark.asyncio
async def test_memory_center_updates_and_lists_long_term_memory(
    client: AsyncClient, session_factory: sessionmaker[Session]
) -> None:
    await _create_profile(client)

    updated = await client.put(
        "/memory/user-001/fitness_goal",
        json={"memory_value": "增肌"},
    )
    assert updated.status_code == 200
    assert updated.json()["memory_key"] == "fitness_goal"
    assert updated.json()["memory_value"] == "增肌"

    second = await client.put(
        "/memory/user-001/experience_level",
        json={"memory_value": "beginner"},
    )
    assert second.status_code == 200

    listed = await client.get("/memory/user-001")
    assert listed.status_code == 200
    memories = {item["memory_key"]: item["memory_value"] for item in listed.json()["memories"]}
    assert memories == {"fitness_goal": "增肌", "experience_level": "beginner"}

    replaced = await client.put(
        "/memory/user-001/fitness_goal",
        json={"memory_value": "减脂"},
    )
    assert replaced.status_code == 200
    assert replaced.json()["memory_value"] == "减脂"

    with session_factory() as session:
        rows = session.scalars(
            select(UserMemory).where(UserMemory.user_id == "user-001")
        ).all()
        assert {(row.memory_key, row.memory_value) for row in rows} == {
            ("fitness_goal", "减脂"),
            ("experience_level", "beginner"),
        }


@pytest.mark.asyncio
async def test_memory_center_updates_row_that_already_exists(client: AsyncClient) -> None:
    await _create_profile(client)
    assert (
        await client.put(
            "/memory/user-001/training_preference",
            json={"memory_value": "不喜欢跑步，更喜欢力量训练"},
        )
    ).status_code == 200

    updated = await client.put(
        "/memory/user-001/training_preference",
        json={"memory_value": "更喜欢力量训练"},
    )
    assert updated.status_code == 200

    listed = await client.get("/memory/user-001")
    assert [item["memory_value"] for item in listed.json()["memories"]] == [
        "更喜欢力量训练",
    ]


@pytest.mark.asyncio
async def test_memory_center_rejects_unknown_memory_key(client: AsyncClient) -> None:
    await _create_profile(client)

    response = await client.put(
        "/memory/user-001/secret",
        json={"memory_value": "invalid"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_memory_center_update_works_without_profile(client: AsyncClient) -> None:
    """Parity with MemoryService.remember(): web users may have no profile row."""
    response = await client.put(
        "/memory/user-001/fitness_goal",
        json={"memory_value": "增肌"},
    )

    assert response.status_code == 200
    listed = await client.get("/memory/user-001")
    assert [item["memory_value"] for item in listed.json()["memories"]] == ["增肌"]


@pytest.mark.asyncio
async def test_memory_center_reset_clears_every_long_term_item(
    client: AsyncClient, session_factory: sessionmaker[Session]
) -> None:
    await _create_profile(client)
    for key, value in (
        ("fitness_goal", "增肌"),
        ("training_preference", "力量训练"),
        ("experience_level", "beginner"),
    ):
        assert (
            await client.put(f"/memory/user-001/{key}", json={"memory_value": value})
        ).status_code == 200

    reset = await client.delete("/memory/user-001")

    assert reset.status_code == 200
    assert reset.json() == {"user_id": "user-001", "reset_count": 3}

    listed = await client.get("/memory/user-001")
    assert listed.json()["memories"] == []

    with session_factory() as session:
        rows = session.scalars(
            select(UserMemory).where(UserMemory.user_id == "user-001")
        ).all()
        assert rows == []


@pytest.mark.asyncio
async def test_memory_center_reset_reports_zero_when_nothing_to_clear(
    client: AsyncClient,
) -> None:
    response = await client.delete("/memory/user-001")

    assert response.status_code == 200
    assert response.json() == {"user_id": "user-001", "reset_count": 0}


@pytest.mark.asyncio
async def test_memory_items_expose_created_and_updated_timestamps(
    client: AsyncClient,
) -> None:
    await _create_profile(client)
    created = await client.put(
        "/memory/user-001/fitness_goal",
        json={"memory_value": "增肌"},
    )
    assert created.status_code == 200
    body = created.json()
    # created_at/updated_at come from separate utc_now() calls on insert, so
    # they may differ by microseconds; the contract is both exist in order.
    assert body["created_at"] is not None
    assert body["updated_at"] is not None
    assert body["created_at"] <= body["updated_at"]

    listed = await client.get("/memory/user-001")
    item = listed.json()["memories"][0]
    assert item["created_at"] is not None
    assert item["updated_at"] is not None


@pytest.mark.asyncio
async def test_memory_center_summary_reports_count_and_latest_update(
    client: AsyncClient,
) -> None:
    await _create_profile(client)

    empty = await client.get("/memory/user-001")
    assert empty.json()["count"] == 0
    assert empty.json()["last_updated_at"] is None

    for key, value in (
        ("fitness_goal", "增肌"),
        ("training_preference", "力量训练"),
    ):
        assert (
            await client.put(f"/memory/user-001/{key}", json={"memory_value": value})
        ).status_code == 200

    listed = await client.get("/memory/user-001")
    payload = listed.json()
    assert payload["count"] == 2
    assert payload["last_updated_at"] is not None
    latest = max(item["updated_at"] for item in payload["memories"])
    assert payload["last_updated_at"] == latest


@pytest.mark.asyncio
async def test_memory_reset_clears_summary(client: AsyncClient) -> None:
    await _create_profile(client)
    assert (
        await client.put(
            "/memory/user-001/experience_level",
            json={"memory_value": "advanced"},
        )
    ).status_code == 200

    await client.delete("/memory/user-001")

    listed = await client.get("/memory/user-001")
    assert listed.json()["count"] == 0
    assert listed.json()["last_updated_at"] is None
