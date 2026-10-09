from collections.abc import Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base
from app.database.session import get_db
from app.main import app
from tests.test_training_records import PROFILE_PAYLOAD


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'workout-api.db'}",
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


async def test_workout_checkin_api_persists_and_lists_history(client: AsyncClient) -> None:
    assert (await client.post("/profile", json=PROFILE_PAYLOAD)).status_code == 201

    created = await client.post(
        "/workouts",
        json={
            "user_id": "user-001",
            "date": "2026-10-08",
            "muscle_group": "胸",
            "exercise": "卧推",
            "weight": 60,
            "sets": 4,
            "reps": 8,
            "feeling": "状态良好",
            "note": "上斜哑铃40kg",
        },
    )
    assert created.status_code == 201
    assert created.json()["muscle_group"] == "胸"

    listed = await client.get("/workouts/user-001")
    assert listed.status_code == 200
    assert listed.json()[0]["exercise"] == "卧推"


async def test_workout_checkin_api_rejects_invalid_sets(client: AsyncClient) -> None:
    response = await client.post(
        "/workouts",
        json={
            "user_id": "user-001",
            "date": "2026-10-08",
            "muscle_group": "胸",
            "exercise": "卧推",
            "weight": 60,
            "sets": 0,
            "reps": 8,
            "feeling": "状态良好",
            "note": "",
        },
    )
    assert response.status_code == 422
