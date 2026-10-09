from collections.abc import Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes.profile import get_db
from app.database.models import Base, UserProfile
from app.main import app

PROFILE_PAYLOAD = {
    "user_id": "user-001",
    "nickname": "Alex",
    "age": 30,
    "gender": "prefer_not_to_say",
    "height": 178.0,
    "weight": 75.5,
    "fitness_level": "intermediate",
    "goal": "提升力量并改善体能",
    "training_frequency": 4,
    "diet_preference": "高蛋白，不过度限制碳水",
    "lifestyle": "久坐办公，晚上训练",
}


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'profile-test.db'}",
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
        transport = ASGITransport(app=app)
        yield AsyncClient(transport=transport, base_url="http://test")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_user_profile(client: AsyncClient) -> None:
    response = await client.post("/profile", json=PROFILE_PAYLOAD)

    assert response.status_code == 201
    body = response.json()
    assert body["user_id"] == "user-001"
    assert body["nickname"] == "Alex"
    assert body["fitness_level"] == "intermediate"
    assert body["id"] >= 1
    assert body["created_at"]
    assert body["updated_at"]


@pytest.mark.asyncio
async def test_get_user_profile(client: AsyncClient) -> None:
    await client.post("/profile", json=PROFILE_PAYLOAD)

    response = await client.get("/profile/user-001")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user-001"
    assert body["goal"] == "提升力量并改善体能"
    assert body["training_frequency"] == 4


@pytest.mark.asyncio
async def test_get_unknown_user_profile_returns_404(client: AsyncClient) -> None:
    response = await client.get("/profile/unknown-user")

    assert response.status_code == 404
    assert response.json() == {"detail": "User profile not found"}


def test_user_profile_persists_across_sessions(session_factory: sessionmaker[Session]) -> None:
    first_session = session_factory()
    first_session.add(UserProfile(**PROFILE_PAYLOAD))
    first_session.commit()
    first_session.close()

    second_session = session_factory()
    profile = second_session.scalar(select(UserProfile).where(UserProfile.user_id == "user-001"))
    second_session.close()

    assert profile is not None
    assert profile.nickname == "Alex"
    assert profile.weight == 75.5

@pytest.mark.asyncio
async def test_city_can_be_saved_for_web_user_without_full_profile(client: AsyncClient) -> None:
    response = await client.patch("/profile/web-user/city", json={"city": "广州"})

    assert response.status_code == 200
    assert response.json()["city"] == "广州"

    fetched = await client.get("/profile/web-user")
    assert fetched.status_code == 200
    assert fetched.json()["city"] == "广州"
