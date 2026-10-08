from collections.abc import Iterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base
from app.database.session import get_db
from app.main import app
from tests.test_training_records import PROFILE_PAYLOAD, TRAINING_PAYLOAD


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'training-api.db'}",
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
async def test_training_record_api(client: AsyncClient) -> None:
    await client.post("/profile", json=PROFILE_PAYLOAD)

    created = await client.post("/training-records", json=TRAINING_PAYLOAD)
    assert created.status_code == 201
    record_id = created.json()["id"]

    fetched = await client.get(f"/training-records/{record_id}", params={"user_id": "user-001"})
    assert fetched.status_code == 200
    assert fetched.json()["training_type"] == "力量训练"


@pytest.mark.asyncio
async def test_training_record_api_requires_existing_profile(client: AsyncClient) -> None:
    response = await client.post("/training-records", json=TRAINING_PAYLOAD)

    assert response.status_code == 404
    assert response.json()["detail"] == "User profile not found"