from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import Base, UserProfile
from app.database.training_repository import SQLTrainingRecordRepository
from app.schemas.training import TrainingRecordCreate

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
    "diet_preference": "高蛋白",
    "lifestyle": "久坐办公，晚上训练",
}

TRAINING_PAYLOAD = {
    "user_id": "user-001",
    "training_date": date(2026, 10, 7).isoformat(),
    "training_type": "力量训练",
    "duration_minutes": 60,
    "intensity": "中等",
    "body_parts": ["胸部", "肱三头肌"],
    "sets": [
        {"exercise": "卧推", "value": 4},
        {"exercise": "上斜哑铃卧推", "value": 3},
    ],
    "reps": [
        {"exercise": "卧推", "value": 8},
        {"exercise": "上斜哑铃卧推", "value": 10},
    ],
    "weight": [
        {"exercise": "卧推", "value": 60},
        {"exercise": "上斜哑铃卧推", "value": 22.5},
    ],
    "notes": "状态良好，最后一次卧推保留两次余力。",
}


@pytest.fixture
def session_factory(tmp_path) -> sessionmaker[Session]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'training-test.db'}",
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
def db_session(session_factory: sessionmaker[Session]) -> Session:
    session = session_factory()
    session.add(UserProfile(**PROFILE_PAYLOAD))
    session.commit()
    try:
        yield session
    finally:
        session.close()


def test_create_training_record(db_session: Session) -> None:
    payload = TrainingRecordCreate(**TRAINING_PAYLOAD)

    record = SQLTrainingRecordRepository(db_session).create(payload)

    assert record.id >= 1
    assert record.user_id == "user-001"
    assert record.training_type == "力量训练"
    assert record.body_parts == ["胸部", "肱三头肌"]
    assert record.sets[0].exercise == "卧推"


def test_get_training_record(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    created = repository.create(TrainingRecordCreate(**TRAINING_PAYLOAD))

    record = repository.get(created.id, "user-001")

    assert record is not None
    assert record.id == created.id
    assert record.weight[1].value == 22.5


def test_training_record_persists_across_sessions(session_factory: sessionmaker[Session]) -> None:
    first_session = session_factory()
    first_session.add(UserProfile(**PROFILE_PAYLOAD))
    first_session.commit()
    repository = SQLTrainingRecordRepository(first_session)
    created = repository.create(TrainingRecordCreate(**TRAINING_PAYLOAD))
    first_session.commit()
    first_session.close()

    second_session = session_factory()
    record = SQLTrainingRecordRepository(second_session).get(created.id, "user-001")
    second_session.close()

    assert record is not None
    assert record.notes.startswith("状态良好")
    assert record.created_at
    assert record.updated_at


def test_get_record_for_other_user_returns_none(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    created = repository.create(TrainingRecordCreate(**TRAINING_PAYLOAD))

    record = repository.get(created.id, "other-user")

    assert record is None