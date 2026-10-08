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


def training_payload(
    *,
    user_id: str = "user-001",
    training_date: date = date(2026, 10, 1),
    training_type: str = "力量训练",
    duration_minutes: int = 60,
) -> TrainingRecordCreate:
    return TrainingRecordCreate(
        user_id=user_id,
        training_date=training_date,
        training_type=training_type,
        duration_minutes=duration_minutes,
        intensity="中等",
        body_parts=["胸部"],
        sets=[{"exercise": "卧推", "value": 4}],
        reps=[{"exercise": "卧推", "value": 8}],
        weight=[{"exercise": "卧推", "value": 60}],
        notes="测试记录",
    )


@pytest.fixture
def session_factory(tmp_path) -> sessionmaker[Session]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'repository-query-test.db'}",
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


def test_list_by_user_filters_and_sorts_stably(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    in_range_old = repository.create(
        training_payload(training_date=date(2026, 10, 2), training_type="跑步")
    )
    in_range_new = repository.create(
        training_payload(training_date=date(2026, 10, 2), training_type="力量训练")
    )
    repository.create(training_payload(training_date=date(2026, 9, 30)))
    repository.create(training_payload(training_date=date(2026, 10, 5)))

    records = repository.list_by_user(
        "user-001",
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 3),
        limit=10,
    )

    assert [record.id for record in records] == [in_range_new.id, in_range_old.id]


def test_list_by_user_supports_limit_and_empty_result(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    repository.create(training_payload())
    repository.create(training_payload(training_date=date(2026, 10, 2)))

    limited = repository.list_by_user(
        "user-001", date(2026, 10, 1), date(2026, 10, 31), limit=1
    )
    empty = repository.list_by_user(
        "user-001", date(2027, 1, 1), date(2027, 1, 31), limit=10
    )

    assert len(limited) == 1
    assert empty == []


def test_get_latest_by_user_uses_stable_ordering(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    repository.create(training_payload(training_date=date(2026, 10, 1)))
    latest = repository.create(training_payload(training_date=date(2026, 10, 3)))

    record = repository.get_latest_by_user("user-001")

    assert record is not None
    assert record.id == latest.id


def test_get_latest_by_user_returns_none_for_empty_user(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)

    assert repository.get_latest_by_user("unknown-user") is None


def test_summarize_by_user_counts_records_days_duration_and_types(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    repository.create(
        training_payload(
            training_date=date(2026, 10, 1),
            training_type="力量训练",
            duration_minutes=60,
        )
    )
    repository.create(
        training_payload(
            training_date=date(2026, 10, 1),
            training_type="力量训练",
            duration_minutes=30,
        )
    )
    repository.create(
        training_payload(
            training_date=date(2026, 10, 3),
            training_type="跑步",
            duration_minutes=45,
        )
    )
    repository.create(
        training_payload(
            training_date=date(2026, 10, 20),
            training_type="恢复",
            duration_minutes=10,
        )
    )

    summary = repository.summarize_by_user(
        "user-001",
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 7),
    )

    assert summary.record_count == 3
    assert summary.training_days == 2
    assert summary.total_duration_minutes == 135
    assert summary.type_distribution == {"力量训练": 2, "跑步": 1}
    assert summary.start_date == date(2026, 10, 1)
    assert summary.end_date == date(2026, 10, 7)


def test_queries_are_isolated_by_user(db_session: Session) -> None:
    other_profile = dict(PROFILE_PAYLOAD, user_id="user-002", nickname="Blair")
    db_session.add(UserProfile(**other_profile))
    db_session.commit()
    repository = SQLTrainingRecordRepository(db_session)
    repository.create(training_payload(training_date=date(2026, 10, 1)))
    repository.create(training_payload(user_id="user-002", training_date=date(2026, 10, 2)))

    records = repository.list_by_user(
        "user-001", date(2026, 10, 1), date(2026, 10, 31), 10
    )
    assert len(records) == 1
    assert repository.get_latest_by_user("user-001") is not None
    summary = repository.summarize_by_user(
        "user-001", date(2026, 10, 1), date(2026, 10, 31)
    )
    assert summary.record_count == 1


def test_invalid_date_range_is_rejected(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)

    with pytest.raises(ValueError):
        repository.list_by_user(
            "user-001", date(2026, 10, 2), date(2026, 10, 1), limit=10
        )

    with pytest.raises(ValueError):
        repository.summarize_by_user(
            "user-001", date(2026, 10, 2), date(2026, 10, 1)
        )


def test_single_day_date_range_is_valid(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)
    repository.create(training_payload(training_date=date(2026, 10, 1)))

    summary = repository.summarize_by_user(
        "user-001", date(2026, 10, 1), date(2026, 10, 1)
    )

    assert summary.record_count == 1
    assert summary.training_days == 1


def test_limit_must_be_positive(db_session: Session) -> None:
    repository = SQLTrainingRecordRepository(db_session)

    with pytest.raises(ValueError):
        repository.list_by_user(
            "user-001", date(2026, 10, 1), date(2026, 10, 2), limit=0
        )