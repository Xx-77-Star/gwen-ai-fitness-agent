"""Shared real-model evaluation fixtures."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from app.database.models import UserProfile
from app.database.session import get_session_factory
from app.database.training_repository import SQLTrainingRecordRepository
from app.schemas.training import TrainingRecordCreate

USER_ID = "demo_user"
START_DATE = date(2026, 10, 1)
END_DATE = date(2026, 10, 7)
DEMO_RECORDS = (
    (date(2026, 10, 1), "Chest", 60),
    (date(2026, 10, 3), "Leg", 70),
    (date(2026, 10, 5), "Back", 50),
)


def _record_payload(
    training_date: date, training_type: str, duration_minutes: int
) -> TrainingRecordCreate:
    return TrainingRecordCreate(
        user_id=USER_ID,
        training_date=training_date,
        training_type=training_type,
        duration_minutes=duration_minutes,
        intensity="中等",
        body_parts=[training_type],
        sets=[{"exercise": f"{training_type}-main", "value": 3}],
        reps=[{"exercise": f"{training_type}-main", "value": 10}],
        weight=[{"exercise": f"{training_type}-main", "value": 20}],
        notes="Phase E-4 real model evaluation fixture",
    )


@pytest.fixture
def demo_training_repository():
    session = get_session_factory()()
    try:
        if session.scalar(select(UserProfile).where(UserProfile.user_id == USER_ID)) is None:
            session.add(
                UserProfile(
                    user_id=USER_ID,
                    nickname="Demo User",
                    age=30,
                    gender="prefer_not_to_say",
                    height=175.0,
                    weight=70.0,
                    fitness_level="intermediate",
                    goal="真实模型评估",
                    training_frequency=3,
                    diet_preference="均衡",
                    lifestyle="测试环境",
                )
            )
            session.flush()

        repository = SQLTrainingRecordRepository(session)
        existing = repository.list_by_user(USER_ID, START_DATE, END_DATE, 50)
        existing_keys = {
            (record.training_date, record.training_type, record.duration_minutes)
            for record in existing
        }
        for key in DEMO_RECORDS:
            if key not in existing_keys:
                repository.create(_record_payload(*key))
        session.commit()
        yield repository
    finally:
        session.close()