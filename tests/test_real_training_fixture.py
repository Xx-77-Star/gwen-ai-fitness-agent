from datetime import date

import pytest
from sqlalchemy import select

from app.database.models import UserProfile
from app.database.session import get_session_factory
from app.database.training_repository import SQLTrainingRecordRepository
from app.schemas.training import TrainingRecordCreate
from app.tools.executor import ToolExecutor
from app.tools.schemas import ToolCall
from app.tools.training_tools import build_training_tool_registry

USER_ID = "demo_user"
START_DATE = date(2026, 10, 1)
END_DATE = date(2026, 10, 7)


def training_record(
    training_date: date,
    training_type: str,
    duration_minutes: int,
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
        notes="Phase E-1-A fixed tool calling fixture",
    )


DEMO_RECORDS = (
    training_record(date(2026, 10, 1), "Chest", 60),
    training_record(date(2026, 10, 3), "Leg", 70),
    training_record(date(2026, 10, 5), "Back", 50),
)


@pytest.fixture
def demo_training_repository() -> SQLTrainingRecordRepository:
    """Create fixed demo data through the project repository on app SQLite."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        profile = session.scalar(select(UserProfile).where(UserProfile.user_id == USER_ID))
        if profile is None:
            session.add(
                UserProfile(
                    user_id=USER_ID,
                    nickname="Demo User",
                    age=30,
                    gender="prefer_not_to_say",
                    height=175.0,
                    weight=70.0,
                    fitness_level="intermediate",
                    goal="真实 Tool Calling 测试",
                    training_frequency=3,
                    diet_preference="均衡",
                    lifestyle="测试环境",
                )
            )
            session.flush()

        repository = SQLTrainingRecordRepository(session)
        existing = repository.list_by_user(
            USER_ID,
            start_date=START_DATE,
            end_date=END_DATE,
            limit=50,
        )
        existing_keys = {
            (record.training_date, record.training_type, record.duration_minutes)
            for record in existing
        }
        for payload in DEMO_RECORDS:
            key = (
                payload.training_date,
                payload.training_type,
                payload.duration_minutes,
            )
            if key not in existing_keys:
                repository.create(payload)

        session.commit()
        yield repository
    finally:
        session.close()


def test_get_training_summary_returns_fixed_demo_data(
    demo_training_repository: SQLTrainingRecordRepository,
) -> None:
    executor = ToolExecutor(build_training_tool_registry())

    result = executor.execute(
        ToolCall(
            call_id="phase-e1-a-summary",
            name="get_training_summary",
            arguments={
                "start_date": START_DATE.isoformat(),
                "end_date": END_DATE.isoformat(),
            },
        ),
        user_id=USER_ID,
        repository=demo_training_repository,
    )

    assert result.status == "success"
    assert result.data == {
        "summary": {
            "record_count": 3,
            "training_days": 3,
            "total_duration_minutes": 180,
            "type_distribution": {
                "Back": 1,
                "Chest": 1,
                "Leg": 1,
            },
            "start_date": START_DATE.isoformat(),
            "end_date": END_DATE.isoformat(),
        }
    }
