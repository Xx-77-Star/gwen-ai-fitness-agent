import re
from datetime import date

import pytest
from sqlalchemy import select

from app.agent.graph import get_agent_graph_with_repository
from app.database.models import UserProfile
from app.database.session import get_session_factory
from app.database.training_repository import SQLTrainingRecordRepository
from app.schemas.training import TrainingRecordCreate
from app.tools.registry import ToolRegistry
from app.tools.training_tools import build_training_tool_registry

pytestmark = pytest.mark.real_llm

USER_ID = "demo_user"
START_DATE = date(2026, 10, 1)
END_DATE = date(2026, 10, 7)


def record_payload(training_date: date, training_type: str, duration: int) -> TrainingRecordCreate:
    return TrainingRecordCreate(
        user_id=USER_ID,
        training_date=training_date,
        training_type=training_type,
        duration_minutes=duration,
        intensity="中等",
        body_parts=[training_type],
        sets=[{"exercise": f"{training_type}-main", "value": 3}],
        reps=[{"exercise": f"{training_type}-main", "value": 10}],
        weight=[{"exercise": f"{training_type}-main", "value": 20}],
        notes="Phase E-1-B real tool calling fixture",
    )


@pytest.fixture
def real_training_repository():
    session = get_session_factory()()
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
                    goal="真实 Tool Calling 验证",
                    training_frequency=3,
                    diet_preference="均衡",
                    lifestyle="测试环境",
                )
            )
            session.flush()

        repository = SQLTrainingRecordRepository(session)
        expected = {
            (date(2026, 10, 1), "Chest", 60),
            (date(2026, 10, 3), "Leg", 70),
            (date(2026, 10, 5), "Back", 50),
        }
        existing = repository.list_by_user(USER_ID, START_DATE, END_DATE, 50)
        existing_keys = {
            (item.training_date, item.training_type, item.duration_minutes) for item in existing
        }
        for key in expected - existing_keys:
            repository.create(record_payload(*key))
        session.commit()
        yield repository
    finally:
        session.close()


@pytest.mark.asyncio
async def test_real_bailian_langgraph_tool_calling_loop(real_training_repository) -> None:
    session = real_training_repository._session
    graph = get_agent_graph_with_repository(session)

    result = await graph.ainvoke(
        {
            "user_id": USER_ID,
            "input": "查询我最近训练情况",
            "conversation_history": [],
            "rag_context": [],
            "tool_calls": [],
            "tool_results": [],
            "tool_rounds": 0,
            "tool_call_count": 0,
            "answer_draft": None,
            "stop_reason": "completed",
            "messages": [],
        }
    )

    tool_calls = result.get("metadata", {}).get("_previous_tool_calls", [])
    assert tool_calls, f"Tool Decision did not return tool_calls; state={result}"
    assert tool_calls[0]["tool_name"] == "get_training_summary"

    registry: ToolRegistry = build_training_tool_registry()
    definition = registry.get("get_training_summary")
    assert definition is not None
    validated = definition.input_model.model_validate(tool_calls[0]["arguments"])

    tool_results = result.get("tool_results", [])
    assert len(tool_results) == 1
    assert tool_results[0]["status"] == "success"
    summary = tool_results[0]["result"]["summary"]
    assert summary["record_count"] == 3
    assert summary["training_days"] == 3
    assert summary["total_duration_minutes"] == 180

    response = result["response"]
    assert re.search(r"(3|三)", response)
    assert re.search(r"(180|三小时|3小时)", response)
    assert validated.start_date == START_DATE
    assert validated.end_date == END_DATE
