from datetime import date

from app.schemas.training import TrainingRecordResponse, TrainingSummary
from app.tools.executor import ToolExecutor
from app.tools.schemas import ToolCall
from app.tools.training_tools import build_training_tool_registry

RECORD = TrainingRecordResponse(
    id=1,
    user_id="user-001",
    training_date=date(2026, 10, 7),
    training_type="力量训练",
    duration_minutes=60,
    intensity="中等",
    body_parts=["胸部"],
    sets=[{"exercise": "卧推", "value": 4}],
    reps=[{"exercise": "卧推", "value": 8}],
    weight=[{"exercise": "卧推", "value": 60}],
    notes="状态良好",
    created_at="2026-10-07T12:00:00Z",
    updated_at="2026-10-07T12:00:00Z",
)


class FakeTrainingRepository:
    def __init__(self) -> None:
        self.calls = []

    def get_latest_by_user(self, user_id):
        self.calls.append(("get_latest_by_user", user_id))
        return RECORD

    def list_by_user(self, user_id, start_date, end_date, limit):
        self.calls.append(
            ("list_by_user", user_id, start_date, end_date, limit)
        )
        return [RECORD]

    def summarize_by_user(self, user_id, start_date, end_date):
        self.calls.append(("summarize_by_user", user_id, start_date, end_date))
        return TrainingSummary(
            record_count=1,
            training_days=1,
            total_duration_minutes=60,
            type_distribution={"力量训练": 1},
            start_date=start_date,
            end_date=end_date,
        )


def test_get_last_training_record_uses_context_user_id() -> None:
    repository = FakeTrainingRepository()
    executor = ToolExecutor(build_training_tool_registry())

    result = executor.execute(
        ToolCall(call_id="call-1", name="get_last_training_record", arguments={}),
        user_id="context-user",
        repository=repository,
    )

    assert result.status == "success"
    assert result.data["record"]["id"] == 1
    assert repository.calls == [("get_latest_by_user", "context-user")]


def test_list_training_records_calls_repository_with_injected_user_id() -> None:
    repository = FakeTrainingRepository()
    executor = ToolExecutor(build_training_tool_registry())

    result = executor.execute(
        ToolCall(
            call_id="call-1",
            name="list_training_records",
            arguments={
                "start_date": "2026-10-01",
                "end_date": "2026-10-07",
                "limit": 20,
            },
        ),
        user_id="context-user",
        repository=repository,
    )

    assert result.status == "success"
    assert result.data["returned_count"] == 1
    assert repository.calls == [
        ("list_by_user", "context-user", date(2026, 10, 1), date(2026, 10, 7), 20)
    ]


def test_summary_tool_returns_repository_summary() -> None:
    repository = FakeTrainingRepository()
    executor = ToolExecutor(build_training_tool_registry())

    result = executor.execute(
        ToolCall(
            call_id="call-1",
            name="get_training_summary",
            arguments={
                "start_date": "2026-10-01",
                "end_date": "2026-10-07",
            },
        ),
        user_id="context-user",
        repository=repository,
    )

    assert result.status == "success"
    assert result.data["summary"]["record_count"] == 1
    assert repository.calls == [
        ("summarize_by_user", "context-user", date(2026, 10, 1), date(2026, 10, 7))
    ]


def test_empty_latest_record_is_success() -> None:
    repository = FakeTrainingRepository()
    repository.get_latest_by_user = lambda user_id: None
    executor = ToolExecutor(build_training_tool_registry())

    result = executor.execute(
        ToolCall(call_id="call-1", name="get_last_training_record", arguments={}),
        user_id="context-user",
        repository=repository,
    )

    assert result.status == "success"
    assert result.data == {"record": None}


def test_tool_arguments_cannot_override_user_id() -> None:
    repository = FakeTrainingRepository()
    executor = ToolExecutor(build_training_tool_registry())

    result = executor.execute(
        ToolCall(
            call_id="call-1",
            name="list_training_records",
            arguments={
                "start_date": "2026-10-01",
                "end_date": "2026-10-07",
                "limit": 20,
                "user_id": "other-user",
            },
        ),
        user_id="context-user",
        repository=repository,
    )

    assert result.status == "error"
    assert result.error_code == "forbidden_argument"
    assert repository.calls == []