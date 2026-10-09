from datetime import UTC, datetime

from app.tools.recommendation_tools import build_recommendation
from app.tools.time_tools import SystemTimeProvider, get_current_time
from app.tools.workout_tools import build_workout_tool_registry


class FixedTimeProvider:
    def now(self, *, timezone: str = "Asia/Shanghai") -> datetime:
        del timezone
        return datetime(2026, 10, 9, 14, 35, tzinfo=UTC)


def test_current_time_returns_date_weekday_and_time() -> None:
    result = get_current_time({}, "user-001", {"time_provider": FixedTimeProvider()})

    assert result["date"] == "2026-10-09"
    assert result["weekday"] == "星期五"
    assert result["time"] == "14:35"
    assert result["datetime"] == "2026-10-09T14:35:00+00:00"
    assert result["timezone"] == "Asia/Shanghai"
    assert result["yesterday"] == "2026-10-08"
    assert result["tomorrow"] == "2026-10-10"


def test_time_tool_registry_contains_current_time() -> None:
    from app.tools.time_tools import build_time_tool_registry

    registry = build_time_tool_registry()
    assert registry.names() == ("get_current_time",)
    schema = registry.schema_payloads()[0]["function"]
    assert "今天" in schema["description"]
    assert "训练连续天数" in schema["description"]


def test_workout_tools_return_history_context() -> None:
    class Record:
        id = 1
        user_id = "user-001"
        date = "2026-10-08"
        muscle_group = "胸"
        exercise = "卧推"
        weight = 60.0
        sets = 4
        reps = 8
        feeling = "状态良好"
        note = "上斜哑铃40kg"

    class Repository:
        def list_by_user(self, user_id, *, limit):
            assert user_id == "user-001"
            assert limit == 50
            return [Record()]

    registry = build_workout_tool_registry()
    definition = registry.get("get_workout_history_context")
    assert definition is not None
    result = definition.handler(
        {"current_date": "2026-10-09"},
        "user-001",
        Repository(),
    )

    assert result["record_count"] == 1
    assert result["muscle_groups"] == ["胸"]
    assert result["records"][0]["exercise"] == "卧推"
    assert "推荐原因" in result["reasoning_hint"]


def test_recommendation_uses_workout_history_to_explain_recommendation() -> None:
    recommendation = build_recommendation(
        user_goal="增肌",
        recent_training=["胸", "卧推"],
        weather_condition="sunny",
        workout_history=[
            {
                "date": "2026-10-08",
                "muscle_group": "胸",
                "exercise": "卧推",
                "weight": 60,
                "sets": 4,
                "reps": 8,
                "feeling": "状态良好",
            }
        ],
    )

    assert recommendation.recommendation_type == "indoor_strength"
    assert "背部" in recommendation.workout
    assert "二头" in recommendation.workout
    assert "推力训练" in recommendation.recommendation_reason


def test_system_time_provider_works_without_tzdata() -> None:
    result = get_current_time({}, "user-001", {"time_provider": SystemTimeProvider()})

    assert result["date"]
    assert result["weekday"] in {
        "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"
    }
    assert result["time"]
    assert result["timezone"] == "Asia/Shanghai"
    assert result["yesterday"] == "2026-10-08"
    assert result["tomorrow"] == "2026-10-10"




def test_city_text_is_extracted_as_long_term_weather_memory() -> None:
    from app.memory.extractor import RuleBasedMemoryExtractor

    result = RuleBasedMemoryExtractor().extract(
        [{"role": "user", "content": "我的所在地是广州，今天适合跑步吗？"}]
    )
    assert result.memories[0].key == "weather_location"
    assert result.memories[0].value == "广州"



def test_same_day_workout_is_marked_today_not_yesterday() -> None:
    class Record:
        id = 1
        user_id = "user-001"
        date = "2026-10-09"
        muscle_group = "胸"
        exercise = "卧推"
        weight = 100.0
        sets = 4
        reps = 5
        feeling = "疲惫"
        note = ""

    class Repository:
        def list_by_user(self, user_id, *, limit):
            return [Record()]

    definition = build_workout_tool_registry().get("get_workout_history_context")
    result = definition.handler(
        {"current_date": "2026-10-09"},
        "user-001",
        Repository(),
    )

    assert result["records"][0]["date_relation"] == "today"
    assert result["yesterday_date"] == "2026-10-08"
