"""Offline Fitness Recommendation Tool tests."""

from __future__ import annotations

from app.tools.executor import ToolExecutor
from app.tools.recommendation_tools import (
    TrainingRecommendationInput,
    build_recommendation,
    build_recommendation_tool_registry,
    get_training_recommendation,
)
from app.tools.schemas import ToolCall


def test_recommendation_schema_and_registry_are_registered() -> None:
    registry = build_recommendation_tool_registry()
    payload = registry.schema_payloads()[0]
    definition = registry.get("get_training_recommendation")

    assert registry.names() == ("get_training_recommendation",)
    assert definition is not None
    assert definition.input_model is TrainingRecommendationInput
    assert set(payload["function"]["parameters"]["properties"]) == {
        "user_goal",
        "recent_training",
        "weather_condition",
    }


def test_recommendation_uses_goal_recent_training_and_weather() -> None:
    recommendation = build_recommendation(
        user_goal="提升体能",
        recent_training=["跑步", "力量训练"],
        weather_condition="rainy",
    )

    assert recommendation.goal == "提升体能"
    assert recommendation.weather_condition == "rainy"
    assert recommendation.recommendation_type == "indoor_strength"
    assert recommendation.workout
    assert recommendation.duration_minutes > 0
    assert recommendation.focus_areas


def test_recommendation_returns_structured_data_through_executor() -> None:
    registry = build_recommendation_tool_registry()
    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            call_id="recommendation-1",
            name="get_training_recommendation",
            arguments={
                "user_goal": "改善心肺",
                "recent_training": ["跑步"],
                "weather_condition": "sunny",
            },
        ),
        user_id="user-001",
        repository=None,
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["recommendation_type"] == "indoor_strength"
    assert result.data["duration_minutes"] == 45


def test_recommendation_rejects_invalid_weather_without_execution() -> None:
    registry = build_recommendation_tool_registry()
    executor = ToolExecutor(registry)

    result = executor.execute(
        ToolCall(
            call_id="recommendation-1",
            name="get_training_recommendation",
            arguments={
                "user_goal": "改善心肺",
                "recent_training": [],
                "weather_condition": "stormy",
            },
        ),
        user_id="user-001",
        repository=None,
    )

    assert result.status == "error"
    assert result.error_code == "invalid_arguments"
    assert result.safe_error_message == "Tool arguments are invalid"


def test_recommendation_handler_is_available_for_tool_selection() -> None:
    result = get_training_recommendation(
        {
            "user_goal": "保持健康",
            "recent_training": [],
            "weather_condition": "cold",
        },
        "user-001",
        None,
    )

    assert result["recommendation_type"] == "mobility"
    assert result["intensity"] == "low"
