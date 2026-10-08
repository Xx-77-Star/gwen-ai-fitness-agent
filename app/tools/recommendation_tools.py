"""Deterministic fitness recommendation tool with structured output."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition

WeatherCondition = Literal["sunny", "cloudy", "rainy", "snowy", "windy", "hot", "cold"]
Intensity = Literal["low", "moderate"]


class TrainingRecommendationInput(BaseModel):
    """Input for get_training_recommendation."""

    model_config = ConfigDict(extra="forbid")

    user_goal: str = Field(min_length=1, max_length=500)
    recent_training: list[str] = Field(default_factory=list, max_length=20)
    weather_condition: WeatherCondition

    @field_validator("recent_training")
    @classmethod
    def validate_recent_training(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class TrainingRecommendation(BaseModel):
    """Structured recommendation returned to the Agent."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    weather_condition: WeatherCondition
    recommendation_type: Literal["outdoor_cardio", "indoor_strength", "indoor_cardio", "mobility"]
    workout: str
    intensity: Intensity
    duration_minutes: int = Field(ge=10, le=120)
    focus_areas: list[str] = Field(default_factory=list)
    recovery_note: str


def build_recommendation(
    user_goal: str,
    recent_training: list[str],
    weather_condition: WeatherCondition,
) -> TrainingRecommendation:
    normalized = [item.lower() for item in recent_training]
    recent_text = " ".join(normalized)
    if weather_condition in {"rainy", "snowy", "windy"}:
        recommendation_type = "indoor_strength"
        workout = "安排一次室内全身力量训练，优先选择深蹲、卧推和划船等基础动作。"
        intensity = "moderate"
        duration_minutes = 45
        focus_areas = ["全身力量", "动作质量"]
    elif weather_condition in {"hot", "cold"}:
        recommendation_type = "mobility"
        workout = "安排一次低冲击活动或动态热身，根据体感控制训练强度。"
        intensity = "low"
        duration_minutes = 30
        focus_areas = ["活动度", "恢复"]
    elif "跑步" in recent_text or "有氧" in recent_text:
        recommendation_type = "indoor_strength"
        workout = "建议安排室内力量训练，补充上下肢推拉动作和核心训练。"
        intensity = "moderate"
        duration_minutes = 45
        focus_areas = ["力量", "核心"]
    else:
        recommendation_type = "outdoor_cardio"
        workout = "天气条件适合时可安排一次中等强度户外有氧训练。"
        intensity = "moderate"
        duration_minutes = 40
        focus_areas = ["心肺", "基础耐力"]

    recovery_note = "训练后根据疲劳和睡眠情况调整下一次训练量。"
    return TrainingRecommendation(
        goal=user_goal,
        weather_condition=weather_condition,
        recommendation_type=recommendation_type,
        workout=workout,
        intensity=intensity,
        duration_minutes=duration_minutes,
        focus_areas=focus_areas,
        recovery_note=recovery_note,
    )


def get_training_recommendation(
    arguments: dict[str, Any],
    user_id: str,
    repository: Any,
) -> dict[str, Any]:
    """Build a recommendation without depending on database or external APIs."""
    payload = TrainingRecommendationInput.model_validate(arguments)
    return build_recommendation(
        payload.user_goal,
        payload.recent_training,
        payload.weather_condition,
    ).model_dump(mode="json")


def register_recommendation_tool(registry: ToolRegistry) -> None:
    """Register the fitness recommendation tool on an existing registry."""
    registry.register(
        ToolDefinition(
            name="get_training_recommendation",
            description=(
                "用于根据用户目标、最近训练和天气条件生成结构化训练建议，"
                "例如帮我安排今天怎么训练、根据最近训练推荐课程、下雨天练什么。"
                "不要用于查询历史训练记录或当前天气。"
            ),
            input_model=TrainingRecommendationInput,
            handler=get_training_recommendation,
        )
    )


def build_recommendation_tool_registry() -> ToolRegistry:
    """Create a registry containing the fitness recommendation tool."""
    registry = ToolRegistry()
    register_recommendation_tool(registry)
    return registry
