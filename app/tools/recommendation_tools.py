from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition

WeatherCondition = Literal["sunny", "cloudy", "rainy", "snowy", "windy", "hot", "cold"]
Intensity = Literal["low", "moderate"]
RecommendationType = Literal["outdoor_cardio", "indoor_strength", "indoor_cardio", "mobility"]


class TrainingRecommendationInput(BaseModel):
    """Input for get_training_recommendation."""

    model_config = ConfigDict(extra="forbid")

    user_goal: str = Field(min_length=1, max_length=500)
    recent_training: list[str] = Field(default_factory=list, max_length=20)
    weather_condition: WeatherCondition
    workout_history: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    current_date: Any = None
    today_records: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    yesterday_records: list[dict[str, Any]] = Field(default_factory=list, max_length=30)

    @field_validator("recent_training")
    @classmethod
    def validate_recent_training(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class TrainingRecommendation(BaseModel):
    """Structured recommendation returned to the Agent."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    weather_condition: WeatherCondition
    recommendation_type: RecommendationType
    workout: str
    intensity: Intensity
    duration_minutes: int = Field(ge=10, le=120)
    focus_areas: list[str] = Field(default_factory=list)
    recovery_note: str
    recommendation_reason: str


def _history_muscles(history: list[dict[str, Any]]) -> list[str]:
    values = []
    for item in history:
        if isinstance(item, dict):
            value = str(item.get("muscle_group", "")).strip()
            if value and value not in values:
                values.append(value)
    return values


def build_recommendation(
    user_goal: str,
    recent_training: list[str],
    weather_condition: WeatherCondition,
    workout_history: list[dict[str, Any]] | None = None,
    current_date: Any = None,
    today_records: list[dict[str, Any]] | None = None,
    yesterday_records: list[dict[str, Any]] | None = None,
) -> TrainingRecommendation:
    normalized = [item.lower() for item in recent_training]
    recent_text = " ".join(normalized)
    history = workout_history or []
    today_records = today_records or []
    yesterday_records = yesterday_records or []
    muscles = _history_muscles(history)
    recent_muscle_text = "、".join(muscles)
    today_muscles = _history_muscles(today_records)
    yesterday_muscles = _history_muscles(yesterday_records)

    if weather_condition in {"rainy", "snowy", "windy"}:
        recommendation_type = "indoor_strength"
        workout = "安排一次室内全身力量训练，优先选择深蹲、卧推和划船等基础动作。"
        intensity = "moderate"
        duration_minutes = 45
        focus_areas = ["全身力量", "动作质量"]
        reason = "天气不适合户外训练，因此优先安排室内力量训练。"
    elif weather_condition in {"hot", "cold"}:
        recommendation_type = "mobility"
        workout = "安排一次低冲击活动或动态热身，根据体感控制训练强度。"
        intensity = "low"
        duration_minutes = 30
        focus_areas = ["活动度", "恢复"]
        reason = "极端体感条件下，先以低冲击活动和恢复为主。"
    elif any(item in recent_text for item in ("胸", "胸肌", "推", "卧推")) and (
        "背" not in recent_muscle_text and "二头" not in recent_muscle_text
    ):
        recommendation_type = "indoor_strength"
        workout = "今天推荐背部和二头训练，例如划船、引体向上或高位下拉，再加入二头弯举。"
        intensity = "moderate"
        duration_minutes = 45
        focus_areas = ["背部", "二头", "拉力平衡"]
        relation = "昨天" if yesterday_muscles else "最近"
        reason = (
            f"{relation}训练记录显示已练{'、'.join(yesterday_muscles or muscles) or '胸'}，"
            "判断为推力训练；今天改练背部和二头，平衡前后侧肌群。"
        )
        if today_muscles:
            reason = (
                f"今天已经练过{'、'.join(today_muscles)}；"
                "如果继续训练，请选择低强度恢复或不同肌群，避免重复刺激。"
            )
    elif "跑步" in recent_text or "有氧" in recent_text:
        recommendation_type = "indoor_strength"
        workout = "建议安排室内力量训练，补充上下肢推拉动作和核心训练。"
        intensity = "moderate"
        duration_minutes = 45
        focus_areas = ["力量", "核心"]
        reason = "最近已有跑步或有氧训练，今天补充力量训练以平衡训练结构。"
    else:
        recommendation_type = "outdoor_cardio"
        workout = "天气条件适合时可安排一次中等强度户外有氧训练。"
        intensity = "moderate"
        duration_minutes = 40
        focus_areas = ["心肺", "基础耐力"]
        reason = "暂无明显肌群失衡或天气限制，优先安排基础心肺训练。"

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
        recommendation_reason=reason,
    )


def get_training_recommendation(
    arguments: dict[str, Any],
    user_id: str,
    repository: Any,
) -> dict[str, Any]:
    """Build a recommendation from user goal, history and weather context."""
    payload = TrainingRecommendationInput.model_validate(arguments)
    return build_recommendation(
        payload.user_goal,
        payload.recent_training,
        payload.weather_condition,
        payload.workout_history,
        payload.current_date,
        payload.today_records,
        payload.yesterday_records,
    ).model_dump(mode="json")


def register_recommendation_tool(registry: ToolRegistry) -> None:
    """Register the fitness recommendation tool on an existing registry."""
    registry.register(
        ToolDefinition(
            name="get_training_recommendation",
            description=(
                "用于根据用户目标、最近训练和天气条件生成结构化训练建议，"
                "例如帮我安排今天怎么训练、根据最近训练推荐课程、下雨天练什么。"
                "如果工具返回 workout_history，请结合最近训练肌群、动作和连续天数生成推荐原因。"
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
