from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.database.workout_repository import WorkoutCheckInRepository
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition

MAX_WORKOUT_LIMIT = 50


class ListWorkoutInput(BaseModel):
    """Input for list_workout_check_ins."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=10, ge=1, le=MAX_WORKOUT_LIMIT)


class WorkoutHistoryInput(BaseModel):
    """Input for get_workout_history_context."""

    model_config = ConfigDict(extra="forbid")

    current_date: Any = None
    days: int = Field(default=14, ge=1, le=90)


def _date_text(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _date_relation(record_date: date | None, current_date: date | None) -> str:
    if record_date is None or current_date is None:
        return "unknown"
    if record_date == current_date:
        return "today"
    if record_date == current_date - timedelta(days=1):
        return "yesterday"
    return "earlier"


def _as_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def list_workout_check_ins(
    arguments: dict[str, Any],
    user_id: str,
    repository: WorkoutCheckInRepository,
) -> dict[str, Any]:
    payload = ListWorkoutInput.model_validate(arguments)
    records = repository.list_by_user(user_id, limit=payload.limit)
    return {
        "records": [record.model_dump(mode="json") for record in records],
        "returned_count": len(records),
    }


def get_workout_history_context(
    arguments: dict[str, Any],
    user_id: str,
    repository: WorkoutCheckInRepository,
) -> dict[str, Any]:
    payload = WorkoutHistoryInput.model_validate(arguments)
    records = repository.list_by_user(user_id, limit=MAX_WORKOUT_LIMIT)
    current_date = _as_date(payload.current_date)
    selected = records
    if current_date is not None:
        selected = [
            item for item in records if _as_date(item.date) <= current_date
        ][: payload.days * 3]
    muscle_groups = list(dict.fromkeys(item.muscle_group for item in selected))
    exercises = [
        {
            "date": _date_text(_as_date(item.date)),
            "date_relation": _date_relation(_as_date(item.date), current_date),
            "muscle_group": item.muscle_group,
            "exercise": item.exercise,
            "weight": item.weight,
            "sets": item.sets,
            "reps": item.reps,
            "feeling": item.feeling,
        }
        for item in selected
    ]
    return {
        "current_date": _date_text(current_date),
        "yesterday_date": _date_text(current_date - timedelta(days=1)) if current_date else None,
        "records": exercises,
        "record_count": len(exercises),
        "muscle_groups": muscle_groups,
        "reasoning_hint": (
            "请根据最近训练部位判断训练连续性、昨天训练内容和今天适合安排的肌群，并说明推荐原因。"
        ),
    }


def register_workout_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="list_workout_check_ins",
            description="读取用户主动记录的训练打卡历史，包括日期、训练部位、动作、重量、组数、次数、训练感受和备注。用于用户查看训练打卡记录。",
            input_model=ListWorkoutInput,
            handler=list_workout_check_ins,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_workout_history_context",
            description=(
                "读取最近训练打卡历史，并返回用于训练推荐的上下文。"
                "用户询问今天练什么、昨天练什么、本周训练安排、训练连续天数、最近训练肌群时必须调用。"
                "返回最近记录的训练部位和动作，让后续推荐结合实际历史。"
            ),
            input_model=WorkoutHistoryInput,
            handler=get_workout_history_context,
        )
    )


def build_workout_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_workout_tools(registry)
    return registry





