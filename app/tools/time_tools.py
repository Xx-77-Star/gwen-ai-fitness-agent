from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition

DEFAULT_TIMEZONE = "Asia/Shanghai"
WEEKDAY_NAMES = (
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
)


class TimeProvider(Protocol):
    """Provider-neutral boundary for local date and time facts."""

    def now(self, *, timezone: str = DEFAULT_TIMEZONE) -> datetime: ...


class SystemTimeProvider:
    """System clock provider with a timezone-aware Windows-safe fallback."""

    def now(self, *, timezone: str = DEFAULT_TIMEZONE) -> datetime:
        if timezone.upper() in {"UTC", "GMT"}:
            return datetime.now(UTC)
        try:
            return datetime.now(ZoneInfo(timezone))
        except Exception:
            # Windows images may not ship tzdata. Use the host local clock,
            # which is Asia/Shanghai in the deployed/local target environment.
            return datetime.now().astimezone()


class CurrentTimeInput(BaseModel):
    """Input for get_current_time."""

    model_config = ConfigDict(extra="forbid")

    timezone: str = DEFAULT_TIMEZONE


def build_current_time_payload(now: datetime, timezone: str | None = None) -> dict[str, Any]:
    """Return the stable time facts used by the Agent and UI."""
    local = now.replace(second=0, microsecond=0)
    return {
        "date": local.date().isoformat(),
        "weekday": WEEKDAY_NAMES[local.weekday()],
        "time": local.strftime("%H:%M"),
        "datetime": local.isoformat(),
        "timezone": timezone or (str(local.tzinfo) if local.tzinfo else None),
        "yesterday": (local.date() - timedelta(days=1)).isoformat(),
        "tomorrow": (local.date() + timedelta(days=1)).isoformat(),
    }


def get_current_time(
    arguments: dict[str, Any],
    user_id: str,
    runtime_context: Any,
) -> dict[str, Any]:
    """Handler-compatible wrapper for the system time provider."""
    payload = CurrentTimeInput.model_validate(arguments)
    context = (
        runtime_context
        if isinstance(runtime_context, dict)
        else {"time_provider": runtime_context}
    )
    provider = context.get("time_provider") or SystemTimeProvider()
    now = provider.now(timezone=payload.timezone)
    return build_current_time_payload(now, payload.timezone)


def register_time_tool(registry: ToolRegistry) -> None:
    """Register the deterministic current-date tool on an existing registry."""
    registry.register(
        ToolDefinition(
            name="get_current_time",
            description=(
                "获取当前系统准确日期、星期和时间。用户询问今天几月几日、现在几点、今天/昨天/明天、"
                "本周训练安排、训练连续天数，或需要基于当前日期计算训练记录时，必须先调用此工具。"
                "默认使用 Asia/Shanghai 时区；日期计算必须以工具返回的 date 为唯一当前日期。"
                "昨天是 yesterday，明天是 tomorrow，不得把记录日期当作当前日期。"
            ),
            input_model=CurrentTimeInput,
            handler=get_current_time,
        )
    )


def build_time_tool_registry() -> ToolRegistry:
    """Create a registry containing only the current-time tool."""
    registry = ToolRegistry()
    register_time_tool(registry)
    return registry
