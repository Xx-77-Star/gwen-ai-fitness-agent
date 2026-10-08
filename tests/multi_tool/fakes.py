"""Shared fakes for Multi Tool Calling tests."""

from __future__ import annotations

from typing import Any

from app.llm.types import LLMResponse, LLMToolCall
from app.tools.recommendation_tools import register_recommendation_tool
from app.tools.registry import ToolRegistry
from app.tools.training_tools import build_training_tool_registry
from app.tools.weather_tools import register_weather_tool
from tests.weather.fakes import FakeWeatherClient


def multi_tool_registry() -> tuple[ToolRegistry, FakeWeatherClient]:
    registry = build_training_tool_registry()
    weather_client = FakeWeatherClient()
    register_weather_tool(registry, weather_client)
    register_recommendation_tool(registry)
    return registry, weather_client


class MultiToolDecisionLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    call_id="summary-call",
                    name="get_training_summary",
                    arguments={"start_date": "2026-10-01", "end_date": "2026-10-07"},
                ),
                LLMToolCall(
                    call_id="weather-call",
                    name="get_current_weather",
                    arguments={"latitude": 31.2304, "longitude": 121.4737},
                ),
                LLMToolCall(
                    call_id="recommendation-call",
                    name="get_training_recommendation",
                    arguments={
                        "user_goal": "提升体能",
                        "recent_training": ["力量训练", "跑步"],
                        "weather_condition": "rainy",
                    },
                ),
            ],
            finish_reason="tool_calls",
        )
