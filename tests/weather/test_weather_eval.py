"""Offline Weather Tool selection evaluation."""

from __future__ import annotations

import pytest

from app.llm.types import LLMResponse, LLMToolCall
from app.tools.weather_tools import build_weather_tool_registry
from tests.weather.fakes import FakeWeatherClient


class WeatherDecisionLLM:
    def __init__(self) -> None:
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    call_id="weather-call",
                    name="get_current_weather",
                    arguments={"latitude": 31.2304, "longitude": 121.4737},
                )
            ],
            finish_reason="tool_calls",
        )


@pytest.mark.asyncio
async def test_weather_tool_selection_eval_exercises_external_tool() -> None:
    from app.agent.nodes.tool_decision import tool_decision_node

    registry = build_weather_tool_registry(FakeWeatherClient())
    llm = WeatherDecisionLLM()

    result = await tool_decision_node(
        {"input": "上海今天天气怎么样", "user_id": "user-001", "conversation_history": []},
        llm_client=llm,
        tool_registry=registry,
    )

    assert result["stop_reason"] == "tool_call"
    assert result["tool_calls"] == [
        {
            "call_id": "weather-call",
            "tool_name": "get_current_weather",
            "arguments": {"latitude": 31.2304, "longitude": 121.4737},
        }
    ]
    assert llm.calls[0]["tools"][0]["function"]["name"] == "get_current_weather"
