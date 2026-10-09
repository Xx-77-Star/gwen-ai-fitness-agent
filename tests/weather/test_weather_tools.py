"""Offline Weather Tool tests with a fake external API."""

from __future__ import annotations

import pytest

from app.agent.graph import create_agent_graph
from app.agent.nodes.tool_execution import tool_execution_node
from app.llm.types import LLMResponse, LLMToolCall
from app.observability.runner import invoke_agent_with_trace
from app.observability.tracing import InMemoryAgentTraceSink
from app.tools.executor import ToolExecutor
from app.tools.weather_tools import (
    WeatherInput,
    build_weather_tool_registry,
    get_current_weather,
)
from tests.fakes import FakeLLMClient, FakeProfileRepository, FakeTrainingRepository
from tests.weather.fakes import FakeWeatherClient


def test_weather_schema_and_registry_expose_external_tool() -> None:
    registry = build_weather_tool_registry(FakeWeatherClient())
    payloads = registry.schema_payloads()
    definition = registry.get("get_current_weather")

    assert registry.names() == ("get_current_weather",)
    assert definition is not None
    assert definition.input_model is WeatherInput
    assert payloads[0]["function"]["name"] == "get_current_weather"
    assert set(payloads[0]["function"]["parameters"]["properties"]) == {
        "latitude",
        "longitude",
        "city",
    }


def test_weather_tool_calls_external_client_and_validates_coordinates() -> None:
    client = FakeWeatherClient()
    result = get_current_weather(
        {"latitude": 31.2304, "longitude": 121.4737},
        "user-001",
        {"weather_client": client},
    )

    assert result["temperature_celsius"] == 22.5
    assert client.calls == [(31.2304, 121.4737)]


def test_weather_tool_rejects_invalid_coordinates_without_external_call() -> None:
    client = FakeWeatherClient()

    with pytest.raises(ValueError):
        get_current_weather(
            {"latitude": 100, "longitude": 0},
            "user-001",
            {"weather_client": client},
        )

    assert client.calls == []


def test_weather_tool_execution_keeps_training_tool_loop_compatible() -> None:
    client = FakeWeatherClient()
    registry = build_weather_tool_registry(client)
    executor = ToolExecutor(registry)

    result = tool_execution_node(
        {
            "tool_calls": [
                {
                    "call_id": "weather-1",
                    "tool_name": "get_current_weather",
                    "arguments": {"latitude": 31.2304, "longitude": 121.4737},
                }
            ]
        },
        tool_executor=executor,
        tool_registry=registry,
        runtime_context={"user_id": "user-001", "weather_client": client},
    )

    assert result["tool_results"][0]["status"] == "success"
    assert result["tool_results"][0]["result"]["temperature_celsius"] == 22.5
    assert result["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_weather_tool_is_visible_to_agent_and_trace() -> None:
    client = FakeWeatherClient()
    registry = build_weather_tool_registry(client)
    fake_llm = FakeLLMClient(
        decision_responses=[
            LLMResponse(
                tool_calls=[
                    LLMToolCall(
                        call_id="weather-trace-call",
                        name="get_current_weather",
                        arguments={"latitude": 31.2304, "longitude": 121.4737},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="当前气温适合户外训练。", finish_reason="stop"),
        ]
    )
    graph = create_agent_graph(
        fake_llm,
        FakeProfileRepository(),
        training_repository=FakeTrainingRepository(),
        weather_client=client,
    )
    sink = InMemoryAgentTraceSink()
    state = {
        "user_id": "user-001",
        "input": "上海今天天气怎么样？",
        "conversation_history": [],
        "messages": [],
        "tool_calls": [],
        "tool_results": [],
        "tool_messages": [],
        "tool_rounds": 0,
        "tool_call_count": 0,
        "answer_draft": None,
        "stop_reason": "completed",
        "metadata": {},
    }

    result, trace = await invoke_agent_with_trace(graph, state, sink=sink)

    assert result["response"] == "当前气温适合户外训练。"
    assert [node.node_name for node in trace.nodes] == [
        "load_profile",
        "memory_retrieve",
            "rag_retrieve",
        "intent",
        "tool_decision",
        "tool_execution",
        "tool_decision",
        "response",
        "memory_update",
        "conversation_persistence",
    ]
    tool_trace = next(node for node in trace.nodes if node.node_name == "tool_execution")
    assert tool_trace.tool_calls[0]["tool_name"] == "get_current_weather"
    assert tool_trace.tool_results_summary[0]["status"] == "success"
    assert client.calls == [(31.2304, 121.4737)]
    assert registry.get("get_current_weather") is not None
