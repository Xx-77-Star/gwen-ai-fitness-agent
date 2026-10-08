"""Offline Multi Tool Calling integration tests."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.graph import create_agent_graph
from app.agent.nodes.tool_decision import tool_decision_node
from app.agent.nodes.tool_execution import tool_execution_node
from app.observability.runner import invoke_agent_with_trace
from app.observability.tracing import InMemoryAgentTraceSink
from app.tools.executor import ToolExecutor
from tests.fakes import FakeProfileRepository, FakeTrainingRepository
from tests.multi_tool.fakes import MultiToolDecisionLLM, multi_tool_registry


def multi_state() -> dict[str, Any]:
    return {
        "user_id": "user-001",
        "input": "查看我最近训练情况，并结合天气给我今天训练建议",
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


@pytest.mark.asyncio
async def test_tool_decision_can_return_multiple_calls() -> None:
    registry, _weather_client = multi_tool_registry()

    result = await tool_decision_node(
        multi_state(),
        llm_client=MultiToolDecisionLLM(),
        tool_registry=registry,
    )

    assert result["stop_reason"] == "tool_call"
    assert [call["tool_name"] for call in result["tool_calls"]] == [
        "get_training_summary",
        "get_current_weather",
        "get_training_recommendation",
    ]


def test_tool_execution_runs_multiple_calls_in_order() -> None:
    registry, weather_client = multi_tool_registry()
    decision = MultiToolDecisionLLM()
    calls = [
        {
            "call_id": "summary-call",
            "tool_name": "get_training_summary",
            "arguments": {"start_date": "2026-10-01", "end_date": "2026-10-07"},
        },
        {
            "call_id": "weather-call",
            "tool_name": "get_current_weather",
            "arguments": {"latitude": 31.2304, "longitude": 121.4737},
        },
        {
            "call_id": "recommendation-call",
            "tool_name": "get_training_recommendation",
            "arguments": {
                "user_goal": "提升体能",
                "recent_training": ["力量训练", "跑步"],
                "weather_condition": "rainy",
            },
        },
    ]

    result = tool_execution_node(
        {"tool_calls": calls},
        tool_executor=ToolExecutor(registry),
        tool_registry=registry,
        runtime_context={
            "user_id": "user-001",
            "repository": FakeTrainingRepository(),
            "weather_client": weather_client,
        },
    )

    assert [item["tool_name"] for item in result["tool_results"]] == [
        "get_training_summary",
        "get_current_weather",
        "get_training_recommendation",
    ]
    assert result["tool_call_count"] == 3
    assert result["tool_rounds"] == 1
    assert weather_client.calls == [(31.2304, 121.4737)]
    assert decision.calls == []


@pytest.mark.asyncio
async def test_multi_tool_response_and_trace_include_all_results() -> None:
    registry, weather_client = multi_tool_registry()
    graph = create_agent_graph(
        MultiToolDecisionLLM(),
        FakeProfileRepository(),
        training_repository=FakeTrainingRepository(),
        weather_client=weather_client,
    )
    sink = InMemoryAgentTraceSink()

    result, trace = await invoke_agent_with_trace(graph, multi_state(), sink=sink)

    assert result["response"]
    assert len(result["tool_results"]) == 6
    tool_trace = next(node for node in trace.nodes if node.node_name == "tool_execution")
    assert len(tool_trace.tool_calls) == 3
    assert len(tool_trace.tool_results_summary) == 3
    assert [item["tool_name"] for item in tool_trace.tool_results_summary] == [
        "get_training_summary",
        "get_current_weather",
        "get_training_recommendation",
    ]
