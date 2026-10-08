"""Offline Agent Run Trace tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.observability.runner import invoke_agent_with_trace
from app.observability.tracing import AgentTraceCallback, InMemoryAgentTraceSink
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition
from tests.fakes import FakeLLMClient, FakeProfileRepository, FakeTrainingRepository


def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_training_summary",
            description="查询训练汇总。",
            input_model=_SummaryInput,
            handler=lambda arguments, user_id, repository: {"summary": {"record_count": 1}},
        )
    )
    return registry


class _SummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: date = Field()
    end_date: date = Field()


class BaselineGraph:
    """Thin fake exposing the same ainvoke contract as LangGraph."""

    def __init__(self, *, fail: bool = False, tool: bool = False) -> None:
        self.fail = fail
        self.tool = tool
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        state: Mapping[str, Any],
        *,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.configs.append(dict(config))
        callbacks = config["callbacks"]
        callback = callbacks[-1]
        assert isinstance(callback, AgentTraceCallback)
        await callback.on_chain_start(
            None,
            dict(state),
            run_id="graph-run",
            name="LangGraph",
        )
        await callback.on_chain_start(
            None,
            dict(state),
            run_id="profile-run",
            parent_run_id="graph-run",
            name="load_profile",
        )
        await callback.on_chain_end(
            {"profile": None, "profile_loaded": False},
            run_id="profile-run",
            parent_run_id="graph-run",
        )
        await callback.on_chain_start(
            None,
            dict(state),
            run_id="intent-run",
            parent_run_id="graph-run",
            name="intent",
        )
        await callback.on_chain_end(
            {"intent": "training", "conversation_history": []},
            run_id="intent-run",
            parent_run_id="graph-run",
        )
        await callback.on_chain_start(
            None,
            dict(state),
            run_id="decision-run",
            parent_run_id="graph-run",
            name="tool_decision",
        )
        if self.tool:
            calls = [
                {
                    "call_id": "call-1",
                    "tool_name": "get_training_summary",
                    "arguments": {
                        "start_date": "2026-10-01",
                        "end_date": "2026-10-07",
                    },
                }
            ]
            await callback.on_chain_end(
                {"tool_calls": calls, "answer_draft": None, "stop_reason": "tool_call"},
                run_id="decision-run",
                parent_run_id="graph-run",
            )
            await callback.on_chain_start(
                None,
                {"tool_calls": calls},
                run_id="tool-run",
                parent_run_id="graph-run",
                name="tool_execution",
            )
            await callback.on_chain_end(
                {
                    "tool_results": [
                        {
                            "call_id": "call-1",
                            "tool_name": "get_training_summary",
                            "status": "success",
                            "result": {"summary": {"record_count": 1}},
                        }
                    ],
                    "tool_call_count": 1,
                    "tool_rounds": 1,
                    "stop_reason": "completed",
                },
                run_id="tool-run",
                parent_run_id="graph-run",
            )
        else:
            await callback.on_chain_end(
                {
                    "tool_calls": [],
                    "answer_draft": "建议先安排一次全身力量训练。",
                    "stop_reason": "completed",
                },
                run_id="decision-run",
                parent_run_id="graph-run",
            )
        await callback.on_chain_start(
            None,
            dict(state),
            run_id="response-run",
            parent_run_id="graph-run",
            name="response",
        )
        if self.fail:
            error = RuntimeError("upstream unavailable")
            await callback.on_chain_error(error, run_id="response-run", parent_run_id="graph-run")
            await callback.on_chain_error(error, run_id="graph-run")
            raise error
        result = {"response": "建议先安排一次全身力量训练。", "stop_reason": "completed"}
        await callback.on_chain_end(result, run_id="response-run", parent_run_id="graph-run")
        await callback.on_chain_end(result, run_id="graph-run")
        return result


@pytest.mark.asyncio
async def test_every_node_is_recorded_and_response_is_in_trace() -> None:
    sink = InMemoryAgentTraceSink()
    graph = BaselineGraph()

    result, trace = await invoke_agent_with_trace(
        graph,
        {"user_id": "user-001", "input": "你好"},
        sink=sink,
    )

    assert result["response"].startswith("建议")
    assert trace.user_id == "user-001"
    assert trace.input == "你好"
    assert trace.final_response == result["response"]
    assert trace.stop_reason == "completed"
    assert [node.node_name for node in trace.nodes] == [
        "load_profile",
        "intent",
        "tool_decision",
        "response",
    ]
    assert all(node.duration_ms >= 0 for node in trace.nodes)
    assert sink.traces == (trace,)


@pytest.mark.asyncio
async def test_tool_execution_records_calls_and_results_summary() -> None:
    sink = InMemoryAgentTraceSink()
    graph = BaselineGraph(tool=True)

    _result, trace = await invoke_agent_with_trace(
        graph,
        {"user_id": "user-001", "input": "查询训练汇总"},
        sink=sink,
    )

    tool_trace = next(node for node in trace.nodes if node.node_name == "tool_execution")
    assert tool_trace.tool_calls == [
        {
            "call_id": "call-1",
            "tool_name": "get_training_summary",
            "arguments": {"start_date": "2026-10-01", "end_date": "2026-10-07"},
        }
    ]
    assert tool_trace.tool_results_summary == [
        {
            "call_id": "call-1",
            "tool_name": "get_training_summary",
            "status": "success",
            "error_code": None,
        }
    ]


@pytest.mark.asyncio
async def test_error_is_recorded_without_changing_the_exception() -> None:
    sink = InMemoryAgentTraceSink()
    graph = BaselineGraph(fail=True)

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await invoke_agent_with_trace(graph, {"user_id": "user-001", "input": "你好"}, sink=sink)

    trace = sink.traces[0]
    assert trace.status == "error"
    assert trace.error == {"type": "RuntimeError", "message": "upstream unavailable"}
    response_trace = next(node for node in trace.nodes if node.node_name == "response")
    assert response_trace.status == "error"
    assert response_trace.error == {"type": "RuntimeError", "message": "upstream unavailable"}


@pytest.mark.asyncio
async def test_real_graph_callback_does_not_require_node_edits() -> None:
    from app.agent.graph import create_agent_graph

    sink = InMemoryAgentTraceSink()
    graph = create_agent_graph(
        FakeLLMClient(),
        FakeProfileRepository(),
        training_repository=FakeTrainingRepository(),
    )

    _result, trace = await invoke_agent_with_trace(
        graph,
        {
            "user_id": "user-001",
            "input": "你好",
            "conversation_history": [],
            "rag_context": [],
            "tool_calls": [],
            "tool_results": [],
            "tool_rounds": 0,
            "tool_call_count": 0,
            "answer_draft": None,
            "stop_reason": "completed",
            "messages": [],
        },
        sink=sink,
    )

    assert [node.node_name for node in trace.nodes] == [
        "load_profile",
        "memory_retrieve",
        "rag_retrieve",
        "intent",
        "tool_decision",
        "response",
        "memory_update",
        "conversation_persistence",
    ]
    assert trace.final_response


@pytest.mark.asyncio
async def test_trace_summary_keeps_multi_tool_call_ids_and_errors_without_result_payload() -> None:
    sink = InMemoryAgentTraceSink()
    graph = BaselineGraph(tool=True)

    result, trace = await invoke_agent_with_trace(
        graph,
        {"user_id": "user-001", "input": "查询训练汇总"},
        sink=sink,
    )
    node = next(item for item in trace.nodes if item.node_name == "tool_execution")
    assert node.tool_calls[0]["call_id"] == "call-1"
    assert node.tool_results_summary[0]["call_id"] == "call-1"
    assert "result" not in node.tool_results_summary[0]


@pytest.mark.asyncio
async def test_error_node_status_and_error_payload_are_preserved() -> None:
    sink = InMemoryAgentTraceSink()
    graph = BaselineGraph(fail=True)
    with pytest.raises(RuntimeError):
        await invoke_agent_with_trace(graph, {"user_id": "user-001", "input": "你好"}, sink=sink)
    response_node = next(item for item in sink.traces[0].nodes if item.node_name == "response")
    assert response_node.status == "error"
    assert response_node.error == {"type": "RuntimeError", "message": "upstream unavailable"}

