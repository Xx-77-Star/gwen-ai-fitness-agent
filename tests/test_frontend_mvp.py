from datetime import UTC, datetime
from pathlib import Path

from app.api.routes.chat import _frontend_trace
from app.observability.tracing import AgentNodeTrace, AgentRunTrace


def _node(name: str, *, tool_calls=None, tool_results=None, error=None, stop_reason=None):
    return AgentNodeTrace(
        node_name=name,
        node_start_time=datetime.now(UTC),
        node_end_time=datetime.now(UTC),
        duration_ms=1.5,
        tool_calls=tool_calls or [],
        tool_results_summary=tool_results or [],
        stop_reason=stop_reason,
        error=error,
    )


def _trace(nodes):
    return AgentRunTrace(
        run_id="run-1",
        user_id="user-001",
        input="今天训练安排",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        duration_ms=123.0,
        nodes=tuple(nodes),
        stop_reason="completed",
    )


def test_frontend_assets_have_agent_trace_dashboard_structure() -> None:
    html = __import__("pathlib").Path("frontend/index.html").read_text(encoding="utf-8")
    js = __import__("pathlib").Path("frontend/app.js").read_text(encoding="utf-8")
    css = __import__("pathlib").Path("frontend/styles.css").read_text(encoding="utf-8")
    for marker in (
        "Agent Trace",
        "RUN SUMMARY",
        "EXECUTION TIMELINE",
        "EVIDENCE",
        "run-summary",
        "execution-timeline",
    ):
        assert marker in html
    for marker in (
        "normalizeTrace",
        "renderRunSummary",
        "renderTimeline",
        "renderToolExchanges",
        "renderEvidence",
    ):
        assert marker in js
    assert "innerHTML" not in js
    for marker in (
        "run-summary-grid",
        "timeline-list",
        "tool-exchange",
        "rag-card",
        "memory-card",
        "json-details",
    ):
        assert marker in css


def test_frontend_trace_adds_run_summary_timeline_and_evidence() -> None:
    nodes = [
        _node("load_profile"),
        _node("memory_retrieve"),
        _node("rag_retrieve"),
        _node("response"),
    ]
    result = {
        "memory_context": {
            "short_term": [{"role": "user", "content": "你好"}],
            "long_term": [{"memory_key": "goal", "memory_value": "增肌"}],
        },
        "rag_context": [{"content": "训练知识", "source": "training/muscle.md", "score": 0.9}],
    }
    payload = _frontend_trace(_trace(nodes), result)

    assert payload["run"] == {
        "run_id": "run-1",
        "status": "success",
        "duration_ms": 123.0,
        "stop_reason": "completed",
        "error": None,
    }
    assert payload["summary"] == {
        "status": "success",
        "duration_ms": 123.0,
        "node_count": 4,
        "tool_call_count": 0,
        "tool_result_count": 0,
        "rag_chunk_count": 1,
        "memory_fact_count": 1,
        "memory_turn_count": 1,
        "source_count": 1,
    }
    timeline = [
        (item["sequence"], item["node_name"], item["label"], item["category"])
        for item in payload["timeline"]
    ]
    assert timeline == [
        (1, "load_profile", "读取用户画像", "profile"),
        (2, "memory_retrieve", "读取记忆", "memory"),
        (3, "rag_retrieve", "检索知识库", "retrieval"),
        (4, "response", "生成回答", "response"),
    ]
    assert payload["evidence"]["rag"]["chunks"][0]["source"] == "training/muscle.md"
    assert payload["evidence"]["memory"]["facts"][0]["memory_key"] == "goal"


def test_tool_calls_are_merged_from_trace_nodes_and_state_results() -> None:
    calls = [
        {"call_id": "call-1", "tool_name": "get_training_summary", "arguments": {"limit": 3}},
        {"call_id": "call-2", "tool_name": "get_weather", "arguments": {"city": "上海"}},
    ]
    summaries = [
        {
            "call_id": "call-1",
            "tool_name": "get_training_summary",
            "status": "success",
            "error_code": None,
        },
        {
            "call_id": "call-2",
            "tool_name": "get_weather",
            "status": "error",
            "error_code": "invalid_arguments",
        },
    ]
    nodes = [
        _node("tool_decision"),
        _node("tool_execution", tool_calls=calls, tool_results=summaries),
    ]
    result = {
        "tool_results": [
            {
                "call_id": "call-1",
                "tool_name": "get_training_summary",
                "status": "success",
                "result": {"count": 2},
            },
            {
                "call_id": "call-2",
                "tool_name": "get_weather",
                "status": "error",
                "error": {"code": "invalid_arguments"},
            },
        ]
    }
    payload = _frontend_trace(_trace(nodes), result)

    assert payload["tool_exchanges"] == [
        {
            "call_id": "call-1",
            "tool_name": "get_training_summary",
            "arguments": {"limit": 3},
            "status": "success",
            "result": {"count": 2},
            "error": None,
            "result_summary": summaries[0],
            "has_result": True,
        },
        {
            "call_id": "call-2",
            "tool_name": "get_weather",
            "arguments": {"city": "上海"},
            "status": "error",
            "result": None,
            "error": {"code": "invalid_arguments"},
            "result_summary": summaries[1],
            "has_result": True,
        },
    ]
    assert payload["summary"]["tool_call_count"] == 2
    assert payload["summary"]["tool_result_count"] == 2


def test_multi_tool_rounds_preserve_execution_order_and_match_by_call_id() -> None:
    first = [
        {"call_id": "a", "tool_name": "summary", "arguments": {"n": 1}},
        {"call_id": "b", "tool_name": "weather", "arguments": {"n": 2}},
    ]
    second = [{"call_id": "c", "tool_name": "recommend", "arguments": {"n": 3}}]
    first_results = [
        {
            "call_id": "b",
            "tool_name": "weather",
            "status": "success",
            "result": {"temperature": 22},
        },
        {"call_id": "a", "tool_name": "summary", "status": "success", "result": {"count": 1}},
    ]
    second_results = [
        {"call_id": "c", "tool_name": "recommend", "status": "success", "result": {"plan": "腿部"}}
    ]
    nodes = [
        _node("tool_decision"),
        _node(
            "tool_execution",
            tool_calls=first,
            tool_results=[
                {"call_id": "a", "status": "success"},
                {"call_id": "b", "status": "success"},
            ],
        ),
        _node("tool_decision"),
        _node(
            "tool_execution",
            tool_calls=second,
            tool_results=[{"call_id": "c", "status": "success"}],
        ),
    ]
    payload = _frontend_trace(_trace(nodes), {"tool_results": [*first_results, *second_results]})

    assert [item["call_id"] for item in payload["tool_exchanges"]] == ["a", "b", "c"]
    assert [item["tool_name"] for item in payload["tool_exchanges"]] == [
        "summary",
        "weather",
        "recommend",
    ]
    assert [item["result"] for item in payload["tool_exchanges"]] == [
        {"count": 1},
        {"temperature": 22},
        {"plan": "腿部"},
    ]


def test_missing_results_keep_trace_summary_and_empty_trace_is_safe() -> None:
    call = {"call_id": "call-1", "tool_name": "get_summary", "arguments": {}}
    summary = {
        "call_id": "call-1",
        "tool_name": "get_summary",
        "status": "pending",
        "error_code": None,
    }
    payload = _frontend_trace(
        _trace([_node("tool_execution", tool_calls=[call], tool_results=[summary])]), {}
    )
    assert payload["tool_exchanges"][0]["status"] == "pending"
    assert payload["tool_exchanges"][0]["result"] is None
    assert payload["tool_exchanges"][0]["result_summary"] == summary

    empty = _frontend_trace(_trace([]), {})
    assert empty["run"]["status"] == "success"
    assert empty["timeline"] == []
    assert empty["tool_exchanges"] == []
    assert empty["evidence"]["sources"] == []


def test_error_node_is_visible_in_timeline() -> None:
    error = {"type": "RuntimeError", "message": "tool failed"}
    payload = _frontend_trace(_trace([_node("tool_execution", error=error)]), {})
    timeline = payload["timeline"][0]
    assert timeline["status"] == "error"
    assert timeline["details"]["error"] == error


def test_product_redesign_has_user_and_developer_modes() -> None:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    js = Path("frontend/app.js").read_text(encoding="utf-8")
    for marker in (
        "TRAIN WITH CLARITY",
        "SET YOUR FOCUS",
        "用户模式",
        "开发者模式",
        "Agent Trace",
        "trace-drawer",
        "goal-card",
        "quick-prompts",
    ):
        assert marker in html
    for marker in ("setMode", "setDrawer", "FitLifeTrace"):
        assert marker in js


def test_product_visual_refinement_covers_empty_loading_and_responsive_states() -> None:
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    js = Path("frontend/app.js").read_text(encoding="utf-8")
    css = Path("frontend/styles.css").read_text(encoding="utf-8")
    for marker in (
        "floating-avatar",
        "empty-glyph",
        "loading-state",
        "Agent Trace",
        "trace-drawer",
    ):
        assert marker in html
    for marker in ("loadingState", "setBusy", "textContent", "prefers-reduced-motion"):
        if marker == "prefers-reduced-motion":
            assert marker in css
        elif marker == "textContent":
            assert marker in js
        else:
            assert marker in js
    assert "innerHTML" not in js
    assert "@media" in css
