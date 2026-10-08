"""LangGraph callback-based Agent Run Trace instrumentation.

The callback observes graph and node lifecycle events without changing node
business logic. Traces are emitted through an injectable sink so callers can
persist JSON logs or tests can inspect traces in memory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.callbacks import AsyncCallbackHandler

from app.observability.streaming import NodeEvent, get_agent_run_bus

_SAFE_TEXT_LIMIT = 4000
_TRACE_LOGGER = logging.getLogger("fitlife.agent.trace")


@dataclass(frozen=True)
class AgentNodeTrace:
    """Lifecycle trace for one LangGraph node."""

    node_name: str
    node_start_time: datetime
    node_end_time: datetime | None
    duration_ms: float | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results_summary: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    stop_reason: str | None = None
    error: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        return "success" if self.node_end_time is not None else "running"

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRunTrace:
    """Complete trace for one Agent invocation."""

    run_id: str
    user_id: str
    input: str
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    nodes: tuple[AgentNodeTrace, ...] = ()
    final_response: str | None = None
    stop_reason: str | None = None
    error: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "error"
        return "success" if self.finished_at is not None else "running"

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class AgentTraceSink(Protocol):
    """Destination for completed Agent Run Traces."""

    def emit(self, trace: AgentRunTrace) -> None:
        """Store or forward one completed trace."""
        ...


class InMemoryAgentTraceSink:
    """Thread-safe sink used by tests and local tooling."""

    def __init__(self) -> None:
        self._traces: list[AgentRunTrace] = []
        self._lock = RLock()

    def emit(self, trace: AgentRunTrace) -> None:
        with self._lock:
            self._traces.append(trace)

    @property
    def traces(self) -> tuple[AgentRunTrace, ...]:
        with self._lock:
            return tuple(self._traces)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


class AgentNodeStreamingSink:
    """Relay node lifecycle events to the user-facing execution status stream.

    Pure observability wiring: no Agent / LangGraph / Memory / RAG logic changes.
    """

    def __init__(self, conversation_id: str, user_id: str = "") -> None:
        self._conversation_id = conversation_id
        self._user_id = user_id

    async def emit_node(self, node_name: str, phase: str, error: bool = False) -> None:
        """Publish inline so events can never race the run's completion."""
        bus = get_agent_run_bus()
        event = NodeEvent(node_name=node_name, phase=phase, error=error)
        await bus.publish(self._conversation_id, event)

    def emit(self, trace: AgentRunTrace) -> None:
        """No-op for completed traces; node events were already streamed."""
        del trace


class LoggingAgentTraceSink:
    """Emit one structured log record for each completed Agent Run Trace."""

    def emit(self, trace: AgentRunTrace) -> None:
        _TRACE_LOGGER.info(
            "agent_run_trace",
            extra={"agent_run_trace": trace.model_dump()},
        )


class JSONAgentTraceSink:
    """Append one JSON trace object per line to a local file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, trace: AgentRunTrace) -> None:
        with self._path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(trace.model_dump(), ensure_ascii=False, default=str) + "\n"
            )


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_text(value: Any, *, limit: int = _SAFE_TEXT_LIMIT) -> str:
    text = value if isinstance(value, str) else str(value)
    return text[:limit]


def _error_payload(error: BaseException | Any) -> dict[str, Any]:
    return {"type": type(error).__name__, "message": _safe_text(error)}


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "<truncated>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return _safe_text(value) if isinstance(value, str) else value
    if isinstance(value, Mapping):
        return {
            _safe_text(key, limit=100): _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:50]]
    return _safe_text(value)


def summarize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    summaries: list[dict[str, Any]] = []
    for item in list(value)[:20]:
        if isinstance(item, Mapping):
            call = dict(item)
            summaries.append(
                {
                    "call_id": _safe_text(call.get("call_id", ""), limit=100),
                    "tool_name": _safe_text(call.get("tool_name", ""), limit=100),
                    "arguments": _json_safe(call.get("arguments", {})),
                }
            )
    return summaries


def summarize_tool_results(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    summaries: list[dict[str, Any]] = []
    for item in list(value)[:20]:
        if isinstance(item, Mapping):
            result = dict(item)
            error_code = result.get("error_code")
            summaries.append(
                {
                    "call_id": _safe_text(result.get("call_id", ""), limit=100),
                    "tool_name": _safe_text(result.get("tool_name", ""), limit=100),
                    "status": _safe_text(result.get("status", "unknown"), limit=20),
                    "error_code": _safe_text(error_code, limit=100) if error_code else None,
                }
            )
    return summaries


class AgentTraceCallback(AsyncCallbackHandler):
    """Record top-level LangGraph node lifecycles through callbacks."""

    def __init__(
        self,
        sink: AgentTraceSink | None = None,
        stream_sink: AgentNodeStreamingSink | None = None,
    ) -> None:
        self._sink = sink
        self._stream_sink = stream_sink
        self._run_id = str(uuid4())
        self._user_id = ""
        self._input = ""
        self._started_at = _now()
        self._finished_at: datetime | None = None
        self._error: dict[str, Any] | None = None
        self._nodes: list[AgentNodeTrace] = []
        self._active_nodes: dict[Any, dict[str, Any]] = {}
        self._child_to_parent: dict[Any, Any] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def trace(self) -> AgentRunTrace:
        return self._build_trace()

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any] | Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(kwargs.get("name") or (serialized or {}).get("name") or "unknown")
        if name == "LangGraph":
            self._started_at = _now()
            self._user_id = _safe_text((inputs or {}).get("user_id", ""), limit=100)
            self._input = _safe_text((inputs or {}).get("input", ""), limit=_SAFE_TEXT_LIMIT)
            return
        if parent_run_id is not None:
            self._child_to_parent[run_id] = parent_run_id
        parent_name = self._node_name_for(parent_run_id)
        if parent_name is not None and parent_name != "LangGraph":
            return
        if self._stream_sink is not None:
            await self._stream_sink.emit_node(name, "start")
        self._active_nodes[run_id] = {
            "node_name": name,
            "node_start_time": _now(),
            "inputs": inputs if isinstance(inputs, Mapping) else {},
        }

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        active = self._active_nodes.pop(run_id, None)
        if active is None:
            return
        node_end_time = _now()
        node_start_time = active["node_start_time"]
        duration_ms = (node_end_time - node_start_time).total_seconds() * 1000
        inputs = active["inputs"]
        outputs = outputs if isinstance(outputs, Mapping) else {}
        if self._stream_sink is not None:
            await self._stream_sink.emit_node(str(active["node_name"]), "end")
        self._nodes.append(
            AgentNodeTrace(
                node_name=str(active["node_name"]),
                node_start_time=node_start_time,
                node_end_time=node_end_time,
                duration_ms=duration_ms,
                tool_calls=summarize_tool_calls(inputs.get("tool_calls", [])),
                tool_results_summary=summarize_tool_results(
                    outputs.get("tool_results", inputs.get("tool_results", []))
                ),
                final_response=_optional_text(outputs.get("response")),
                stop_reason=_optional_text(outputs.get("stop_reason"), limit=50),
            )
        )

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        active = self._active_nodes.pop(run_id, None)
        if active is not None:
            if self._stream_sink is not None:
                await self._stream_sink.emit_node(str(active["node_name"]), "end", error=True)
            node_end_time = _now()
            node_start_time = active["node_start_time"]
            self._nodes.append(
                AgentNodeTrace(
                    node_name=str(active["node_name"]),
                    node_start_time=node_start_time,
                    node_end_time=node_end_time,
                    duration_ms=(node_end_time - node_start_time).total_seconds() * 1000,
                    error=_error_payload(error),
                )
            )
        if self._error is None:
            self._error = _error_payload(error)

    def finish(self, result: Mapping[str, Any] | None = None) -> AgentRunTrace:
        if self._finished_at is None:
            self._finished_at = _now()
        if result is not None:
            self._nodes = [
                self._with_response(
                    node,
                    _optional_text(result.get("response")),
                    _optional_text(result.get("stop_reason"), limit=50),
                )
                if node.node_name == "response"
                else node
                for node in self._nodes
            ]
        trace = self._build_trace()
        if self._sink is not None:
            self._sink.emit(trace)
        return trace

    @staticmethod
    def _with_response(
        node: AgentNodeTrace,
        final_response: str | None,
        stop_reason: str | None,
    ) -> AgentNodeTrace:
        return AgentNodeTrace(
            node_name=node.node_name,
            node_start_time=node.node_start_time,
            node_end_time=node.node_end_time,
            duration_ms=node.duration_ms,
            tool_calls=node.tool_calls,
            tool_results_summary=node.tool_results_summary,
            final_response=final_response if final_response is not None else node.final_response,
            stop_reason=stop_reason if stop_reason is not None else node.stop_reason,
            error=node.error,
        )

    def _node_name_for(self, run_id: Any) -> str | None:
        if run_id in self._active_nodes:
            return str(self._active_nodes[run_id]["node_name"])
        parent = self._child_to_parent.get(run_id)
        return self._node_name_for(parent) if parent is not None else None

    def _build_trace(self) -> AgentRunTrace:
        final_response = next(
            (
                node.final_response
                for node in reversed(self._nodes)
                if node.final_response is not None
            ),
            None,
        )
        stop_reason = next(
            (node.stop_reason for node in reversed(self._nodes) if node.stop_reason is not None),
            None,
        )
        duration_ms = (
            (self._finished_at - self._started_at).total_seconds() * 1000
            if self._finished_at
            else None
        )
        return AgentRunTrace(
            run_id=self._run_id,
            user_id=self._user_id,
            input=self._input,
            started_at=self._started_at,
            finished_at=self._finished_at,
            duration_ms=duration_ms,
            nodes=tuple(self._nodes),
            final_response=final_response,
            stop_reason=stop_reason,
            error=self._error,
        )


def _optional_text(value: Any, *, limit: int = _SAFE_TEXT_LIMIT) -> str | None:
    return _safe_text(value, limit=limit) if value is not None else None


def new_agent_trace_callback(
    sink: AgentTraceSink | None = None,
    stream_sink: AgentNodeStreamingSink | None = None,
) -> AgentTraceCallback:
    """Create a callback for one Agent Run Trace."""
    return AgentTraceCallback(sink=sink, stream_sink=stream_sink)
