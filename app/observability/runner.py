"""Tracing wrapper around the existing LangGraph invocation contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.observability.tracing import (
    AgentRunTrace,
    AgentTraceSink,
    new_agent_trace_callback,
)


async def invoke_agent_with_trace(
    graph: Any,
    state: Mapping[str, Any],
    *,
    sink: AgentTraceSink | None = None,
    stream_sink: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], AgentRunTrace]:
    """Invoke a LangGraph Agent and return its result and completed run trace."""
    callback = new_agent_trace_callback(sink, stream_sink=stream_sink)
    run_config = dict(config or {})
    callbacks = list(run_config.get("callbacks") or [])
    callbacks.append(callback)
    run_config["callbacks"] = callbacks
    try:
        result = await graph.ainvoke(dict(state), config=run_config)
    except BaseException as exc:
        callback.finish()
        raise exc
    normalized = dict(result) if isinstance(result, Mapping) else {}
    return normalized, callback.finish(normalized)
