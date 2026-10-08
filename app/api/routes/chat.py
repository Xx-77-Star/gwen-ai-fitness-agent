import json
from collections.abc import Mapping, Sequence
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.graph import get_agent_graph_with_repository
from app.database.session import get_db
from app.llm.errors import LLMError
from app.memory.conversation_repository import SQLConversationRepository
from app.observability.runner import invoke_agent_with_trace
from app.observability.streaming import get_agent_run_bus
from app.observability.tracing import (
    AgentNodeStreamingSink,
    AgentNodeTrace,
    AgentRunTrace,
    LoggingAgentTraceSink,
)
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

DbSession = Annotated[Session, Depends(get_db)]

_NODE_DISPLAY = {
    "load_profile": ("读取用户画像", "profile"),
    "memory_retrieve": ("读取记忆", "memory"),
    "rag_retrieve": ("检索知识库", "retrieval"),
    "intent": ("识别用户意图", "reasoning"),
    "tool_decision": ("判断工具需求", "decision"),
    "tool_execution": ("执行工具", "tools"),
    "response": ("生成回答", "response"),
    "memory_update": ("更新记忆", "memory"),
    "conversation_persistence": ("保存会话", "persistence"),
}


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: DbSession) -> ChatResponse:
    """Run one turn through the LangGraph workflow and persist conversation turns."""
    conversation_repository = SQLConversationRepository(db)
    # A client-supplied run id lets the status stream subscribe before the work
    # starts; generate one when the caller omitted it so the stream id is stable.
    conversation_id = (payload.conversation_id or "").strip() or str(uuid4())
    agent_graph = get_agent_graph_with_repository(db)
    # Live node events for the user-facing execution status card (observability
    # only: no Agent / Memory / RAG logic is touched).
    run_bus = get_agent_run_bus()
    stream_sink = AgentNodeStreamingSink(conversation_id, payload.user_id)
    await run_bus.register(conversation_id, payload.user_id)
    try:
        conversation_id = conversation_repository.create_conversation(
            payload.user_id, conversation_id
        )
        result, trace = await invoke_agent_with_trace(
            agent_graph,
            {
                "user_id": payload.user_id,
                "conversation_id": conversation_id,
                "input": payload.message,
                "conversation_history": [],
                "rag_context": [],
                "tool_calls": [],
                "tool_rounds": 0,
                "tool_call_count": 0,
                "answer_draft": None,
                "stop_reason": "completed",
                "messages": [],
                "tool_messages": [],
                "tool_results": [],
            },
            sink=LoggingAgentTraceSink(),
            stream_sink=stream_sink,
        )
        response = result.get("response")
        if not isinstance(response, str):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent workflow returned an invalid state",
            )
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM service is unavailable",
        ) from exc
    finally:
        # Always close the execution-status stream, including validation or
        # upstream failures, so the client never waits on a dead run.
        await run_bus.finish(conversation_id)

    return ChatResponse(
        response=response,
        conversation_id=conversation_id,
        trace=_frontend_trace(trace, result),
    )




@router.get("/{conversation_id}/events")
async def chat_events(conversation_id: str, user_id: str):
    """Stream node lifecycle events for one chat run (execution status only).

    The payload contains only workflow stage names and start/end phases - never
    model reasoning, prompts or tool payloads.
    """
    run_bus = get_agent_run_bus()
    owner = await run_bus.owner(conversation_id)
    if owner is not None and owner != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="conversation does not belong to user",
        )
    subscription = await run_bus.subscribe(conversation_id)

    async def event_stream():
        try:
            while True:
                event = await subscription.queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event.payload(), ensure_ascii=False)}\n\n"
        finally:
            await run_bus.unsubscribe(conversation_id, subscription)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

def _frontend_trace(trace: AgentRunTrace, result: dict[str, object]) -> dict[str, object]:
    """Build a UI-friendly trace while preserving the existing API fields."""
    node_payloads = [node.model_dump() for node in trace.nodes]
    memory = _memory_payload(result.get("memory_context"))
    rag_chunks = _mapping_list(result.get("rag_context"))
    source_names = _source_names(rag_chunks)
    tool_exchanges = _tool_exchanges(trace.nodes, result)
    timeline = _timeline(trace.nodes)
    tool_calls = [
        {
            "call_id": item["call_id"],
            "tool_name": item["tool_name"],
            "arguments": item["arguments"],
        }
        for item in tool_exchanges
    ]
    tool_results = [
        {
            "call_id": item["call_id"],
            "tool_name": item["tool_name"],
            "status": item["status"],
            "result": item["result"],
            "error": item["error"],
            "result_summary": item["result_summary"],
        }
        for item in tool_exchanges
        if item["result_summary"] is not None
        or item["result"] is not None
        or item["error"] is not None
    ]
    stop_reason = _first_text(trace.stop_reason, result.get("stop_reason")) or "unknown"
    run = {
        "run_id": trace.run_id,
        "status": trace.status,
        "duration_ms": trace.duration_ms,
        "stop_reason": stop_reason,
        "error": trace.error,
    }
    summary = {
        "status": trace.status,
        "duration_ms": trace.duration_ms,
        "node_count": len(trace.nodes),
        "tool_call_count": len(tool_exchanges),
        "tool_result_count": sum(item["has_result"] for item in tool_exchanges),
        "rag_chunk_count": len(rag_chunks),
        "memory_fact_count": _list_length(memory.get("long_term")),
        "memory_turn_count": _list_length(memory.get("short_term")),
        "source_count": len(source_names),
    }
    evidence = {
        "memory": {
            "fact_count": summary["memory_fact_count"],
            "turn_count": summary["memory_turn_count"],
            "facts": _mapping_list(memory.get("long_term")),
            "turns": _mapping_list(memory.get("short_term")),
        },
        "rag": {"chunk_count": len(rag_chunks), "chunks": rag_chunks},
        "tools": {"exchange_count": len(tool_exchanges), "exchanges": tool_exchanges},
        "sources": source_names,
    }
    return {
        "run_id": trace.run_id,
        "status": trace.status,
        "duration_ms": trace.duration_ms,
        "nodes": node_payloads,
        "memory": memory,
        "rag": {"chunks": rag_chunks},
        "tools": {"calls": tool_calls, "results": tool_results},
        "sources": [item for item in _source_names(rag_chunks, unique=False)],
        "run": run,
        "summary": summary,
        "timeline": timeline,
        "tool_exchanges": tool_exchanges,
        "evidence": evidence,
    }


def _timeline(nodes: Sequence[AgentNodeTrace]) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    for sequence, node in enumerate(nodes, start=1):
        label, category = _NODE_DISPLAY.get(node.node_name, (node.node_name, "other"))
        timeline.append(
            {
                "sequence": sequence,
                "node_name": node.node_name,
                "label": label,
                "category": category,
                "status": node.status,
                "duration_ms": node.duration_ms,
                "details": {
                    "tool_calls": node.tool_calls,
                    "tool_results": node.tool_results_summary,
                    "final_response": node.final_response,
                    "stop_reason": node.stop_reason,
                    "error": node.error,
                },
            }
        )
    return timeline


def _tool_exchanges(
    nodes: Sequence[AgentNodeTrace],
    result: Mapping[str, object],
) -> list[dict[str, object]]:
    trace_calls = _trace_tool_calls(nodes)
    trace_summaries = _trace_tool_summaries(nodes)
    state_results = _mapping_list(result.get("tool_results"))
    if not trace_calls:
        trace_calls = _calls_from_results(state_results)
    if not trace_calls and not state_results:
        trace_calls = _mapping_list(result.get("tool_calls"))

    exchanges: list[dict[str, object]] = []
    remaining_results = list(state_results)
    remaining_summaries = list(trace_summaries)
    used_results: set[int] = set()
    used_summaries: set[int] = set()

    for call in trace_calls:
        result_index = _matching_index(remaining_results, call, used_results)
        summary_index = _matching_index(remaining_summaries, call, used_summaries)
        state_result = remaining_results[result_index] if result_index is not None else None
        summary = remaining_summaries[summary_index] if summary_index is not None else None
        if result_index is not None:
            used_results.add(result_index)
        if summary_index is not None:
            used_summaries.add(summary_index)
        exchanges.append(_tool_exchange(call, state_result, summary))

    for index, state_result in enumerate(remaining_results):
        if index in used_results:
            continue
        exchanges.append(_tool_exchange(_call_from_result(state_result), state_result, None))
    for index, summary in enumerate(remaining_summaries):
        if index in used_summaries:
            continue
        exchanges.append(_tool_exchange(_call_from_summary(summary), None, summary))
    return exchanges


def _tool_exchange(
    call: Mapping[str, object],
    state_result: Mapping[str, object] | None,
    summary: Mapping[str, object] | None,
) -> dict[str, object]:
    status = (
        _first_text(
            state_result.get("status") if state_result else None,
            summary.get("status") if summary else None,
        )
        or "pending"
    )
    error = state_result.get("error") if state_result else None
    if error is None and summary is not None and summary.get("error_code"):
        error = {"code": summary["error_code"]}
    result_value = state_result.get("result") if state_result is not None else None
    summary_payload = dict(summary) if summary is not None else None
    return {
        "call_id": _first_text(
            call.get("call_id"),
            state_result.get("call_id") if state_result else None,
        )
        or "",
        "tool_name": _first_text(
            call.get("tool_name"),
            state_result.get("tool_name") if state_result else None,
            summary.get("tool_name") if summary else None,
        )
        or "",
        "arguments": call.get("arguments", {}),
        "status": status,
        "result": result_value,
        "error": error,
        "result_summary": summary_payload,
        "has_result": state_result is not None or summary is not None,
    }


def _trace_tool_calls(nodes: Sequence[AgentNodeTrace]) -> list[dict[str, object]]:
    return [dict(item) for node in nodes for item in _mapping_list(node.tool_calls)]


def _trace_tool_summaries(nodes: Sequence[AgentNodeTrace]) -> list[dict[str, object]]:
    return [dict(item) for node in nodes for item in _mapping_list(node.tool_results_summary)]


def _matching_index(
    records: Sequence[Mapping[str, object]],
    call: Mapping[str, object],
    used: set[int],
) -> int | None:
    call_id = _first_text(call.get("call_id"))
    tool_name = _first_text(call.get("tool_name"))
    candidates = [index for index, item in enumerate(records) if index not in used]
    if call_id:
        for index in candidates:
            record_id = _first_text(
                records[index].get("call_id"),
                records[index].get("tool_call_id"),
            )
            if record_id == call_id:
                return index
    if tool_name:
        for index in candidates:
            record = records[index]
            record_id = _first_text(record.get("call_id"), record.get("tool_call_id"))
            if not record_id and _first_text(record.get("tool_name")) == tool_name:
                return index
    return None


def _calls_from_results(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for item in results:
        call = _call_from_result(item)
        if call:
            calls.append(call)
    return calls


def _call_from_result(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "call_id": _first_text(item.get("call_id"), item.get("tool_call_id")) or "",
        "tool_name": _first_text(item.get("tool_name")) or "",
        "arguments": item.get("arguments", {}),
    }


def _call_from_summary(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "call_id": _first_text(item.get("call_id")) or "",
        "tool_name": _first_text(item.get("tool_name")) or "",
        "arguments": {},
    }


def _memory_payload(value: object) -> dict[str, object]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _source_names(chunks: Sequence[Mapping[str, object]], *, unique: bool = True) -> list[str]:
    names = [_first_text(item.get("source")) for item in chunks]
    values = [name for name in names if name]
    return list(dict.fromkeys(values)) if unique else values


def _list_length(value: object) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return 0
    return len(value)


def _first_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None
