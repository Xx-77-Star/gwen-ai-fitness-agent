from typing import Any

from app.agent.state import AgentState, ToolCall, ToolResult
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolCall as ExecutorToolCall

DEFAULT_MAX_TOOL_CALLS = 4


def tool_execution_node(
    state: AgentState,
    *,
    tool_executor: ToolExecutor,
    tool_registry: ToolRegistry,
    runtime_context: dict[str, Any] | None = None,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> dict[str, Any]:
    """Execute the current round of validated tool calls."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")

    calls = list(state.get("tool_calls", []))
    previous_count = int(state.get("tool_call_count", 0))
    previous_rounds = int(state.get("tool_rounds", 0))
    runtime = runtime_context or {}
    user_id = runtime.get("user_id")

    if len(calls) > max_tool_calls:
        return _limit_error(calls, previous_count, previous_rounds, max_tool_calls)

    results: list[ToolResult] = []
    for call in calls:
        validation_error = _validate_call(call, tool_registry, user_id)
        if validation_error is not None:
            results.append(validation_error)
            continue
        try:
            result = tool_executor.execute(
                ExecutorToolCall(
                    call_id=call["call_id"],
                    name=call["tool_name"],
                    arguments=dict(call.get("arguments", {})),
                ),
                user_id=str(user_id),
                repository=_tool_runtime_dependency(call.get("tool_name"), runtime),
            )
        except Exception:
            results.append(_error_result(call, "tool_execution_error", "Tool execution failed"))
            continue
        results.append(_to_state_result(result))

    return {
        "tool_calls": [],
        "tool_results": results,
        "tool_call_count": previous_count + len(calls),
        "tool_rounds": previous_rounds + (1 if calls else 0),
        "metadata": {"_previous_tool_calls": calls},
        "stop_reason": "completed" if calls else state.get("stop_reason", "completed"),
    }


def _tool_runtime_dependency(tool_name: str | None, runtime: dict[str, Any]) -> Any:
    if tool_name == "get_current_weather":
        return runtime.get("weather_client")
    return runtime.get("repository")


def _validate_call(
    call: ToolCall,
    tool_registry: ToolRegistry,
    user_id: Any,
) -> ToolResult | None:
    call_id = call.get("call_id")
    tool_name = call.get("tool_name")
    arguments = call.get("arguments", {})
    if not call_id or not tool_name:
        return _error_result(call, "invalid_tool_call", "Tool call is invalid")
    if tool_registry.get(tool_name) is None:
        return _error_result(call, "unknown_tool", "Requested tool is not registered")
    if not user_id:
        return _error_result(call, "missing_user_context", "User context is required")
    if "user_id" in arguments:
        return _error_result(
            call, "forbidden_argument", "user_id cannot be supplied in tool arguments"
        )
    return None


def _to_state_result(result: Any) -> ToolResult:
    data = getattr(result, "data", None)
    if getattr(result, "status", "error") == "success":
        return ToolResult(
            call_id=result.call_id,
            tool_call_id=result.call_id,
            tool_name=result.tool_name,
            status="success",
            result=data or {},
        )
    return ToolResult(
        call_id=result.call_id,
        tool_call_id=result.call_id,
        tool_name=result.tool_name,
        status="error",
        error={
            "code": getattr(result, "error_code", "tool_execution_error"),
            "message": getattr(result, "safe_error_message", "Tool execution failed"),
        },
    )


def _error_result(call: ToolCall, code: str, message: str) -> ToolResult:
    call_id = call.get("call_id", "")
    return ToolResult(
        call_id=call_id,
        tool_call_id=call_id,
        tool_name=call.get("tool_name", ""),
        status="error",
        error={"code": code, "message": message},
    )


def _limit_error(
    calls: list[ToolCall],
    previous_count: int,
    previous_rounds: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    return {
        "tool_calls": [],
        "tool_results": [],
        "tool_call_count": previous_count,
        "tool_rounds": previous_rounds,
        "stop_reason": "tool_limit",
        "metadata": {
            "tool_execution_error": {
                "code": "tool_call_limit",
                "message": f"Tool call count exceeds the maximum of {max_tool_calls}",
                "requested_count": len(calls),
            }
        },
    }
