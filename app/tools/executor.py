from typing import Any

from pydantic import ValidationError

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolCall, ToolResult


class ToolExecutor:
    """Validate and execute fixed tool calls without exposing internals."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        tool_call: ToolCall,
        *,
        user_id: str,
        repository: Any,
    ) -> ToolResult:
        definition = self._registry.get(tool_call.name)
        if definition is None:
            return self._error(tool_call, "unknown_tool", "Requested tool is not registered")

        if not user_id:
            return self._error(tool_call, "missing_user_context", "User context is required")
        if "user_id" in tool_call.arguments:
            return self._error(
                tool_call,
                "forbidden_argument",
                "user_id cannot be supplied in tool arguments",
            )

        try:
            validated = definition.input_model.model_validate(tool_call.arguments)
        except ValidationError:
            return self._error(tool_call, "invalid_arguments", "Tool arguments are invalid")

        try:
            data = definition.handler(validated.model_dump(), user_id, repository)
        except ValueError:
            return self._error(tool_call, "invalid_arguments", "Tool arguments are invalid")
        except Exception:
            return self._error(tool_call, "tool_execution_error", "Tool execution failed")

        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.name,
            status="success",
            data=data,
        )

    @staticmethod
    def _error(
        tool_call: ToolCall,
        error_code: str,
        safe_error_message: str,
    ) -> ToolResult:
        return ToolResult(
            call_id=tool_call.call_id,
            tool_name=tool_call.name,
            status="error",
            error_code=error_code,
            safe_error_message=safe_error_message,
        )