from pydantic import BaseModel

from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolCall, ToolDefinition


class ValidInput(BaseModel):
    value: int


class EmptyRepository:
    def __init__(self) -> None:
        self.calls = []


def ok_handler(arguments, user_id, repository):
    repository.calls.append((arguments, user_id))
    return {"value": arguments["value"], "user_id": user_id}


def error_handler(arguments, user_id, repository):
    raise RuntimeError("database connection string must not escape")


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="ok_tool",
            description="A valid tool.",
            input_model=ValidInput,
            handler=ok_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="error_tool",
            description="A failing tool.",
            input_model=ValidInput,
            handler=error_handler,
        )
    )
    return registry


def test_execute_registered_tool() -> None:
    repository = EmptyRepository()
    executor = ToolExecutor(build_registry())

    result = executor.execute(
        ToolCall(call_id="call-1", name="ok_tool", arguments={"value": 7}),
        user_id="user-001",
        repository=repository,
    )

    assert result.status == "success"
    assert result.data == {"value": 7, "user_id": "user-001"}
    assert repository.calls == [({"value": 7}, "user-001")]


def test_execute_unknown_tool_returns_safe_error() -> None:
    executor = ToolExecutor(ToolRegistry())

    result = executor.execute(
        ToolCall(call_id="call-1", name="unknown", arguments={}),
        user_id="user-001",
        repository=EmptyRepository(),
    )

    assert result.status == "error"
    assert result.error_code == "unknown_tool"


def test_execute_invalid_arguments_returns_error() -> None:
    executor = ToolExecutor(build_registry())

    result = executor.execute(
        ToolCall(call_id="call-1", name="ok_tool", arguments={"value": "bad"}),
        user_id="user-001",
        repository=EmptyRepository(),
    )

    assert result.status == "error"
    assert result.error_code == "invalid_arguments"


def test_execute_rejects_user_id_argument() -> None:
    executor = ToolExecutor(build_registry())

    result = executor.execute(
        ToolCall(
            call_id="call-1",
            name="ok_tool",
            arguments={"value": 7, "user_id": "other-user"},
        ),
        user_id="user-001",
        repository=EmptyRepository(),
    )

    assert result.status == "error"
    assert result.error_code == "forbidden_argument"


def test_handler_exception_is_not_exposed() -> None:
    executor = ToolExecutor(build_registry())

    result = executor.execute(
        ToolCall(call_id="call-1", name="error_tool", arguments={"value": 1}),
        user_id="user-001",
        repository=EmptyRepository(),
    )

    assert result.status == "error"
    assert result.error_code == "tool_execution_error"
    assert result.safe_error_message == "Tool execution failed"