from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.nodes.tool_execution import tool_execution_node
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class SummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: str = Field()
    end_date: str = Field()


class FakeRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def query(self, user_id: str) -> dict[str, Any]:
        self.calls.append(("query", user_id))
        return {"record_count": 2}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_training_summary",
            description="查询训练汇总。",
            input_model=SummaryInput,
            handler=lambda args, user_id, repository: {
                "user_id": user_id,
                "repository_calls": repository.calls,
                **args,
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="get_training_records",
            description="查询训练记录。",
            input_model=SummaryInput,
            handler=lambda args, user_id, repository: {"user_id": user_id, **args},
        )
    )
    registry.register(
        ToolDefinition(
            name="explode",
            description="异常工具。",
            input_model=SummaryInput,
            handler=lambda args, user_id, repository: (_ for _ in ()).throw(
                RuntimeError("SQL password must not escape")
            ),
        )
    )
    return registry


def executor() -> ToolExecutor:
    return ToolExecutor(build_registry())


def call(name: str, call_id: str = "call-1", arguments: dict[str, Any] | None = None):
    return {
        "call_id": call_id,
        "tool_name": name,
        "arguments": arguments or {"start_date": "2026-10-01", "end_date": "2026-10-07"},
    }


def runtime(repository: FakeRepository) -> dict[str, Any]:
    return {"user_id": "user-001", "repository": repository}


def test_executes_get_training_summary_and_injects_runtime_user_id() -> None:
    repository = FakeRepository()

    result = tool_execution_node(
        {"tool_calls": [call("get_training_summary")]},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context=runtime(repository),
    )

    assert result["tool_results"] == [
        {
            "call_id": "call-1",
            "tool_call_id": "call-1",
            "tool_name": "get_training_summary",
            "status": "success",
            "result": {
                "user_id": "user-001",
                "repository_calls": [],
                "start_date": "2026-10-01",
                "end_date": "2026-10-07",
            },
        }
    ]
    assert result["tool_call_count"] == 1
    assert result["tool_rounds"] == 1
    assert result["stop_reason"] == "completed"


def test_multiple_tool_calls_execute_in_order() -> None:
    repository = FakeRepository()
    calls = [
        call("get_training_summary", "call-1"),
        call("get_training_records", "call-2"),
    ]

    result = tool_execution_node(
        {"tool_calls": calls},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context=runtime(repository),
    )

    assert [item["call_id"] for item in result["tool_results"]] == ["call-1", "call-2"]
    assert result["tool_call_count"] == 2
    assert result["tool_rounds"] == 1


def test_unknown_tool_returns_safe_error_without_execution() -> None:
    repository = FakeRepository()

    result = tool_execution_node(
        {"tool_calls": [call("unknown_tool")]},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context=runtime(repository),
    )

    assert result["tool_results"] == [
        {
            "call_id": "call-1",
            "tool_call_id": "call-1",
            "tool_name": "unknown_tool",
            "status": "error",
            "error": {
                "code": "unknown_tool",
                "message": "Requested tool is not registered",
            },
        }
    ]
    assert repository.calls == []


def test_handler_exception_is_converted_to_safe_error() -> None:
    repository = FakeRepository()

    result = tool_execution_node(
        {"tool_calls": [call("explode")]},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context=runtime(repository),
    )

    assert result["tool_results"][0]["status"] == "error"
    assert result["tool_results"][0]["error"] == {
        "code": "tool_execution_error",
        "message": "Tool execution failed",
    }
    assert "SQL password" not in str(result)


def test_runtime_user_id_is_required_and_model_user_id_is_rejected() -> None:
    repository = FakeRepository()
    result_without_user = tool_execution_node(
        {"tool_calls": [call("get_training_summary")]},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context={"repository": repository},
    )
    result_with_model_user = tool_execution_node(
        {
            "tool_calls": [
                call(
                    "get_training_summary",
                    arguments={
                        "start_date": "2026-10-01",
                        "end_date": "2026-10-07",
                        "user_id": "model-user",
                    },
                )
            ]
        },
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context=runtime(repository),
    )

    assert result_without_user["tool_results"][0]["error"]["code"] == "missing_user_context"
    assert result_with_model_user["tool_results"][0]["error"]["code"] == "forbidden_argument"
    assert repository.calls == []


def test_execution_does_not_access_database_beyond_runtime_repository() -> None:
    repository = FakeRepository()

    result = tool_execution_node(
        {"tool_calls": [call("get_training_summary")]},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context={"user_id": "user-001", "repository": repository},
    )

    assert result["tool_results"][0]["result"]["repository_calls"] == []
    assert repository.calls == []


def test_empty_tool_calls_are_normal_noop() -> None:
    result = tool_execution_node(
        {"tool_calls": [], "tool_call_count": 3, "tool_rounds": 2},
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context={"user_id": "user-001", "repository": FakeRepository()},
    )

    assert result["tool_results"] == []
    assert result["tool_call_count"] == 3
    assert result["tool_rounds"] == 2
    assert result["stop_reason"] == "completed"


def test_tool_call_count_limit_rejects_batch_before_execution() -> None:
    repository = FakeRepository()

    result = tool_execution_node(
        {
            "tool_calls": [
                call("get_training_summary", "call-1"),
                call("get_training_records", "call-2"),
            ]
        },
        tool_executor=executor(),
        tool_registry=build_registry(),
        runtime_context=runtime(repository),
        max_tool_calls=1,
    )

    assert result["tool_results"] == []
    assert result["tool_call_count"] == 0
    assert result["tool_rounds"] == 0
    assert result["stop_reason"] == "tool_limit"
    assert result["metadata"]["tool_execution_error"]["code"] == "tool_call_limit"
    assert repository.calls == []
