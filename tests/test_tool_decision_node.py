from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.nodes.tool_decision import tool_decision_node
from app.llm.types import LLMResponse, LLMToolCall
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class TrainingQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: str = Field(description="YYYY-MM-DD")
    end_date: str = Field(description="YYYY-MM-DD")


class FakeLLM:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class UnusedRepository:
    def __init__(self) -> None:
        self.calls = []

    def never_called(self):
        raise AssertionError("tool must not execute")


def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="list_training_records",
            description="查询训练记录。",
            input_model=TrainingQueryInput,
            handler=lambda args, user_id, repository: {"records": []},
        )
    )
    return registry


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "list_training_records",
            "description": "查询训练记录。",
            "parameters": TrainingQueryInput.model_json_schema(),
        },
    }


def state(message: str = "今天应该怎么训练？") -> dict[str, Any]:
    return {
        "input": message,
        "user_id": "user-001",
        "conversation_history": [],
    }


async def test_direct_question_returns_answer_draft_without_tools() -> None:
    response = LLMResponse(content="建议先做一次全身力量训练。", finish_reason="stop")
    llm = FakeLLM(response)

    result = await tool_decision_node(state("你好"), llm_client=llm, tool_registry=registry())

    assert result["tool_calls"] == []
    assert result["answer_draft"] == "建议先做一次全身力量训练。"
    assert result["stop_reason"] == "completed"
    assert llm.calls[0]["tools"] == [tool_definition()]
    assert llm.calls[0]["tool_choice"] == "auto"


async def test_training_query_generates_tool_call() -> None:
    response = LLMResponse(
        tool_calls=[
            LLMToolCall(
                call_id="call-1",
                name="list_training_records",
                arguments={"start_date": "2026-10-01", "end_date": "2026-10-07"},
            )
        ],
        finish_reason="tool_calls",
    )
    llm = FakeLLM(response)

    result = await tool_decision_node(
        state("查询我本周的训练记录"),
        llm_client=llm,
        tool_registry=registry(),
    )

    assert result["tool_calls"] == [
        {
            "call_id": "call-1",
            "tool_name": "list_training_records",
            "arguments": {
                "start_date": "2026-10-01",
                "end_date": "2026-10-07",
            },
        }
    ]
    assert result["answer_draft"] is None
    assert result["stop_reason"] == "tool_call"
    assert UnusedRepository().calls == []


async def test_tool_arguments_are_parsed_and_validated() -> None:
    response = LLMResponse(
        tool_calls=[
            LLMToolCall(
                call_id="call-1",
                name="list_training_records",
                arguments={"start_date": "2026-10-01", "end_date": "2026-10-07"},
            )
        ],
        finish_reason="tool_calls",
    )

    result = await tool_decision_node(
        state(),
        llm_client=FakeLLM(response),
        tool_registry=registry(),
    )

    assert result["tool_calls"][0]["arguments"]["start_date"] == "2026-10-01"
    assert result["tool_calls"][0]["arguments"]["end_date"] == "2026-10-07"


async def test_unknown_tool_is_rejected_safely() -> None:
    response = LLMResponse(
        tool_calls=[
            LLMToolCall(call_id="call-1", name="delete_everything", arguments={})
        ],
        finish_reason="tool_calls",
    )

    result = await tool_decision_node(
        state(),
        llm_client=FakeLLM(response),
        tool_registry=registry(),
    )

    assert result["tool_calls"] == []
    assert result["answer_draft"] is None
    assert result["stop_reason"] == "error"
    assert result["metadata"]["tool_decision_error"] == {
        "code": "unknown_tool",
        "message": "Requested tool is not registered",
    }


async def test_invalid_tool_arguments_are_rejected() -> None:
    response = LLMResponse(
        tool_calls=[
            LLMToolCall(
                call_id="call-1",
                name="list_training_records",
                arguments={"start_date": "not-a-date"},
            )
        ],
        finish_reason="tool_calls",
    )

    result = await tool_decision_node(
        state(),
        llm_client=FakeLLM(response),
        tool_registry=registry(),
    )

    assert result["tool_calls"] == []
    assert result["stop_reason"] == "error"
    assert result["metadata"]["tool_decision_error"]["code"] == "invalid_arguments"


async def test_user_id_argument_is_rejected() -> None:
    response = LLMResponse(
        tool_calls=[
            LLMToolCall(
                call_id="call-1",
                name="list_training_records",
                arguments={
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-07",
                    "user_id": "other-user",
                },
            )
        ],
        finish_reason="tool_calls",
    )

    result = await tool_decision_node(
        state(),
        llm_client=FakeLLM(response),
        tool_registry=registry(),
    )

    assert result["tool_calls"] == []
    assert result["stop_reason"] == "error"
    assert result["metadata"]["tool_decision_error"]["code"] == "forbidden_argument"


async def test_answer_draft_is_preserved_with_tool_request() -> None:
    response = LLMResponse(
        content="正在为你整理训练数据。",
        tool_calls=[
            LLMToolCall(
                call_id="call-1",
                name="list_training_records",
                arguments={"start_date": "2026-10-01", "end_date": "2026-10-07"},
            )
        ],
        finish_reason="tool_calls",
    )

    result = await tool_decision_node(
        state(),
        llm_client=FakeLLM(response),
        tool_registry=registry(),
    )

    assert result["answer_draft"] == "正在为你整理训练数据。"
    assert result["tool_calls"][0]["tool_name"] == "list_training_records"


async def test_llm_failure_returns_safe_error() -> None:
    class FailingLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("internal provider secret")

    result = await tool_decision_node(
        state(),
        llm_client=FailingLLM(),  # type: ignore[arg-type]
        tool_registry=registry(),
    )

    assert result["tool_calls"] == []
    assert result["answer_draft"] is None
    assert result["stop_reason"] == "error"
    assert result["metadata"]["tool_decision_error"] == {
        "code": "llm_decision_failed",
        "message": "Tool decision request failed",
    }
