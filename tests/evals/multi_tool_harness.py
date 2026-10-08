"""Deterministic offline Multi Tool Calling evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.nodes.response import response_node
from app.agent.nodes.tool_decision import tool_decision_node
from app.llm.types import LLMResponse, LLMToolCall

DATASET_PATH = Path(__file__).parent / "datasets" / "multi_tool_selection.jsonl"
RESPONSE_DATASET_PATH = Path(__file__).parent / "datasets" / "multi_tool_response.jsonl"


@dataclass(frozen=True)
class MultiToolSelectionCase:
    id: str
    question: str
    expected_tools: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MultiToolSelectionCase:
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            expected_tools=tuple(str(item) for item in payload["expected_tools"]),
        )


@dataclass(frozen=True)
class MultiToolSelectionPrediction:
    case: MultiToolSelectionCase
    selected_tools: tuple[str, ...]
    stop_reason: str

    @property
    def is_correct(self) -> bool:
        return self.selected_tools == self.case.expected_tools and self.stop_reason == "tool_call"


@dataclass(frozen=True)
class MultiToolSelectionReport:
    predictions: tuple[MultiToolSelectionPrediction, ...]

    @property
    def total(self) -> int:
        return len(self.predictions)

    @property
    def correct(self) -> int:
        return sum(item.is_correct for item in self.predictions)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"Multi Tool Selection Accuracy: {self.accuracy:.2%} "
            f"({self.correct}/{self.total})",
        )


class FakeMultiToolSelectionLLM:
    def __init__(self, expected_tools: Sequence[str]) -> None:
        self.expected_tools = tuple(expected_tools)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    call_id=f"call-{index + 1}",
                    name=name,
                    arguments=_arguments_for(name),
                )
                for index, name in enumerate(self.expected_tools)
            ],
            finish_reason="tool_calls",
        )


@dataclass(frozen=True)
class MultiToolResponseCase:
    id: str
    question: str
    tool_results: tuple[dict[str, Any], ...]
    candidate_response: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MultiToolResponseCase:
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            tool_results=tuple(dict(item) for item in payload["tool_results"]),
            candidate_response=str(payload["candidate_response"]),
            required_terms=tuple(str(item) for item in payload["required_terms"]),
            forbidden_terms=tuple(str(item) for item in payload.get("forbidden_terms", [])),
        )


@dataclass(frozen=True)
class MultiToolResponsePrediction:
    case: MultiToolResponseCase
    response: str

    @property
    def passed(self) -> bool:
        return all(term in self.response for term in self.case.required_terms) and not any(
            term in self.response for term in self.case.forbidden_terms
        )


@dataclass(frozen=True)
class MultiToolResponseReport:
    predictions: tuple[MultiToolResponsePrediction, ...]

    @property
    def total(self) -> int:
        return len(self.predictions)

    @property
    def passed_count(self) -> int:
        return sum(item.passed for item in self.predictions)

    @property
    def score(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"Multi Tool Response Quality: {self.score:.2%} "
            f"({self.passed_count}/{self.total} pass)",
        )


class FakeMultiToolResponseLLM:
    def __init__(self, response_by_question: Mapping[str, str]) -> None:
        self.response_by_question = dict(response_by_question)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        question = _latest_user_question(kwargs["messages"])
        return self.response_by_question[question]


def load_multi_tool_selection_dataset(
    path: Path = DATASET_PATH,
) -> tuple[MultiToolSelectionCase, ...]:
    return tuple(
        MultiToolSelectionCase.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_multi_tool_response_dataset(
    path: Path = RESPONSE_DATASET_PATH,
) -> tuple[MultiToolResponseCase, ...]:
    return tuple(
        MultiToolResponseCase.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


async def evaluate_multi_tool_selection(
    cases: Iterable[MultiToolSelectionCase], *, tool_registry: Any
) -> MultiToolSelectionReport:
    predictions = []
    for case in cases:
        llm = FakeMultiToolSelectionLLM(case.expected_tools)
        result = await tool_decision_node(
            {"input": case.question, "user_id": "multi-eval-user", "conversation_history": []},
            llm_client=llm,
            tool_registry=tool_registry,
        )
        calls = result.get("tool_calls", [])
        predictions.append(
            MultiToolSelectionPrediction(
                case=case,
                selected_tools=tuple(call["tool_name"] for call in calls),
                stop_reason=str(result.get("stop_reason", "unknown")),
            )
        )
    return MultiToolSelectionReport(tuple(predictions))


async def evaluate_multi_tool_response(
    cases: Iterable[MultiToolResponseCase],
) -> MultiToolResponseReport:
    case_list = tuple(cases)
    llm = FakeMultiToolResponseLLM({case.question: case.candidate_response for case in case_list})
    predictions = []
    for case in case_list:
        result = await response_node(
            {
                "input": case.question,
                "conversation_history": [{"role": "user", "content": case.question}],
                "tool_results": list(case.tool_results),
                "tool_calls": [],
                "metadata": {},
            },
            llm_client=llm,
        )
        predictions.append(MultiToolResponsePrediction(case=case, response=str(result["response"])))
    return MultiToolResponseReport(tuple(predictions))


def _arguments_for(tool_name: str) -> dict[str, Any]:
    if tool_name == "get_training_recommendation":
        return {
            "user_goal": "提升体能",
            "recent_training": ["力量训练", "跑步"],
            "weather_condition": "rainy",
        }
    if tool_name == "get_current_weather":
        return {"latitude": 31.2304, "longitude": 121.4737}
    if tool_name == "get_last_training_record":
        return {}
    if tool_name == "get_training_summary":
        return {"start_date": "2026-10-01", "end_date": "2026-10-07"}
    return {
        "start_date": "2026-10-01",
        "end_date": "2026-10-07",
        "limit": 20,
    }


def _latest_user_question(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        payload = message.model_dump(mode="json") if hasattr(message, "model_dump") else message
        if isinstance(payload, Mapping) and payload.get("role") == "user":
            return str(payload.get("content", "")).strip()
    raise AssertionError("Multi Tool Response Evaluation requires a user question")
