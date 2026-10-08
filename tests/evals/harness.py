"""Deterministic local harness for Tool Selection Evaluation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.nodes.tool_decision import tool_decision_node
from app.llm.types import LLMResponse, LLMToolCall
from app.tools.registry import ToolRegistry

DATASET_PATH = Path(__file__).parent / "datasets" / "tool_selection.jsonl"
EVAL_USER_ID = "eval-user"
EVAL_START_DATE = "2026-10-01"
EVAL_END_DATE = "2026-10-07"
VALID_TOOLS = frozenset(
    {
        "get_training_summary",
        "get_last_training_record",
        "list_training_records",
        "get_current_weather",
        "get_training_recommendation",
    }
)

_latest_report: ToolSelectionEvaluationReport | None = None


@dataclass(frozen=True)
class ToolSelectionCase:
    """One labeled natural-language tool selection example."""

    id: str
    question: str
    expected_tool: str
    arguments: dict[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ToolSelectionCase:
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            expected_tool=str(payload["expected_tool"]),
            arguments=dict(payload.get("arguments", {})),
        )


@dataclass(frozen=True)
class ToolSelectionPrediction:
    """Observed selection for one dataset case."""

    case: ToolSelectionCase
    selected_tool: str | None
    stop_reason: str
    error_code: str | None

    @property
    def is_correct(self) -> bool:
        return self.selected_tool == self.case.expected_tool and self.stop_reason == "tool_call"


@dataclass(frozen=True)
class ToolSelectionEvaluationReport:
    """Aggregate accuracy and per-tool metrics for one evaluation run."""

    predictions: tuple[ToolSelectionPrediction, ...]

    @property
    def total(self) -> int:
        return len(self.predictions)

    @property
    def correct(self) -> int:
        return sum(prediction.is_correct for prediction in self.predictions)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def failures(self) -> tuple[ToolSelectionPrediction, ...]:
        return tuple(prediction for prediction in self.predictions if not prediction.is_correct)

    @property
    def per_tool_accuracy(self) -> dict[str, float]:
        metrics: dict[str, tuple[int, int]] = {}
        for prediction in self.predictions:
            correct, total = metrics.get(prediction.case.expected_tool, (0, 0))
            metrics[prediction.case.expected_tool] = (
                correct + int(prediction.is_correct),
                total + 1,
            )
        return {
            tool: tool_correct / tool_total
            for tool, (tool_correct, tool_total) in sorted(metrics.items())
        }

    def summary_lines(self) -> tuple[str, ...]:
        lines = [
            f"Tool Selection Accuracy: {self.accuracy:.2%} ({self.correct}/{self.total})"
        ]
        lines.extend(
            f"- {tool}: {accuracy:.2%}"
            for tool, accuracy in self.per_tool_accuracy.items()
        )
        return tuple(lines)


class FakeToolSelectionLLM:
    """Offline Fake LLM that emits deterministic training tool decisions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs.get("tools") is None:
            raise AssertionError("Tool Selection Evaluation must use tool calling")
        question = _latest_user_question(kwargs["messages"])
        tool_name = self._select_tool(question)
        call_id = f"fake-call-{len(self.calls)}"
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    call_id=call_id,
                    name=tool_name,
                    arguments=_arguments_for(tool_name),
                )
            ],
            finish_reason="tool_calls",
        )

    @staticmethod
    def _select_tool(question: str) -> str:
        normalized = question.strip()
        last_training_markers = (
            "上一次",
            "上次",
            "最近一次",
            "最后一次",
            "最后练",
            "训练是什么",
            "练了什么",
        )
        list_markers = (
            "列出",
            "记录列表",
            "记录明细",
            "有哪些训练",
            "所有训练记录",
            "训练清单",
            "都列出来",
        )
        recommendation_markers = (
            "训练建议",
            "怎么训练",
            "怎么练",
            "推荐训练",
            "今天练什么",
            "安排训练",
            "安排今天",
            "课程推荐",
        )
        weather_markers = (
            "天气",
            "气温",
            "温度",
            "风速",
            "户外训练",
        )
        summary_markers = (
            "最近训练情况",
            "最近练得怎么样",
            "最近的训练情况",
            "训练总结",
            "总结训练",
            "训练趋势",
            "训练次数",
            "训练时长",
            "类型分布",
        )

        if any(marker in normalized for marker in recommendation_markers):
            return "get_training_recommendation"
        if any(marker in normalized for marker in weather_markers):
            return "get_current_weather"
        if normalized in {"最近练得怎么样", "训练总结", "训练趋势"}:
            return "get_training_summary"
        if normalized.startswith("查询我最近训练") or normalized.startswith("我最近训练"):
            return "get_training_summary"
        if any(marker in normalized for marker in last_training_markers):
            return "get_last_training_record"
        if any(marker in normalized for marker in list_markers):
            return "list_training_records"
        if any(marker in normalized for marker in summary_markers):
            return "get_training_summary"
        raise AssertionError(f"Fake LLM has no deterministic decision for: {question}")


def load_tool_selection_dataset(
    path: Path = DATASET_PATH,
) -> tuple[ToolSelectionCase, ...]:
    cases: list[ToolSelectionCase] = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on dataset line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Dataset line {line_number} must be an object")
            cases.append(ToolSelectionCase.from_mapping(payload))
    return tuple(cases)


async def evaluate_tool_selection(
    cases: Iterable[ToolSelectionCase],
    *,
    tool_registry: ToolRegistry,
) -> ToolSelectionEvaluationReport:
    """Run all cases through the real Tool Decision Node with a Fake LLM."""
    global _latest_report

    llm = FakeToolSelectionLLM()
    predictions: list[ToolSelectionPrediction] = []
    for case in cases:
        result = await tool_decision_node(
            {
                "input": case.question,
                "user_id": EVAL_USER_ID,
                "conversation_history": [],
            },
            llm_client=llm,
            tool_registry=tool_registry,
        )
        calls = result.get("tool_calls", [])
        selected_tool = calls[0]["tool_name"] if len(calls) == 1 else None
        error = result.get("metadata", {}).get("tool_decision_error", {})
        predictions.append(
            ToolSelectionPrediction(
                case=case,
                selected_tool=selected_tool,
                stop_reason=str(result.get("stop_reason", "unknown")),
                error_code=error.get("code"),
            )
        )

    report = ToolSelectionEvaluationReport(tuple(predictions))
    _latest_report = report
    return report


def latest_tool_selection_report() -> ToolSelectionEvaluationReport | None:
    """Return the most recent report for pytest terminal reporting."""
    return _latest_report


def reset_latest_report() -> None:
    """Clear the report between independent evaluation runs."""
    global _latest_report

    _latest_report = None


def write_accuracy_report(
    path: Path,
    cases: Iterable[ToolSelectionCase],
    *,
    tool_registry: ToolRegistry,
) -> ToolSelectionEvaluationReport:
    """Evaluate the full dataset and persist the accuracy summary."""
    report = asyncio.run(evaluate_tool_selection(cases, tool_registry=tool_registry))
    path.write_text("\n".join(report.summary_lines()) + "\n", encoding="utf-8")
    return report


def _arguments_for(tool_name: str) -> dict[str, Any]:
    if tool_name == "get_last_training_record":
        return {}
    if tool_name == "get_training_recommendation":
        return {
            "user_goal": "提升体能",
            "recent_training": ["跑步", "力量训练"],
            "weather_condition": "rainy",
        }
    if tool_name == "get_current_weather":
        return {"latitude": 31.2304, "longitude": 121.4737}
    if tool_name == "get_training_summary":
        return {"start_date": EVAL_START_DATE, "end_date": EVAL_END_DATE}
    return {
        "start_date": EVAL_START_DATE,
        "end_date": EVAL_END_DATE,
        "limit": 20,
    }


def _latest_user_question(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        payload = message.model_dump(mode="json") if hasattr(message, "model_dump") else message
        if isinstance(payload, Mapping) and payload.get("role") == "user":
            return str(payload.get("content", "")).strip()
    raise AssertionError("Tool Selection Evaluation requires a user question")