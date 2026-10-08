"""Real Bailian evaluation harness with sanitized behavior logs."""

from __future__ import annotations

import json
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.llm.client import OpenAICompatibleLLMClient
from app.llm.types import LLMResponse
from app.tools.registry import ToolRegistry

RESULTS_DIR = Path(__file__).parent.parent / "results"
TOOL_SELECTION_RESULT_PATH = RESULTS_DIR / "real_tool_selection_result.json"
RESPONSE_QUALITY_RESULT_PATH = RESULTS_DIR / "real_response_quality_result.json"
START_DATE = "2026-10-01"
END_DATE = "2026-10-07"

TOOL_SELECTION_CASES = (
    ("tool-001", "查询我最近训练情况", "get_training_summary"),
    ("tool-002", "我上一次训练是什么", "get_last_training_record"),
    ("tool-003", "列出我的训练记录", "list_training_records"),
    ("tool-004", "最近训练趋势如何", "get_training_summary"),
    ("tool-005", "看看我的训练历史", "list_training_records"),
    ("tool-006", "上海今天天气怎么样", "get_current_weather"),
    ("tool-007", "根据最近训练帮我安排今天的训练建议", "get_training_recommendation"),
)


@dataclass
class RealToolCaseResult:
    id: str
    question: str
    expected_tool: str
    selected_tools: list[str] = field(default_factory=list)
    returned_tool_calls: int = 0
    parameter_error: bool = False
    duplicate_tool_calls: bool = False
    arguments: list[dict[str, Any]] = field(default_factory=list)
    correct: bool = False
    error: str | None = None


@dataclass
class RealToolSelectionResult:
    model: str
    accuracy: float
    cases: list[RealToolCaseResult]
    tool_calls_stable: bool
    tool_misselection_count: int
    parameter_error_count: int
    duplicate_call_count: int
    recorded_at: str


@dataclass
class RealResponseResult:
    model: str
    question: str
    response: str
    real_record_count: int
    real_total_duration_minutes: int
    real_types: list[str]
    has_record_count: bool
    has_total_duration: bool
    has_type_information: bool
    non_empty: bool
    groundedness_pass: bool
    fabrication_detected: bool
    tool_result_ignored: bool
    record_count_mentions: list[str] = field(default_factory=list)
    duration_mentions: list[str] = field(default_factory=list)
    type_mentions: list[str] = field(default_factory=list)


@dataclass
class RealResponseQualityResult:
    model: str
    score: float
    cases: list[RealResponseResult]
    recorded_at: str


def build_real_llm_client(settings: Settings | None = None) -> OpenAICompatibleLLMClient:
    """Build the configured real client without exposing its API key."""
    return OpenAICompatibleLLMClient(settings or Settings())


async def run_real_tool_selection(
    *,
    model: str,
    tool_registry: ToolRegistry,
    cases: tuple[tuple[str, str, str], ...] = TOOL_SELECTION_CASES,
) -> RealToolSelectionResult:
    """Call real qwen-plus and record sanitized Tool Calling behavior."""
    llm = build_real_llm_client()
    payloads = tool_registry.schema_payloads()
    definitions = {name: tool_registry.get(name) for name in tool_registry.names()}
    results: list[RealToolCaseResult] = []

    for case_id, question, expected_tool in cases:
        result = RealToolCaseResult(
            id=case_id,
            question=question,
            expected_tool=expected_tool,
        )
        try:
            with _capture_provider_debug_output():
                response = await llm.complete(
                    system_prompt="你是 FitLife AI 的工具选择评估助手。只返回需要调用的合适工具。",
                    messages=[{"role": "user", "content": question}],
                    tools=payloads,
                    tool_choice="auto",
                )
            if not isinstance(response, LLMResponse):
                raise TypeError("real LLM returned a non-tool response")
            result.returned_tool_calls = len(response.tool_calls)
            result.selected_tools = [call.name for call in response.tool_calls]
            result.arguments = [dict(call.arguments) for call in response.tool_calls]
            result.duplicate_tool_calls = len(result.selected_tools) != len(
                set(result.selected_tools)
            )
            result.parameter_error = any(
                _has_parameter_error(definitions.get(call.name), call.arguments)
                for call in response.tool_calls
            )
            result.correct = (
                result.returned_tool_calls == 1
                and result.selected_tools == [expected_tool]
                and not result.parameter_error
            )
        except Exception as exc:
            result.error = type(exc).__name__
        results.append(result)

    correct_count = sum(result.correct for result in results)
    accuracy = correct_count / len(results) if results else 0.0
    await _close_real_client(llm)
    return RealToolSelectionResult(
        model=model,
        accuracy=accuracy,
        cases=results,
        tool_calls_stable=all(result.returned_tool_calls == 1 for result in results),
        tool_misselection_count=sum(
            result.returned_tool_calls == 1 and result.selected_tools != [result.expected_tool]
            for result in results
        ),
        parameter_error_count=sum(result.parameter_error for result in results),
        duplicate_call_count=sum(result.duplicate_tool_calls for result in results),
        recorded_at=datetime.now(UTC).isoformat(),
    )


async def run_real_response_groundedness(
    *,
    model: str,
    tool_registry: ToolRegistry,
    training_result: dict[str, Any],
) -> RealResponseResult:
    """Generate one final answer and verify it reflects actual tool result data."""
    from app.agent.nodes.response import response_node

    summary = training_result.get("summary", training_result)
    record_count = int(summary.get("record_count", 0))
    total_duration = int(summary.get("total_duration_minutes", 0))
    type_distribution = dict(summary.get("type_distribution", {}))
    types = sorted(type_distribution)
    question = "查询我最近训练情况"
    state = {
        "input": question,
        "user_id": "real-eval-user",
        "conversation_history": [{"role": "user", "content": question}],
        "answer_draft": None,
        "tool_calls": [
            {
                "call_id": "real-response-call",
                "tool_name": "get_training_summary",
                "arguments": {"start_date": START_DATE, "end_date": END_DATE},
            }
        ],
        "tool_results": [
            {
                "call_id": "real-response-call",
                "tool_name": "get_training_summary",
                "status": "success",
                "result": {"summary": summary},
            }
        ],
    }
    llm = build_real_llm_client()
    try:
        with _capture_provider_debug_output():
            generated = await response_node(state, llm_client=llm)
    finally:
        await _close_real_client(llm)
    response = str(generated.get("response", "")).strip()
    count_mentions = _matching_numbers(response, (record_count,))
    duration_mentions = _matching_numbers(response, (total_duration,))
    type_mentions = _matching_training_types(response, types)
    fabrication_markers = ("没有训练记录", "暂无训练记录", "没有任何训练")
    fabrication_detected = record_count > 0 and any(
        marker in response for marker in fabrication_markers
    )
    ignored = not count_mentions or not duration_mentions or not type_mentions
    non_empty = bool(response)
    groundedness_pass = (
        non_empty
        and bool(count_mentions)
        and bool(duration_mentions)
        and len(type_mentions) == len(types)
        and not fabrication_detected
        and not ignored
    )
    return RealResponseResult(
        model=model,
        question=question,
        response=response,
        real_record_count=record_count,
        real_total_duration_minutes=total_duration,
        real_types=types,
        has_record_count=bool(count_mentions),
        has_total_duration=bool(duration_mentions),
        has_type_information=len(type_mentions) == len(types),
        non_empty=non_empty,
        groundedness_pass=groundedness_pass,
        fabrication_detected=fabrication_detected,
        tool_result_ignored=ignored,
        record_count_mentions=count_mentions,
        duration_mentions=duration_mentions,
        type_mentions=type_mentions,
    )


def write_real_tool_selection_result(
    result: RealToolSelectionResult,
    path: Path = TOOL_SELECTION_RESULT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_real_response_quality_result(
    result: RealResponseQualityResult,
    path: Path = RESPONSE_QUALITY_RESULT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def real_response_quality_result(
    model: str,
    cases: list[RealResponseResult],
) -> RealResponseQualityResult:
    score = sum(case.groundedness_pass for case in cases) / len(cases) if cases else 0.0
    return RealResponseQualityResult(
        model=model,
        score=score,
        cases=cases,
        recorded_at=datetime.now(UTC).isoformat(),
    )


@contextmanager
def _capture_provider_debug_output():
    """Prevent provider debug output, including key previews, from reaching pytest."""
    with redirect_stdout(StringIO()):
        yield


async def _close_real_client(llm: Any) -> None:
    close = getattr(llm, "_client", None)
    close = getattr(close, "close", None)
    if close is not None:
        await close()


def _matching_training_types(response: str, training_types: list[str]) -> list[str]:
    aliases = {
        "Back": ("Back", "背部", "背"),
        "Chest": ("Chest", "胸部", "胸"),
        "Leg": ("Leg", "腿部", "下肢", "腿"),
    }
    return [
        training_type
        for training_type in training_types
        if any(alias in response for alias in aliases.get(training_type, (training_type,)))
    ]


def _has_parameter_error(definition: Any, arguments: dict[str, Any]) -> bool:
    if definition is None:
        return True
    try:
        definition.input_model.model_validate(arguments)
    except Exception:
        return True
    return False


def _matching_numbers(response: str, expected: tuple[int, ...]) -> list[str]:
    numbers: list[str] = []
    normalized = response.replace(",", "")
    for number in expected:
        if str(number) in normalized:
            numbers.append(str(number))
    return numbers