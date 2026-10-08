"""Real qwen-plus Multi Tool Calling evaluation with sanitized logs."""

from __future__ import annotations

import json
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from app.agent.nodes.response import response_node
from app.config.settings import Settings
from app.llm.client import OpenAICompatibleLLMClient
from app.llm.types import LLMResponse
from app.tools.registry import ToolRegistry

RESULTS_DIR = Path(__file__).parent.parent / "results"
MULTI_TOOL_SELECTION_RESULT_PATH = RESULTS_DIR / "real_multi_tool_selection_result.json"
MULTI_TOOL_RESPONSE_RESULT_PATH = RESULTS_DIR / "real_multi_tool_response_result.json"
QUESTION = "查看我最近训练情况，并结合天气给我今天训练建议"
EXPECTED_TOOLS = (
    "get_training_summary",
    "get_current_weather",
    "get_training_recommendation",
)


@dataclass
class RealMultiToolCaseResult:
    question: str
    expected_tools: list[str]
    selected_tools: list[str] = field(default_factory=list)
    returned_tool_calls: int = 0
    arguments: list[dict[str, Any]] = field(default_factory=list)
    parameter_error: bool = False
    duplicate_tool_calls: bool = False
    correct: bool = False
    error: str | None = None


@dataclass
class RealMultiToolSelectionResult:
    model: str
    accuracy: float
    cases: list[RealMultiToolCaseResult]
    multi_tool_calls_stable: bool
    tool_misselection_count: int
    parameter_error_count: int
    duplicate_call_count: int
    recorded_at: str


@dataclass
class RealMultiToolResponseResult:
    model: str
    question: str
    response: str
    has_training_data: bool
    has_weather_data: bool
    has_recommendation: bool
    multiple_tool_results_referenced: bool
    non_empty: bool
    groundedness_pass: bool
    error: str | None = None


@dataclass
class RealMultiToolResponseReport:
    model: str
    score: float
    cases: list[RealMultiToolResponseResult]
    recorded_at: str


def build_real_multi_tool_registry() -> ToolRegistry:
    from app.tools.recommendation_tools import register_recommendation_tool
    from app.tools.training_tools import build_training_tool_registry
    from app.tools.weather_tools import OpenMeteoWeatherClient, register_weather_tool

    registry = build_training_tool_registry()
    register_weather_tool(registry, OpenMeteoWeatherClient())
    register_recommendation_tool(registry)
    return registry


async def run_real_multi_tool_selection(
    *,
    model: str,
    tool_registry: ToolRegistry,
) -> RealMultiToolSelectionResult:
    llm = OpenAICompatibleLLMClient(Settings())
    definitions = {name: tool_registry.get(name) for name in tool_registry.names()}
    try:
        with _capture_provider_debug_output():
            response = await llm.complete(
                system_prompt=(
                    "你是 FitLife AI 的多工具规划评估助手。"
                    "根据任务一次性返回所有必要的工具调用，并保持合理执行顺序。"
                ),
                messages=[{"role": "user", "content": QUESTION}],
                tools=tool_registry.schema_payloads(),
                tool_choice="auto",
            )
        if not isinstance(response, LLMResponse):
            raise TypeError("real LLM returned a non-tool response")
        selected_tools = [call.name for call in response.tool_calls]
        arguments = [dict(call.arguments) for call in response.tool_calls]
        parameter_error = any(
            _has_parameter_error(definitions.get(call.name), call.arguments)
            for call in response.tool_calls
        )
        duplicate = len(selected_tools) != len(set(selected_tools))
        correct = selected_tools == list(EXPECTED_TOOLS) and not parameter_error and not duplicate
        case = RealMultiToolCaseResult(
            question=QUESTION,
            expected_tools=list(EXPECTED_TOOLS),
            selected_tools=selected_tools,
            returned_tool_calls=len(response.tool_calls),
            arguments=arguments,
            parameter_error=parameter_error,
            duplicate_tool_calls=duplicate,
            correct=correct,
        )
    except Exception as exc:
        case = RealMultiToolCaseResult(
            question=QUESTION,
            expected_tools=list(EXPECTED_TOOLS),
            error=type(exc).__name__,
        )
    await _close_real_client(llm)
    return RealMultiToolSelectionResult(
        model=model,
        accuracy=float(case.correct),
        cases=[case],
        multi_tool_calls_stable=case.returned_tool_calls == len(EXPECTED_TOOLS),
        tool_misselection_count=int(case.selected_tools != list(EXPECTED_TOOLS)),
        parameter_error_count=int(case.parameter_error),
        duplicate_call_count=int(case.duplicate_tool_calls),
        recorded_at=datetime.now(UTC).isoformat(),
    )


async def run_real_multi_tool_response(
    *,
    model: str,
    tool_results: list[dict[str, Any]],
) -> RealMultiToolResponseResult:
    llm = OpenAICompatibleLLMClient(Settings())
    try:
        with _capture_provider_debug_output():
            generated = await response_node(
                {
                    "input": QUESTION,
                    "conversation_history": [{"role": "user", "content": QUESTION}],
                    "answer_draft": None,
                    "tool_calls": [
                        {
                            "call_id": item["call_id"],
                            "tool_name": item["tool_name"],
                            "arguments": {},
                        }
                        for item in tool_results
                    ],
                    "tool_results": tool_results,
                    "metadata": {},
                },
                llm_client=llm,
            )
        response = str(generated.get("response", "")).strip()
        has_training = _contains_any(response, ("训练", "力量", "跑步", "分钟", "次"))
        has_weather = _contains_any(response, ("天气", "气温", "摄氏", "雨", "晴", "风"))
        has_recommendation = _contains_any(response, ("建议", "推荐", "安排", "训练计划"))
        referenced = has_training and has_weather and has_recommendation
        return RealMultiToolResponseResult(
            model=model,
            question=QUESTION,
            response=response,
            has_training_data=has_training,
            has_weather_data=has_weather,
            has_recommendation=has_recommendation,
            multiple_tool_results_referenced=referenced,
            non_empty=bool(response),
            groundedness_pass=bool(response) and referenced,
        )
    except Exception as exc:
        return RealMultiToolResponseResult(
            model=model,
            question=QUESTION,
            response="",
            has_training_data=False,
            has_weather_data=False,
            has_recommendation=False,
            multiple_tool_results_referenced=False,
            non_empty=False,
            groundedness_pass=False,
            error=type(exc).__name__,
        )
    finally:
        await _close_real_client(llm)


def write_real_multi_tool_selection_result(
    result: RealMultiToolSelectionResult,
    path: Path = MULTI_TOOL_SELECTION_RESULT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_real_multi_tool_response_result(
    result: RealMultiToolResponseReport,
    path: Path = MULTI_TOOL_RESPONSE_RESULT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def real_multi_tool_response_report(
    model: str, cases: list[RealMultiToolResponseResult]
) -> RealMultiToolResponseReport:
    score = sum(case.groundedness_pass for case in cases) / len(cases) if cases else 0.0
    return RealMultiToolResponseReport(
        model=model,
        score=score,
        cases=cases,
        recorded_at=datetime.now(UTC).isoformat(),
    )


def _has_parameter_error(definition: Any, arguments: dict[str, Any]) -> bool:
    if definition is None:
        return True
    try:
        definition.input_model.model_validate(arguments)
    except Exception:
        return True
    return False


def _contains_any(response: str, terms: tuple[str, ...]) -> bool:
    return any(term in response for term in terms)


@contextmanager
def _capture_provider_debug_output():
    with redirect_stdout(StringIO()):
        yield


async def _close_real_client(llm: Any) -> None:
    close = getattr(llm, "_client", None)
    close = getattr(close, "close", None)
    if close is not None:
        await close()
