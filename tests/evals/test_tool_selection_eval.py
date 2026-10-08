"""Tool Selection Evaluation tests driven entirely by a Fake LLM."""

from pathlib import Path

import pytest

from app.tools.recommendation_tools import register_recommendation_tool
from app.tools.training_tools import build_training_tool_registry
from app.tools.weather_tools import register_weather_tool
from tests.evals import harness
from tests.evals.harness import (
    VALID_TOOLS,
    FakeToolSelectionLLM,
    evaluate_tool_selection,
    load_tool_selection_dataset,
)
from tests.weather.fakes import FakeWeatherClient


@pytest.fixture(autouse=True)
def reset_tool_selection_report() -> None:
    harness.reset_latest_report()
    yield


def build_eval_registry():
    registry = build_training_tool_registry()
    register_weather_tool(registry, FakeWeatherClient())
    register_recommendation_tool(registry)
    return registry


def test_tool_selection_dataset_is_well_formed() -> None:
    cases = load_tool_selection_dataset()

    assert cases
    assert len({case.id for case in cases}) == len(cases)
    assert len({case.question for case in cases}) == len(cases)
    assert {case.expected_tool for case in cases} == set(VALID_TOOLS)
    assert all(case.question.strip() for case in cases)
    assert all(isinstance(case.arguments, dict) for case in cases)


async def test_fake_llm_tool_selection_evaluation() -> None:
    cases = load_tool_selection_dataset()
    registry = build_eval_registry()

    report = await evaluate_tool_selection(cases, tool_registry=registry)

    assert isinstance(report.predictions, tuple)
    assert report.total == len(cases)
    assert report.accuracy == 1.0
    assert report.failures == ()
    assert set(report.per_tool_accuracy) == set(VALID_TOOLS)


async def test_fake_llm_is_used_without_real_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden_real_llm_call(*args, **kwargs):
        raise AssertionError("Agent Evaluation must not call a real LLM API")

    monkeypatch.setattr(
        "app.llm.client.OpenAICompatibleLLMClient.complete",
        forbidden_real_llm_call,
    )
    cases = load_tool_selection_dataset()[:1]

    report = await evaluate_tool_selection(
        cases,
        tool_registry=build_eval_registry(),
    )

    assert report.accuracy == 1.0
    assert isinstance(FakeToolSelectionLLM, type)


def test_full_dataset_accuracy_report(tmp_path: Path) -> None:
    cases = load_tool_selection_dataset()
    report_path = tmp_path / "tool-selection-accuracy.txt"

    harness.write_accuracy_report(
        report_path,
        cases,
        tool_registry=build_eval_registry(),
    )

    assert report_path.read_text(encoding="utf-8").startswith(
        "Tool Selection Accuracy: 100.00% (23/23)"
    )