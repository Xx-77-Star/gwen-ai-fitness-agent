"""Offline Multi Tool Calling evaluation tests."""

from __future__ import annotations

from tests.evals.multi_tool_harness import (
    evaluate_multi_tool_response,
    evaluate_multi_tool_selection,
    load_multi_tool_response_dataset,
    load_multi_tool_selection_dataset,
)
from tests.multi_tool.fakes import multi_tool_registry


async def test_multi_tool_selection_eval_covers_ten_tasks() -> None:
    cases = load_multi_tool_selection_dataset()
    registry, _weather_client = multi_tool_registry()

    report = await evaluate_multi_tool_selection(cases, tool_registry=registry)

    assert report.total >= 10
    assert report.accuracy == 1.0
    assert all(len(case.expected_tools) >= 2 for case in cases)


async def test_multi_tool_response_quality_eval_references_all_sources() -> None:
    cases = load_multi_tool_response_dataset()

    report = await evaluate_multi_tool_response(cases)

    assert report.total >= 5
    assert report.score == 1.0
    assert all(prediction.passed for prediction in report.predictions)
