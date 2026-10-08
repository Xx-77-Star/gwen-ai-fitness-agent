"""Real qwen-plus Multi Tool Calling evaluation. Skipped by default."""

import pytest

from app.config.settings import Settings
from tests.evals.real_llm.multi_tool_harness import (
    MULTI_TOOL_RESPONSE_RESULT_PATH,
    MULTI_TOOL_SELECTION_RESULT_PATH,
    build_real_multi_tool_registry,
    real_multi_tool_response_report,
    run_real_multi_tool_response,
    run_real_multi_tool_selection,
    write_real_multi_tool_response_result,
    write_real_multi_tool_selection_result,
)

pytestmark = pytest.mark.real_llm


async def test_real_qwen_plus_multi_tool_selection() -> None:
    settings = Settings()
    registry = build_real_multi_tool_registry()

    result = await run_real_multi_tool_selection(
        model=settings.llm_model,
        tool_registry=registry,
    )
    write_real_multi_tool_selection_result(result)

    payload = MULTI_TOOL_SELECTION_RESULT_PATH.read_text(encoding="utf-8")
    assert "api_key" not in payload.lower()
    assert result.model == settings.llm_model
    assert result.accuracy == 1.0
    assert result.multi_tool_calls_stable
    assert result.tool_misselection_count == 0
    assert result.parameter_error_count == 0
    assert result.duplicate_call_count == 0


async def test_real_qwen_plus_multi_tool_response_grounding() -> None:
    settings = Settings()
    tool_results = [
        {
            "call_id": "training-summary",
            "tool_name": "get_training_summary",
            "status": "success",
            "result": {
                "summary": {
                    "record_count": 3,
                    "total_duration_minutes": 180,
                    "type_distribution": {"Chest": 1, "Leg": 1, "Back": 1},
                }
            },
        },
        {
            "call_id": "weather",
            "tool_name": "get_current_weather",
            "status": "success",
            "result": {"temperature_celsius": 12, "weather_condition": "rainy"},
        },
        {
            "call_id": "recommendation",
            "tool_name": "get_training_recommendation",
            "status": "success",
            "result": {
                "recommendation_type": "indoor_strength",
                "duration_minutes": 45,
                "workout": "室内全身力量训练",
            },
        },
    ]

    response = await run_real_multi_tool_response(
        model=settings.llm_model,
        tool_results=tool_results,
    )
    report = real_multi_tool_response_report(settings.llm_model, [response])
    write_real_multi_tool_response_result(report)

    assert response.non_empty
    assert response.has_training_data
    assert response.has_weather_data
    assert response.has_recommendation
    assert response.multiple_tool_results_referenced
    assert response.groundedness_pass
    assert report.score == 1.0
    assert "api_key" not in MULTI_TOOL_RESPONSE_RESULT_PATH.read_text(encoding="utf-8").lower()
