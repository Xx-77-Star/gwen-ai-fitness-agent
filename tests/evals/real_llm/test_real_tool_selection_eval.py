"""Real qwen-plus Tool Selection Evaluation. Skipped by default."""

import pytest

from app.config.settings import Settings
from app.tools.recommendation_tools import register_recommendation_tool
from app.tools.training_tools import build_training_tool_registry
from tests.evals.real_llm.harness import (
    TOOL_SELECTION_CASES,
    TOOL_SELECTION_RESULT_PATH,
    run_real_tool_selection,
    write_real_tool_selection_result,
)

pytestmark = pytest.mark.real_llm


async def test_real_qwen_plus_tool_selection_and_stability() -> None:
    settings = Settings()
    registry = build_training_tool_registry()
    register_recommendation_tool(registry)

    result = await run_real_tool_selection(model=settings.llm_model, tool_registry=registry)

    write_real_tool_selection_result(result)
    payload = TOOL_SELECTION_RESULT_PATH.read_text(encoding="utf-8")
    assert "api_key" not in payload.lower()
    assert result.model == settings.llm_model
    assert len(result.cases) == len(TOOL_SELECTION_CASES)
    assert result.accuracy == 1.0
    assert result.tool_calls_stable
    assert result.tool_misselection_count == 0
    assert result.parameter_error_count == 0
    assert result.duplicate_call_count == 0