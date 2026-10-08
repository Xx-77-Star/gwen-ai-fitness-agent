"""Real qwen-plus response groundedness evaluation. Skipped by default."""

import pytest

from app.config.settings import Settings
from app.tools.executor import ToolExecutor
from app.tools.schemas import ToolCall
from app.tools.training_tools import build_training_tool_registry
from tests.evals.real_llm.harness import (
    END_DATE,
    START_DATE,
    real_response_quality_result,
    run_real_response_groundedness,
    write_real_response_quality_result,
)

pytestmark = pytest.mark.real_llm


async def test_real_qwen_plus_response_groundedness(demo_training_repository) -> None:
    settings = Settings()
    registry = build_training_tool_registry()
    executor = ToolExecutor(registry)
    tool_result = executor.execute(
        ToolCall(
            call_id="real-response-summary",
            name="get_training_summary",
            arguments={"start_date": START_DATE, "end_date": END_DATE},
        ),
        user_id="demo_user",
        repository=demo_training_repository,
    )
    assert tool_result.status == "success"
    assert tool_result.data is not None

    response = await run_real_response_groundedness(
        model=settings.llm_model,
        tool_registry=registry,
        training_result=tool_result.data,
    )
    report = real_response_quality_result(settings.llm_model, [response])
    write_real_response_quality_result(report)

    assert response.non_empty
    assert response.has_record_count
    assert response.has_total_duration
    assert response.has_type_information
    assert not response.fabrication_detected
    assert not response.tool_result_ignored
    assert report.score == 1.0