"""Response Quality Evaluation tests driven by a Fake LLM."""

from pathlib import Path

import pytest

from tests.evals import response_harness
from tests.evals.response_harness import (
    DIMENSION_WEIGHTS,
    QUALITY_DIMENSIONS,
    FakeResponseEvaluationLLM,
    evaluate_response_quality,
    load_response_quality_dataset,
)


@pytest.fixture(autouse=True)
def reset_response_quality_report() -> None:
    response_harness.reset_latest_response_report()
    yield


def test_response_quality_dataset_is_well_formed() -> None:
    cases = load_response_quality_dataset()

    assert len(cases) == 15
    assert len({case.id for case in cases}) == len(cases)
    assert len({case.question for case in cases}) == len(cases)
    assert {case.scenario for case in cases} == {
        "direct_draft",
        "direct_generated",
        "tool_result",
    }
    assert all(case.candidate_response.strip() for case in cases)
    assert sum(DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)


async def test_fake_llm_response_quality_evaluation() -> None:
    cases = load_response_quality_dataset()

    report = await evaluate_response_quality(cases)

    assert report.total == len(cases)
    assert report.score == 1.0
    assert report.passed_count == len(cases)
    assert report.failures == ()
    assert set(report.dimension_scores) == set(QUALITY_DIMENSIONS)
    assert all(score == 1.0 for score in report.dimension_scores.values())


async def test_fake_llm_is_used_without_real_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden_real_llm_call(*args, **kwargs):
        raise AssertionError("Response Evaluation must not call a real LLM API")

    monkeypatch.setattr(
        "app.llm.client.OpenAICompatibleLLMClient.complete",
        forbidden_real_llm_call,
    )
    cases = load_response_quality_dataset()

    report = await evaluate_response_quality(cases)

    assert report.score == 1.0
    assert isinstance(FakeResponseEvaluationLLM, type)


def test_full_dataset_score_report(tmp_path: Path) -> None:
    cases = load_response_quality_dataset()
    report_path = tmp_path / "response-quality-score.txt"

    report = response_harness.write_response_quality_report(report_path, cases)

    assert report.score == 1.0
    assert report_path.read_text(encoding="utf-8").startswith(
        "Response Quality Score: 100.00% (15/15 pass)"
    )