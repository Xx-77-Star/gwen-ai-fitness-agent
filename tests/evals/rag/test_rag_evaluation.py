"""Offline RAG evaluation tests with Fake Embedding and Fake VectorStore."""

import pytest

from app.rag.retriever import KnowledgeRetriever
from app.rag.service import KnowledgeRetrievalService
from tests.evals.rag.evaluation import RAGEvalCase, evaluate_rag
from tests.evals.rag.fakes import FakeRAGEmbeddingProvider, FakeRAGVectorStore, knowledge_chunk

CASES = (
    RAGEvalCase(
        id="muscle-gain",
        query="增肌训练应该如何渐进增加重量？",
        category="增肌",
        expected_sources=("training/muscle_gain.md",),
        grounded_response="增肌应逐步增加训练刺激。来源：training/muscle_gain.md",
        hallucinated_response="每天必须完成一百组训练，来源：training/secret.md",
    ),
    RAGEvalCase(
        id="fat-loss",
        query="减脂训练如何保留肌肉？",
        category="减脂",
        expected_sources=("training/fat_loss.md",),
        grounded_response="减脂期间保留力量训练有助于保留肌肉。来源：training/fat_loss.md",
        hallucinated_response="完全停止力量训练才能减脂，来源：nutrition/secret.md",
    ),
    RAGEvalCase(
        id="protein",
        query="蛋白质摄入应该怎样分配？",
        category="蛋白质",
        expected_sources=("nutrition/protein.md",),
        grounded_response="蛋白质应分配到多餐。来源：nutrition/protein.md",
        hallucinated_response="蛋白质每天只吃一顿即可，来源：training/secret.md",
    ),
    RAGEvalCase(
        id="sleep",
        query="睡眠如何帮助训练恢复？",
        category="睡眠",
        expected_sources=("recovery/sleep.md",),
        grounded_response="睡眠支持训练恢复。来源：recovery/sleep.md",
        hallucinated_response="睡眠完全不影响恢复，来源：nutrition/secret.md",
    ),
)


def knowledge_chunks():
    return [
        knowledge_chunk("增肌训练渐进增加重量和次数。", "training/muscle_gain.md", "增肌"),
        knowledge_chunk("减脂训练保留力量和肌肉。", "training/fat_loss.md", "减脂"),
        knowledge_chunk("蛋白质摄入分配到多餐。", "nutrition/protein.md", "蛋白质"),
        knowledge_chunk("睡眠支持训练恢复。", "recovery/sleep.md", "睡眠"),
    ]


def retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever(
        FakeRAGEmbeddingProvider(),
        FakeRAGVectorStore(knowledge_chunks()),
        top_k=2,
    )


def test_rag_dataset_covers_training_nutrition_and_recovery() -> None:
    assert len(CASES) == 4
    assert {case.category for case in CASES} == {"增肌", "减脂", "蛋白质", "睡眠"}
    assert all(
        case.category in case.grounded_response
        for case in CASES
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_each_question_retrieves_its_expected_knowledge(case: RAGEvalCase) -> None:
    report = evaluate_rag([case], retriever=retriever())

    assert report.retrieval_accuracy == 1.0
    assert report.top_k_recall == 1.0
    assert case.expected_sources[0] in report.retrieval_predictions[0].retrieved_sources
    assert report.context_relevance > 0.0


def test_full_rag_metrics_are_scored_with_fake_components() -> None:
    report = evaluate_rag(list(CASES), retriever=retriever())

    assert report.retrieval_accuracy == 1.0
    assert report.top_k_recall == 1.0
    assert report.context_relevance > 0.0
    assert report.response_groundedness == 1.0
    assert all(
        prediction.grounded_score > prediction.hallucinated_score
        for prediction in report.response_predictions
    )


def test_response_groundedness_distinguishes_source_aware_answers() -> None:
    report = evaluate_rag(list(CASES), retriever=retriever())

    for prediction in report.response_predictions:
        assert prediction.grounded_score == 1.0
        assert prediction.hallucinated_score < 1.0


def test_context_formatting_keeps_source_and_score() -> None:
    results = retriever().retrieve_results("增肌训练", top_k=1)
    context = [
        {
            "content": result.chunk.content,
            "source": result.chunk.metadata["source"],
            "score": result.score,
        }
        for result in results
    ]

    formatted = KnowledgeRetrievalService.format_context(context)

    assert "Knowledge Context:" in formatted
    assert "Source: training/muscle_gain.md" in formatted
    assert "score:" in formatted


def test_rag_eval_uses_no_real_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_provider(*args: object, **kwargs: object):
        raise AssertionError("RAG Evaluation must not call a real embedding API")

    monkeypatch.setattr(
        "app.rag.embedding.BailianEmbeddingProvider.embed_text",
        forbidden_provider,
    )
    monkeypatch.setattr(
        "app.rag.embedding.BailianEmbeddingProvider.embed_documents",
        forbidden_provider,
    )

    report = evaluate_rag(list(CASES), retriever=retriever())

    assert report.retrieval_accuracy == 1.0
    assert isinstance(FakeRAGEmbeddingProvider, type)
    assert isinstance(FakeRAGVectorStore, type)