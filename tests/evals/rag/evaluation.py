"""Deterministic metrics and report for the local RAG evaluation dataset."""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.retriever import KnowledgeRetriever


@dataclass(frozen=True)
class RAGEvalCase:
    id: str
    query: str
    category: str
    expected_sources: tuple[str, ...]
    grounded_response: str
    hallucinated_response: str


@dataclass(frozen=True)
class RAGRetrievalPrediction:
    case: RAGEvalCase
    retrieved_sources: tuple[str, ...]
    expected_relevant: int
    retrieved_relevant: int
    context_relevance: float

    @property
    def retrieval_accuracy(self) -> float:
        if not self.case.expected_sources:
            return 1.0
        return self.retrieved_relevant / len(self.case.expected_sources)

    @property
    def top_k_recall(self) -> float:
        if not self.case.expected_sources:
            return 1.0
        return min(1.0, self.retrieved_relevant / len(self.case.expected_sources))


@dataclass(frozen=True)
class RAGResponsePrediction:
    case: RAGEvalCase
    grounded_response: str
    hallucinated_response: str
    grounded_score: float
    hallucinated_score: float


@dataclass(frozen=True)
class RAGEvaluationReport:
    retrieval_predictions: tuple[RAGRetrievalPrediction, ...]
    response_predictions: tuple[RAGResponsePrediction, ...]

    @property
    def retrieval_accuracy(self) -> float:
        return (
            sum(prediction.retrieval_accuracy for prediction in self.retrieval_predictions)
            / len(self.retrieval_predictions)
        )

    @property
    def top_k_recall(self) -> float:
        return (
            sum(prediction.top_k_recall for prediction in self.retrieval_predictions)
            / len(self.retrieval_predictions)
        )

    @property
    def context_relevance(self) -> float:
        return (
            sum(prediction.context_relevance for prediction in self.retrieval_predictions)
            / len(self.retrieval_predictions)
        )

    @property
    def response_groundedness(self) -> float:
        return (
            sum(prediction.grounded_score for prediction in self.response_predictions)
            / len(self.response_predictions)
        )

    def summary_lines(self) -> tuple[str, ...]:
        return (
            f"RAG Retrieval Accuracy: {self.retrieval_accuracy:.2%}",
            f"RAG Top-K Recall: {self.top_k_recall:.2%}",
            f"RAG Context Relevance: {self.context_relevance:.2%}",
            f"RAG Response Groundedness: {self.response_groundedness:.2%}",
        )


def evaluate_rag(
    cases: list[RAGEvalCase],
    *,
    retriever: KnowledgeRetriever,
    top_k: int = 2,
) -> RAGEvaluationReport:
    retrieval_predictions = []
    response_predictions = []
    for case in cases:
        results = retriever.retrieve_results(case.query, top_k=top_k)
        retrieved_sources = tuple(result.chunk.metadata.get("source", "") for result in results)
        retrieved_relevant = sum(
            1 for source in retrieved_sources if source in case.expected_sources
        )
        scores = [result.score for result in results]
        relevance = sum(scores) / len(scores) if scores else 0.0
        retrieval_predictions.append(
            RAGRetrievalPrediction(
                case=case,
                retrieved_sources=retrieved_sources,
                expected_relevant=retrieved_relevant,
                retrieved_relevant=retrieved_relevant,
                context_relevance=relevance,
            )
        )
        response_predictions.append(
            RAGResponsePrediction(
                case=case,
                grounded_response=case.grounded_response,
                hallucinated_response=case.hallucinated_response,
                grounded_score=_groundedness_score(case, case.grounded_response),
                hallucinated_score=_groundedness_score(case, case.hallucinated_response),
            )
        )
    return RAGEvaluationReport(tuple(retrieval_predictions), tuple(response_predictions))


def _groundedness_score(case: RAGEvalCase, response: str) -> float:
    expected_terms = (case.category, *case.expected_sources)
    return sum(term in response for term in expected_terms) / len(expected_terms)