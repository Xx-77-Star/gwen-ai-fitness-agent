from collections.abc import Sequence

from app.rag.models import DocumentChunk, VectorSearchResult


class FakeEmbeddingProvider:
    """Deterministic embeddings for tests without provider or network access."""

    def __init__(self) -> None:
        self.text_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        self.text_calls.append(text)
        return self._vector(text)

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        inputs = list(documents)
        self.document_calls.append(inputs)
        return [self._vector(document) for document in inputs]

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [
            float(text.count("训练")),
            float(text.count("蛋白质")),
            float(text.count("睡眠")),
            float(len(text) % 7),
        ]


class FakeVectorStore:
    """Scriptable vector store used to verify the retrieval boundary."""

    def __init__(self, results: Sequence[VectorSearchResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[Sequence[float], int]] = []

    def similarity_search(
        self,
        query: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        self.calls.append((query, limit))
        return self.results[:limit]


def search_result(
    content: str,
    *,
    source: str,
    score: float,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk=DocumentChunk(
            id=source,
            content=content,
            metadata={"source": source, "category": "training"},
        ),
        score=score,
    )