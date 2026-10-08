from collections.abc import Sequence

from app.rag.models import DocumentChunk, VectorSearchResult


class FakeRAGEmbeddingProvider:
    """Category-aware deterministic embeddings with no provider or network access."""

    def embed_text(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
        return [self._vector(document) for document in documents]

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("增肌" in normalized or "muscle" in normalized),
            float("减脂" in normalized or "fat" in normalized),
            float("蛋白质" in normalized or "protein" in normalized),
            float("睡眠" in normalized or "sleep" in normalized),
        ]


class FakeRAGVectorStore:
    """Deterministic vector store ranked by category-aware vector similarity."""

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.calls: list[tuple[Sequence[float], int]] = []

    def similarity_search(
        self,
        query: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        self.calls.append((query, limit))
        scored = []
        for chunk in self.chunks:
            vector = FakeRAGEmbeddingProvider._vector(chunk.content)
            score = sum(left * right for left, right in zip(query, vector, strict=True))
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            VectorSearchResult(chunk=chunk.model_copy(deep=True), score=float(score))
            for score, chunk in scored[:limit]
            if score > 0
        ]


def knowledge_chunk(content: str, source: str, category: str) -> DocumentChunk:
    return DocumentChunk(
        id=source,
        content=content,
        metadata={"source": source, "category": category},
    )