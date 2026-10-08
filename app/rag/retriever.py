"""Query embedding and Top-K retrieval over a local knowledge vector store."""

from collections.abc import Sequence
from typing import Protocol

from app.rag.embedding import EmbeddingProvider
from app.rag.models import DocumentChunk, VectorSearchResult


class SimilaritySearchStore(Protocol):
    """Minimal vector store contract used by knowledge retrieval."""

    def similarity_search(
        self,
        query: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[VectorSearchResult]:
        """Return ranked chunks for an embedding vector."""
        ...


class KnowledgeRetriever:
    """Embed user queries and retrieve the most relevant knowledge chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: SimilaritySearchStore,
        *,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[DocumentChunk]:
        """Return Top-K chunks without their similarity scores."""
        return [result.chunk for result in self.retrieve_results(query, top_k=top_k)]

    def retrieve_results(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Embed a query and return Top-K scored chunks in relevance order."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty")

        limit = self._top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self._embedding_provider.embed_text(normalized_query)
        return self._vector_store.similarity_search(query_embedding, limit=limit)