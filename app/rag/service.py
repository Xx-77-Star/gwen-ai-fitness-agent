"""Knowledge retrieval and prompt-safe context formatting for the Agent."""

from collections.abc import Mapping
from typing import Any

from app.rag.models import DocumentChunk
from app.rag.retriever import KnowledgeRetriever


class KnowledgeRetrievalService:
    """Coordinate embedding-based retrieval and format source-aware context."""

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    @property
    def top_k(self) -> int:
        return self._retriever.top_k

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[DocumentChunk]:
        return self._retriever.retrieve(query, top_k=top_k)

    def retrieve_context(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized chunks containing content, source, and score."""
        return [
            {
                "content": result.chunk.content,
                "source": result.chunk.metadata.get("source", "unknown"),
                "score": result.score,
            }
            for result in self._retriever.retrieve_results(query, top_k=top_k)
        ]

    @staticmethod
    def format_context(context: Any) -> str:
        """Render retrieved knowledge while preserving source attribution."""
        items = _normalize_context_items(context)
        if not items:
            return ""

        sections = ["Knowledge Context:"]
        for index, item in enumerate(items, start=1):
            content = str(item.get("content", "")).strip()
            source = str(item.get("source", "unknown")).strip() or "unknown"
            score = item.get("score")
            score_label = f" (score: {score:.4f})" if isinstance(score, (int, float)) else ""
            sections.append(f"[{index}] Source: {source}{score_label}\n{content}")
        return "\n\n".join(sections)


def _normalize_context_items(context: Any) -> list[dict[str, Any]]:
    if isinstance(context, Mapping):
        return [dict(context)]
    if isinstance(context, (list, tuple)):
        normalized = []
        for item in context:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
                continue
            if isinstance(item, DocumentChunk):
                normalized.append(
                    {
                        "content": item.content,
                        "source": item.metadata.get("source", "unknown"),
                        "score": 0.0,
                    }
                )
        return normalized
    return []