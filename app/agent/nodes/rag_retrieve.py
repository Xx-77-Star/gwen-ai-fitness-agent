"""Knowledge retrieval node between Memory retrieval and Intent classification."""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.rag.service import KnowledgeRetrievalService


def rag_retrieve_node(
    state: AgentState,
    *,
    knowledge_service: KnowledgeRetrievalService,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Retrieve source-aware knowledge context for the current user query."""
    query = state.get("input", "").strip()
    if not query:
        return {"rag_context": []}

    try:
        context = knowledge_service.retrieve_context(query, top_k=top_k)
    except Exception:
        return {
            "rag_context": [],
            "metadata": {
                **state.get("metadata", {}),
                "rag_retrieval_error": "Knowledge retrieval failed",
            },
        }
    return {"rag_context": context}