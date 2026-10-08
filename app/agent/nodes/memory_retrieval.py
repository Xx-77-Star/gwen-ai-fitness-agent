"""Memory retrieval node for normalized short- and long-term context."""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.memory.conversation_repository import ConversationRepository
from app.memory.interface import MemoryServiceInterface


def memory_retrieve_node(
    state: AgentState,
    *,
    memory_service: MemoryServiceInterface,
    conversation_repository: ConversationRepository | None = None,
    conversation_window: int = 20,
) -> dict[str, Any]:
    """Load persisted recent conversation context and long-term user memory."""
    user_id = state.get("user_id", "").strip()
    conversation_id = state.get("conversation_id", "").strip()
    persisted_turns: list[dict[str, Any]] = []
    if conversation_repository is not None and conversation_id and user_id:
        try:
            persisted_turns = [
                {"role": turn.role, "content": turn.content}
                for turn in conversation_repository.list_recent_turns(
                    conversation_id,
                    user_id,
                    limit=conversation_window,
                )
            ]
        except Exception:
            persisted_turns = []

    short_term = persisted_turns or state.get("messages") or state.get("conversation_history", [])
    context = memory_service.retrieve(user_id, short_term=short_term)
    return {
        "memory_context": context,
        "memory_loaded": True,
        "memory_turns": [turn.model_dump() for turn in context.short_term],
        "user_memories": [memory.model_dump() for memory in context.long_term],
    }
