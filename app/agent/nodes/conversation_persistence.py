"""Post-response persistence of cross-request conversation turns."""

from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState
from app.memory.conversation_repository import ConversationRepository

_LOGGER = logging.getLogger("fitlife.conversation.persistence")


def conversation_persist_node(
    state: AgentState,
    *,
    conversation_repository: ConversationRepository,
) -> dict[str, Any]:
    """Save the current user and assistant turns without changing the response."""
    user_id = str(state.get("user_id", "")).strip()
    conversation_id = str(state.get("conversation_id", "")).strip()
    input_text = str(state.get("input", "")).strip()
    response_text = str(state.get("response", "")).strip()
    metadata = dict(state.get("metadata", {}))
    if not user_id or not conversation_id or not input_text or not response_text:
        metadata["conversation_persistence"] = {
            "status": "skipped",
            "saved_turns": 0,
        }
        return {"metadata": metadata}

    try:
        saved_turns = 0
        conversation_repository.append_turn(
            conversation_id, user_id, "user", input_text
        )
        saved_turns += 1
        conversation_repository.append_turn(
            conversation_id, user_id, "assistant", response_text
        )
        saved_turns += 1
        metadata["conversation_persistence"] = {
            "status": "success",
            "saved_turns": saved_turns,
            "conversation_id": conversation_id,
        }
    except Exception as exc:
        _LOGGER.warning(
            "conversation_persistence_failed",
            extra={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "error_type": type(exc).__name__,
            },
        )
        metadata["conversation_persistence"] = {
            "status": "error",
            "saved_turns": 0,
            "error": {
                "type": type(exc).__name__,
                "message": "Conversation persistence failed",
            },
        }
    return {"metadata": metadata}
