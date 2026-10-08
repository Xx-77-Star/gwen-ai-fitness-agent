"""Post-response persistence of stable long-term user facts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.agent.state import AgentState
from app.memory.extractor import MemoryExtractionResult, RuleBasedMemoryExtractor
from app.memory.interface import MemoryServiceInterface

_LOGGER = logging.getLogger("fitlife.memory.update")
_ALLOWED_KEYS = {"fitness_goal", "training_preference", "experience_level"}


def memory_update_node(
    state: AgentState,
    *,
    memory_service: MemoryServiceInterface,
    extractor: RuleBasedMemoryExtractor | None = None,
) -> dict[str, Any]:
    """Persist stable facts after the final response without changing its outcome."""
    user_id = str(state.get("user_id", "")).strip()
    conversation = _completed_conversation(state)
    extraction = MemoryExtractionResult()
    update_metadata = {
        "memory_update": {
            "attempted": False,
            "written_keys": [],
            "status": "skipped",
        }
    }
    if user_id and conversation:
        extraction = (extractor or RuleBasedMemoryExtractor()).extract(
            conversation,
            tool_results=state.get("tool_results", []),
        )

    validated = _validated_memories(extraction)
    if validated:
        update_metadata["memory_update"]["attempted"] = True
        try:
            written_keys = []
            for memory in validated:
                memory_service.remember(user_id, memory["key"], memory["value"])
                written_keys.append(memory["key"])
            update_metadata["memory_update"].update(
                {"written_keys": written_keys, "status": "success"}
            )
        except Exception as exc:
            _LOGGER.warning(
                "memory_update_failed",
                extra={"user_id": user_id, "error_type": type(exc).__name__},
            )
            update_metadata["memory_update"].update(
                {
                    "written_keys": [],
                    "status": "error",
                    "error": {
                        "type": type(exc).__name__,
                        "message": "Memory update failed",
                    },
                }
            )

    return {
        "memory_extractions": [memory.model_dump() for memory in extraction.memories],
        "metadata": {**dict(state.get("metadata", {})), **update_metadata},
    }


def _completed_conversation(state: AgentState) -> list[dict[str, Any]]:
    messages = list(state.get("messages", []) or state.get("conversation_history", []))
    conversation = [
        {"role": item.get("role"), "content": item.get("content")}
        for item in messages
        if isinstance(item, Mapping)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    input_text = state.get("input")
    if isinstance(input_text, str) and input_text.strip():
        normalized_input = input_text.strip()
        if not any(
            item["role"] == "user" and item["content"] == normalized_input for item in conversation
        ):
            conversation.insert(0, {"role": "user", "content": normalized_input})
    response_text = state.get("response")
    if isinstance(response_text, str) and response_text.strip():
        normalized_response = response_text.strip()
        if not any(
            item["role"] == "assistant" and item["content"] == normalized_response
            for item in conversation
        ):
            conversation.append({"role": "assistant", "content": normalized_response})
    return conversation


def _validated_memories(result: MemoryExtractionResult) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    seen: set[str] = set()
    for memory in result.memories:
        key = memory.key
        value = memory.value.strip()
        if key in _ALLOWED_KEYS and key not in seen and value:
            validated.append({"key": key, "value": value})
            seen.add(key)
    return validated
