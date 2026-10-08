"""Prompt-safe formatting for Memory context already loaded into Agent State."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_memory_prompt(state: Mapping[str, Any]) -> str:
    """Return stable long-term facts and normalized short-term turns for prompts.

    ``memory_context`` is the canonical retrieved context. ``user_memories`` is
    also consulted so persisted memory snapshot data is not silently dropped.
    Duplicate facts are displayed once while preserving the first retrieved
    value for each memory key.
    """
    context = _mapping(state.get("memory_context"))
    long_term = _memory_facts(
        _sequence(context.get("long_term")),
        _sequence(state.get("user_memories")),
    )
    short_term = _memory_turns(_sequence(context.get("short_term")))
    sections = [
        "用户长期记忆（稳定信息，可用于判断和个性化回答）：",
        *(f"- {key}: {value}" for key, value in long_term),
    ]
    if short_term:
        sections.extend(
            [
                "用户短期记忆（本轮对话上下文）：",
                *(f"- {role}: {content}" for role, content in short_term),
            ]
        )
    return "\n".join(sections)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _memory_facts(*sources: list[Any]) -> list[tuple[str, str]]:
    facts: dict[str, str] = {}
    for source in sources:
        for item in source:
            record = _mapping(item)
            key = str(record.get("memory_key", "")).strip()
            value = str(record.get("memory_value", "")).strip()
            if key and value and key not in facts:
                facts[key] = value
    return list(facts.items())


def _memory_turns(turns: list[Any]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for turn in turns:
        record = _mapping(turn)
        role = str(record.get("role", "")).strip()
        content = str(record.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append((role, content))
    return normalized
