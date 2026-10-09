"""Structured extraction of durable fitness facts from one completed conversation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

StableMemoryKey = Literal[
    "fitness_goal",
    "training_preference",
    "experience_level",
    "weather_location",
]


class ExtractedMemory(BaseModel):
    """One stable user fact allowed in long-term Memory."""

    model_config = ConfigDict(extra="forbid")

    key: StableMemoryKey
    value: str = Field(min_length=1, max_length=4000)


class MemoryExtractionResult(BaseModel):
    """Structured extractor output consumed by the Memory Update Node."""

    model_config = ConfigDict(extra="forbid")

    memories: list[ExtractedMemory] = Field(default_factory=list)


class RuleBasedMemoryExtractor:
    """Deterministic extractor that only recognizes explicit durable statements."""

    def extract(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tool_results: Sequence[Mapping[str, Any]] | None = None,
    ) -> MemoryExtractionResult:
        del tool_results
        user_text = _conversation_text(conversation, "user")
        memories: list[ExtractedMemory] = []
        seen: set[str] = set()

        fitness_goal = _extract_fitness_goal(user_text)
        if fitness_goal is not None:
            memories.append(ExtractedMemory(key="fitness_goal", value=fitness_goal))
            seen.add("fitness_goal")

        training_preference = _extract_training_preference(user_text)
        if training_preference is not None:
            memories.append(
                ExtractedMemory(key="training_preference", value=training_preference)
            )
            seen.add("training_preference")

        experience_level = _extract_experience_level(user_text)
        weather_location = _extract_weather_location(user_text)
        if weather_location is not None:
            memories.append(
                ExtractedMemory(key="weather_location", value=weather_location)
            )
            seen.add("weather_location")

        if experience_level is not None and experience_level not in seen:
            memories.append(ExtractedMemory(key="experience_level", value=experience_level))

        return MemoryExtractionResult(memories=memories)


def _conversation_text(conversation: Sequence[Mapping[str, Any]], role: str) -> str:
    chunks: list[str] = []
    for turn in conversation:
        if not isinstance(turn, Mapping) or turn.get("role") != role:
            continue
        content = turn.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content.strip())
    return "\n".join(chunks)


def _extract_fitness_goal(text: str) -> str | None:
    normalized = text.replace(" ", "")
    markers = ("我的健身目标是", "我的目标是")
    for marker in markers:
        start = normalized.find(marker)
        if start >= 0:
            value = normalized[start + len(marker) :].strip("，。！？,.;；\t\n")
            if value:
                return value[:100]
    if "增肌" in normalized:
        return "增肌"
    if "减脂" in normalized or "减肥" in normalized:
        return "减脂"
    return None


def _extract_training_preference(text: str) -> str | None:
    normalized = text.replace(" ", "")
    if "不喜欢跑步" in normalized and "更喜欢力量训练" in normalized:
        return "不喜欢跑步，更喜欢力量训练"
    return None


def _extract_weather_location(text: str) -> str | None:
    normalized = text.replace(" ", "")
    for marker in ("我的所在地是", "所在地是", "我的城市是", "我住在", "我在"):
        start = normalized.find(marker)
        if start < 0:
            continue
        value = normalized[start + len(marker) :]
        value = re.split(
            r"[，。！？,.;；\t\n]|今天|明天|昨天|适合|天气|跑步|训练",
            value,
            maxsplit=1,
        )[0]
        value = value.strip("，。！？,.;；\t\n")
        if value:
            return value[:100]
    return None


def _extract_experience_level(text: str) -> str | None:
    normalized = text.replace(" ", "")
    if "我是初学者" in normalized or "我是健身小白" in normalized:
        return "beginner"
    if "我是高级训练者" in normalized:
        return "advanced"
    if "我是中级训练者" in normalized:
        return "intermediate"
    return None
