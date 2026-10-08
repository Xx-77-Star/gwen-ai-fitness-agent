"""Offline Fitness Recommendation Tool Selection evaluation."""

from __future__ import annotations

import pytest

from app.agent.nodes.tool_decision import tool_decision_node
from app.llm.types import LLMResponse, LLMToolCall
from app.tools.recommendation_tools import build_recommendation_tool_registry


class RecommendationDecisionLLM:
    def __init__(self) -> None:
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(
            tool_calls=[
                LLMToolCall(
                    call_id="recommendation-call",
                    name="get_training_recommendation",
                    arguments={
                        "user_goal": "提升体能",
                        "recent_training": ["跑步", "力量训练"],
                        "weather_condition": "rainy",
                    },
                )
            ],
            finish_reason="tool_calls",
        )


@pytest.mark.asyncio
async def test_recommendation_tool_selection_eval() -> None:
    registry = build_recommendation_tool_registry()
    llm = RecommendationDecisionLLM()

    result = await tool_decision_node(
        {
            "input": "根据最近训练帮我安排今天的训练建议",
            "user_id": "user-001",
            "conversation_history": [],
        },
        llm_client=llm,
        tool_registry=registry,
    )

    assert result["stop_reason"] == "tool_call"
    assert result["tool_calls"][0]["tool_name"] == "get_training_recommendation"
    assert result["tool_calls"][0]["arguments"]["weather_condition"] == "rainy"
    assert llm.calls[0]["tools"][0]["function"]["name"] == "get_training_recommendation"
