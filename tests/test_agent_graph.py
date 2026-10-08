from datetime import date

import pytest

from app.agent.graph import create_agent_graph, make_tool_execution_router, route_tool_decision
from app.agent.nodes.intent import classify_intent
from app.llm.types import LLMResponse, LLMToolCall
from app.schemas.user import UserProfileResponse
from tests.fakes import FakeLLMClient, FakeProfileRepository, FakeTrainingRepository

PROFILE = UserProfileResponse(
    id=1,
    user_id="user-001",
    nickname="Alex",
    age=30,
    gender="prefer_not_to_say",
    height=178.0,
    weight=75.5,
    fitness_level="intermediate",
    goal="提升力量并改善体能",
    training_frequency=4,
    diet_preference="高蛋白",
    lifestyle="久坐办公，晚上训练",
    created_at="2026-10-07T12:00:00Z",
    updated_at="2026-10-07T12:00:00Z",
)


def state(message: str = "今天应该怎么训练？", user_id: str = "user-001") -> dict:
    return {
        "user_id": user_id,
        "input": message,
        "conversation_history": [],
        "rag_context": [],
        "tool_calls": [],
        "tool_results": [],
        "tool_rounds": 0,
        "tool_call_count": 0,
        "answer_draft": None,
        "stop_reason": "completed",
        "messages": [],
    }


def tool_response(call_id: str = "call-1", name: str = "get_training_summary") -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            LLMToolCall(
                call_id=call_id,
                name=name,
                arguments={"start_date": "2026-10-01", "end_date": "2026-10-07"}
                if name == "get_training_summary"
                else {},
            )
        ],
        finish_reason="tool_calls",
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("今天应该怎么训练？", "training"),
        ("晚餐应该吃什么？", "nutrition"),
        ("最近睡眠不好怎么恢复？", "recovery"),
        ("帮我安排这一周的生活", "lifestyle"),
        ("你好", "general"),
    ],
)
def test_classify_intent(message: str, expected: str) -> None:
    assert classify_intent(message) == expected


@pytest.mark.asyncio
async def test_agent_reads_existing_profile() -> None:
    fake_llm = FakeLLMClient(text_response="建议先安排一次全身力量训练，并根据体感调整强度。")
    repository = FakeProfileRepository({"user-001": PROFILE})
    graph = create_agent_graph(fake_llm, repository, training_repository=FakeTrainingRepository())

    result = await graph.ainvoke(state())

    assert repository.calls == ["user-001"]
    assert result["profile_loaded"] is True
    assert result["profile"] == PROFILE
    assert "Alex" in fake_llm.calls[0]["system_prompt"]
    assert result["response"].startswith("建议")


@pytest.mark.asyncio
async def test_agent_runs_without_profile() -> None:
    fake_llm = FakeLLMClient()
    repository = FakeProfileRepository()
    graph = create_agent_graph(fake_llm, repository, training_repository=FakeTrainingRepository())

    result = await graph.ainvoke(state(user_id="unknown-user"))

    assert repository.calls == ["unknown-user"]
    assert result["profile_loaded"] is False
    assert result["profile"] is None
    assert result["response"].startswith("建议")


@pytest.mark.asyncio
async def test_response_node_receives_profile() -> None:
    fake_llm = FakeLLMClient()
    repository = FakeProfileRepository({"user-001": PROFILE})
    graph = create_agent_graph(fake_llm, repository, training_repository=FakeTrainingRepository())

    await graph.ainvoke(state())

    system_prompt = fake_llm.calls[0]["system_prompt"]
    assert "当前用户画像" in system_prompt
    assert "健身等级：intermediate" in system_prompt


@pytest.mark.asyncio
async def test_direct_decision_goes_straight_to_response_without_second_llm_call() -> None:
    fake_llm = FakeLLMClient()
    graph = create_agent_graph(
        fake_llm,
        FakeProfileRepository({"user-001": PROFILE}),
        training_repository=FakeTrainingRepository(),
    )

    result = await graph.ainvoke(state("你好"))

    assert result["tool_calls"] == []
    assert result["tool_results"] == []
    assert result["response"] == "建议先安排一次全身力量训练，并根据体感调整强度。"
    assert len(fake_llm.calls) == 1


@pytest.mark.asyncio
async def test_tool_loop_passes_tool_results_to_second_decision() -> None:
    training_repository = FakeTrainingRepository()
    fake_llm = FakeLLMClient(
        decision_responses=[
            tool_response(),
            LLMResponse(content="本周完成 1 次训练，累计 30 分钟。", finish_reason="stop"),
        ],
        text_response="最终回答必须使用工具结果。",
    )
    graph = create_agent_graph(
        fake_llm,
        FakeProfileRepository({"user-001": PROFILE}),
        training_repository=training_repository,
    )

    result = await graph.ainvoke(state("查询我本周的训练记录"))

    assert result["tool_rounds"] == 1
    assert result["tool_call_count"] == 1
    assert result["tool_results"][0]["status"] == "success"
    assert result["response"] == "本周完成 1 次训练，累计 30 分钟。"
    assert training_repository.calls == [
        ("summarize_by_user", "user-001", date(2026, 10, 1), date(2026, 10, 7))
    ]
    second_decision_messages = fake_llm.calls[1]["messages"]
    assert any(message.get("role") == "tool" for message in second_decision_messages)


@pytest.mark.asyncio
async def test_old_tool_calls_are_replaced_before_second_execution() -> None:
    training_repository = FakeTrainingRepository()
    fake_llm = FakeLLMClient(
        decision_responses=[
            tool_response("call-1", "get_last_training_record"),
            tool_response("call-2", "get_training_summary"),
            LLMResponse(content="工具结果已足够。", finish_reason="stop"),
        ]
    )
    graph = create_agent_graph(
        fake_llm,
        FakeProfileRepository({"user-001": PROFILE}),
        training_repository=training_repository,
    )

    result = await graph.ainvoke(state("分析我的训练"))

    assert result["tool_call_count"] == 2
    assert result["tool_rounds"] == 2
    assert [item["call_id"] for item in result["tool_results"]] == ["call-1", "call-2"]
    assert training_repository.calls == [
        ("get_latest_by_user", "user-001"),
        ("summarize_by_user", "user-001", date(2026, 10, 1), date(2026, 10, 7)),
    ]


@pytest.mark.asyncio
async def test_max_tool_rounds_stops_loop() -> None:
    repository = FakeTrainingRepository()
    fake_llm = FakeLLMClient(
        decision_responses=[
            tool_response("call-1", "get_last_training_record"),
            tool_response("call-2", "get_last_training_record"),
        ]
    )
    graph = create_agent_graph(
        fake_llm,
        FakeProfileRepository({"user-001": PROFILE}),
        training_repository=repository,
        max_tool_rounds=2,
        max_tool_calls=4,
    )

    result = await graph.ainvoke(state("持续查询训练"))

    assert result["tool_rounds"] == 2
    assert result["tool_call_count"] == 2
    assert result["response"] == "建议先安排一次全身力量训练，并根据体感调整强度。"
    assert repository.calls == [("get_latest_by_user", "user-001")] * 2


@pytest.mark.asyncio
async def test_response_uses_tool_results_when_draft_is_absent() -> None:
    fake_llm = FakeLLMClient(
        decision_responses=[
            tool_response(),
            LLMResponse(content="草稿", finish_reason="stop"),
        ],
        text_response="训练记录显示 1 次训练。",
    )
    graph = create_agent_graph(
        fake_llm,
        FakeProfileRepository({"user-001": PROFILE}),
        training_repository=FakeTrainingRepository(),
    )

    result = await graph.ainvoke(state("查询训练汇总"))

    assert result["tool_results"][0]["status"] == "success"
    assert result["response"] == "草稿"


def test_tool_decision_routes_on_current_calls_only() -> None:
    assert route_tool_decision({"tool_calls": []}) == "response"
    assert route_tool_decision({"tool_calls": [{"call_id": "x"}]}) == "tool_execution"


def test_tool_execution_router_enforces_round_and_total_limits() -> None:
    router = make_tool_execution_router(2, 4)
    assert (
        router({"tool_rounds": 1, "tool_call_count": 1, "stop_reason": "completed"})
        == "tool_decision"
    )
    assert (
        router({"tool_rounds": 2, "tool_call_count": 2, "stop_reason": "completed"}) == "response"
    )
    assert (
        router({"tool_rounds": 1, "tool_call_count": 4, "stop_reason": "completed"}) == "response"
    )
    assert (
        router({"tool_rounds": 1, "tool_call_count": 1, "stop_reason": "tool_limit"}) == "response"
    )
