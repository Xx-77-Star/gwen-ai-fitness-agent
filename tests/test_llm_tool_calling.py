from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.llm.client import LLMClient, OpenAICompatibleLLMClient
from app.llm.errors import LLMError
from app.llm.messages import AssistantMessage, ToolResultMessage
from app.llm.types import LLMToolCall, LLMToolDefinition


class FakeChatCompletions:
    def __init__(self, completion: object) -> None:
        self.completion = completion
        self.request: dict[str, object] | None = None

    async def create(self, **kwargs):
        self.request = kwargs
        return self.completion


class FakeOpenAI:
    def __init__(self, completion: object) -> None:
        self.completions = FakeChatCompletions(completion)
        self.chat = SimpleNamespace(completions=self.completions)


class RecordingFakeLLM:
    """Scriptable fake used as an LLMClient without calling a provider."""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def tool_completion(*, finish_reason: str = "tool_calls", arguments: str | None = None) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(
                                name="get_last_training_record",
                                arguments=arguments if arguments is not None else "{}",
                            ),
                        )
                    ],
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


def settings() -> Settings:
    return Settings(LLM_API_KEY="test-key")


def tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        function={
            "name": "get_last_training_record",
            "description": "查询当前用户最近一次训练记录。",
            "parameters": {"type": "object", "properties": {}},
        }
    )


@pytest.mark.asyncio
async def test_fake_llm_client_implements_provider_agnostic_contract() -> None:
    expected = SimpleNamespace(content="已完成", tool_calls=[], finish_reason="stop")
    fake: LLMClient = RecordingFakeLLM(expected)

    result = await fake.complete(
        system_prompt="FitLife",
        messages=[{"role": "user", "content": "你好"}],
    )

    assert result is expected
    assert "tools" not in fake.calls[0]
    assert "tool_choice" not in fake.calls[0]


@pytest.mark.asyncio
async def test_openai_compatible_client_normalizes_tool_calls_and_finish_reason() -> None:
    fake_openai = FakeOpenAI(tool_completion(arguments='{"limit":3}'))
    client = OpenAICompatibleLLMClient(settings(), fake_openai)

    result = await client.complete(
        system_prompt="FitLife",
        messages=[{"role": "user", "content": "查询最近训练"}],
        tools=[tool_definition()],
        tool_choice="auto",
    )

    request = fake_openai.completions.request
    assert request["tools"] == [tool_definition().model_dump(mode="json")]
    assert request["tool_choice"] == "auto"
    assert result.content is None
    assert result.finish_reason == "tool_calls"
    assert result.requires_tool_execution is True
    assert result.tool_calls == [
        LLMToolCall(
            call_id="call-1",
            name="get_last_training_record",
            arguments={"limit": 3},
        )
    ]
    assert result.usage is not None
    assert result.usage.total_tokens == 13


@pytest.mark.asyncio
async def test_client_round_trips_assistant_tool_call_and_tool_result_messages() -> None:
    fake_openai = FakeOpenAI(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="训练记录是 1 次。", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
    )
    client = OpenAICompatibleLLMClient(settings(), fake_openai)
    messages = [
        {"role": "user", "content": "查询最近训练"},
        AssistantMessage(
            tool_calls=[
                LLMToolCall(
                    call_id="call-1",
                    name="get_last_training_record",
                    arguments={"limit": 3},
                )
            ]
        ),
        ToolResultMessage(
            tool_call_id="call-1",
            name="get_last_training_record",
            content={"status": "success", "data": {"record": None}},
        ),
    ]

    result = await client.complete(messages=messages, tools=[tool_definition()])

    request_messages = fake_openai.completions.request["messages"]
    assert request_messages[1]["content"] is None
    assert request_messages[1]["tool_calls"][0]["id"] == "call-1"
    assert request_messages[1]["tool_calls"][0]["function"]["arguments"] == '{"limit":3}'
    assert request_messages[2]["role"] == "tool"
    assert request_messages[2]["tool_call_id"] == "call-1"
    assert '"status":"success"' in request_messages[2]["content"]
    assert result.content == "训练记录是 1 次。"
    assert result.finish_reason == "stop"
    assert result.requires_tool_execution is False


@pytest.mark.asyncio
async def test_legacy_conversation_turn_remains_supported() -> None:
    fake_openai = FakeOpenAI(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" 保持规律训练。 ", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
    )
    client = OpenAICompatibleLLMClient(settings(), fake_openai)

    result = await client.complete(
        system_prompt="FitLife",
        messages=[{"role": "user", "content": "怎么坚持训练？"}],
    )

    assert result == "保持规律训练。"
    assert "tools" not in fake_openai.completions.request
    assert "tool_choice" not in fake_openai.completions.request


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_rejected() -> None:
    fake_openai = FakeOpenAI(tool_completion(arguments="not-json"))
    client = OpenAICompatibleLLMClient(settings(), fake_openai)

    with pytest.raises(LLMError, match="invalid tool"):
        await client.complete(
            messages=[{"role": "user", "content": "查询"}],
            tools=[tool_definition()],
        )


@pytest.mark.asyncio
async def test_tool_result_requires_call_id() -> None:
    client = OpenAICompatibleLLMClient(settings(), FakeOpenAI(tool_completion()))
    invalid = {"role": "tool", "content": {"ok": True}}

    with pytest.raises(LLMError, match="protocol"):
        await client.complete(messages=[invalid], tools=[tool_definition()])








