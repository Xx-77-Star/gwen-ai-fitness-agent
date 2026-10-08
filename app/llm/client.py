import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, overload

from openai import AsyncOpenAI

from app.agent.state import ConversationTurn
from app.config.settings import Settings
from app.llm.errors import LLMError
from app.llm.messages import (
    AssistantMessage,
    LLMMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from app.llm.types import FinishReason, LLMResponse, LLMToolCall, LLMToolDefinition, LLMUsage

LLMMessageInput = LLMMessage | ConversationTurn
ToolSchema = Mapping[str, Any]
ToolChoice = str | dict[str, Any]


class LLMClient(Protocol):
    """Provider-agnostic LLM contract with optional OpenAI-compatible tools."""

    @overload
    async def complete(
        self,
        *,
        system_prompt: str | None = ...,
        messages: Sequence[LLMMessageInput],
        tools: None = ...,
        tool_choice: None = ...,
    ) -> str: ...

    @overload
    async def complete(
        self,
        *,
        system_prompt: str | None = ...,
        messages: Sequence[LLMMessageInput],
        tools: Sequence[LLMToolDefinition | ToolSchema],
        tool_choice: ToolChoice | None = ...,
    ) -> LLMResponse: ...

    async def complete(
        self,
        *,
        system_prompt: str | None = None,
        messages: Sequence[LLMMessageInput],
        tools: Sequence[LLMToolDefinition | ToolSchema] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> str | LLMResponse:
        """Return text, or a normalized tool-calling response when tools are supplied."""
        ...


class OpenAICompatibleLLMClient:
    """Alibaba Cloud Bailian/DashScope client using the OpenAI-compatible API.

    Calls without ``tools`` remain text-only and return ``str`` for compatibility
    with existing Agent nodes. Calls with ``tools`` return provider-neutral
    ``LLMResponse`` values, including assistant tool calls, normalized finish
    reasons, and token usage. The client never executes tools or integrates
    with LangGraph.
    """

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self._settings = settings
        self._client = client or AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    @overload
    async def complete(
        self,
        *,
        system_prompt: str | None = ...,
        messages: Sequence[LLMMessageInput],
        tools: None = ...,
        tool_choice: None = ...,
    ) -> str: ...

    @overload
    async def complete(
        self,
        *,
        system_prompt: str | None = ...,
        messages: Sequence[LLMMessageInput],
        tools: Sequence[LLMToolDefinition | ToolSchema],
        tool_choice: ToolChoice | None = ...,
    ) -> LLMResponse: ...

    async def complete(
        self,
        *,
        system_prompt: str | None = None,
        messages: Sequence[LLMMessageInput],
        tools: Sequence[LLMToolDefinition | ToolSchema] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> str | LLMResponse:
        if tool_choice is not None and tools is None:
            raise LLMError("tool_choice requires tools")

        key = self._settings.llm_api_key.get_secret_value()
        key_preview = f"{key[:4]}...{key[-4:]}" if len(key) >= 8 else "***"
        print(
            "[LLM-DEBUG] entry=complete "
            f"key_preview={key_preview} "
            f"key_length={len(key)} "
            f"base_url={self._settings.llm_base_url} "
            f"model={self._settings.llm_model}"
        )
        request_tools = self._build_tools(tools)
        request: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": self._build_messages(system_prompt, messages),
            "temperature": self._settings.llm_temperature,
        }
        if tools is not None:
            request["tools"] = request_tools
            if tool_choice is not None:
                request["tool_choice"] = tool_choice

        try:
            completion = await self._client.chat.completions.create(**request)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("LLM completion request failed") from exc

        response = self._normalize_response(completion)
        if tools is None:
            if not response.content:
                raise LLMError("LLM returned an empty completion")
            return response.content
        return response

    def _build_messages(
        self,
        system_prompt: str | None,
        messages: Sequence[LLMMessageInput],
    ) -> list[dict[str, Any]]:
        payloads = [
            self._message_payload(message) for message in self._normalize_messages(messages)
        ]
        if system_prompt:
            payloads.insert(0, self._message_payload(SystemMessage(content=system_prompt)))
        return payloads

    @staticmethod
    def _normalize_messages(messages: Sequence[LLMMessageInput]) -> list[LLMMessage]:
        normalized: list[LLMMessage] = []
        for message in messages:
            if isinstance(
                message, (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage)
            ):
                normalized.append(message)
                continue
            if not isinstance(message, Mapping):
                raise LLMError("LLM messages must be mappings or message models")
            try:
                normalized.append(_message_from_mapping(dict(message)))
            except (TypeError, ValueError) as exc:
                raise LLMError("LLM message protocol is invalid") from exc
        return normalized

    @staticmethod
    def _message_payload(message: LLMMessage) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, AssistantMessage):
            payload: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for call in message.tool_calls
                ]
            return payload
        if isinstance(message, ToolResultMessage):
            payload = {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": _tool_result_content(message.content),
            }
            if message.name:
                payload["name"] = message.name
            return payload
        raise LLMError("Unsupported LLM message role")

    @staticmethod
    def _build_tools(
        tools: Sequence[LLMToolDefinition | ToolSchema] | None,
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for tool in tools or []:
            payload = (
                tool.model_dump(mode="json") if isinstance(tool, LLMToolDefinition) else dict(tool)
            )
            if payload.get("type") != "function" or not payload.get("function"):
                raise LLMError("LLM tool protocol is invalid")
            payloads.append(payload)
        return payloads

    def _normalize_response(self, completion: Any) -> LLMResponse:
        choices = getattr(completion, "choices", None)
        if not choices:
            raise LLMError("LLM returned no completion choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMError("LLM returned an invalid completion message")

        content = getattr(message, "content", None)
        content = content.strip() if isinstance(content, str) and content.strip() else None
        tool_calls = self._normalize_tool_calls(getattr(message, "tool_calls", None))
        if not content and not tool_calls:
            raise LLMError("LLM returned an empty completion")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._normalize_finish_reason(getattr(choice, "finish_reason", None)),
            usage=self._normalize_usage(getattr(completion, "usage", None)),
            raw=self._dump_raw(completion),
        )

    @staticmethod
    def _normalize_tool_calls(raw_tool_calls: Any) -> list[LLMToolCall]:
        normalized: list[LLMToolCall] = []
        for raw in raw_tool_calls or []:
            if isinstance(raw, LLMToolCall):
                normalized.append(raw)
                continue
            function = getattr(raw, "function", None)
            call_id = getattr(raw, "id", None)
            name = getattr(function, "name", None) if function is not None else None
            if not call_id or not name:
                raise LLMError("LLM returned an invalid tool call")
            try:
                normalized.append(
                    LLMToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=_parse_tool_arguments(getattr(function, "arguments", None)),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise LLMError("LLM returned invalid tool arguments") from exc
        return normalized

    @staticmethod
    def _normalize_finish_reason(reason: Any) -> FinishReason:
        normalized = str(reason or "").strip().lower()
        if normalized in {"stop", "tool_calls", "length", "content_filter"}:
            return normalized  # type: ignore[return-value]
        return "unknown"

    @staticmethod
    def _normalize_usage(usage: Any) -> LLMUsage | None:
        if usage is None:
            return None
        try:
            if isinstance(usage, Mapping):
                return LLMUsage.model_validate(dict(usage))
            return LLMUsage.model_validate(
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            )
        except (TypeError, ValueError) as exc:
            raise LLMError("LLM returned invalid usage data") from exc

    @staticmethod
    def _dump_raw(completion: Any) -> dict[str, Any] | None:
        dump = getattr(completion, "model_dump", None)
        return dump(mode="json") if callable(dump) else None


def _message_from_mapping(payload: dict[str, Any]) -> LLMMessage:
    role = payload.get("role")
    if role == "system":
        return SystemMessage.model_validate(payload)
    if role == "user":
        return UserMessage.model_validate(payload)
    if role == "assistant":
        raw_calls = payload.get("tool_calls") or []
        calls = [
            LLMToolCall.model_validate(call)
            if "call_id" in call
            else LLMToolCall(
                call_id=call["id"],
                name=call["function"]["name"],
                arguments=_parse_tool_arguments(call["function"].get("arguments")),
            )
            for call in raw_calls
        ]
        return AssistantMessage(content=payload.get("content"), tool_calls=calls)
    if role == "tool":
        return ToolResultMessage.model_validate(payload)
    raise ValueError(f"unsupported role: {role!r}")


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be an object")
    return parsed


def _tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)
