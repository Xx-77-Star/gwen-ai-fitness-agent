import pytest
from pydantic import BaseModel

from app.tools.registry import ToolRegistry, register_tools
from app.tools.schemas import ToolDefinition


class EmptyInput(BaseModel):
    pass


def handler(arguments, user_id, repository):
    return {"ok": True}


def definition(name: str = "sample_tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="A test tool.",
        input_model=EmptyInput,
        handler=handler,
    )


def test_register_and_get_tool() -> None:
    registry = ToolRegistry()

    registry.register(definition())

    assert registry.get("sample_tool") is not None
    assert registry.names() == ("sample_tool",)


def test_schema_payload_contains_name_description_and_parameters() -> None:
    registry = ToolRegistry()
    registry.register(definition())

    payloads = registry.schema_payloads()

    assert payloads[0]["type"] == "function"
    assert payloads[0]["function"]["name"] == "sample_tool"
    assert payloads[0]["function"]["description"] == "A test tool."
    assert "properties" in payloads[0]["function"]["parameters"]


def test_get_unknown_tool_returns_none() -> None:
    registry = ToolRegistry()

    assert registry.get("unknown_tool") is None


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(definition())

    with pytest.raises(ValueError):
        registry.register(definition())


def test_register_multiple_tools() -> None:
    registry = ToolRegistry()

    register_tools(registry, [definition("first"), definition("second")])

    assert registry.names() == ("first", "second")