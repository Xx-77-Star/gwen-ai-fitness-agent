from collections.abc import Iterable

from app.tools.schemas import ToolDefinition


class ToolRegistry:
    """Registry for a fixed, application-defined set of tool handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def schema_payloads(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_model.model_json_schema(),
                },
            }
            for definition in self._tools.values()
        ]


def register_tools(
    registry: ToolRegistry,
    definitions: Iterable[ToolDefinition],
) -> None:
    for definition in definitions:
        registry.register(definition)