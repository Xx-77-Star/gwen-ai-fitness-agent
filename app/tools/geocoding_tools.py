from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class GeocodingClient(Protocol):
    def geocode(self, *, city: str, count: int = 1) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OpenMeteoGeocodingClient:
    timeout_seconds: float = 10.0

    def geocode(self, *, city: str, count: int = 1) -> dict[str, Any]:
        response = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": count, "language": "zh", "format": "json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            raise ValueError("city was not found")
        item = results[0]
        return {
            "city": item.get("name") or city,
            "country": item.get("country"),
            "admin1": item.get("admin1"),
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "timezone": item.get("timezone"),
        }


class CityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=1, max_length=100)


def geocode_city(arguments: dict[str, Any], user_id: str, runtime_context: Any) -> dict[str, Any]:
    payload = CityInput.model_validate(arguments)
    context = (
        runtime_context
        if isinstance(runtime_context, dict)
        else {"geocoding_client": runtime_context}
    )
    client = context.get("geocoding_client") or OpenMeteoGeocodingClient()
    return client.geocode(city=payload.city)


def register_geocoding_tool(
    registry: ToolRegistry,
    geocoding_client: GeocodingClient | None = None,
) -> None:
    registry.register(
        ToolDefinition(
            name="geocode_city",
            description="把用户填写的城市转换为经纬度，供天气工具使用。不要用于获取天气。",
            input_model=CityInput,
            handler=lambda arguments, user_id, repository: geocode_city(
                arguments, user_id, {"geocoding_client": geocoding_client}
            ),
        )
    )


def build_geocoding_tool_registry(geocoding_client: GeocodingClient | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    register_geocoding_tool(registry, geocoding_client)
    return registry
