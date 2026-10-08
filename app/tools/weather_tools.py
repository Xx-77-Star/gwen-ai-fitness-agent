"""Weather tool definition and external weather client boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class WeatherClient(Protocol):
    """Provider-neutral external weather API boundary."""

    def get_current_weather(self, *, latitude: float, longitude: float) -> dict[str, Any]:
        """Return normalized current weather data for one coordinate pair."""
        ...


class WeatherInput(BaseModel):
    """Input for get_current_weather."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@dataclass(frozen=True)
class OpenMeteoWeatherClient:
    """Read-only Open-Meteo client with a normalized response contract."""

    timeout_seconds: float = 10.0

    def get_current_weather(self, *, latitude: float, longitude: float) -> dict[str, Any]:
        response = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current_weather")
        if not isinstance(current, dict):
            raise ValueError("weather response is invalid")
        return {
            "latitude": latitude,
            "longitude": longitude,
            "temperature_celsius": current.get("temperature"),
            "wind_speed_kmh": current.get("windspeed"),
            "wind_direction_degrees": current.get("winddirection"),
            "is_day": current.get("is_day"),
            "weather_code": current.get("weathercode"),
            "observed_at": current.get("time"),
        }


def get_current_weather(
    arguments: dict[str, Any],
    user_id: str,
    runtime_context: Any,
) -> dict[str, Any]:
    """Handler-compatible wrapper for the external weather client."""
    payload = WeatherInput.model_validate(arguments)
    context = (
        runtime_context
        if isinstance(runtime_context, dict)
        else {"weather_client": runtime_context}
    )
    client = context.get("weather_client")
    if client is None:
        raise ValueError("weather client is not configured")
    return client.get_current_weather(
        latitude=payload.latitude,
        longitude=payload.longitude,
    )


def register_weather_tool(registry: ToolRegistry, weather_client: WeatherClient) -> None:
    """Register the external weather tool on an existing registry."""
    registry.register(
        ToolDefinition(
            name="get_current_weather",
            description=(
                "只用于用户询问指定地点当前天气、气温、风速或是否适合户外训练。"
                "需要提供 latitude 和 longitude 坐标；不要用于训练记录、训练统计或个人档案查询。"
            ),
            input_model=WeatherInput,
            handler=lambda arguments, user_id, repository: get_current_weather(
                arguments,
                user_id,
                repository,
            ),
        )
    )


def build_weather_tool_registry(weather_client: WeatherClient) -> ToolRegistry:
    """Create a registry containing the external weather tool."""
    registry = ToolRegistry()
    register_weather_tool(registry, weather_client)
    return registry
