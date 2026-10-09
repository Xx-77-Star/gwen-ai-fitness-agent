from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition


class WeatherClient(Protocol):
    def get_current_weather(self, *, latitude: float, longitude: float) -> dict[str, Any]:
        ...


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    city: str = Field(default="", max_length=100)


@dataclass(frozen=True)
class OpenMeteoWeatherClient:
    timeout_seconds: float = 10.0

    def get_current_weather(self, *, latitude: float, longitude: float) -> dict[str, Any]:
        response = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,rain,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        current = response.json().get("current")
        if not isinstance(current, dict):
            raise ValueError("weather response is invalid")
        code = current.get("weather_code")
        return {
            "temperature": current.get("temperature_2m"),
            "temperature_celsius": current.get("temperature_2m"),
            "weather_code": code,
            "weather": _weather_label(code),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "precipitation": current.get("precipitation"),
            "rain": current.get("rain"),
            "observed_at": current.get("time"),
        }


def _weather_label(code: Any) -> str:
    try:
        value = int(code)
    except (TypeError, ValueError):
        return "unknown"
    if value == 0:
        return "clear"
    if value in {1, 2, 3}:
        return "cloudy"
    if value in {45, 48}:
        return "fog"
    if value in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "rain"
    if value in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if value in {95, 96, 99}:
        return "thunderstorm"
    return "unknown"


def get_current_weather(
    arguments: dict[str, Any],
    user_id: str,
    runtime_context: Any,
) -> dict[str, Any]:
    payload = WeatherInput.model_validate(arguments)
    context = (
        runtime_context
        if isinstance(runtime_context, dict)
        else {"weather_client": runtime_context}
    )
    client = context.get("weather_client")
    if client is None:
        raise ValueError("weather client is not configured")
    result = client.get_current_weather(latitude=payload.latitude, longitude=payload.longitude)
    return {"city": payload.city, **result}


def register_weather_tool(registry: ToolRegistry, weather_client: WeatherClient) -> None:
    registry.register(
        ToolDefinition(
            name="get_current_weather",
            description=(
                "获取指定经纬度的实时天气：温度、天气状态、湿度、风速和降雨。"
                "用户询问今天天气、是否适合跑步、是否适合户外训练或训练安排是否需要调整时调用。"
                "如果用户没有提供坐标，应先调用 geocode_city 获取城市经纬度。"
            ),
            input_model=WeatherInput,
            handler=lambda arguments, user_id, repository: get_current_weather(
                arguments, user_id, repository
            ),
        )
    )


def build_weather_tool_registry(weather_client: WeatherClient) -> ToolRegistry:
    registry = ToolRegistry()
    register_weather_tool(registry, weather_client)
    return registry
