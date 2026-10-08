"""Fake external weather clients for offline tests."""

from __future__ import annotations

from typing import Any


class FakeWeatherClient:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def get_current_weather(self, *, latitude: float, longitude: float) -> dict[str, Any]:
        self.calls.append((latitude, longitude))
        return {
            "latitude": latitude,
            "longitude": longitude,
            "temperature_celsius": 22.5,
            "wind_speed_kmh": 8.0,
            "wind_direction_degrees": 180,
            "is_day": 1,
            "weather_code": 1,
            "observed_at": "2026-10-07T12:00:00Z",
        }
