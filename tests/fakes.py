from collections.abc import Sequence
from typing import Any

from app.agent.state import ConversationTurn
from app.llm.types import LLMResponse
from app.memory.profile_repository import ProfileRepository
from app.schemas.training import TrainingSummary
from app.schemas.user import UserProfileResponse


class FakeLLMClient:
    def __init__(
        self,
        decision_responses: Sequence[LLMResponse] | None = None,
        text_response: str | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.decision_responses = list(decision_responses or [])
        self.decision_response = (
            self.decision_responses[0]
            if self.decision_responses
            else LLMResponse(
                content="建议先安排一次全身力量训练，并根据体感调整强度。", finish_reason="stop"
            )
        )
        self.text_response = text_response or "建议先安排一次全身力量训练，并根据体感调整强度。"

    async def complete(self, *, system_prompt: str, messages: Sequence[ConversationTurn], **kwargs):
        self.calls.append({"system_prompt": system_prompt, "messages": messages, **kwargs})
        if kwargs.get("tools") is not None:
            if self.decision_responses:
                return self.decision_responses.pop(0)
            return self.decision_response
        return self.text_response


class FakeProfileRepository(ProfileRepository):
    def __init__(self, profiles: dict[str, UserProfileResponse] | None = None) -> None:
        self.profiles = profiles or {}
        self.calls: list[str] = []

    def get_profile(self, user_id: str) -> UserProfileResponse | None:
        self.calls.append(user_id)
        return self.profiles.get(user_id)


class FakeTrainingRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def get_latest_by_user(self, user_id):
        self.calls.append(("get_latest_by_user", user_id))
        return {"id": 1, "training_date": "2026-10-07"}

    def list_by_user(self, user_id, start_date, end_date, limit):
        self.calls.append(("list_by_user", user_id, start_date, end_date, limit))
        return []

    def summarize_by_user(self, user_id, start_date, end_date):
        self.calls.append(("summarize_by_user", user_id, start_date, end_date))
        return TrainingSummary(
            record_count=1,
            training_days=1,
            total_duration_minutes=30,
            type_distribution={"力量训练": 1},
            start_date=start_date,
            end_date=end_date,
        )
