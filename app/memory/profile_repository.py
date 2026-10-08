from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import UserProfile
from app.schemas.user import UserProfileResponse


class ProfileRepository(Protocol):
    """Profile lookup contract used by Agent nodes."""

    def get_profile(self, user_id: str) -> UserProfileResponse | None:
        """Return the profile for a user, or None when it does not exist."""
        ...


class SQLProfileRepository:
    """SQLAlchemy-backed profile repository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_profile(self, user_id: str) -> UserProfileResponse | None:
        profile = self._session.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
        if profile is None:
            return None
        return UserProfileResponse.model_validate(profile)