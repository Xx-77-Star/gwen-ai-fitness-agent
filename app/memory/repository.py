"""Repository contracts and SQL implementation for long-term user memory."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import UserMemory as UserMemoryModel
from app.memory.models import UserMemory


class UserMemoryRepository(Protocol):
    """Persistence boundary used by MemoryService."""

    def list_by_user(self, user_id: str) -> list[UserMemory]:
        """Return all long-term memory items for one user."""
        ...

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        """Create or update one memory key for one user."""
        ...


class SQLUserMemoryRepository:
    """SQLAlchemy-backed long-term user memory repository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_by_user(self, user_id: str) -> list[UserMemory]:
        rows = self._session.scalars(
            select(UserMemoryModel)
            .where(UserMemoryModel.user_id == user_id)
            .order_by(UserMemoryModel.updated_at.desc(), UserMemoryModel.id.desc())
        ).all()
        return [UserMemory.model_validate(row) for row in rows]

    def upsert(self, user_id: str, memory_key: str, memory_value: str) -> UserMemory:
        row = self._session.scalar(
            select(UserMemoryModel).where(
                UserMemoryModel.user_id == user_id,
                UserMemoryModel.memory_key == memory_key,
            )
        )
        if row is None:
            row = UserMemoryModel(
                user_id=user_id,
                memory_key=memory_key,
                memory_value=memory_value,
            )
            self._session.add(row)
        else:
            row.memory_value = memory_value
        self._session.flush()
        return UserMemory.model_validate(row)
