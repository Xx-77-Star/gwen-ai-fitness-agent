from datetime import datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import WorkoutRecord
from app.schemas.workout import WorkoutCheckInCreate, WorkoutCheckInResponse


class WorkoutCheckInRepository(Protocol):
    """Workout check-in persistence and query contract."""

    def create(self, payload: WorkoutCheckInCreate) -> WorkoutCheckInResponse:
        """Persist and return one workout check-in."""
        ...

    def list_by_user(self, user_id: str, *, limit: int) -> list[WorkoutCheckInResponse]:
        """Return the latest workout check-ins for one user."""
        ...


class SQLWorkoutCheckInRepository:
    """SQLAlchemy-backed workout check-in repository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, payload: WorkoutCheckInCreate) -> WorkoutCheckInResponse:
        record = WorkoutRecord(**payload.model_dump())
        self._session.add(record)
        self._session.flush()
        return WorkoutCheckInResponse.model_validate(record)

    def list_by_user(self, user_id: str, *, limit: int) -> list[WorkoutCheckInResponse]:
        if limit < 1:
            raise ValueError("limit must be positive")

        records = self._session.scalars(
            select(WorkoutRecord)
            .where(WorkoutRecord.user_id == user_id)
            .order_by(WorkoutRecord.date.desc(), WorkoutRecord.id.desc())
            .limit(limit)
        ).all()
        return [WorkoutCheckInResponse.model_validate(record) for record in records]


class InMemoryWorkoutCheckInRepository:
    """Small repository implementation for isolated Agent and tool tests."""

    def __init__(self, records: list[WorkoutCheckInResponse] | None = None) -> None:
        self.records = list(records or [])
        self.calls: list[tuple[str, str, int]] = []

    def create(self, payload: WorkoutCheckInCreate) -> WorkoutCheckInResponse:
        record = WorkoutCheckInResponse(
            id=len(self.records) + 1,
            created_at=datetime.now(),
            **payload.model_dump(),
        )
        self.records.append(record)
        return record

    def list_by_user(self, user_id: str, *, limit: int) -> list[WorkoutCheckInResponse]:
        self.calls.append(("list_by_user", user_id, limit))
        matching = [item for item in self.records if item.user_id == user_id]
        return sorted(matching, key=lambda item: (item.date, item.id), reverse=True)[:limit]

