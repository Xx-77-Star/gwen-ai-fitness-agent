from collections import Counter
from datetime import date
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import TrainingRecord, UserProfile
from app.schemas.training import (
    TrainingRecordCreate,
    TrainingRecordResponse,
    TrainingSummary,
)


class TrainingRecordRepository(Protocol):
    """Training persistence and query contract used by higher layers."""

    def create(self, payload: TrainingRecordCreate) -> TrainingRecordResponse:
        """Persist and return one training record."""
        ...

    def get(self, record_id: int, user_id: str) -> TrainingRecordResponse | None:
        """Return one record belonging to a user."""
        ...

    def list_by_user(
        self,
        user_id: str,
        start_date: date,
        end_date: date,
        limit: int,
    ) -> list[TrainingRecordResponse]:
        """Return a user's records in a date range with stable ordering."""
        ...

    def get_latest_by_user(self, user_id: str) -> TrainingRecordResponse | None:
        """Return a user's latest training record."""
        ...

    def summarize_by_user(
        self,
        user_id: str,
        start_date: date,
        end_date: date,
    ) -> TrainingSummary:
        """Return basic training behavior counts for a user and date range."""
        ...


class SQLTrainingRecordRepository:
    """SQLAlchemy-backed training record repository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, payload: TrainingRecordCreate) -> TrainingRecordResponse:
        has_profile = self._session.scalar(
            select(UserProfile.id).where(UserProfile.user_id == payload.user_id)
        )
        if has_profile is None:
            raise ValueError("User profile not found")

        record = TrainingRecord(**payload.model_dump())
        self._session.add(record)
        self._session.flush()
        return TrainingRecordResponse.model_validate(record)

    def get(self, record_id: int, user_id: str) -> TrainingRecordResponse | None:
        record = self._session.scalar(
            select(TrainingRecord).where(
                TrainingRecord.id == record_id,
                TrainingRecord.user_id == user_id,
            )
        )
        if record is None:
            return None
        return TrainingRecordResponse.model_validate(record)

    def list_by_user(
        self,
        user_id: str,
        start_date: date,
        end_date: date,
        limit: int,
    ) -> list[TrainingRecordResponse]:
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")
        if limit < 1:
            raise ValueError("limit must be positive")

        records = self._session.scalars(
            select(TrainingRecord)
            .where(
                TrainingRecord.user_id == user_id,
                TrainingRecord.training_date >= start_date,
                TrainingRecord.training_date <= end_date,
            )
            .order_by(TrainingRecord.training_date.desc(), TrainingRecord.id.desc())
            .limit(limit)
        ).all()
        return [TrainingRecordResponse.model_validate(record) for record in records]

    def get_latest_by_user(self, user_id: str) -> TrainingRecordResponse | None:
        record = self._session.scalar(
            select(TrainingRecord)
            .where(TrainingRecord.user_id == user_id)
            .order_by(TrainingRecord.training_date.desc(), TrainingRecord.id.desc())
            .limit(1)
        )
        if record is None:
            return None
        return TrainingRecordResponse.model_validate(record)

    def summarize_by_user(
        self,
        user_id: str,
        start_date: date,
        end_date: date,
    ) -> TrainingSummary:
        if start_date > end_date:
            raise ValueError("start_date must be on or before end_date")

        rows = self._session.execute(
            select(
                TrainingRecord.training_date,
                TrainingRecord.training_type,
                TrainingRecord.duration_minutes,
            ).where(
                TrainingRecord.user_id == user_id,
                TrainingRecord.training_date >= start_date,
                TrainingRecord.training_date <= end_date,
            )
        ).all()

        type_distribution = Counter(row.training_type for row in rows)
        return TrainingSummary(
            record_count=len(rows),
            training_days=len({row.training_date for row in rows}),
            total_duration_minutes=sum(row.duration_minutes for row in rows),
            type_distribution=dict(sorted(type_distribution.items())),
            start_date=start_date,
            end_date=end_date,
        )