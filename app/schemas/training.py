from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ExerciseSet(BaseModel):
    """Per-exercise strength metrics kept as structured JSON."""

    exercise: str = Field(min_length=1, max_length=100)
    value: float = Field(ge=0)


class TrainingRecordCreate(BaseModel):
    """Request model for saving one training session."""

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    training_date: date
    training_type: str = Field(min_length=1, max_length=50)
    duration_minutes: int = Field(gt=0, le=1440)
    intensity: str = Field(min_length=1, max_length=32)
    body_parts: list[str] = Field(min_length=1, max_length=20)
    sets: list[ExerciseSet] = Field(min_length=1, max_length=50)
    reps: list[ExerciseSet] = Field(min_length=1, max_length=50)
    weight: list[ExerciseSet] = Field(min_length=1, max_length=50)
    notes: str = Field(max_length=1000)


class TrainingRecordResponse(TrainingRecordCreate):
    """Persistent training record returned by Training APIs."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class TrainingSummary(BaseModel):
    """Basic behavior summary for one user and date range.

    One TrainingRecord represents one recorded training session. This model
    intentionally does not estimate volume, calories, muscle growth, or progress.
    """

    record_count: int = Field(ge=0)
    training_days: int = Field(ge=0)
    total_duration_minutes: int = Field(ge=0)
    type_distribution: dict[str, int]
    start_date: date
    end_date: date