from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

MuscleGroup = str


class WorkoutCheckInCreate(BaseModel):
    """Request model for saving one user-submitted workout check-in."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    date: date
    muscle_group: str = Field(min_length=1, max_length=20)
    exercise: str = Field(min_length=1, max_length=100)
    weight: float = Field(ge=0, le=1000)
    sets: int = Field(ge=1, le=100)
    reps: int = Field(ge=1, le=1000)
    feeling: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=1000)


class WorkoutCheckInResponse(WorkoutCheckInCreate):
    """Persistent workout check-in returned by workout APIs and Agent tools."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
