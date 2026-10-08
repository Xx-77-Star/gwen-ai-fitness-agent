from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FitnessLevel = Literal["beginner", "intermediate", "advanced", "unknown"]


class UserProfileCreate(BaseModel):
    """Request model for creating a user profile."""

    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    nickname: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=1, le=120)
    gender: str = Field(min_length=1, max_length=32)
    height: float = Field(gt=50, le=250, description="Height in centimeters")
    weight: float = Field(gt=20, le=500, description="Weight in kilograms")
    fitness_level: FitnessLevel
    goal: str = Field(min_length=1, max_length=500)
    training_frequency: int = Field(ge=0, le=14, description="Training sessions per week")
    diet_preference: str = Field(min_length=1, max_length=200)
    lifestyle: str = Field(min_length=1, max_length=500)


class UserProfileResponse(UserProfileCreate):
    """Persistent user profile returned by Profile APIs."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime