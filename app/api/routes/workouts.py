from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.workout_repository import SQLWorkoutCheckInRepository
from app.schemas.workout import WorkoutCheckInCreate, WorkoutCheckInResponse

router = APIRouter(prefix="/workouts", tags=["workout-check-ins"])

DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=WorkoutCheckInResponse, status_code=status.HTTP_201_CREATED)
def create_workout_check_in(payload: WorkoutCheckInCreate, db: DbSession) -> WorkoutCheckInResponse:
    """Persist one user-submitted workout check-in."""
    repository = SQLWorkoutCheckInRepository(db)
    try:
        return repository.create(payload)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found",
        ) from exc


@router.get("/{user_id}", response_model=list[WorkoutCheckInResponse])
def list_workout_check_ins(
    user_id: str,
    db: DbSession,
    limit: int = Query(default=20, ge=1, le=50),
) -> list[WorkoutCheckInResponse]:
    """Return the latest workout check-ins for one user."""
    return SQLWorkoutCheckInRepository(db).list_by_user(user_id, limit=limit)
