from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import UserProfile
from app.database.session import get_db
from app.schemas.user import UserProfileCreate, UserProfileResponse

router = APIRouter(prefix="/profile", tags=["profile"])

DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
def create_profile(payload: UserProfileCreate, db: DbSession) -> UserProfile:
    """Create and persist a user profile."""
    existing = db.scalar(select(UserProfile.id).where(UserProfile.user_id == payload.user_id))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A profile already exists for this user_id",
        )

    profile = UserProfile(**payload.model_dump())
    db.add(profile)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A profile already exists for this user_id",
        ) from exc
    return profile


@router.get("/{user_id}", response_model=UserProfileResponse)
def get_profile(user_id: str, db: DbSession) -> UserProfile:
    """Return the profile for a user_id."""
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found",
        )
    return profile