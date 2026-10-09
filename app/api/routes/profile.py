from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import UserMemory, UserProfile
from app.database.session import get_db
from app.schemas.user import UserProfileCreate, UserProfileResponse, UserProfileUpdate

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


@router.patch("/{user_id}/city", response_model=UserProfileResponse)
def update_profile_city(user_id: str, payload: UserProfileUpdate, db: DbSession) -> UserProfile:
    """Save the user's self-reported city for weather-aware recommendations."""
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if profile is None:
        # Web users can set weather location before completing the optional
        # training profile. Keep this lightweight and preserve the profile flow.
        profile = UserProfile(
            user_id=user_id,
            nickname="Gwen User",
            age=30,
            gender="prefer_not_to_say",
            height=170.0,
            weight=65.0,
            fitness_level="unknown",
            goal="保持健康",
            training_frequency=0,
            diet_preference="均衡",
            lifestyle="未填写",
            city=payload.city.strip(),
        )
        db.add(profile)
    else:
        profile.city = payload.city.strip()
        db.add(profile)
    db.flush()
    for memory_key in ("weather_location", "city"):
        memory = db.scalar(
            select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.memory_key == memory_key,
            )
        )
        if memory is None:
            db.add(
                UserMemory(
                    user_id=user_id,
                    memory_key=memory_key,
                    memory_value=payload.city.strip(),
                )
            )
        else:
            memory.memory_value = payload.city.strip()
            db.add(memory)
    db.flush()
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