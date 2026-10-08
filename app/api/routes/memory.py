"""Gwen Memory Center API: view, update and reset long-term user memory.

This is a thin management layer over the existing user_memory table. It
deliberately does not touch the app.memory service or repository, so the
Agent memory flow stays exactly as it is.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import UserMemory
from app.database.session import get_db
from app.schemas.memory import (
    MemoryCenterResponse,
    MemoryItemResponse,
    MemoryKey,
    MemoryResetResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])

DbSession = Annotated[Session, Depends(get_db)]


@router.get("/{user_id}", response_model=MemoryCenterResponse)
def get_memory_center(user_id: str, db: DbSession) -> MemoryCenterResponse:
    """Return every long-term memory item Gwen currently holds for a user."""
    rows = db.scalars(
        select(UserMemory)
        .where(UserMemory.user_id == user_id)
        .order_by(UserMemory.updated_at.desc(), UserMemory.id.desc())
    ).all()
    memories = [MemoryItemResponse.model_validate(row) for row in rows]
    return MemoryCenterResponse(
        user_id=user_id,
        memories=memories,
        count=len(memories),
        last_updated_at=max((item.updated_at for item in memories), default=None),
    )


@router.put("/{user_id}/{memory_key}", response_model=MemoryUpdateResponse)
def update_memory_item(
    user_id: str,
    memory_key: MemoryKey,
    payload: MemoryUpdateRequest,
    db: DbSession,
) -> UserMemory:
    """Create or replace one long-term memory value for a user."""
    row = db.scalar(
        select(UserMemory).where(
            UserMemory.user_id == user_id,
            UserMemory.memory_key == memory_key,
        )
    )
    if row is None:
        # Match MemoryService.remember(): long-term memory is keyed by user_id
        # alone and must work for web users that have no user_profiles row.
        row = UserMemory(
            user_id=user_id,
            memory_key=memory_key,
            memory_value=payload.memory_value,
        )
        db.add(row)
    else:
        row.memory_value = payload.memory_value
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="记忆保存失败，请稍后重试",
        ) from exc
    return row


@router.delete("/{user_id}", response_model=MemoryResetResponse)
def reset_memory_center(user_id: str, db: DbSession) -> MemoryResetResponse:
    """Clear every long-term memory item so the experience can start fresh."""
    result = db.execute(delete(UserMemory).where(UserMemory.user_id == user_id))
    return MemoryResetResponse(
        user_id=user_id,
        reset_count=int(result.rowcount or 0),
    )
