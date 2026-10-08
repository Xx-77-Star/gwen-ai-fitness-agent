from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.training_repository import SQLTrainingRecordRepository
from app.schemas.training import TrainingRecordCreate, TrainingRecordResponse

router = APIRouter(prefix="/training-records", tags=["training-records"])

DbSession = Annotated[Session, Depends(get_db)]


@router.post("", response_model=TrainingRecordResponse, status_code=status.HTTP_201_CREATED)
def create_training_record(payload: TrainingRecordCreate, db: DbSession) -> TrainingRecordResponse:
    """Persist one training record through the repository layer."""
    repository = SQLTrainingRecordRepository(db)
    try:
        return repository.create(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found",
        ) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training record could not be created",
        ) from exc


@router.get("/{record_id}", response_model=TrainingRecordResponse)
def get_training_record(record_id: int, user_id: str, db: DbSession) -> TrainingRecordResponse:
    """Return one training record for the requesting user."""
    record = SQLTrainingRecordRepository(db).get(record_id, user_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training record not found",
        )
    return record