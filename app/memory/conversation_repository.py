"""Persistence for cross-request conversation history."""

from __future__ import annotations

from typing import Literal, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import ConversationTurn
from app.memory.models import ConversationTurnRecord

ConversationRole = Literal["user", "assistant", "system"]


class ConversationRepository(Protocol):
    """Conversation persistence contract used by Agent nodes and chat requests."""

    def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None = None,
    ) -> str:
        """Return a conversation identifier, creating its identity when needed."""
        ...

    def append_turn(
        self,
        conversation_id: str,
        user_id: str,
        role: ConversationRole,
        content: str,
    ) -> ConversationTurnRecord:
        """Append one turn and return its persisted record."""
        ...

    def list_recent_turns(
        self,
        conversation_id: str,
        user_id: str,
        *,
        limit: int,
    ) -> list[ConversationTurnRecord]:
        """Return the latest turns in chronological order."""
        ...


class SQLConversationRepository:
    """SQLAlchemy-backed cross-request conversation repository."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None = None,
    ) -> str:
        normalized_user_id = user_id.strip()
        normalized_conversation_id = (conversation_id or "").strip()
        if not normalized_user_id:
            raise ValueError("user_id is required")
        if not normalized_conversation_id:
            from uuid import uuid4

            normalized_conversation_id = str(uuid4())
        owner = self._session.scalar(
            select(ConversationTurn.user_id)
            .where(ConversationTurn.conversation_id == normalized_conversation_id)
            .limit(1)
        )
        if owner is not None and owner != normalized_user_id:
            raise ValueError("conversation does not belong to user")
        return normalized_conversation_id

    def append_turn(
        self,
        conversation_id: str,
        user_id: str,
        role: ConversationRole,
        content: str,
    ) -> ConversationTurnRecord:
        normalized_conversation_id = conversation_id.strip()
        normalized_user_id = user_id.strip()
        normalized_content = content.strip()
        if not normalized_conversation_id or not normalized_user_id or not normalized_content:
            raise ValueError("conversation_id, user_id, and content are required")
        owner = self._session.scalar(
            select(ConversationTurn.user_id)
            .where(ConversationTurn.conversation_id == normalized_conversation_id)
            .limit(1)
        )
        if owner is not None and owner != normalized_user_id:
            raise ValueError("conversation does not belong to user")
        next_sequence = int(
            self._session.scalar(
                select(func.coalesce(func.max(ConversationTurn.sequence), 0)).where(
                    ConversationTurn.conversation_id == normalized_conversation_id
                )
            )
            or 0
        ) + 1
        record = ConversationTurn(
            conversation_id=normalized_conversation_id,
            user_id=normalized_user_id,
            role=role,
            content=normalized_content,
            sequence=next_sequence,
        )
        self._session.add(record)
        self._session.flush()
        return ConversationTurnRecord.model_validate(record)

    def list_recent_turns(
        self,
        conversation_id: str,
        user_id: str,
        *,
        limit: int,
    ) -> list[ConversationTurnRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = self._session.scalars(
            select(ConversationTurn)
            .where(
                ConversationTurn.conversation_id == conversation_id,
                ConversationTurn.user_id == user_id,
            )
            .order_by(ConversationTurn.sequence.desc(), ConversationTurn.id.desc())
            .limit(limit)
        ).all()
        return [ConversationTurnRecord.model_validate(row) for row in reversed(rows)]
