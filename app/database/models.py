from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for database records."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for FitLife AI database models."""


class UserProfile(Base):
    """Persistent fitness profile for one user."""

    __tablename__ = "user_profiles"
    __table_args__ = (
        CheckConstraint("age >= 1 AND age <= 120", name="ck_user_profiles_age_range"),
        CheckConstraint("height >= 50 AND height <= 250", name="ck_user_profiles_height_range"),
        CheckConstraint("weight >= 20 AND weight <= 500", name="ck_user_profiles_weight_range"),
        CheckConstraint(
            "training_frequency >= 0 AND training_frequency <= 14",
            name="ck_user_profiles_training_frequency_range",
        ),
        CheckConstraint(
            "fitness_level IN ('beginner', 'intermediate', 'advanced', 'unknown')",
            name="ck_user_profiles_fitness_level",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    nickname: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(32), nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    fitness_level: Mapped[str] = mapped_column(String(32), nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    training_frequency: Mapped[int] = mapped_column(Integer, nullable=False)
    diet_preference: Mapped[str] = mapped_column(String(200), nullable=False)
    lifestyle: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<UserProfile id={self.id} user_id={self.user_id!r}>"


class UserMemory(Base):
    """One durable key/value memory item for a user."""

    __tablename__ = "user_memory"
    __table_args__ = (
        CheckConstraint("length(memory_key) > 0", name="ck_user_memory_key_not_empty"),
        CheckConstraint("length(memory_value) > 0", name="ck_user_memory_value_not_empty"),
        CheckConstraint("created_at <= updated_at", name="ck_user_memory_timestamps"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("user_profiles.user_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    memory_key: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )


class ConversationTurn(Base):
    """One persisted user or assistant turn for cross-request conversation memory."""

    __tablename__ = "conversation_turns"
    __table_args__ = (
        CheckConstraint("sequence > 0", name="ck_conversation_turns_sequence"),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="ck_conversation_turns_role",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("user_profiles.user_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )


class TrainingRecord(Base):
    """One completed or planned training session for a user."""

    __tablename__ = "training_records"
    __table_args__ = (
        CheckConstraint("duration_minutes > 0", name="ck_training_records_duration"),
        CheckConstraint("weight >= 0", name="ck_training_records_weight"),
        CheckConstraint("created_at <= updated_at", name="ck_training_records_timestamps"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("user_profiles.user_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    training_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    training_type: Mapped[str] = mapped_column(String(50), nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    intensity: Mapped[str] = mapped_column(String(32), nullable=False)
    body_parts: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    sets: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    reps: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    weight: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<TrainingRecord id={self.id} user_id={self.user_id!r}>"
