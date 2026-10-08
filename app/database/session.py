from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.database.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the process-wide SQLAlchemy engine from application settings."""
    settings = get_settings()
    connect_args = (
        {"check_same_thread": False} if settings.database_url.startswith("sqlite:") else {}
    )
    return create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Create the process-wide SQLAlchemy session factory."""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create database tables for the current phase.

    Phase 2.1 intentionally uses `create_all`; Alembic migrations can be added
    when the data model starts evolving across deployed environments.
    """
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency that commits successful requests and rolls back errors."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()