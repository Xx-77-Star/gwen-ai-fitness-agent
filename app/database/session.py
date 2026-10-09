from collections.abc import Iterator
from contextlib import suppress
from functools import lru_cache

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import OperationalError
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
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "sqlite":
        inspector = inspect(engine)
        if "user_profiles" in inspector.get_table_names():
            columns = {item["name"] for item in inspector.get_columns("user_profiles")}
            if "city" not in columns:
                with engine.begin() as connection, suppress(OperationalError):
                    connection.execute(
                        text(
                            "ALTER TABLE user_profiles ADD COLUMN city "
                            "VARCHAR(100) NOT NULL DEFAULT ''"
                        )
                    )


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