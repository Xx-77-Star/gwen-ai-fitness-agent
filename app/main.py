from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.memory import router as memory_router
from app.api.routes.profile import router as profile_router
from app.api.routes.training import router as training_router
from app.api.routes.workouts import router as workout_router
from app.config.settings import get_settings
from app.database.session import init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Validate configuration and initialize the current database schema."""
    get_settings()
    init_db()
    yield


def create_app() -> FastAPI:
    """Application factory used by Uvicorn and tests."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.4.0",
        description="Production-style foundation for a fitness life-management Agent",
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(memory_router)
    application.include_router(chat_router)
    application.include_router(profile_router)
    application.include_router(training_router)
    application.include_router(workout_router)
    application.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
    return application


app = create_app()
