from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness endpoint for deployment and local verification."""
    return {"status": "ok"}