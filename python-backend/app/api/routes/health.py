import time

from fastapi import APIRouter, Request

from app.config.settings import settings
from app.models.responses import DetailedHealthResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    start_time: float = request.app.state.start_time
    uptime = time.time() - start_time
    return HealthResponse(status="ok", version=settings.version, uptime_seconds=round(uptime, 2))


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def health_detailed(request: Request) -> DetailedHealthResponse:
    start_time: float = request.app.state.start_time
    uptime = time.time() - start_time

    engine_registry = getattr(request.app.state, "engine_registry", None)
    engines = engine_registry.available_engines() if engine_registry else {}

    return DetailedHealthResponse(
        status="ok",
        version=settings.version,
        uptime_seconds=round(uptime, 2),
        engines=engines,
        database={"connected": False},
    )
