from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_session, get_tracking_service
from app.infra.persistence.session_store import Session
from app.models.enums import TrackingDirection
from app.services.tracking_service import TrackingService

router = APIRouter(prefix="/tracking", tags=["Tracking"])


class InitializeRequest(BaseModel):
    frame_index: int
    roi: list[int]  # [x, y, w, h]


class PropagateRequest(BaseModel):
    direction: TrackingDirection = TrackingDirection.FORWARD
    max_frames: int | None = None
    auto_segment: bool = False


@router.post("/initialize")
async def initialize_tracking(
    request: InitializeRequest,
    session: Session = Depends(get_session),
    service: TrackingService = Depends(get_tracking_service),
):
    if len(request.roi) != 4:
        raise HTTPException(
            400,
            detail={
                "code": "INVALID_ROI",
                "message": "ROI must be [x, y, w, h]",
            },
        )
    roi = (request.roi[0], request.roi[1], request.roi[2], request.roi[3])
    try:
        return service.initialize(session, request.frame_index, roi)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, detail={"code": "INIT_FAILED", "message": str(exc)})


@router.post("/propagate")
async def propagate_tracking(
    request: PropagateRequest,
    session: Session = Depends(get_session),
    service: TrackingService = Depends(get_tracking_service),
):
    try:
        return service.propagate(
            session,
            direction=request.direction,
            max_frames=request.max_frames,
            auto_segment=request.auto_segment,
        )
    except ValueError as exc:
        raise HTTPException(400, detail={"code": "PROPAGATE_FAILED", "message": str(exc)})


@router.get("/state")
async def get_tracking_state(
    session: Session = Depends(get_session),
    service: TrackingService = Depends(get_tracking_service),
):
    return service.get_state(session)


@router.delete("/clear")
async def clear_tracking(
    session: Session = Depends(get_session),
    service: TrackingService = Depends(get_tracking_service),
):
    return service.clear(session)
