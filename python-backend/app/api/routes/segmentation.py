from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.api.dependencies import get_session, get_segmentation_service
from app.infra.persistence.session_store import Session
from app.services.segmentation_service import SegmentationService
from app.models.enums import SegmentationEngine

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])


class SegmentRequest(BaseModel):
    frame_index: int
    engine: SegmentationEngine = SegmentationEngine.NNUNET
    roi: list[int] | None = None  # [x, y, w, h]
    seed_points: list[list[int]] | None = None  # [[x, y], ...]


@router.post("/segment")
async def segment_frame(
    request: SegmentRequest,
    session: Session = Depends(get_session),
    service: SegmentationService = Depends(get_segmentation_service),
):
    roi = tuple(request.roi) if request.roi and len(request.roi) == 4 else None
    seeds = [tuple(s) for s in request.seed_points] if request.seed_points else None

    result = service.segment_frame(
        session,
        request.frame_index,
        request.engine,
        roi=roi,
        seed_points=seeds,
    )
    return result


@router.post("/segment-and-extract")
async def segment_and_extract(
    request: SegmentRequest,
    session: Session = Depends(get_session),
    service: SegmentationService = Depends(get_segmentation_service),
):
    roi = tuple(request.roi) if request.roi and len(request.roi) == 4 else None
    seeds = [tuple(s) for s in request.seed_points] if request.seed_points else None

    result = service.segment_and_extract(
        session,
        request.frame_index,
        request.engine,
        roi=roi,
        seed_points=seeds,
    )
    return result


@router.get("/engines")
async def get_engines(
    service: SegmentationService = Depends(get_segmentation_service),
):
    return {"engines": service.get_available_engines()}


@router.get("/mask/{frame_index}")
async def get_mask(
    frame_index: int,
    session: Session = Depends(get_session),
):
    """Get segmentation mask as PNG binary."""
    seg_data = session.segmentations.get(frame_index)
    if seg_data is None:
        raise HTTPException(
            404,
            detail={
                "code": "NO_SEGMENTATION",
                "message": f"No segmentation for frame {frame_index}",
            },
        )

    from app.infra.dicom_handler import DicomHandler

    mask = seg_data["mask"]
    png_bytes = DicomHandler.frame_to_png(mask)
    return Response(content=png_bytes, media_type="image/png")
