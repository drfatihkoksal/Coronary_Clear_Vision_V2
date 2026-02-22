from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.api.dependencies import get_session
from app.infra.persistence.session_store import Session
from app.core.mask_ops import (
    apply_brush,
    apply_smart_brush,
    apply_flood_fill,
    apply_morphological_op,
)
from app.infra.dicom_handler import DicomHandler

router = APIRouter(prefix="/mask-edit", tags=["Mask Edit"])


class BrushRequest(BaseModel):
    frame_index: int
    points: list[list[int]]  # [[x, y], ...]
    radius: int = 5
    is_erasing: bool = False


class SmartBrushRequest(BaseModel):
    frame_index: int
    points: list[list[int]]
    radius: int = 5
    tolerance: int = 30
    is_erasing: bool = False


class FloodFillRequest(BaseModel):
    frame_index: int
    seed_point: list[int]  # [x, y]
    tolerance: int = 0


class MorphologicalRequest(BaseModel):
    frame_index: int
    operation: str  # dilate, erode, open, close
    kernel_size: int = 3
    iterations: int = 1


def _get_mask(session: Session, frame_index: int):
    seg = session.segmentations.get(frame_index)
    if seg is None:
        raise HTTPException(
            404,
            detail={
                "code": "NO_SEGMENTATION",
                "message": f"No segmentation for frame {frame_index}",
            },
        )
    return seg["mask"]


def _set_mask(session: Session, frame_index: int, mask):
    session.segmentations[frame_index]["mask"] = mask


@router.post("/brush")
async def brush(request: BrushRequest, session: Session = Depends(get_session)):
    mask = _get_mask(session, request.frame_index)
    value = 0 if request.is_erasing else 255
    points = [(p[0], p[1]) for p in request.points]
    new_mask = apply_brush(mask, points, request.radius, value)
    _set_mask(session, request.frame_index, new_mask)
    png = DicomHandler.frame_to_png(new_mask)
    return Response(content=png, media_type="image/png")


@router.post("/smart-brush")
async def smart_brush(request: SmartBrushRequest, session: Session = Depends(get_session)):
    mask = _get_mask(session, request.frame_index)
    if session.frames is None or request.frame_index >= len(session.frames):
        raise HTTPException(
            400,
            detail={"code": "NO_FRAME", "message": "Frame not available"},
        )
    image = session.frames[request.frame_index]
    value = 0 if request.is_erasing else 255
    points = [(p[0], p[1]) for p in request.points]
    new_mask = apply_smart_brush(mask, image, points, request.radius, request.tolerance, value)
    _set_mask(session, request.frame_index, new_mask)
    png = DicomHandler.frame_to_png(new_mask)
    return Response(content=png, media_type="image/png")


@router.post("/flood-fill")
async def flood_fill(request: FloodFillRequest, session: Session = Depends(get_session)):
    mask = _get_mask(session, request.frame_index)
    seed = (request.seed_point[0], request.seed_point[1])
    new_mask = apply_flood_fill(mask, seed, tolerance=request.tolerance)
    _set_mask(session, request.frame_index, new_mask)
    png = DicomHandler.frame_to_png(new_mask)
    return Response(content=png, media_type="image/png")


@router.post("/morphological")
async def morphological(request: MorphologicalRequest, session: Session = Depends(get_session)):
    mask = _get_mask(session, request.frame_index)
    new_mask = apply_morphological_op(mask, request.operation, request.kernel_size, request.iterations)
    _set_mask(session, request.frame_index, new_mask)
    png = DicomHandler.frame_to_png(new_mask)
    return Response(content=png, media_type="image/png")
