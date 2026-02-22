"""QFR (Quantitative Flow Ratio) API routes."""
import logging

import numpy as np
from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.api.dependencies import get_dicom_handler, get_segmentation_service, get_session, get_session_store, get_qfr_service
from app.infra.dicom_handler import DicomHandler
from app.infra.persistence.session_store import Session, SessionStore
from app.services.qfr_service import QFRService, QFRSession, QFRProjection
from app.services.segmentation_service import SegmentationService
from app.models.enums import SegmentationEngine
from app.core.calibration import calibrate_from_mask, CATHETER_SIZES_FR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qfr", tags=["QFR"])


class SegmentProjectionRequest(BaseModel):
    projection_id: int  # 1 or 2
    frame_index: int = 0  # Which frame to segment (must match where seed points were placed)
    engine: SegmentationEngine = SegmentationEngine.NNUNET
    roi: list[int] | None = None  # [x, y, w, h]
    seed_points: list[list[int]] | None = None


class CalibrateProjectionRequest(BaseModel):
    projection_id: int
    catheter_size_fr: int
    catheter_diameter_px: float


class SetTimiFramesRequest(BaseModel):
    projection_id: int  # 1 or 2
    timi_start: int
    timi_end: int


class CalibrateFromSegmentationRequest(BaseModel):
    projection_id: int  # 1 or 2
    catheter_size_fr: int = 6


class ReconstructRequest(BaseModel):
    mode: str = "fQFR"
    kt: float = 1.52
    timi_start_p1: int | None = None
    timi_end_p1: int | None = None
    timi_start_p2: int | None = None
    timi_end_p2: int | None = None


class RecalculateRequest(BaseModel):
    stenosis_indices: list[int]
    mode: str = "fQFR"
    kt: float = 1.52


def _ensure_qfr_session(session: Session) -> QFRSession:
    if session.qfr_session is None:
        session.qfr_session = QFRSession()
    return session.qfr_session


def _get_projection(qfr_session: QFRSession, proj_id: int) -> QFRProjection:
    if proj_id == 1:
        if qfr_session.projection1 is None:
            raise ValueError("Projection 1 not loaded")
        return qfr_session.projection1
    elif proj_id == 2:
        if qfr_session.projection2 is None:
            raise ValueError("Projection 2 not loaded")
        return qfr_session.projection2
    else:
        raise ValueError("projection_id must be 1 or 2")


@router.post("/projection/upload")
async def upload_projection(
    projection_id: int = Query(..., ge=1, le=2),
    file: UploadFile = File(...),
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    store: SessionStore = Depends(get_session_store),
    dicom_handler: DicomHandler = Depends(get_dicom_handler),
):
    """Upload a DICOM projection for QFR analysis.

    Creates a new session if X-Session-ID is not provided or session not found.
    """
    data = await file.read()
    ds, study, frames = dicom_handler.load_from_bytes(data, anonymize=True)

    # Get or create session
    session = store.get(x_session_id) if x_session_id else None
    if session is None:
        session = store.create()

    qfr_session = _ensure_qfr_session(session)

    # Extract projection-specific DICOM metadata
    angle_deg = float(getattr(ds, "PositionerPrimaryAngle", 0))
    secondary_angle_deg = float(getattr(ds, "PositionerSecondaryAngle", 0))
    sod = float(getattr(ds, "DistanceSourceToPatient", 1000))
    sid = float(getattr(ds, "DistanceSourceToDetector", 1000))

    pixel_spacing = 0.3
    if study.pixel_spacing:
        pixel_spacing = study.pixel_spacing.row_spacing

    # Log pixel spacing diagnostics for magnification debugging
    raw_imager_ps = getattr(ds, "ImagerPixelSpacing", None)
    raw_pixel_ps = getattr(ds, "PixelSpacing", None)
    logger.info(
        "QFR Projection %d pixel spacing: final=%.4fmm source=%s | "
        "ImagerPixelSpacing=%s PixelSpacing=%s | "
        "SOD=%.1fmm SID=%.1fmm mag_factor=%.3f | "
        "angles=%.1f°/%.1f°",
        projection_id,
        pixel_spacing,
        study.pixel_spacing.source if study.pixel_spacing else "default",
        [float(x) for x in raw_imager_ps] if raw_imager_ps else "N/A",
        [float(x) for x in raw_pixel_ps] if raw_pixel_ps else "N/A",
        sod, sid,
        sod / sid if sid > 0 else 0,
        angle_deg, secondary_angle_deg,
    )

    # Extract cine frame rate from DICOM (try multiple tags)
    frame_rate = 15.0
    cine_rate = getattr(ds, "CineRate", None)
    if cine_rate is not None:
        frame_rate = float(cine_rate)
    else:
        recommended_fps = getattr(ds, "RecommendedDisplayFrameRate", None)
        if recommended_fps is not None:
            frame_rate = float(recommended_fps)
        else:
            frame_time = getattr(ds, "FrameTime", None)
            if frame_time is not None and float(frame_time) > 0:
                frame_rate = 1000.0 / float(frame_time)

    proj = QFRProjection(
        frames=frames,
        num_frames=study.num_frames,
        image_width=study.image_width,
        image_height=study.image_height,
        pixel_spacing=pixel_spacing,
        angle_deg=angle_deg,
        secondary_angle_deg=secondary_angle_deg,
        sod=sod,
        sid=sid,
        frame_rate=frame_rate,
    )

    if projection_id == 1:
        qfr_session.projection1 = proj
    else:
        qfr_session.projection2 = proj

    # Clear previous results when new projection uploaded
    qfr_session.result_3d = None
    qfr_session.mesh = None
    qfr_session.qfr_result = None

    return {
        "session_id": session.id,
        "projection_id": projection_id,
        "num_frames": study.num_frames,
        "image_width": study.image_width,
        "image_height": study.image_height,
        "angle_deg": angle_deg,
        "secondary_angle_deg": secondary_angle_deg,
        "pixel_spacing": pixel_spacing,
        "sod": sod,
        "sid": sid,
        "frame_rate": frame_rate,
    }


@router.post("/projection/segment")
async def segment_projection(
    request: SegmentProjectionRequest,
    session: Session = Depends(get_session),
    seg_service: SegmentationService = Depends(get_segmentation_service),
):
    """Segment vessel in a QFR projection.

    Calls SegmentationService.segment_image() — the same core code used by
    the normal segmentation endpoints — to guarantee identical inference.
    """
    qfr_session = _ensure_qfr_session(session)
    proj = _get_projection(qfr_session, request.projection_id)

    if proj.frames is None or len(proj.frames) == 0:
        raise ValueError("No frames in projection")

    frame_idx = request.frame_index
    if frame_idx < 0 or frame_idx >= len(proj.frames):
        raise ValueError(f"Frame index {frame_idx} out of range (0-{len(proj.frames)-1})")

    image = proj.frames[frame_idx]
    roi = tuple(request.roi) if request.roi and len(request.roi) == 4 else None
    seeds = [tuple(s) for s in request.seed_points] if request.seed_points else None

    # Call the same service method used by /segmentation/segment
    # No num_centerline_points override — use natural resolution from mask,
    # same as the main segmentation flow. Epipolar matching handles resampling.
    result = seg_service.segment_image(
        image, request.engine, roi=roi, seed_points=seeds,
    )

    # Store results in QFR projection state
    proj.centerline = result["centerline"]
    proj.diameters_px = result["diameters_px"]
    proj.diameters_mm = [d * proj.pixel_spacing for d in proj.diameters_px]
    proj.mask = result["mask"]

    return {
        "projection_id": request.projection_id,
        "confidence": result["confidence"],
        "centerline": [{"x": p[0], "y": p[1]} for p in result["centerline"]],
        "diameters_mm": proj.diameters_mm,
        "num_points": len(result["centerline"]),
    }


@router.post("/projection/calibrate")
async def calibrate_projection(
    request: CalibrateProjectionRequest,
    session: Session = Depends(get_session),
):
    """Calibrate pixel spacing for a QFR projection using catheter reference."""
    qfr_session = _ensure_qfr_session(session)
    proj = _get_projection(qfr_session, request.projection_id)

    # French size to mm: 1 Fr = 1/3 mm
    catheter_mm = request.catheter_size_fr / 3.0
    new_spacing = catheter_mm / request.catheter_diameter_px

    proj.pixel_spacing = new_spacing

    # Recalculate diameters from original pixel values
    if proj.diameters_px:
        proj.diameters_mm = [d * new_spacing for d in proj.diameters_px]

    return {
        "projection_id": request.projection_id,
        "pixel_spacing": new_spacing,
        "catheter_size_fr": request.catheter_size_fr,
        "catheter_mm": catheter_mm,
    }


@router.post("/projection/set-timi-frames")
async def set_timi_frames(
    request: SetTimiFramesRequest,
    session: Session = Depends(get_session),
):
    """Set TIMI T0/T-End frames for a QFR projection."""
    qfr_session = _ensure_qfr_session(session)
    proj = _get_projection(qfr_session, request.projection_id)

    if request.timi_start < 0 or request.timi_end < 0:
        raise ValueError("TIMI frame indices must be non-negative")
    if request.timi_start >= proj.num_frames or request.timi_end >= proj.num_frames:
        raise ValueError("TIMI frame indices out of range")
    if request.timi_start >= request.timi_end:
        raise ValueError("TIMI start must be before end")

    proj.timi_start = request.timi_start
    proj.timi_end = request.timi_end

    # Clear previous QFR result since flow parameters changed
    qfr_session.qfr_result = None

    return {
        "projection_id": request.projection_id,
        "timi_start": proj.timi_start,
        "timi_end": proj.timi_end,
        "timi_frame_count": proj.timi_end - proj.timi_start,
    }


@router.post("/projection/calibrate-from-segmentation")
async def calibrate_from_segmentation(
    request: CalibrateFromSegmentationRequest,
    session: Session = Depends(get_session),
):
    """Calibrate pixel spacing using segmentation mask as catheter reference."""
    qfr_session = _ensure_qfr_session(session)
    proj = _get_projection(qfr_session, request.projection_id)

    if proj.mask is None:
        raise ValueError("No segmentation mask available. Segment the projection first.")

    catheter_mm = CATHETER_SIZES_FR.get(request.catheter_size_fr)
    if catheter_mm is None:
        raise ValueError(f"Unknown catheter size: {request.catheter_size_fr}Fr")

    pixel_spacing = calibrate_from_mask(proj.mask, catheter_mm)

    # Quality score based on mask measurement consistency
    binary = (proj.mask > 127).astype(bool)
    row_widths = binary.sum(axis=1)
    nonzero_widths = row_widths[row_widths > 0]
    quality_score = 1.0 - float(np.std(nonzero_widths) / (np.mean(nonzero_widths) + 1e-6)) if len(nonzero_widths) > 0 else 0.0
    quality_score = max(0.0, min(1.0, quality_score))

    # Measure diameter used
    measured_diameter_px = catheter_mm / pixel_spacing if pixel_spacing > 0 else 0.0

    proj.pixel_spacing = pixel_spacing

    # Recalculate diameters from original pixel values
    if proj.diameters_px:
        proj.diameters_mm = [d * pixel_spacing for d in proj.diameters_px]

    return {
        "projection_id": request.projection_id,
        "pixel_spacing": round(pixel_spacing, 6),
        "catheter_size_fr": request.catheter_size_fr,
        "measured_diameter_px": round(measured_diameter_px, 2),
        "quality_score": round(quality_score, 3),
    }


@router.get("/projection/mask/{frame_index}")
async def get_projection_mask(
    frame_index: int,
    projection_id: int = Query(..., ge=1, le=2),
    session: Session = Depends(get_session),
):
    """Get segmentation mask for a QFR projection frame as PNG binary."""
    qfr_session = session.qfr_session
    if qfr_session is None:
        raise ValueError("No QFR session")

    proj = _get_projection(qfr_session, projection_id)

    if proj.mask is None:
        raise ValueError("No segmentation mask available")

    # Currently we store a single mask (frame 0 segmentation)
    # Return it for any frame_index request
    png_bytes = DicomHandler.frame_to_png(proj.mask)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Frame-Index": str(frame_index),
            "X-Projection-ID": str(projection_id),
            "Cache-Control": "private, max-age=60",
        },
    )


@router.post("/reconstruct")
async def reconstruct(
    request: ReconstructRequest,
    session: Session = Depends(get_session),
    qfr_service: QFRService = Depends(get_qfr_service),
):
    """Run 3D reconstruction and QFR calculation."""
    # Apply TIMI frames from request to projections if provided
    qfr_session = _ensure_qfr_session(session)
    if request.timi_start_p1 is not None and request.timi_end_p1 is not None:
        if qfr_session.projection1:
            qfr_session.projection1.timi_start = request.timi_start_p1
            qfr_session.projection1.timi_end = request.timi_end_p1
    if request.timi_start_p2 is not None and request.timi_end_p2 is not None:
        if qfr_session.projection2:
            qfr_session.projection2.timi_start = request.timi_start_p2
            qfr_session.projection2.timi_end = request.timi_end_p2

    result = qfr_service.reconstruct_and_calculate(session, mode=request.mode, kt=request.kt)
    return result


@router.post("/recalculate")
async def recalculate(
    request: RecalculateRequest,
    session: Session = Depends(get_session),
    qfr_service: QFRService = Depends(get_qfr_service),
):
    """Recalculate QFR with user-specified stenosis positions.

    Uses cached 3D data — no re-reconstruction needed.
    """
    result = qfr_service.recalculate_qfr(
        session, request.stenosis_indices, request.mode, request.kt,
    )
    return result


@router.get("/result")
async def get_result(session: Session = Depends(get_session)):
    """Get the last QFR result."""
    qfr_session = session.qfr_session
    if qfr_session is None or qfr_session.qfr_result is None:
        return {"qfr": None, "message": "No QFR result available"}

    return {
        "qfr": qfr_session.qfr_result,
        "mesh": qfr_session.mesh,
        "result_3d": qfr_session.result_3d,
    }


@router.get("/diagnostics")
async def get_diagnostics(session: Session = Depends(get_session)):
    """Get detailed QFR pipeline diagnostics for debugging.

    Returns per-view diameter breakdowns, reference/MLD values,
    pixel spacings, and view disagreement metrics.
    """
    qfr_session = session.qfr_session
    if qfr_session is None or qfr_session.qfr_result is None:
        return {"error": "No QFR result available. Run reconstruct first."}

    p1, p2 = qfr_session.projection1, qfr_session.projection2
    if p1 is None or p2 is None:
        return {"error": "Both projections required"}

    result = qfr_session.qfr_result
    diameters_3d = qfr_session.result_3d.get("diameters_3d", []) if qfr_session.result_3d else []

    # Re-measure individual view diameters for diagnostics
    d1_mm_list = [d * p1.pixel_spacing for d in p1.diameters_px] if p1.diameters_px else []
    d2_mm_list = [d * p2.pixel_spacing for d in p2.diameters_px] if p2.diameters_px else []

    return {
        "qfr": result.get("qfr"),
        "mode": result.get("mode"),
        "vessel_length_mm": result.get("vessel_length_mm"),
        "reference_diameter_mm": result.get("reference_diameter_mm"),
        "mld_mm": result.get("mld_mm"),
        "num_lesions": result.get("num_lesions"),
        "lesions": result.get("lesions"),
        "v_hyp_m_s": result.get("v_hyp_m_s"),
        "v_rest_m_s": result.get("v_rest_m_s"),
        "timi_frame_count": result.get("timi_frame_count"),
        "frame_rate": result.get("frame_rate"),
        "angular_separation": qfr_session.result_3d.get("angular_separation", 0) if qfr_session.result_3d else 0,
        "projection1": {
            "pixel_spacing": p1.pixel_spacing,
            "angle_deg": p1.angle_deg,
            "secondary_angle_deg": p1.secondary_angle_deg,
            "sod": p1.sod,
            "sid": p1.sid,
            "num_centerline_pts": len(p1.centerline),
            "num_diameter_pts": len(p1.diameters_px) if p1.diameters_px else 0,
            "diameters_mm_summary": {
                "min": round(min(d1_mm_list), 3) if d1_mm_list else 0,
                "max": round(max(d1_mm_list), 3) if d1_mm_list else 0,
                "mean": round(sum(d1_mm_list) / len(d1_mm_list), 3) if d1_mm_list else 0,
            },
            "has_mask": p1.mask is not None,
            "timi_start": p1.timi_start,
            "timi_end": p1.timi_end,
        },
        "projection2": {
            "pixel_spacing": p2.pixel_spacing,
            "angle_deg": p2.angle_deg,
            "secondary_angle_deg": p2.secondary_angle_deg,
            "sod": p2.sod,
            "sid": p2.sid,
            "num_centerline_pts": len(p2.centerline),
            "num_diameter_pts": len(p2.diameters_px) if p2.diameters_px else 0,
            "diameters_mm_summary": {
                "min": round(min(d2_mm_list), 3) if d2_mm_list else 0,
                "max": round(max(d2_mm_list), 3) if d2_mm_list else 0,
                "mean": round(sum(d2_mm_list) / len(d2_mm_list), 3) if d2_mm_list else 0,
            },
            "has_mask": p2.mask is not None,
            "timi_start": p2.timi_start,
            "timi_end": p2.timi_end,
        },
        "combined_diameters_mm": [round(d, 3) for d in diameters_3d] if diameters_3d else [],
        "pressure_profile": result.get("pressure_profile", []),
        "diameters_mm_profile": result.get("diameters_mm", []),
    }


@router.get("/projection/frame/{frame_index}")
async def get_projection_frame(
    frame_index: int,
    projection_id: int = Query(..., ge=1, le=2),
    session: Session = Depends(get_session),
):
    """Get a frame from a QFR projection as PNG binary."""
    qfr_session = session.qfr_session
    if qfr_session is None:
        raise ValueError("No QFR session")

    proj = _get_projection(qfr_session, projection_id)

    if proj.frames is None or frame_index < 0 or frame_index >= len(proj.frames):
        raise IndexError(f"Frame index {frame_index} out of range")

    png_bytes = DicomHandler.frame_to_png(proj.frames[frame_index])
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Frame-Index": str(frame_index),
            "X-Projection-ID": str(projection_id),
            "Cache-Control": "private, max-age=3600",
            "Vary": "X-Session-ID",
        },
    )
