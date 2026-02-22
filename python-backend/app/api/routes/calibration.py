from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.api.dependencies import get_session, get_analysis_service
from app.infra.persistence.session_store import Session
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/calibration", tags=["Calibration"])


class CatheterCalibrationRequest(BaseModel):
    catheter_diameter_px: float
    catheter_size_fr: int


class CatheterFromSegmentationRequest(BaseModel):
    frame_index: int
    catheter_size_fr: int


class ManualCalibrationRequest(BaseModel):
    known_distance_mm: float
    measured_distance_px: float


@router.post("/catheter")
async def calibrate_catheter(
    request: CatheterCalibrationRequest,
    session: Session = Depends(get_session),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.calibrate_catheter(session, request.catheter_diameter_px, request.catheter_size_fr)


@router.post("/catheter-from-segmentation")
async def calibrate_catheter_from_segmentation(
    request: CatheterFromSegmentationRequest,
    session: Session = Depends(get_session),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.calibrate_catheter_from_segmentation(
        session, request.frame_index, request.catheter_size_fr
    )


@router.post("/manual")
async def calibrate_manual(
    request: ManualCalibrationRequest,
    session: Session = Depends(get_session),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.calibrate_manual(session, request.known_distance_mm, request.measured_distance_px)


@router.get("/current")
async def get_calibration(session: Session = Depends(get_session)):
    if session.calibration:
        return session.calibration.model_dump()
    if session.study and session.study.pixel_spacing:
        return session.study.pixel_spacing.model_dump()
    return {"row_spacing": 0.1, "col_spacing": 0.1, "source": "default", "confidence": 0.0}
