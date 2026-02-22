from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.dependencies import get_session, get_analysis_service
from app.infra.persistence.session_store import Session
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/qca", tags=["QCA"])


class QCARequest(BaseModel):
    frame_index: int
    method: str = "gaussian"
    num_points: int | None = None


@router.post("/calculate")
async def calculate_qca(
    request: QCARequest,
    session: Session = Depends(get_session),
    service: AnalysisService = Depends(get_analysis_service),
):
    return service.compute_qca(session, request.frame_index, request.method, request.num_points)


@router.get("/measurements/{frame_index}")
async def get_measurements(
    frame_index: int,
    session: Session = Depends(get_session),
):
    result = session.qca_measurements.get(frame_index)
    if result is None:
        raise HTTPException(404, detail={"code": "NO_QCA", "message": f"No QCA for frame {frame_index}"})
    return result
