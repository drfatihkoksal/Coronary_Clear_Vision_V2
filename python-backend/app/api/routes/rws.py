from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.dependencies import get_session
from app.infra.persistence.session_store import Session
from app.core.rws_calculator import calculate_rws
from app.models.enums import OutlierMethod

router = APIRouter(prefix="/rws", tags=["RWS"])


class RWSRequest(BaseModel):
    start_frame: int
    end_frame: int
    outlier_method: OutlierMethod = OutlierMethod.HAMPEL
    vessel: str | None = None


@router.post("/calculate")
async def calculate(
    request: RWSRequest,
    session: Session = Depends(get_session),
):
    """Calculate RWS for a frame range."""
    # Build diameter profiles from QCA measurements
    diameter_profiles = {}
    for frame_idx, qca in session.qca_measurements.items():
        if isinstance(qca, dict) and "diameter_profile_mm" in qca:
            diameter_profiles[frame_idx] = qca["diameter_profile_mm"]

    if not diameter_profiles:
        raise HTTPException(400, detail={"code": "NO_QCA", "message": "No QCA measurements available. Segment and run QCA first."})

    result = calculate_rws(
        diameter_profiles,
        request.start_frame,
        request.end_frame,
        request.outlier_method,
        request.vessel,
    )

    # Store result
    session.rws_results.append(result)

    return result


@router.get("/results")
async def get_results(session: Session = Depends(get_session)):
    """Get all RWS results for this session."""
    return {"results": session.rws_results}


@router.delete("/results/{index}")
async def delete_result(
    index: int,
    session: Session = Depends(get_session),
):
    """Delete an RWS result by index."""
    if index < 0 or index >= len(session.rws_results):
        raise HTTPException(404, detail={"code": "NOT_FOUND", "message": f"RWS result {index} not found"})
    session.rws_results.pop(index)
    return {"status": "deleted"}
