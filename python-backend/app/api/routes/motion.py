from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.dependencies import get_session
from app.infra.persistence.session_store import Session
from app.core.motion_analyzer import compute_motion_signal, detect_motion_peaks

router = APIRouter(prefix="/motion", tags=["Motion"])


@router.post("/calculate")
async def calculate_motion(session: Session = Depends(get_session)):
    """Calculate motion signal from all frames."""
    if session.frames is None:
        raise HTTPException(400, detail={"code": "NO_STUDY", "message": "No study loaded"})

    frame_rate = session.study.frame_rate if session.study else 15.0

    signal = compute_motion_signal(session.frames)
    peaks = detect_motion_peaks(signal, frame_rate=frame_rate)

    result = {
        "signal": signal,
        "peaks": peaks,
        "num_frames": len(signal),
    }
    session.motion_signal = result
    return result


@router.get("/signal")
async def get_motion_signal(session: Session = Depends(get_session)):
    if session.motion_signal is None:
        raise HTTPException(404, detail={"code": "NO_MOTION", "message": "Motion signal not calculated"})
    return session.motion_signal


class MotionPeaksUpdate(BaseModel):
    peaks: list[int]


@router.post("/peaks")
async def set_motion_peaks(
    request: MotionPeaksUpdate,
    session: Session = Depends(get_session),
):
    if session.motion_signal is None:
        raise HTTPException(400, detail={"code": "NO_MOTION", "message": "Calculate motion first"})
    session.motion_signal["peaks"] = sorted(request.peaks)
    return session.motion_signal
