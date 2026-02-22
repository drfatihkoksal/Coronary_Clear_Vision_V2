from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.dependencies import get_session
from app.infra.persistence.session_store import Session
from app.infra.ecg_parser import extract_ecg_from_dicom
from app.core.ecg_analyzer import (
    detect_r_peaks,
    compute_heart_rate,
    compute_beat_boundaries,
    add_r_peak,
    remove_r_peak,
    move_r_peak,
)

router = APIRouter(prefix="/ecg", tags=["ECG"])


def _build_ecg_response(session: Session) -> dict:
    """Build standard ECG response from session data."""
    data = session.ecg_data
    return {
        "signal": data["signal"],
        "sample_rate": data["sample_rate"],
        "num_samples": data["num_samples"],
        "r_peaks": data["r_peaks"],
        "heart_rate": data.get("heart_rate"),
        "beat_boundaries": data.get("beat_boundaries", []),
    }


def _recalculate_from_peaks(session: Session, new_peaks: list[int]) -> dict:
    """Update session R-peaks, recalculate HR and beats, return response."""
    data = session.ecg_data
    data["r_peaks"] = sorted(new_peaks)

    hr = compute_heart_rate(new_peaks, data["sample_rate"])
    data["heart_rate"] = round(hr, 1) if hr else None

    num_frames = session.study.num_frames if session.study else 0
    frame_rate = session.study.frame_rate if session.study else 15.0
    beats = compute_beat_boundaries(
        new_peaks, num_frames, frame_rate, data["sample_rate"],
    )
    data["beat_boundaries"] = beats

    return _build_ecg_response(session)


def _ensure_ecg_loaded(session: Session) -> None:
    """Ensure ECG data is loaded, extract from DICOM if needed."""
    if session.ecg_data is not None:
        return

    ds = getattr(session, 'dataset', None)
    if ds is None:
        raise HTTPException(404, detail={"code": "NO_ECG", "message": "No ECG data available"})

    import numpy as np
    ecg_raw = extract_ecg_from_dicom(ds)
    if ecg_raw is None:
        raise HTTPException(404, detail={"code": "NO_ECG", "message": "No ECG waveform in DICOM"})

    signal = np.array(ecg_raw["signal"])
    sample_rate = ecg_raw["sample_rate"]

    r_peaks = detect_r_peaks(signal, sample_rate)
    hr = compute_heart_rate(r_peaks, sample_rate)

    num_frames = session.study.num_frames if session.study else 0
    frame_rate = session.study.frame_rate if session.study else 15.0
    beats = compute_beat_boundaries(r_peaks, num_frames, frame_rate, sample_rate)

    session.ecg_data = {
        "signal": ecg_raw["signal"],
        "sample_rate": sample_rate,
        "num_samples": ecg_raw["num_samples"],
        "r_peaks": r_peaks,
        "heart_rate": round(hr, 1) if hr else None,
        "beat_boundaries": beats,
    }


@router.get("/signal")
async def get_ecg_signal(session: Session = Depends(get_session)):
    """Get ECG signal, R-peaks, and beat boundaries."""
    _ensure_ecg_loaded(session)
    return _build_ecg_response(session)


class RPeaksUpdate(BaseModel):
    r_peaks: list[int]


@router.post("/r-peaks")
async def set_r_peaks(
    request: RPeaksUpdate,
    session: Session = Depends(get_session),
):
    """Manually set/update all R-peak locations."""
    if session.ecg_data is None:
        raise HTTPException(400, detail={"code": "NO_ECG", "message": "Load ECG first"})

    return _recalculate_from_peaks(session, request.r_peaks)


class RPeakAdd(BaseModel):
    sample_index: int


@router.post("/r-peaks/add")
async def add_rpeak(
    request: RPeakAdd,
    session: Session = Depends(get_session),
):
    """Add a single R-peak at the given sample index."""
    if session.ecg_data is None:
        raise HTTPException(400, detail={"code": "NO_ECG", "message": "Load ECG first"})

    data = session.ecg_data
    try:
        new_peaks = add_r_peak(
            data["r_peaks"],
            request.sample_index,
            data["sample_rate"],
            data["num_samples"],
        )
    except ValueError as e:
        raise HTTPException(400, detail={"code": "INVALID_PEAK", "message": str(e)})

    return _recalculate_from_peaks(session, new_peaks)


class RPeakRemove(BaseModel):
    sample_index: int


@router.post("/r-peaks/remove")
async def remove_rpeak(
    request: RPeakRemove,
    session: Session = Depends(get_session),
):
    """Remove the R-peak nearest to the given sample index."""
    if session.ecg_data is None:
        raise HTTPException(400, detail={"code": "NO_ECG", "message": "Load ECG first"})

    data = session.ecg_data
    try:
        new_peaks = remove_r_peak(data["r_peaks"], request.sample_index)
    except ValueError as e:
        raise HTTPException(400, detail={"code": "INVALID_PEAK", "message": str(e)})

    return _recalculate_from_peaks(session, new_peaks)


class RPeakMove(BaseModel):
    from_index: int
    to_index: int


@router.post("/r-peaks/move")
async def move_rpeak(
    request: RPeakMove,
    session: Session = Depends(get_session),
):
    """Move an R-peak from one position to another."""
    if session.ecg_data is None:
        raise HTTPException(400, detail={"code": "NO_ECG", "message": "Load ECG first"})

    data = session.ecg_data
    try:
        new_peaks = move_r_peak(
            data["r_peaks"],
            request.from_index,
            request.to_index,
            data["num_samples"],
        )
    except ValueError as e:
        raise HTTPException(400, detail={"code": "INVALID_PEAK", "message": str(e)})

    return _recalculate_from_peaks(session, new_peaks)
