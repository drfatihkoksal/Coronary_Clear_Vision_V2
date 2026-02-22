from fastapi import APIRouter, Depends, File, UploadFile, Query
from fastapi.responses import Response

from app.api.dependencies import get_session, get_study_service
from app.infra.persistence.session_store import Session
from app.services.study_service import StudyService

router = APIRouter(prefix="/dicom", tags=["DICOM"])


@router.post("/upload")
async def upload_dicom(
    file: UploadFile = File(...),
    anonymize: bool = Query(True),
    service: StudyService = Depends(get_study_service),
):
    """Upload and parse a DICOM file. Returns session info + metadata."""
    data = await file.read()
    session = service.load_study(data, anonymize=anonymize)
    study = session.study

    return {
        "session_id": session.id,
        "patient": study.patient.model_dump(mode="json"),
        "study_info": study.study_info.model_dump(mode="json"),
        "num_frames": study.num_frames,
        "frame_rate": study.frame_rate,
        "image_width": study.image_width,
        "image_height": study.image_height,
        "pixel_spacing": study.pixel_spacing.model_dump() if study.pixel_spacing else None,
    }


@router.get("/frame/{frame_index}")
async def get_frame(
    frame_index: int,
    session: Session = Depends(get_session),
    service: StudyService = Depends(get_study_service),
):
    """Get a single frame as PNG binary."""
    png_bytes = service.get_frame_png(session, frame_index)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Frame-Index": str(frame_index),
            "Cache-Control": "private, max-age=3600",
            "Vary": "X-Session-ID",
        },
    )


@router.get("/metadata")
async def get_metadata(session: Session = Depends(get_session)):
    """Get study metadata for current session."""
    study = session.study
    if study is None:
        return {"error": {"code": "NO_STUDY", "message": "No study loaded"}}
    return {
        "session_id": session.id,
        "patient": study.patient.model_dump(mode="json"),
        "study_info": study.study_info.model_dump(mode="json"),
        "num_frames": study.num_frames,
        "frame_rate": study.frame_rate,
        "image_width": study.image_width,
        "image_height": study.image_height,
        "pixel_spacing": study.pixel_spacing.model_dump() if study.pixel_spacing else None,
    }


@router.get("/num-frames")
async def get_num_frames(session: Session = Depends(get_session)):
    """Get number of frames in current study."""
    if session.study is None:
        return {"num_frames": 0}
    return {"num_frames": session.study.num_frames}


@router.post("/clear")
async def clear_study(
    session: Session = Depends(get_session),
    service: StudyService = Depends(get_study_service),
):
    """Clear the current study from session."""
    service.clear_study(session)
    return {"status": "cleared"}
