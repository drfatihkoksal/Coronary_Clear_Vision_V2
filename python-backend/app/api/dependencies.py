from functools import lru_cache

from fastapi import Depends, Header, HTTPException

from app.config.settings import settings
from app.infra.dicom_handler import DicomHandler
from app.infra.ml_engines.registry import EngineRegistry
from app.infra.persistence.session_store import SessionStore
from app.services.analysis_service import AnalysisService
from app.services.qfr_service import QFRService
from app.services.segmentation_service import SegmentationService
from app.services.study_service import StudyService
from app.services.tracking_service import TrackingService


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(max_sessions=settings.max_sessions)


@lru_cache
def get_dicom_handler() -> DicomHandler:
    return DicomHandler()


@lru_cache
def get_engine_registry() -> EngineRegistry:
    models_dir = settings.ml_models_path if settings.ml_models_path.exists() else None
    return EngineRegistry(models_dir=models_dir, device=settings.device)


def get_study_service(
    dicom_handler: DicomHandler = Depends(get_dicom_handler),
    store: SessionStore = Depends(get_session_store),
) -> StudyService:
    return StudyService(dicom_handler, store)


def get_segmentation_service(
    registry: EngineRegistry = Depends(get_engine_registry),
    store: SessionStore = Depends(get_session_store),
) -> SegmentationService:
    return SegmentationService(registry, store)


def get_analysis_service() -> AnalysisService:
    return AnalysisService()


def get_tracking_service(
    store: SessionStore = Depends(get_session_store),
) -> TrackingService:
    return TrackingService(store)


def get_qfr_service() -> QFRService:
    return QFRService()


async def get_session(
    x_session_id: str = Header(..., alias="X-Session-ID"),
    store: SessionStore = Depends(get_session_store),
):
    session = store.get(x_session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": f"Session {x_session_id} not found",
            },
        )
    return session
