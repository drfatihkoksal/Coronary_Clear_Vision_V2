import logging

from app.infra.dicom_handler import DicomHandler
from app.infra.persistence.session_store import SessionStore, Session

logger = logging.getLogger(__name__)


class StudyService:
    def __init__(self, dicom_handler: DicomHandler, session_store: SessionStore):
        self._dicom = dicom_handler
        self._store = session_store

    def load_study(self, file_data: bytes, anonymize: bool = True) -> Session:
        """Load a DICOM file, create a session, and store data."""
        ds, study, frames = self._dicom.load_from_bytes(file_data, anonymize=anonymize)

        session = self._store.create()
        session.study = study
        session.frames = frames  # Store numpy arrays in memory
        session.dataset = ds  # Keep for ECG extraction later

        logger.info(
            "Loaded study: %d frames (%dx%d) %.1f fps, session=%s",
            study.num_frames, study.image_width, study.image_height,
            study.frame_rate, session.id,
        )
        return session

    def get_frame_png(self, session: Session, frame_index: int) -> bytes:
        """Get a single frame as PNG bytes."""
        if session.frames is None:
            raise ValueError("No study loaded")
        if frame_index < 0 or frame_index >= len(session.frames):
            raise IndexError(f"Frame index {frame_index} out of range [0, {len(session.frames)})")
        return DicomHandler.frame_to_png(session.frames[frame_index])

    def clear_study(self, session: Session) -> None:
        """Clear study data from session."""
        session.study = None
        session.frames = None
        session.dataset = None
        session.segmentations.clear()
        session.qca_measurements.clear()
        session.rws_results.clear()
