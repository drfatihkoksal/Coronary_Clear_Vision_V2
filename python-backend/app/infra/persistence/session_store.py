from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass
class Session:
    id: str
    study: Any = None  # Will be Study type when loaded
    frames: list | None = None  # list[np.ndarray] stored in memory
    dataset: Any = None  # pydicom Dataset for ECG extraction
    segmentations: dict[int, Any] = field(default_factory=dict)
    qca_measurements: dict[int, Any] = field(default_factory=dict)
    rws_results: list = field(default_factory=list)
    ecg_data: Any = None
    calibration: Any = None
    motion_signal: Any = None
    qfr_session: Any = None
    tracking_state: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionStore:
    def __init__(self, max_sessions: int = 5):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions

    def create(self) -> Session:
        if len(self._sessions) >= self._max_sessions:
            self._evict_lru()
        session = Session(id=str(uuid4()))
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session:
            session.last_accessed = datetime.now(UTC)
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _evict_lru(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_accessed)
        del self._sessions[oldest.id]

    @property
    def active_count(self) -> int:
        return len(self._sessions)
