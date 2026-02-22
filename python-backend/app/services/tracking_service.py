import logging

from app.core.tracking_engine import propagate_tracking, refine_roi_with_optical_flow
from app.infra.persistence.session_store import Session, SessionStore
from app.models.enums import TrackingDirection

logger = logging.getLogger(__name__)


class TrackingService:
    def __init__(self, session_store: SessionStore):
        self._store = session_store

    def initialize(
        self,
        session: Session,
        frame_index: int,
        roi: tuple[int, int, int, int],
    ) -> dict:
        """Initialize tracking state on a given frame with the specified ROI.

        Stores the initial ROI and frame index in the session's tracking_state.
        """
        if session.frames is None:
            raise ValueError("No study loaded")
        if frame_index < 0 or frame_index >= len(session.frames):
            raise IndexError(f"Frame {frame_index} out of range")

        session.tracking_state = {
            "start_frame": frame_index,
            "roi": roi,
            "results": [
                {
                    "frame_index": frame_index,
                    "bbox": roi,
                    "confidence": 1.0,
                    "success": True,
                }
            ],
        }

        logger.info(
            "Initialized tracking on frame %d with ROI %s",
            frame_index,
            roi,
        )

        return {
            "frame_index": frame_index,
            "roi": list(roi),
            "status": "initialized",
        }

    def propagate(
        self,
        session: Session,
        direction: TrackingDirection = TrackingDirection.FORWARD,
        max_frames: int | None = None,
        auto_segment: bool = False,
    ) -> dict:
        """Propagate tracking across frames from the initialized position.

        Optionally refines each tracked bbox with optical flow.
        """
        if session.frames is None:
            raise ValueError("No study loaded")
        if not session.tracking_state:
            raise ValueError("Tracking not initialized")

        start_frame = session.tracking_state["start_frame"]
        roi = tuple(session.tracking_state["roi"])

        results = propagate_tracking(
            frames=session.frames,
            start_frame=start_frame,
            roi=roi,
            direction=direction.value,
            max_frames=max_frames,
        )

        # Refine each tracked frame with optical flow (skip start frame)
        for i in range(1, len(results)):
            if not results[i]["success"]:
                break
            prev_idx = results[i - 1]["frame_index"]
            curr_idx = results[i]["frame_index"]
            refined = refine_roi_with_optical_flow(
                session.frames[prev_idx],
                session.frames[curr_idx],
                results[i]["bbox"],
            )
            results[i]["bbox"] = refined

        # Merge new results into session tracking state
        existing_by_frame = {
            r["frame_index"]: r
            for r in session.tracking_state.get("results", [])
        }
        for r in results:
            existing_by_frame[r["frame_index"]] = r

        session.tracking_state["results"] = sorted(
            existing_by_frame.values(), key=lambda r: r["frame_index"]
        )

        tracked_count = sum(1 for r in results if r["success"])

        logger.info(
            "Propagated tracking %s from frame %d: %d/%d frames tracked",
            direction.value,
            start_frame,
            tracked_count,
            len(results),
        )

        return {
            "direction": direction.value,
            "tracked_frames": tracked_count,
            "total_frames": len(results),
            "results": results,
        }

    def get_state(self, session: Session) -> dict:
        """Return the current tracking state and results."""
        if not session.tracking_state:
            return {"status": "not_initialized", "results": []}

        return {
            "status": "initialized",
            "start_frame": session.tracking_state.get("start_frame"),
            "roi": session.tracking_state.get("roi"),
            "results": session.tracking_state.get("results", []),
            "num_tracked": len(session.tracking_state.get("results", [])),
        }

    def clear(self, session: Session) -> dict:
        """Clear tracking state from session."""
        session.tracking_state = {}
        logger.info("Cleared tracking state")
        return {"status": "cleared"}
