import logging
import time

import numpy as np

from app.core.centerline_extractor import compute_perpendicular_diameters, extract_centerline
from app.infra.ml_engines.registry import EngineRegistry
from app.infra.persistence.session_store import Session, SessionStore
from app.models.enums import SegmentationEngine
from app.models.domain import SegmentationResult

logger = logging.getLogger(__name__)


class SegmentationService:
    def __init__(self, registry: EngineRegistry, session_store: SessionStore):
        self._registry = registry
        self._store = session_store

    def segment_image(
        self,
        image: np.ndarray,
        engine_type: SegmentationEngine = SegmentationEngine.NNUNET,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
        num_centerline_points: int | None = None,
    ) -> dict:
        """Segment an arbitrary image, extract centerline and diameters.

        Core segmentation method used by both normal and QFR paths.
        Does not store results in session — callers manage their own state.
        """
        engine = self._registry.get(engine_type)
        if engine is None:
            available = self._registry.available_engines()
            raise ValueError(
                f"Engine {engine_type} not available. Available: {list(available.keys())}"
            )

        start = time.time()
        mask, confidence = engine.segment(image, roi=roi, seed_points=seed_points)
        inference_ms = (time.time() - start) * 1000

        centerline = extract_centerline(
            mask, method="skeleton", num_points=num_centerline_points
        )

        diameters_px: list[float] = []
        if len(centerline) >= 2:
            diameters_px = compute_perpendicular_diameters(centerline, mask)

        logger.info(
            "Segmented with %s: conf=%.2f, time=%.0fms, centerline=%d pts, mask_nonzero=%d",
            engine_type.value,
            confidence,
            inference_ms,
            len(centerline),
            int(np.count_nonzero(mask)),
        )

        return {
            "engine": engine_type.value,
            "mask": mask,
            "confidence": confidence,
            "inference_time_ms": inference_ms,
            "centerline": centerline,
            "diameters_px": diameters_px,
        }

    def segment_frame(
        self,
        session: Session,
        frame_index: int,
        engine_type: SegmentationEngine = SegmentationEngine.NNUNET,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
    ) -> dict:
        """Segment a single frame and store the result."""
        if session.frames is None:
            raise ValueError("No study loaded")
        if frame_index < 0 or frame_index >= len(session.frames):
            raise IndexError(f"Frame {frame_index} out of range")

        image = session.frames[frame_index]
        result = self.segment_image(image, engine_type, roi=roi, seed_points=seed_points)

        seg_result = SegmentationResult(
            frame_index=frame_index,
            engine=engine_type,
            confidence=result["confidence"],
            inference_time_ms=result["inference_time_ms"],
        )
        session.segmentations[frame_index] = {
            "result": seg_result,
            "mask": result["mask"],
            "probability_map": None,
        }

        return {
            "frame_index": frame_index,
            "engine": engine_type.value,
            "confidence": result["confidence"],
            "inference_time_ms": result["inference_time_ms"],
        }

    def segment_and_extract(
        self,
        session: Session,
        frame_index: int,
        engine_type: SegmentationEngine = SegmentationEngine.NNUNET,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
        num_centerline_points: int | None = None,
    ) -> dict:
        """Segment frame, extract centerline, compute diameters."""
        if session.frames is None:
            raise ValueError("No study loaded")
        if frame_index < 0 or frame_index >= len(session.frames):
            raise IndexError(f"Frame {frame_index} out of range")

        image = session.frames[frame_index]
        result = self.segment_image(
            image, engine_type, roi=roi, seed_points=seed_points,
            num_centerline_points=num_centerline_points,
        )

        seg_result = SegmentationResult(
            frame_index=frame_index,
            engine=engine_type,
            confidence=result["confidence"],
            inference_time_ms=result["inference_time_ms"],
        )
        session.segmentations[frame_index] = {
            "result": seg_result,
            "mask": result["mask"],
            "probability_map": None,
            "centerline": result["centerline"],
            "diameters_px": result["diameters_px"],
        }

        return {
            "frame_index": frame_index,
            "engine": engine_type.value,
            "confidence": result["confidence"],
            "inference_time_ms": result["inference_time_ms"],
            "centerline": [{"x": p[0], "y": p[1]} for p in result["centerline"]],
            "diameters_px": result["diameters_px"],
            "has_centerline": len(result["centerline"]) >= 2,
        }

    def get_available_engines(self) -> dict:
        return self._registry.available_engines()
