from pathlib import Path

import numpy as np

from app.infra.ml_engines.base import BaseSegmentationEngine
from app.infra.ml_engines.nnunet import NNUNetEngine
from app.infra.ml_engines.angiopy import AngioPyEngine
from app.infra.ml_engines.seedmodel import SeedModelEngine
from app.models.enums import SegmentationEngine


class ThresholdEngine(BaseSegmentationEngine):
    """Simple threshold-based fallback engine that is always available."""

    @property
    def engine_type(self) -> SegmentationEngine:
        return SegmentationEngine.NNUNET  # Registered as default

    @property
    def is_available(self) -> bool:
        return True

    def segment(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, float]:
        threshold = np.mean(image) * 0.7
        mask = np.zeros_like(image, dtype=np.uint8)

        if roi is not None:
            x, y, w, h = roi
            region = image[y : y + h, x : x + w]
            mask[y : y + h, x : x + w] = (region < threshold).astype(np.uint8) * 255
        else:
            mask = (image < threshold).astype(np.uint8) * 255

        return mask, 0.3


class EngineRegistry:
    def __init__(self, models_dir: Path | None = None, *, device: str = "cpu"):
        self._engines: dict[SegmentationEngine, BaseSegmentationEngine] = {}
        self._models_dir = models_dir
        self._device = device
        self._register_engines()

    def _register_engines(self):
        """Auto-register all available engines.

        Threshold fallback is always registered as the NNUNET default.
        Real ML engines override the threshold entry if their model files exist.
        """
        # Always register threshold fallback first
        self._engines[SegmentationEngine.NNUNET] = ThresholdEngine()

        if self._models_dir:
            # nnU-Net variants
            for variant in ["roi", "wide", "fullframe"]:
                engine = NNUNetEngine(
                    self._models_dir, variant, device=self._device
                )
                if engine.is_available:
                    self._engines[engine.engine_type] = engine

            # AngioPy
            angiopy = AngioPyEngine(self._models_dir, device=self._device)
            if angiopy.is_available:
                self._engines[SegmentationEngine.ANGIOPY] = angiopy

            # SeedModel
            seedmodel = SeedModelEngine(self._models_dir, device=self._device)
            if seedmodel.is_available:
                self._engines[SegmentationEngine.SEEDMODEL] = seedmodel

    def register(self, engine: BaseSegmentationEngine) -> None:
        self._engines[engine.engine_type] = engine

    def get(self, engine_type: SegmentationEngine) -> BaseSegmentationEngine | None:
        return self._engines.get(engine_type)

    def available_engines(self) -> dict[str, dict]:
        return {
            e.engine_type.value: {"available": e.is_available} for e in self._engines.values()
        }

    def has_engine(self, engine_type: SegmentationEngine) -> bool:
        return engine_type in self._engines
