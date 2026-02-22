from abc import ABC, abstractmethod

import numpy as np

from app.models.enums import SegmentationEngine


class BaseSegmentationEngine(ABC):
    @property
    @abstractmethod
    def engine_type(self) -> SegmentationEngine: ...

    @property
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Returns (mask, confidence)."""
        ...

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass
