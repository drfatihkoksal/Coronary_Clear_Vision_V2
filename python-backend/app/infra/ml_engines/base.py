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
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        """Returns (mask, confidence, probability_map).

        probability_map: Optional float32 array (0..1) with soft edge
        information from the model's sigmoid/softmax output. None when
        the engine doesn't produce probability maps (e.g. threshold fallback).
        """
        ...

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass
