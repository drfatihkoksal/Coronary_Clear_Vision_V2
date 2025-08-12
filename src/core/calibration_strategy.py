"""Calibration strategy pattern for unified calibration system."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import numpy as np
from .domain_models import CalibrationResult, Point


class CalibrationStrategy(ABC):
    """Base class for calibration strategies."""

    @abstractmethod
    def calibrate(self, image: np.ndarray, **kwargs) -> Optional[CalibrationResult]:
        """Perform calibration on the given image."""

    @abstractmethod
    def validate_input(self, image: np.ndarray, **kwargs) -> bool:
        """Validate if the input is suitable for this calibration method."""

    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this calibration method."""


class CatheterCalibrationStrategy(CalibrationStrategy):
    """Calibration using catheter dimensions."""

    def __init__(self, catheter_size_french: int = 6):
        """Initialize with catheter size in French units."""
        self.catheter_size_french = catheter_size_french
        self.catheter_diameter_mm = catheter_size_french / 3.0  # 1 French = 1/3 mm

    def calibrate(self, image: np.ndarray,
                 start_point: Optional[Point] = None,
                 end_point: Optional[Point] = None,
                 **kwargs) -> Optional[CalibrationResult]:
        """Calibrate using two points marking the catheter diameter."""
        if not self.validate_input(image, start_point=start_point, end_point=end_point):
            return None

        # Calculate pixel distance
        pixel_distance = start_point.distance_to(end_point)

        if pixel_distance <= 0:
            return None

        # Calculate pixels per mm
        pixels_per_mm = pixel_distance / self.catheter_diameter_mm

        return CalibrationResult(
            pixels_per_mm=pixels_per_mm,
            method=self.get_method_name(),
            reference_points=[start_point, end_point],
            reference_diameter_mm=self.catheter_diameter_mm,
            confidence=0.95  # High confidence for manual catheter measurement
        )

    def validate_input(self, image: np.ndarray, **kwargs) -> bool:
        """Validate catheter calibration input."""
        start_point = kwargs.get('start_point')
        end_point = kwargs.get('end_point')

        if start_point is None or end_point is None:
            return False

        # Check if points are within image bounds
        h, w = image.shape[:2]
        for point in [start_point, end_point]:
            if not (0 <= point.x < w and 0 <= point.y < h):
                return False

        # Check minimum distance
        if start_point.distance_to(end_point) < 10:  # Minimum 10 pixels
            return False

        return True

    def get_method_name(self) -> str:
        return f"catheter_{self.catheter_size_french}F"


class AngioPyCalibrationStrategy(CalibrationStrategy):
    """Calibration using AngioPy automatic catheter detection."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize with optional model path."""
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        """Lazy load the segmentation model."""
        if self._model is None:
            # This would load the actual AngioPy model
            # For now, we'll keep it as a placeholder
            pass

    def calibrate(self, image: np.ndarray,
                 catheter_size_french: int = 6,
                 **kwargs) -> Optional[CalibrationResult]:
        """Automatically detect and calibrate using catheter."""
        if not self.validate_input(image):
            return None

        # Detect catheter using AngioPy
        catheter_points = self._detect_catheter(image)

        if not catheter_points:
            return None

        # Calculate average diameter from detected points
        pixel_diameter = self._calculate_catheter_diameter(catheter_points)
        catheter_diameter_mm = catheter_size_french / 3.0

        if pixel_diameter <= 0:
            return None

        pixels_per_mm = pixel_diameter / catheter_diameter_mm

        return CalibrationResult(
            pixels_per_mm=pixels_per_mm,
            method=self.get_method_name(),
            reference_points=catheter_points[:2],  # Use first two points
            reference_diameter_mm=catheter_diameter_mm,
            confidence=self._calculate_confidence(catheter_points)
        )

    def _detect_catheter(self, image: np.ndarray) -> List[Point]:
        """Detect catheter points in the image."""
        # Placeholder for actual AngioPy detection
        # This would use the segmentation model to find catheter
        return []

    def _calculate_catheter_diameter(self, points: List[Point]) -> float:
        """Calculate catheter diameter from detected points."""
        if len(points) < 2:
            return 0.0

        # Simple approach: use first two points
        return points[0].distance_to(points[1])

    def _calculate_confidence(self, points: List[Point]) -> float:
        """Calculate confidence based on detection quality."""
        # Placeholder - would use actual detection metrics
        return 0.8 if len(points) >= 2 else 0.5

    def validate_input(self, image: np.ndarray, **kwargs) -> bool:
        """Validate AngioPy calibration input."""
        if image is None or image.size == 0:
            return False

        # Check image dimensions
        if len(image.shape) not in [2, 3]:
            return False

        return True

    def get_method_name(self) -> str:
        return "angiopy_auto"


class ManualCalibrationStrategy(CalibrationStrategy):
    """Manual calibration with known distance."""

    def calibrate(self, image: np.ndarray,
                 start_point: Point,
                 end_point: Point,
                 known_distance_mm: float,
                 **kwargs) -> Optional[CalibrationResult]:
        """Calibrate using two points with known distance."""
        if not self.validate_input(image, start_point=start_point,
                                 end_point=end_point,
                                 known_distance_mm=known_distance_mm):
            return None

        pixel_distance = start_point.distance_to(end_point)

        if pixel_distance <= 0 or known_distance_mm <= 0:
            return None

        pixels_per_mm = pixel_distance / known_distance_mm

        return CalibrationResult(
            pixels_per_mm=pixels_per_mm,
            method=self.get_method_name(),
            reference_points=[start_point, end_point],
            reference_diameter_mm=known_distance_mm,
            confidence=0.9  # Slightly lower than catheter due to manual measurement
        )

    def validate_input(self, image: np.ndarray, **kwargs) -> bool:
        """Validate manual calibration input."""
        start_point = kwargs.get('start_point')
        end_point = kwargs.get('end_point')
        known_distance = kwargs.get('known_distance_mm', 0)

        if start_point is None or end_point is None:
            return False

        if known_distance <= 0:
            return False

        # Check if points are within image bounds
        h, w = image.shape[:2]
        for point in [start_point, end_point]:
            if not (0 <= point.x < w and 0 <= point.y < h):
                return False

        return True

    def get_method_name(self) -> str:
        return "manual"


class CalibrationService:
    """Service for managing calibration strategies."""

    def __init__(self):
        self._strategies: Dict[str, CalibrationStrategy] = {}
        self._current_result: Optional[CalibrationResult] = None

        # Register default strategies
        self.register_strategy('catheter', CatheterCalibrationStrategy())
        self.register_strategy('angiopy', AngioPyCalibrationStrategy())
        self.register_strategy('manual', ManualCalibrationStrategy())

    def register_strategy(self, name: str, strategy: CalibrationStrategy):
        """Register a new calibration strategy."""
        self._strategies[name] = strategy

    def calibrate(self, strategy_name: str, image: np.ndarray, **kwargs) -> Optional[CalibrationResult]:
        """Perform calibration using the specified strategy."""
        if strategy_name not in self._strategies:
            raise ValueError(f"Unknown calibration strategy: {strategy_name}")

        strategy = self._strategies[strategy_name]
        result = strategy.calibrate(image, **kwargs)

        if result:
            self._current_result = result

        return result

    def get_current_calibration(self) -> Optional[CalibrationResult]:
        """Get the current calibration result."""
        return self._current_result

    def set_current_calibration(self, result: CalibrationResult):
        """Set the current calibration result."""
        self._current_result = result

    def get_available_strategies(self) -> List[str]:
        """Get list of available calibration strategies."""
        return list(self._strategies.keys())

    def convert_pixels_to_mm(self, pixels: float) -> Optional[float]:
        """Convert pixels to millimeters using current calibration."""
        if self._current_result is None:
            return None
        return pixels / self._current_result.pixels_per_mm

    def convert_mm_to_pixels(self, mm: float) -> Optional[float]:
        """Convert millimeters to pixels using current calibration."""
        if self._current_result is None:
            return None
        return mm * self._current_result.pixels_per_mm