"""Domain models for Coronary Clear Vision application."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class Point:
    """Represents a 2D point in image coordinates."""
    x: float
    y: float

    def __iter__(self):
        """Allow unpacking like a tuple."""
        return iter((self.x, self.y))

    def __getitem__(self, index):
        """Allow indexing like a tuple."""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Point index out of range")

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to integer tuple."""
        return (int(self.x), int(self.y))

    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class CalibrationResult:
    """Result of a calibration operation."""
    pixels_per_mm: float
    method: str  # 'catheter', 'angiopy', 'manual'
    reference_points: List[Point] = field(default_factory=list)
    reference_diameter_mm: Optional[float] = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pixels_per_mm': self.pixels_per_mm,
            'method': self.method,
            'reference_points': [(p.x, p.y) for p in self.reference_points],
            'reference_diameter_mm': self.reference_diameter_mm,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class VesselSegment:
    """Represents a segment of a vessel."""
    centerline: List[Point]
    diameters: List[float]  # Diameter at each centerline point in mm
    start_point: Point
    end_point: Point
    average_diameter: float
    min_diameter: float
    max_diameter: float
    length_mm: float

    @property
    def diameter_variance(self) -> float:
        """Calculate diameter variance."""
        return np.var(self.diameters) if self.diameters else 0.0

    def get_diameter_at_position(self, position: float) -> float:
        """Get interpolated diameter at a normalized position (0-1)."""
        if not self.diameters:
            return 0.0

        index = int(position * (len(self.diameters) - 1))
        index = max(0, min(index, len(self.diameters) - 1))
        return self.diameters[index]


@dataclass
class StenosisResult:
    """Result of stenosis analysis."""
    location: Point
    percent_stenosis: float
    minimal_luminal_diameter: float  # MLD in mm
    reference_diameter: float  # RVD in mm
    lesion_length: float  # in mm
    confidence: float
    method: str  # 'automatic', 'manual', 'qca'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'location': (self.location.x, self.location.y),
            'percent_stenosis': self.percent_stenosis,
            'minimal_luminal_diameter': self.minimal_luminal_diameter,
            'reference_diameter': self.reference_diameter,
            'lesion_length': self.lesion_length,
            'confidence': self.confidence,
            'method': self.method
        }

    @property
    def severity_category(self) -> str:
        """Categorize stenosis severity."""
        if self.percent_stenosis < 25:
            return "minimal"
        elif self.percent_stenosis < 50:
            return "mild"
        elif self.percent_stenosis < 70:
            return "moderate"
        elif self.percent_stenosis < 90:
            return "severe"
        else:
            return "critical"


@dataclass
class VesselAnalysisResult:
    """Complete result of vessel analysis."""
    vessel_segment: VesselSegment
    stenoses: List[StenosisResult]
    calibration: CalibrationResult
    frame_number: int
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_stenosis(self) -> Optional[StenosisResult]:
        """Get the most severe stenosis."""
        if not self.stenoses:
            return None
        return max(self.stenoses, key=lambda s: s.percent_stenosis)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'vessel_segment': {
                'centerline': [(p.x, p.y) for p in self.vessel_segment.centerline],
                'diameters': self.vessel_segment.diameters,
                'average_diameter': self.vessel_segment.average_diameter,
                'min_diameter': self.vessel_segment.min_diameter,
                'max_diameter': self.vessel_segment.max_diameter,
                'length_mm': self.vessel_segment.length_mm
            },
            'stenoses': [s.to_dict() for s in self.stenoses],
            'calibration': self.calibration.to_dict(),
            'frame_number': self.frame_number,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class SegmentationResult:
    """Result of vessel segmentation."""
    mask: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    method: str = 'angiopy'  # 'angiopy', 'manual', 'threshold'
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def vessel_area_pixels(self) -> int:
        """Calculate vessel area in pixels."""
        return int(np.sum(self.mask > 0))

    def get_skeleton(self) -> np.ndarray:
        """Get vessel skeleton (requires skeletonization)."""
        from skimage.morphology import skeletonize
        return skeletonize(self.mask > 0)


@dataclass
class TrackedPoint:
    """A point tracked across multiple frames."""
    point_id: str
    current_position: Point
    frame_history: Dict[int, Point] = field(default_factory=dict)
    confidence: float = 1.0
    is_active: bool = True

    def add_position(self, frame: int, position: Point):
        """Add a position for a specific frame."""
        self.frame_history[frame] = position
        self.current_position = position

    def get_position_at_frame(self, frame: int) -> Optional[Point]:
        """Get position at a specific frame."""
        return self.frame_history.get(frame)

    def get_trajectory(self) -> List[Tuple[int, Point]]:
        """Get sorted trajectory of the point."""
        return sorted(self.frame_history.items())