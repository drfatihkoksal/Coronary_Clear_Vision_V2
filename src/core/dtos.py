"""
Data Transfer Objects for application data.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class ImageData:
    """DTO for image data."""
    array: np.ndarray
    width: int
    height: int
    channels: int
    dtype: np.dtype
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_array(cls, array: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Create ImageData from numpy array."""
        if array.ndim == 2:
            height, width = array.shape
            channels = 1
        elif array.ndim == 3:
            height, width, channels = array.shape
        else:
            raise ValueError(f"Invalid array dimensions: {array.ndim}")
        
        return cls(
            array=array,
            width=width,
            height=height,
            channels=channels,
            dtype=array.dtype,
            metadata=metadata or {}
        )


@dataclass
class SegmentationResult:
    """DTO for segmentation results."""
    mask: np.ndarray
    confidence: float
    processing_time: float
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationData:
    """DTO for calibration data."""
    point1: Tuple[float, float]
    point2: Tuple[float, float]
    reference_size: float
    unit: str
    pixel_size: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def pixels_per_unit(self) -> float:
        """Calculate pixels per unit."""
        dx = self.point2[0] - self.point1[0]
        dy = self.point2[1] - self.point1[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance / self.reference_size if self.reference_size > 0 else 1.0


@dataclass
class QCAMeasurement:
    """DTO for QCA measurements."""
    vessel_id: str
    frame_number: int
    reference_diameter: float
    minimum_diameter: float
    percent_stenosis: float
    lesion_length: float
    centerline: np.ndarray
    diameters: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DicomMetadata:
    """DTO for DICOM metadata."""
    patient_name: str
    patient_id: str
    study_date: Optional[datetime]
    study_description: str
    series_description: str
    modality: str
    manufacturer: str
    institution: str
    frame_count: int
    frame_rate: Optional[float]
    pixel_spacing: Optional[Tuple[float, float]]
    window_center: Optional[float]
    window_width: Optional[float]
    additional_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRequest:
    """DTO for processing requests."""
    input_path: str
    output_path: Optional[str]
    processing_type: str  # 'segmentation', 'qca', 'tracking', etc.
    options: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingResult:
    """DTO for processing results."""
    request: ProcessingRequest
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)



@dataclass
class ViewerState:
    """DTO for viewer state."""
    current_frame: int
    total_frames: int
    zoom_level: float
    pan_offset: Tuple[float, float]
    rotation_angle: float
    is_playing: bool
    playback_speed: float
    current_mode: str
    overlay_visible: bool
    metadata: Dict[str, Any] = field(default_factory=dict)