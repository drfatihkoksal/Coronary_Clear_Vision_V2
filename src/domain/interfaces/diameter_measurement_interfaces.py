"""
Unified Diameter Measurement Interface

Provides a standardized interface for all diameter measurement methods.
Supports gradient-based, segmentation-based, and other measurement techniques.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, List, Optional, Tuple, Dict, Any, runtime_checkable
import numpy as np
from dataclasses import dataclass


class DiameterMethod(Enum):
    """Available diameter measurement methods"""
    GRADIENT = "gradient"
    SEGMENTATION = "segmentation"
    EDGE_DETECTION = "edge_detection"
    INTENSITY_PROFILE = "intensity_profile"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


@dataclass
class DiameterMeasurement:
    """Result of a diameter measurement at a single point"""
    center_point: Tuple[float, float]  # (y, x) centerline point
    diameter_pixels: float  # Diameter in pixels
    left_edge: Tuple[float, float]  # Left edge point
    right_edge: Tuple[float, float]  # Right edge point
    confidence: float  # Confidence score (0-1)
    method: DiameterMethod  # Method used
    perpendicular_angle: float  # Angle of perpendicular line
    metadata: Dict[str, Any] = None  # Additional method-specific data


@dataclass
class DiameterProfile:
    """Complete diameter profile along a vessel"""
    centerline: np.ndarray  # Centerline points (N x 2)
    measurements: List[DiameterMeasurement]  # Diameter at each point
    average_diameter: float  # Average diameter
    min_diameter: float  # Minimum diameter
    max_diameter: float  # Maximum diameter
    mld_index: int  # Index of minimal lumen diameter
    reference_diameter: Optional[float] = None  # Reference diameter if available
    stenosis_percent: Optional[float] = None  # Stenosis percentage if calculated


@dataclass
class DiameterMeasurementConfig:
    """Configuration for diameter measurement"""
    method: DiameterMethod
    max_search_distance: int = 30  # Maximum search distance in pixels
    smoothing_window: int = 5  # Smoothing window size
    edge_threshold: float = 0.5  # Edge detection threshold
    use_subpixel: bool = True  # Enable subpixel precision
    validate_measurements: bool = True  # Enable validation
    parallel_processing: bool = False  # Enable parallel processing
    custom_params: Dict[str, Any] = None  # Method-specific parameters


@runtime_checkable
class IDiameterMeasurement(Protocol):
    """
    Protocol for diameter measurement implementations.
    
    All diameter measurement methods must implement this interface.
    """
    
    def measure_diameter(self,
                        image: np.ndarray,
                        center_point: Tuple[float, float],
                        perpendicular: np.ndarray,
                        config: DiameterMeasurementConfig) -> Optional[DiameterMeasurement]:
        """
        Measure diameter at a single point.
        
        Args:
            image: Input image (grayscale or binary mask)
            center_point: Center point on vessel (y, x)
            perpendicular: Perpendicular direction vector
            config: Measurement configuration
            
        Returns:
            DiameterMeasurement or None if measurement fails
        """
        ...
    
    def measure_profile(self,
                       image: np.ndarray,
                       centerline: np.ndarray,
                       config: DiameterMeasurementConfig,
                       progress_callback: Optional[callable] = None) -> Optional[DiameterProfile]:
        """
        Measure diameter profile along entire centerline.
        
        Args:
            image: Input image
            centerline: Centerline points (N x 2)
            config: Measurement configuration
            progress_callback: Optional progress callback(current, total)
            
        Returns:
            DiameterProfile or None if measurement fails
        """
        ...
    
    @property
    def method(self) -> DiameterMethod:
        """Get the measurement method type"""
        ...
    
    @property
    def requires_segmentation(self) -> bool:
        """Check if method requires segmentation mask"""
        ...
    
    def validate_measurement(self,
                           measurement: DiameterMeasurement,
                           image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate a diameter measurement.
        
        Args:
            measurement: Measurement to validate
            image: Original image for context
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


class BaseDiameterMeasurement(ABC):
    """
    Abstract base class for diameter measurement implementations.
    
    Provides common functionality and enforces interface.
    """
    
    def __init__(self, method: DiameterMethod):
        self._method = method
        self._last_error: Optional[str] = None
    
    @property
    def method(self) -> DiameterMethod:
        return self._method
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    @abstractmethod
    def measure_diameter(self,
                        image: np.ndarray,
                        center_point: Tuple[float, float],
                        perpendicular: np.ndarray,
                        config: DiameterMeasurementConfig) -> Optional[DiameterMeasurement]:
        """Measure diameter at a single point"""
        pass
    
    def measure_profile(self,
                       image: np.ndarray,
                       centerline: np.ndarray,
                       config: DiameterMeasurementConfig,
                       progress_callback: Optional[callable] = None) -> Optional[DiameterProfile]:
        """
        Default implementation for measuring diameter profile.
        
        Can be overridden for optimized batch processing.
        """
        if centerline is None or len(centerline) < 2:
            self._last_error = "Invalid centerline"
            return None
        
        measurements = []
        total_points = len(centerline)
        
        # Calculate perpendiculars for each point
        perpendiculars = self._calculate_perpendiculars(centerline)
        
        # Measure at each point
        for i, (point, perp) in enumerate(zip(centerline, perpendiculars)):
            if progress_callback:
                progress_callback(i, total_points)
            
            measurement = self.measure_diameter(image, point, perp, config)
            if measurement:
                measurements.append(measurement)
        
        if not measurements:
            self._last_error = "No valid measurements obtained"
            return None
        
        # Calculate statistics
        diameters = [m.diameter_pixels for m in measurements]
        min_idx = np.argmin(diameters)
        
        return DiameterProfile(
            centerline=centerline,
            measurements=measurements,
            average_diameter=np.mean(diameters),
            min_diameter=diameters[min_idx],
            max_diameter=np.max(diameters),
            mld_index=min_idx
        )
    
    def _calculate_perpendiculars(self, centerline: np.ndarray) -> np.ndarray:
        """Calculate perpendicular vectors for each centerline point"""
        n_points = len(centerline)
        perpendiculars = np.zeros((n_points, 2))
        
        for i in range(n_points):
            # Calculate tangent
            if i == 0:
                tangent = centerline[1] - centerline[0]
            elif i == n_points - 1:
                tangent = centerline[i] - centerline[i-1]
            else:
                tangent = centerline[i+1] - centerline[i-1]
            
            # Normalize
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
            
            # Calculate perpendicular (rotate 90 degrees)
            perpendiculars[i] = np.array([-tangent[1], tangent[0]])
        
        return perpendiculars
    
    @abstractmethod
    def validate_measurement(self,
                           measurement: DiameterMeasurement,
                           image: np.ndarray) -> Tuple[bool, str]:
        """Validate measurement - must be implemented"""
        pass
    
    @property
    @abstractmethod
    def requires_segmentation(self) -> bool:
        """Check if method requires segmentation"""
        pass


@runtime_checkable
class IDiameterMeasurementFactory(Protocol):
    """Factory for creating diameter measurement instances"""
    
    def create_measurement(self,
                         method: DiameterMethod,
                         **kwargs) -> IDiameterMeasurement:
        """
        Create a diameter measurement instance.
        
        Args:
            method: Measurement method to use
            **kwargs: Method-specific parameters
            
        Returns:
            IDiameterMeasurement instance
        """
        ...
    
    def get_available_methods(self) -> List[DiameterMethod]:
        """Get list of available measurement methods"""
        ...
    
    def get_default_config(self, method: DiameterMethod) -> DiameterMeasurementConfig:
        """Get default configuration for a method"""
        ...


class DiameterMeasurementAdapter:
    """
    Adapter to unify existing diameter measurement functions.
    
    Wraps legacy functions to conform to the new interface.
    """
    
    def __init__(self, 
                 legacy_function: callable,
                 method: DiameterMethod,
                 requires_segmentation: bool = False):
        """
        Initialize adapter.
        
        Args:
            legacy_function: Existing measurement function
            method: Method type
            requires_segmentation: Whether method needs segmentation
        """
        self.legacy_function = legacy_function
        self._method = method
        self._requires_segmentation = requires_segmentation
    
    def measure_diameter(self,
                        image: np.ndarray,
                        center_point: Tuple[float, float],
                        perpendicular: np.ndarray,
                        config: DiameterMeasurementConfig) -> Optional[DiameterMeasurement]:
        """Adapt legacy function to new interface"""
        try:
            # Call legacy function with appropriate parameters
            result = self.legacy_function(
                image, 
                center_point, 
                perpendicular,
                config.max_search_distance
            )
            
            if result is None:
                return None
            
            # Convert result to standardized format
            left_dist, right_dist = result
            diameter = left_dist + right_dist
            
            # Calculate edge points
            cy, cx = center_point
            left_edge = (
                cy - perpendicular[0] * left_dist,
                cx - perpendicular[1] * left_dist
            )
            right_edge = (
                cy + perpendicular[0] * right_dist,
                cx + perpendicular[1] * right_dist
            )
            
            return DiameterMeasurement(
                center_point=center_point,
                diameter_pixels=diameter,
                left_edge=left_edge,
                right_edge=right_edge,
                confidence=0.8,  # Default confidence
                method=self._method,
                perpendicular_angle=np.arctan2(perpendicular[1], perpendicular[0])
            )
            
        except Exception as e:
            logger.error(f"Error in diameter measurement adapter: {e}")
            return None
    
    @property
    def method(self) -> DiameterMethod:
        return self._method
    
    @property
    def requires_segmentation(self) -> bool:
        return self._requires_segmentation