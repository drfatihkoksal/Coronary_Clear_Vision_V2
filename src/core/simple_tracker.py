"""
Simple Template Matching Tracker for Coronary Angiography
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .domain_models import Point

logger = logging.getLogger(__name__)


@dataclass
class TrackedPoint:
    """Information for a tracked point."""

    point_id: str
    position: Point
    template: np.ndarray
    template_size: int = 31
    search_radius: int = 50
    confidence: float = 1.0

    def __post_init__(self):
        self.half_template = self.template_size // 2


class SimpleTracker:
    """
    Simple template matching tracker using OpenCV.
    Enhanced with FPS-aware parameters and adaptive search.
    """

    def __init__(self, fps: float = 30.0):
        """Initialize the tracker.
        
        Args:
            fps: Video frame rate for parameter optimization
        """
        self.points: Dict[str, TrackedPoint] = {}
        self.fps = fps
        
        # FPS-aware parameters
        if fps < 20:
            # 15 FPS optimizations
            self.template_size = 25  # Larger template for stability
            self.search_radius = 60  # Wider search for large motion
            self.min_confidence = 0.4  # More tolerant threshold
            self.use_pyramid = True
            self.pyramid_levels = 3
        else:
            # Normal FPS parameters
            self.template_size = 21
            self.search_radius = 30
            self.min_confidence = 0.5
            self.use_pyramid = False
            self.pyramid_levels = 2
            
        self.current_frame = None
        self.velocity_history: Dict[str, List[float]] = {}  # Track velocities

        logger.info(f"SimpleTracker initialized for {fps} FPS")

    def set_frame(self, frame: np.ndarray):
        """Set the current frame for tracking."""
        if len(frame.shape) == 3:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.current_frame = frame.copy()

    def add_point(self, point_id: str, position: Point) -> bool:
        """Add a point to track."""
        if self.current_frame is None:
            logger.error("No frame set for tracking")
            return False

        # Extract template
        template = self._extract_template(self.current_frame, position)
        if template is None:
            logger.warning(f"Could not extract template for point {point_id}")
            return False

        # Create tracked point
        tracked_point = TrackedPoint(
            point_id=point_id,
            position=position,
            template=template,
            template_size=self.template_size,
            search_radius=self.search_radius,
        )

        self.points[point_id] = tracked_point
        logger.info(f"Added tracking point {point_id} at ({position.x:.1f}, {position.y:.1f})")
        return True

    def track_in_frame(self, frame: np.ndarray) -> Dict[str, Point]:
        """Track all points in the new frame."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        tracked_positions = {}

        for point_id, point in self.points.items():
            # Find new position
            new_pos, confidence = self._track_point(gray, point)

            if new_pos is not None and confidence >= self.min_confidence:
                # Calculate velocity for adaptive search
                if point_id in self.velocity_history:
                    old_pos = point.position
                    velocity = np.sqrt((new_pos.x - old_pos.x)**2 + (new_pos.y - old_pos.y)**2)
                    self.velocity_history[point_id].append(velocity)
                    if len(self.velocity_history[point_id]) > 10:
                        self.velocity_history[point_id].pop(0)
                else:
                    self.velocity_history[point_id] = [0.0]
                
                # Update position
                point.position = new_pos
                point.confidence = confidence
                tracked_positions[point_id] = new_pos

                # Update template with motion-aware rate
                update_rate = self._get_adaptive_update_rate(point_id, confidence)
                if update_rate > 0:
                    new_template = self._extract_template(gray, new_pos)
                    if new_template is not None:
                        # Ensure both templates are same type
                        old_template = point.template.astype(np.float32)
                        new_template_float = new_template.astype(np.float32)
                        blended = cv2.addWeighted(
                            old_template, 1.0 - update_rate, new_template_float, update_rate, 0
                        )
                        point.template = blended.astype(np.uint8)
            else:
                # Keep last position if tracking failed
                tracked_positions[point_id] = point.position
                logger.debug(f"Tracking failed for {point_id}, keeping last position")

        # Update current frame
        self.current_frame = gray
        return tracked_positions

    def _extract_template(self, frame: np.ndarray, position: Point) -> Optional[np.ndarray]:
        """Extract template around the given position."""
        x, y = int(position.x), int(position.y)
        h, w = frame.shape
        half_size = self.template_size // 2

        # Check bounds
        if x - half_size < 0 or x + half_size >= w or y - half_size < 0 or y + half_size >= h:
            return None

        # Extract template
        template = frame[y - half_size : y + half_size + 1, x - half_size : x + half_size + 1]

        # Convert to uint8 for template matching
        return template.astype(np.uint8)

    def _track_point(self, frame: np.ndarray, point: TrackedPoint) -> Tuple[Optional[Point], float]:
        """Track a single point using template matching with FPS-aware enhancements."""
        x, y = int(point.position.x), int(point.position.y)
        h, w = frame.shape
        
        # Adaptive search radius based on velocity history
        adaptive_radius = self._get_adaptive_search_radius(point.point_id, point.search_radius)

        # Define search region with adaptive radius
        x1 = max(0, x - adaptive_radius)
        y1 = max(0, y - adaptive_radius)
        x2 = min(w, x + adaptive_radius)
        y2 = min(h, y + adaptive_radius)

        # Extract search region and ensure uint8
        search_region = frame[y1:y2, x1:x2].astype(np.uint8)

        # Check if search region is large enough
        if (
            search_region.shape[0] < point.template.shape[0]
            or search_region.shape[1] < point.template.shape[1]
        ):
            return None, 0.0

        # Ensure template is also uint8
        template = point.template.astype(np.uint8)
        
        # Use pyramid tracking for low FPS
        if self.use_pyramid and self.fps < 20:
            new_pos, confidence = self._pyramid_track(
                search_region, template, x1, y1, point.half_template
            )
            if new_pos:
                return new_pos, confidence

        # Regular template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)

        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Convert to global coordinates
        new_x = x1 + max_loc[0] + point.half_template
        new_y = y1 + max_loc[1] + point.half_template

        return Point(float(new_x), float(new_y)), max_val
    
    def _get_adaptive_search_radius(self, point_id: str, base_radius: int) -> int:
        """Calculate adaptive search radius based on velocity history."""
        # FPS compensation
        fps_multiplier = max(1.0, 30.0 / max(self.fps, 10.0))
        
        # Check velocity history
        if point_id in self.velocity_history and len(self.velocity_history[point_id]) > 1:
            recent_velocities = self.velocity_history[point_id][-3:]
            max_velocity = max(recent_velocities) if recent_velocities else 0
            
            # Add extra radius for high velocity
            velocity_addon = int(max_velocity * 1.5)
            adaptive_radius = min(int(base_radius * fps_multiplier) + velocity_addon, 100)
        else:
            # Just FPS compensation
            adaptive_radius = min(int(base_radius * fps_multiplier), 80)
            
        return adaptive_radius
    
    def _pyramid_track(self, search_region: np.ndarray, template: np.ndarray, 
                       offset_x: int, offset_y: int, half_template: int) -> Tuple[Optional[Point], float]:
        """Pyramid-based coarse-to-fine tracking for large motions."""
        try:
            # Build pyramids
            search_pyramid = [search_region]
            template_pyramid = [template]
            
            for _ in range(self.pyramid_levels - 1):
                if search_pyramid[-1].shape[0] > 10 and search_pyramid[-1].shape[1] > 10:
                    search_pyramid.append(cv2.pyrDown(search_pyramid[-1]))
                    template_pyramid.append(cv2.pyrDown(template_pyramid[-1]))
                else:
                    break
            
            # Start from lowest resolution
            level = len(search_pyramid) - 1
            if (search_pyramid[level].shape[0] >= template_pyramid[level].shape[0] and
                search_pyramid[level].shape[1] >= template_pyramid[level].shape[1]):
                
                result = cv2.matchTemplate(search_pyramid[level], template_pyramid[level], cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > self.min_confidence:
                    # Scale up position
                    scale = 2 ** level
                    new_x = offset_x + (max_loc[0] + template_pyramid[level].shape[1] // 2) * scale
                    new_y = offset_y + (max_loc[1] + template_pyramid[level].shape[0] // 2) * scale
                    return Point(float(new_x), float(new_y)), max_val
        except Exception as e:
            logger.debug(f"Pyramid tracking failed: {e}")
            
        return None, 0.0
    
    def _get_adaptive_update_rate(self, point_id: str, confidence: float) -> float:
        """Get adaptive template update rate based on motion and confidence."""
        base_rate = 0.1 if self.fps >= 20 else 0.05
        
        # Reduce update rate for high velocity (motion blur risk)
        if point_id in self.velocity_history and len(self.velocity_history[point_id]) > 0:
            current_velocity = self.velocity_history[point_id][-1]
            if current_velocity > 10:  # High velocity threshold
                base_rate *= 0.2  # Reduce update rate by 80%
        
        # Scale by confidence
        if confidence > 0.9:
            return base_rate
        elif confidence > 0.7:
            return base_rate * 0.5
        else:
            return 0  # Don't update for low confidence

    def remove_point(self, point_id: str):
        """Remove a tracked point."""
        if point_id in self.points:
            del self.points[point_id]
            logger.info(f"Removed tracking point {point_id}")

    def clear(self):
        """Clear all tracked points."""
        self.points.clear()
        self.current_frame = None
        logger.info("Cleared all tracking points")

    def get_tracked_points(self) -> Dict[str, Point]:
        """Get current positions of all tracked points."""
        return {pid: point.position for pid, point in self.points.items()}


# Global tracker instance
_tracker = None


def get_tracker(fps: float = 30.0) -> SimpleTracker:
    """Get global tracker instance.
    
    Args:
        fps: Video frame rate for optimization
        
    Returns:
        SimpleTracker: Optimized tracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = SimpleTracker(fps=fps)
    elif _tracker.fps != fps:
        # Recreate tracker if FPS changed
        logger.info(f"FPS changed from {_tracker.fps} to {fps}, recreating tracker")
        _tracker = SimpleTracker(fps=fps)
    return _tracker
