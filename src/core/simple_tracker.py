"""
Simple Template Matching Tracker for Coronary Angiography
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
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
    """
    
    def __init__(self):
        """Initialize the tracker."""
        self.points: Dict[str, TrackedPoint] = {}
        self.template_size = 31  # Template window size
        self.search_radius = 50  # Search radius in pixels
        self.min_confidence = 0.5  # Minimum correlation threshold
        self.current_frame = None
        
        logger.info("SimpleTracker initialized")
    
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
            search_radius=self.search_radius
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
                # Update position
                point.position = new_pos
                point.confidence = confidence
                tracked_positions[point_id] = new_pos
                
                # Update template occasionally for adaptation
                if confidence > 0.8:
                    new_template = self._extract_template(gray, new_pos)
                    if new_template is not None:
                        # Blend old and new template
                        alpha = 0.9
                        # Ensure both templates are same type
                        old_template = point.template.astype(np.float32)
                        new_template_float = new_template.astype(np.float32)
                        blended = cv2.addWeighted(
                            old_template, alpha,
                            new_template_float, 1 - alpha,
                            0
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
        if (x - half_size < 0 or x + half_size >= w or
            y - half_size < 0 or y + half_size >= h):
            return None
        
        # Extract template
        template = frame[y - half_size:y + half_size + 1,
                        x - half_size:x + half_size + 1]
        
        # Convert to uint8 for template matching
        return template.astype(np.uint8)
    
    def _track_point(self, frame: np.ndarray, point: TrackedPoint) -> Tuple[Optional[Point], float]:
        """Track a single point using template matching."""
        x, y = int(point.position.x), int(point.position.y)
        h, w = frame.shape
        
        # Define search region
        x1 = max(0, x - point.search_radius)
        y1 = max(0, y - point.search_radius)
        x2 = min(w, x + point.search_radius)
        y2 = min(h, y + point.search_radius)
        
        # Extract search region and ensure uint8
        search_region = frame[y1:y2, x1:x2].astype(np.uint8)
        
        # Check if search region is large enough
        if (search_region.shape[0] < point.template.shape[0] or
            search_region.shape[1] < point.template.shape[1]):
            return None, 0.0
        
        # Ensure template is also uint8
        template = point.template.astype(np.uint8)
        
        # Template matching
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Convert to global coordinates
        new_x = x1 + max_loc[0] + point.half_template
        new_y = y1 + max_loc[1] + point.half_template
        
        return Point(float(new_x), float(new_y)), max_val
    
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


def get_tracker() -> SimpleTracker:
    """Get global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SimpleTracker()
    return _tracker