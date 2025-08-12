"""
Minimum Cost Path Centerline Generation

Bu modül tracked points arasında minimum cost path kullanarak
centerline oluşturur. AngioPy segmentasyonundan gelen skeletonize
centerline yerine daha hassas ve kontrollü centerline üretimi sağlar.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
from scipy import ndimage
try:
    from skimage.graph import route_through_array
    from skimage.filters import gaussian
    from skimage.morphology import disk
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available, using fallback methods")
import heapq

logger = logging.getLogger(__name__)


class MinimumCostPathGenerator(object):
    """
    Tracked points arasında minimum cost path ile centerline oluşturucu
    """
    
    def __init__(self):
        """Initialize cost path generator"""
        self.cost_map = None
        self.last_mask = None
        
    def generate_centerline_from_tracked_points(self, 
                                               mask: np.ndarray,
                                               tracked_points: List[Tuple[float, float]],
                                               smooth_factor: float = 1.0,
                                               use_vessel_guidance: bool = True) -> np.ndarray:
        """
        Tracked points arasında minimum cost path ile centerline oluştur
        
        Args:
            mask: Vessel segmentation mask (0-255 or 0-1)
            tracked_points: List of (x, y) tracked points in order
            smooth_factor: Cost map smoothing factor
            use_vessel_guidance: Whether to use vessel mask for cost guidance
            
        Returns:
            np.ndarray: Centerline points as (y, x) coordinates
        """
        if len(tracked_points) < 2:
            raise ValueError("At least 2 tracked points required")
            
        logger.info(f"Generating centerline from {len(tracked_points)} tracked points")
        
        # Normalize mask to 0-1 range
        if mask.max() > 1:
            normalized_mask = mask.astype(np.float32) / 255.0
        else:
            normalized_mask = mask.astype(np.float32)
            
        # Generate cost map
        cost_map = self._generate_cost_map(normalized_mask, smooth_factor, use_vessel_guidance)
        self.cost_map = cost_map
        self.last_mask = normalized_mask
        
        # Convert tracked points to integer coordinates
        points_int = [(int(round(x)), int(round(y))) for x, y in tracked_points]
        
        # Validate points are within bounds
        points_int = self._validate_points_bounds(points_int, mask.shape)
        
        # Generate path through all points
        full_path = self._generate_multi_point_path(cost_map, points_int)
        
        # Smooth the path
        smoothed_path = self._smooth_path(full_path, smooth_factor)
        
        logger.info(f"Generated centerline with {len(smoothed_path)} points")
        
        return smoothed_path
        
    def _generate_cost_map(self, 
                          mask: np.ndarray, 
                          smooth_factor: float,
                          use_vessel_guidance: bool) -> np.ndarray:
        """
        Generate cost map for path finding
        
        Args:
            mask: Normalized vessel mask (0-1)
            smooth_factor: Smoothing factor
            use_vessel_guidance: Use vessel mask for guidance
            
        Returns:
            np.ndarray: Cost map (lower values = preferred paths)
        """
        if use_vessel_guidance:
            # Vessel-guided cost map
            cost_map = self._create_vessel_guided_cost_map(mask, smooth_factor)
        else:
            # Simple distance-based cost map
            cost_map = self._create_distance_cost_map(mask, smooth_factor)
            
        return cost_map
        
    def _create_vessel_guided_cost_map(self, 
                                     mask: np.ndarray, 
                                     smooth_factor: float) -> np.ndarray:
        """
        Create vessel-guided cost map
        
        Lower costs inside vessel, higher costs outside
        """
        logger.debug("Creating vessel-guided cost map")
        
        # Distance transform from vessel edges
        # Inside vessel = low cost, outside = high cost
        vessel_binary = (mask > 0.1).astype(np.uint8)
        
        # Distance to vessel boundary (inside negative, outside positive)
        distance_inside = ndimage.distance_transform_edt(vessel_binary)
        distance_outside = ndimage.distance_transform_edt(1 - vessel_binary)
        
        # Combine distances - prefer center of vessel
        cost_map = np.where(vessel_binary, 
                           1.0 / (distance_inside + 1.0),  # Low cost inside, prefer center
                           distance_outside + 10.0)        # High cost outside
        
        # Add mask intensity guidance - prefer brighter areas (vessel centers)
        intensity_guidance = 1.0 - mask  # Invert so bright areas have low cost
        cost_map = cost_map + intensity_guidance * 0.5
        
        # Smooth cost map
        if smooth_factor > 0:
            if SKIMAGE_AVAILABLE:
                cost_map = gaussian(cost_map, sigma=smooth_factor)
            else:
                # Fallback using scipy
                cost_map = ndimage.gaussian_filter(cost_map, sigma=smooth_factor)
            
        # Ensure positive costs
        cost_map = np.maximum(cost_map, 0.01)
        
        logger.debug(f"Cost map stats: min={cost_map.min():.3f}, max={cost_map.max():.3f}")
        
        return cost_map
        
    def _create_distance_cost_map(self, 
                                mask: np.ndarray, 
                                smooth_factor: float) -> np.ndarray:
        """
        Create simple distance-based cost map
        """
        logger.debug("Creating distance-based cost map")
        
        # Use inverted mask as cost (bright areas = low cost)
        cost_map = 1.0 - mask + 0.1  # Add small constant to avoid zero costs
        
        # Smooth if requested
        if smooth_factor > 0:
            if SKIMAGE_AVAILABLE:
                cost_map = gaussian(cost_map, sigma=smooth_factor)
            else:
                cost_map = ndimage.gaussian_filter(cost_map, sigma=smooth_factor)
            
        return cost_map
        
    def _validate_points_bounds(self, 
                               points: List[Tuple[int, int]], 
                               shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Validate and clamp points to image bounds
        """
        height, width = shape
        validated_points = []
        
        for x, y in points:
            x_clamp = max(0, min(width - 1, x))
            y_clamp = max(0, min(height - 1, y))
            validated_points.append((x_clamp, y_clamp))
            
            if x != x_clamp or y != y_clamp:
                logger.warning(f"Point ({x}, {y}) clamped to ({x_clamp}, {y_clamp})")
                
        return validated_points
        
    def _generate_multi_point_path(self, 
                                  cost_map: np.ndarray, 
                                  points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Generate path through multiple points using Dijkstra's algorithm
        """
        logger.debug(f"Generating path through {len(points)} points")
        
        full_path = []
        
        # Generate path between consecutive points
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            
            logger.debug(f"Finding path from {start_point} to {end_point}")
            
            # Find minimum cost path between these two points
            try:
                segment_path = self._find_minimum_cost_path(cost_map, start_point, end_point)
                
                # Add segment to full path (avoid duplicate points)
                if i == 0:
                    full_path.extend(segment_path)
                else:
                    full_path.extend(segment_path[1:])  # Skip first point to avoid duplication
                    
            except Exception as e:
                logger.error(f"Failed to find path between points {i} and {i+1}: {e}")
                # Fallback: linear interpolation
                segment_path = self._linear_interpolation(start_point, end_point)
                if i == 0:
                    full_path.extend(segment_path)
                else:
                    full_path.extend(segment_path[1:])
                    
        # Convert to numpy array with (y, x) format for consistency with QCA
        path_array = np.array([(y, x) for x, y in full_path])
        
        logger.debug(f"Generated path with {len(path_array)} points")
        
        return path_array
        
    def _find_minimum_cost_path(self, 
                               cost_map: np.ndarray, 
                               start: Tuple[int, int], 
                               end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find minimum cost path using Dijkstra's algorithm
        """
        try:
            if SKIMAGE_AVAILABLE:
                # Use skimage's route_through_array which implements Dijkstra
                # Note: route_through_array expects (row, col) format
                start_rc = (start[1], start[0])  # Convert (x,y) to (row,col)
                end_rc = (end[1], end[0])
                
                # Find path
                indices, weight = route_through_array(
                    cost_map, 
                    start_rc, 
                    end_rc,
                    fully_connected=True  # Allow diagonal movement
                )
                
                # Convert back to (x, y) format
                path = [(col, row) for row, col in indices]
                
                logger.debug(f"Path found with total cost: {weight:.2f}")
                
                return path
            else:
                # Fallback: use simple A* algorithm
                return self._find_path_astar(cost_map, start, end)
            
        except Exception as e:
            logger.error(f"Dijkstra path finding failed: {e}")
            raise
            
    def _find_path_astar(self, cost_map: np.ndarray, start: Tuple[int, int], 
                        end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Fallback A* pathfinding implementation
        """
        # Simple A* implementation
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Priority queue: (f_score, point)
        open_set = [(0, start)]
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        
        height, width = cost_map.shape
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # Check all neighbors (8-connected)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check bounds
                    if (neighbor[0] < 0 or neighbor[0] >= width or 
                        neighbor[1] < 0 or neighbor[1] >= height):
                        continue
                    
                    # Calculate tentative g_score
                    movement_cost = cost_map[neighbor[1], neighbor[0]]
                    if dx != 0 and dy != 0:  # Diagonal movement
                        movement_cost *= 1.414  # sqrt(2)
                    
                    tentative_g_score = g_score[current] + movement_cost
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found - return linear interpolation
        logger.warning("A* pathfinding failed, using linear interpolation")
        return self._linear_interpolation(start, end)
            
    def _linear_interpolation(self, 
                            start: Tuple[int, int], 
                            end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Fallback: linear interpolation between points
        """
        x1, y1 = start
        x2, y2 = end
        
        # Calculate number of points needed
        distance = max(abs(x2 - x1), abs(y2 - y1))
        num_points = max(2, int(distance))
        
        # Generate interpolated points
        path = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            path.append((x, y))
            
        logger.debug(f"Linear interpolation: {len(path)} points")
        
        return path
        
    def _smooth_path(self, 
                    path: np.ndarray, 
                    smooth_factor: float) -> np.ndarray:
        """
        Smooth the generated path
        """
        if len(path) < 3 or smooth_factor <= 0:
            return path
            
        logger.debug(f"Smoothing path with factor {smooth_factor}")
        
        # Apply Gaussian smoothing to coordinates
        sigma = smooth_factor * 2.0  # Scale smoothing factor
        
        # Smooth y and x coordinates separately
        smoothed_y = ndimage.gaussian_filter1d(path[:, 0].astype(np.float32), sigma=sigma)
        smoothed_x = ndimage.gaussian_filter1d(path[:, 1].astype(np.float32), sigma=sigma)
        
        # Combine smoothed coordinates
        smoothed_path = np.column_stack([smoothed_y, smoothed_x])
        
        logger.debug("Path smoothing completed")
        
        return smoothed_path
        
    def visualize_cost_map(self) -> Optional[np.ndarray]:
        """
        Get visualization of the last generated cost map
        
        Returns:
            Optional[np.ndarray]: Cost map as 8-bit image for visualization
        """
        if self.cost_map is None:
            return None
            
        # Normalize cost map to 0-255 for visualization
        normalized = ((self.cost_map - self.cost_map.min()) / 
                     (self.cost_map.max() - self.cost_map.min()) * 255)
        
        return normalized.astype(np.uint8)
        
    def get_path_quality_metrics(self, 
                                path: np.ndarray) -> Dict[str, float]:
        """
        Calculate quality metrics for the generated path
        
        Args:
            path: Generated centerline path
            
        Returns:
            Dict with quality metrics
        """
        if len(path) < 2:
            return {}
            
        metrics = {}
        
        # Path smoothness (average curvature)
        if len(path) > 2:
            curvatures = []
            for i in range(1, len(path) - 1):
                p1, p2, p3 = path[i-1], path[i], path[i+1]
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
                
            metrics['mean_curvature'] = np.mean(curvatures)
            metrics['max_curvature'] = np.max(curvatures)
            
        # Path length
        if len(path) > 1:
            segments = np.diff(path, axis=0)
            segment_lengths = np.linalg.norm(segments, axis=1)
            metrics['total_length'] = np.sum(segment_lengths)
            metrics['mean_segment_length'] = np.mean(segment_lengths)
            
        # Vessel adherence (if mask available)
        if self.last_mask is not None:
            vessel_adherence = []
            for point in path:
                y, x = int(point[0]), int(point[1])
                if (0 <= y < self.last_mask.shape[0] and 
                    0 <= x < self.last_mask.shape[1]):
                    vessel_adherence.append(self.last_mask[y, x])
                    
            if vessel_adherence:
                metrics['vessel_adherence'] = np.mean(vessel_adherence)
                
        return metrics


def create_centerline_from_tracked_points(mask: np.ndarray,
                                         tracked_points: List[Tuple[float, float]],
                                         smooth_factor: float = 1.0,
                                         use_vessel_guidance: bool = True) -> np.ndarray:
    """
    Convenience function to create centerline from tracked points
    
    Args:
        mask: Vessel segmentation mask
        tracked_points: List of (x, y) tracked points
        smooth_factor: Smoothing factor for cost map and path
        use_vessel_guidance: Whether to use vessel mask for guidance
        
    Returns:
        np.ndarray: Centerline points as (y, x) coordinates
    """
    generator = MinimumCostPathGenerator()
    return generator.generate_centerline_from_tracked_points(
        mask, tracked_points, smooth_factor, use_vessel_guidance
    )