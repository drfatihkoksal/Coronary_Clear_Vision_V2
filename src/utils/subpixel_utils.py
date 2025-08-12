"""
Sub-pixel precision utilities for calibration and analysis.
Provides functions for sub-pixel edge detection and measurement.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from scipy import ndimage
from scipy.interpolate import interp1d


def subpixel_edge_detection(mask: np.ndarray, direction: np.ndarray, 
                          center_point: Tuple[float, float], 
                          max_distance: int = 100,
                          is_probability: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect edges with sub-pixel precision along a given direction.
    
    Args:
        mask: Binary mask or probability map
        direction: Direction vector (normalized)
        center_point: Starting point (x, y)
        max_distance: Maximum search distance in pixels
        is_probability: If True, treat mask as probability values (0-1)
        
    Returns:
        Tuple of (left_edge, right_edge) positions with sub-pixel precision
    """
    x_center, y_center = center_point
    dx, dy = direction
    
    # Sample points along the direction
    distances = np.linspace(-max_distance, max_distance, max_distance * 4)  # 4x oversampling
    
    # Calculate sample points
    x_samples = x_center + distances * dx
    y_samples = y_center + distances * dy
    
    # Interpolate mask values at sample points
    mask_values = ndimage.map_coordinates(mask.astype(np.float32), 
                                        [y_samples, x_samples], 
                                        order=1, mode='constant', cval=0)
    
    # Set thresholds based on mask type
    if is_probability:
        # For probability masks, use softer thresholds
        edge_threshold = 0.05  # Lower gradient threshold for better edge detection
        value_threshold = 0.5  # Higher threshold to avoid noise
    else:
        # For binary masks, use harder thresholds
        edge_threshold = 0.1
        value_threshold = 0.5
    
    # Find edges using gradient
    gradient = np.gradient(mask_values)
    
    # Find left edge (negative gradient, mask becomes 0)
    left_edge = None
    for i in range(len(gradient)):
        if gradient[i] < -edge_threshold and mask_values[i] > value_threshold:
            # Sub-pixel interpolation
            if i > 0 and abs(gradient[i] - gradient[i-1]) > 1e-8:
                left_edge = distances[i] - gradient[i] / (gradient[i] - gradient[i-1])
            else:
                left_edge = distances[i]
            break
    
    # Find right edge (positive gradient, mask becomes 0)
    right_edge = None
    for i in range(len(gradient)-1, -1, -1):
        if gradient[i] > edge_threshold and mask_values[i] > value_threshold:
            # Sub-pixel interpolation
            if i < len(gradient) - 1 and abs(gradient[i] - gradient[i+1]) > 1e-8:
                right_edge = distances[i] - gradient[i] / (gradient[i] - gradient[i+1])
            else:
                right_edge = distances[i]
            break
    
    return left_edge, right_edge


def subpixel_contour_width(mask: np.ndarray, center_point: Tuple[float, float], 
                          direction: np.ndarray, method: str = 'gradient', 
                          is_probability: bool = False) -> float:
    """
    Measure width at a point using sub-pixel precision contour analysis.
    
    Args:
        mask: Binary mask or probability map
        center_point: Point to measure width at (x, y)
        direction: Direction perpendicular to vessel (normalized)
        method: Method to use ('gradient' or 'moment')
        is_probability: If True, treat mask as probability values (0-1)
        
    Returns:
        Width in pixels with sub-pixel precision
    """
    if method == 'gradient':
        left_edge, right_edge = subpixel_edge_detection(mask, direction, center_point, 
                                                      max_distance=100,
                                                      is_probability=is_probability)
        if left_edge is not None and right_edge is not None:
            width = abs(right_edge - left_edge)
            # Return actual sub-pixel width without artificial minimum
            return width
    
    elif method == 'moment':
        # Moment-based sub-pixel measurement
        x_center, y_center = center_point
        dx, dy = direction
        
        # Create a profile line
        profile_length = 50
        profile_points = np.linspace(-profile_length, profile_length, profile_length * 2)
        
        x_profile = x_center + profile_points * dx
        y_profile = y_center + profile_points * dy
        
        # Sample mask values
        mask_profile = ndimage.map_coordinates(mask.astype(np.float32),
                                              [y_profile, x_profile],
                                              order=1, mode='constant', cval=0)
        
        # Calculate moments
        total_weight = np.sum(mask_profile)
        if total_weight > 0:
            # Find the extent of the mask
            threshold = 0.25 if is_probability else 0.5
            indices = np.where(mask_profile > threshold)[0]
            if len(indices) > 0:
                # Sub-pixel edge estimation using weighted moments
                left_idx = indices[0]
                right_idx = indices[-1]
                
                # Refine edges with sub-pixel precision
                if left_idx > 0:
                    left_weight = mask_profile[left_idx-1:left_idx+2]
                    if len(left_weight) >= 3:
                        left_offset = subpixel_moment_centroid(left_weight)
                        left_edge = profile_points[left_idx] + left_offset
                    else:
                        left_edge = profile_points[left_idx]
                else:
                    left_edge = profile_points[left_idx]
                
                if right_idx < len(profile_points) - 1:
                    right_weight = mask_profile[right_idx-1:right_idx+2]
                    if len(right_weight) >= 3:
                        right_offset = subpixel_moment_centroid(right_weight)
                        right_edge = profile_points[right_idx] + right_offset
                    else:
                        right_edge = profile_points[right_idx]
                else:
                    right_edge = profile_points[right_idx]
                
                return abs(right_edge - left_edge)
    
    return 0.0


def subpixel_moment_centroid(weights: np.ndarray) -> float:
    """
    Calculate sub-pixel centroid using moment analysis.
    
    Args:
        weights: Array of weights/intensities
        
    Returns:
        Sub-pixel offset from center
    """
    if len(weights) == 0:
        return 0.0
    
    indices = np.arange(len(weights))
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        return 0.0
    
    centroid = np.sum(indices * weights) / total_weight
    center_idx = len(weights) // 2
    
    return centroid - center_idx


def subpixel_catheter_diameter(mask: np.ndarray, method: str = 'moments') -> float:
    """
    Calculate catheter diameter with sub-pixel precision.
    
    Args:
        mask: Binary mask of catheter
        method: Method to use ('moments', 'contours', or 'gradient')
        
    Returns:
        Diameter in pixels with sub-pixel precision
    """
    if method == 'moments':
        # Find catheter contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE  # More precise contour
        )
        
        if not contours:
            return 0.0
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate moments
        moments = cv2.moments(largest_contour)
        
        if moments['m00'] == 0:
            return 0.0
        
        # Calculate centroid
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # Calculate equivalent diameter from area
        area = moments['m00']
        diameter_from_area = 2 * np.sqrt(area / np.pi)
        
        # Alternative: Use minimum enclosing circle
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_contour)
        diameter_from_circle = 2 * radius
        
        # Use the smaller of the two (more conservative)
        return min(diameter_from_area, diameter_from_circle)
    
    elif method == 'contours':
        # Enhanced contour-based measurement
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse for better diameter estimation
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            width, height = ellipse[1]
            return min(width, height)  # Use minor axis as diameter
        else:
            # Fallback to bounding rectangle
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]
            return min(width, height)
    
    elif method == 'gradient':
        # Gradient-based measurement
        # Find the catheter center
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        
        if moments['m00'] == 0:
            return 0.0
        
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        
        # Measure diameter in multiple directions and take average
        angles = np.linspace(0, np.pi, 8)  # 8 directions
        diameters = []
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            left_edge, right_edge = subpixel_edge_detection(mask, direction, (cx, cy))
            
            if left_edge is not None and right_edge is not None:
                diameter = abs(right_edge - left_edge)
                diameters.append(diameter)
        
        if diameters:
            return np.mean(diameters)
    
    return 0.0


def subpixel_vessel_width_profile(mask: np.ndarray, centerline: np.ndarray, 
                                 smoothing: bool = True, is_probability: bool = None) -> np.ndarray:
    """
    Calculate vessel width profile along centerline with sub-pixel precision.
    
    Args:
        mask: Binary vessel mask or probability map
        centerline: Centerline points as array of (y, x) coordinates
        smoothing: Whether to apply smoothing to the profile
        is_probability: If None, auto-detect; if True, treat as probability map
        
    Returns:
        Array of widths with sub-pixel precision
    """
    if len(centerline) < 2:
        return np.array([])
    
    # Auto-detect if mask is probability map
    if is_probability is None:
        is_probability = mask.dtype in [np.float32, np.float64] and mask.max() <= 1.0
    
    widths = []
    
    for i in range(len(centerline)):
        point = centerline[i]
        
        # Calculate perpendicular direction with improved endpoint handling
        if i == 0:
            # First point: use multiple points if available for better tangent estimation
            if len(centerline) >= 3:
                # Use quadratic fit for first 3 points
                tangent = -3*centerline[0] + 4*centerline[1] - centerline[2]
                tangent = tangent / 2.0  # Normalize by step size
            else:
                tangent = centerline[i+1] - centerline[i]
        elif i == len(centerline) - 1:
            # Last point: use multiple points if available for better tangent estimation
            if len(centerline) >= 3:
                # Use quadratic fit for last 3 points
                tangent = centerline[-3] - 4*centerline[-2] + 3*centerline[-1]
                tangent = tangent / 2.0  # Normalize by step size
            else:
                tangent = centerline[i] - centerline[i-1]
        else:
            # Middle points: use average direction
            tangent = (centerline[i+1] - centerline[i-1]) / 2
        
        # Normalize tangent
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm
            
            # Perpendicular direction (rotate 90 degrees)
            # Since tangent is in (y, x) format, perpendicular is (x, -y) which is [tangent[1], -tangent[0]]
            # But we need to provide it in (x, y) format for the function, so it becomes [-tangent[1], tangent[0]]
            perp_direction = np.array([-tangent[1], tangent[0]])
            
            # Measure width with sub-pixel precision
            center_point = (point[1], point[0])  # Convert (y,x) to (x,y)
            width = subpixel_contour_width(mask, center_point, perp_direction, method='gradient', 
                                         is_probability=is_probability)
            
            # Debug: log if width is 0
            if width == 0.0 and i % 10 == 0:  # Log every 10th point to avoid spam
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Zero width at point {i}: center={center_point}, perp_dir={perp_direction}")
            
            widths.append(width)
        else:
            widths.append(0.0)
    
    widths = np.array(widths)
    
    if smoothing and len(widths) > 1:
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        widths = gaussian_filter1d(widths, sigma=1.0)
    
    return widths