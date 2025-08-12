"""
Accurate vessel diameter measurement for coronary angiography
Measures vessel width perpendicular to centerline strictly within binary mask
Returns actual edge positions for precise overlay drawing
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict
import scipy.ndimage
import scipy.signal
from src.utils.diameter_utils import log_diameter_statistics

logger = logging.getLogger(__name__)


def measure_vessel_diameter_accurate(segmentation_map: np.ndarray, centerline: np.ndarray, 
                                   max_search_distance: int = 30) -> np.ndarray:
    """
    Accurate diameter measurement using either a binary mask or a probability map.
    
    Args:
        segmentation_map: Binary vessel mask (0 or 255/1) or grayscale probability map (0-1).
        centerline: Vessel centerline points in (y, x) format.
        max_search_distance: Maximum search distance in pixels (default 30).
        
    Returns:
        Array of diameter measurements in pixels.
    """
    result = measure_vessel_diameter_with_edges(segmentation_map, centerline, max_search_distance)
    return result['diameters']


def measure_vessel_diameter_with_edges(segmentation_map: np.ndarray, centerline: np.ndarray, 
                                     max_search_distance: int = 30) -> Dict[str, np.ndarray]:
    """
    Measure diameter and return exact edge positions for drawing.
    Automatically detects if the input is a binary mask or a probability map.

    Args:
        segmentation_map: Binary vessel mask (0 or 255/1) or grayscale probability map (0-1).
        centerline: Vessel centerline points in (y, x) format.
        max_search_distance: Maximum search distance in pixels (default 30).

    Returns:
        Dictionary with:
        - 'diameters': Array of diameter measurements
        - 'left_edges': Array of left edge distances from centerline
        - 'right_edges': Array of right edge distances from centerline
        - 'perpendiculars': Array of perpendicular unit vectors
    """
    logger.info(f"Starting diameter measurement with edge positions for {len(centerline)} points")
    
    # --- Enhanced map type detection ---
    map_info = _analyze_segmentation_map(segmentation_map)
    logger.info(f"Map analysis: type={map_info['type']}, range=[{map_info['min']:.3f}, {map_info['max']:.3f}], threshold={map_info['threshold']:.3f}")
    
    # Use consistent binary mask for validation
    binary_mask_for_validation = (segmentation_map > map_info['threshold']).astype(np.uint8)
    
    # Validate inputs
    if not _validate_inputs(binary_mask_for_validation, centerline):
        n_points = len(centerline)
        return {
            'diameters': np.zeros(n_points),
            'left_edges': np.zeros(n_points),
            'right_edges': np.zeros(n_points),
            'perpendiculars': np.zeros((n_points, 2))
        }
    
    diameters = []
    left_edges = []
    right_edges = []
    perpendiculars = []
    
    for i in range(len(centerline)):
        perpendicular = _calculate_perpendicular(centerline, i)
        
        if perpendicular is None:
            perpendicular = np.array([0, 1]) # Default horizontal
            left_dist, right_dist = 0.0, 0.0
        else:
            # Unified method that adapts based on map type
            left_dist, right_dist = _measure_perpendicular_edges_unified(
                segmentation_map, centerline[i], perpendicular, max_search_distance, map_info
            )
        
        diameter = left_dist + right_dist
        diameter = _validate_diameter(diameter, i)
        
        diameters.append(diameter)
        left_edges.append(left_dist)
        right_edges.append(right_dist)
        perpendiculars.append(perpendicular)
    
    diameters_array = np.array(diameters)
    _log_statistics(diameters_array)

    return {
        'diameters': diameters_array,
        'left_edges': np.array(left_edges),
        'right_edges': np.array(right_edges),
        'perpendiculars': np.array(perpendiculars)
    }


def _analyze_segmentation_map(segmentation_map: np.ndarray) -> Dict:
    """
    Analyze segmentation map to determine optimal processing parameters
    
    Args:
        segmentation_map: Input segmentation map
        
    Returns:
        Dictionary with map analysis results
    """
    min_val = float(segmentation_map.min())
    max_val = float(segmentation_map.max())
    
    # Determine map type and optimal threshold
    if np.issubdtype(segmentation_map.dtype, np.floating) and max_val <= 1.0:
        map_type = 'probability'
        # For probability maps, use Otsu threshold or 0.5
        unique_vals = np.unique(segmentation_map)
        if len(unique_vals) > 10:  # Continuous probability map
            threshold = 0.5
        else:  # Discrete probability values
            threshold = (max_val + min_val) / 2
    else:
        map_type = 'binary'
        if segmentation_map.dtype == np.uint8 and max_val > 1:
            threshold = 127.5  # For 0-255 range
        else:
            threshold = 0.5   # For 0-1 range
    
    return {
        'type': map_type,
        'min': min_val,
        'max': max_val,
        'threshold': threshold,
        'dtype': str(segmentation_map.dtype)
    }


def _measure_perpendicular_edges_unified(segmentation_map: np.ndarray, center: np.ndarray,
                                       perpendicular: np.ndarray, max_distance: int,
                                       map_info: Dict) -> Tuple[float, float]:
    """
    Unified edge measurement with strict mask boundary enforcement
    
    Args:
        segmentation_map: Input segmentation map
        center: Center point (y, x)
        perpendicular: Perpendicular direction vector
        max_distance: Maximum search distance
        map_info: Map analysis information
        
    Returns:
        Tuple of (left_distance, right_distance)
    """
    # CRITICAL FIX: Use mask-constrained edge detection for all types
    # This prevents diameter lines from extending beyond vessel walls
    return _measure_perpendicular_edges_mask_constrained(
        segmentation_map, center, perpendicular, max_distance, map_info
    )


def _measure_perpendicular_edges_mask_constrained(segmentation_map: np.ndarray, center: np.ndarray,
                                                perpendicular: np.ndarray, max_distance: int,
                                                map_info: Dict) -> Tuple[float, float]:
    """
    Mask-constrained edge detection that strictly respects vessel boundaries
    
    This method ensures diameter measurements never extend beyond the actual
    vessel segmentation, preventing the visual artifacts where diameter lines
    appear outside the vessel walls.
    
    Args:
        segmentation_map: Segmentation map (binary or probability)
        center: Center point (y, x)
        perpendicular: Perpendicular direction vector
        max_distance: Maximum search distance
        map_info: Map analysis information
        
    Returns:
        Tuple of (left_distance, right_distance)
    """
    # Input validation
    if segmentation_map is None or segmentation_map.size == 0:
        logger.error("Invalid segmentation map provided")
        return 0.0, 0.0
    
    if not isinstance(center, np.ndarray) or center.size != 2:
        logger.error("Invalid center point provided")
        return 0.0, 0.0
    
    if not isinstance(perpendicular, np.ndarray) or perpendicular.size != 2:
        logger.error("Invalid perpendicular vector provided")
        return 0.0, 0.0
    
    try:
        cy, cx = float(center[0]), float(center[1])
        h, w = segmentation_map.shape
        threshold = map_info.get('threshold', 0.5)
    except (TypeError, ValueError, KeyError) as e:
        logger.error(f"Error extracting parameters: {e}")
        return 0.0, 0.0
    
    # Validate center is within bounds and inside vessel
    cy_int, cx_int = int(round(cy)), int(round(cx))
    if not (0 <= cy_int < h and 0 <= cx_int < w):
        logger.warning(f"Center point ({cx_int}, {cy_int}) outside bounds ({w}, {h})")
        return 0.0, 0.0
    
    if segmentation_map[cy_int, cx_int] <= threshold:
        logger.debug(f"Center point not inside vessel (value: {segmentation_map[cy_int, cx_int]}, threshold: {threshold})")
        return 0.0, 0.0
    
    # Use fine step size for better sub-pixel accuracy (3 decimal places)
    step_size = 0.01
    
    # Left direction (negative perpendicular)
    left_dist = 0.0
    last_valid_step = 0.0
    debug_first_point = (cy == 50.0 and cx == 128.0)  # Debug first point only
    
    for step in np.arange(step_size, max_distance, step_size):
        y = cy - perpendicular[0] * step
        x = cx - perpendicular[1] * step
        
        # Check bounds first
        if not (0 <= x < w and 0 <= y < h):
            if debug_first_point:
                logger.debug(f"  Step {step:.1f}: OUT OF BOUNDS at ({y:.1f}, {x:.1f})")
            break
        
        # Sample the segmentation value
        if map_info['type'] == 'probability':
            # Use bilinear interpolation for probability maps
            value = _sample_bilinear(segmentation_map, y, x)
        else:
            # Use nearest neighbor for binary maps
            y_int, x_int = int(round(y)), int(round(x))
            if 0 <= y_int < h and 0 <= x_int < w:
                value = segmentation_map[y_int, x_int]
            else:
                value = 0
        
        if debug_first_point and step <= 11.0:
            logger.debug(f"  Step {step:.1f}: pos=({y:.1f}, {x:.1f}), value={value}, threshold={threshold}")
        
        # Check if still inside vessel
        if value > threshold:
            last_valid_step = step
            if debug_first_point and step <= 11.0:
                logger.debug(f"    INSIDE: last_valid = {last_valid_step:.1f}")
        else:
            # Found vessel boundary - use last valid position
            left_dist = last_valid_step
            if debug_first_point:
                logger.debug(f"    BOUNDARY: final left_dist = {left_dist:.1f}")
            break
    else:
        # Reached max distance without finding boundary
        left_dist = last_valid_step
        if debug_first_point:
            logger.debug(f"  MAX DISTANCE: final left_dist = {left_dist:.1f}")
    
    # Right direction (positive perpendicular)
    right_dist = 0.0
    last_valid_step = 0.0
    
    for step in np.arange(step_size, max_distance, step_size):
        y = cy + perpendicular[0] * step
        x = cx + perpendicular[1] * step
        
        # Check bounds first
        if not (0 <= x < w and 0 <= y < h):
            break
        
        # Sample the segmentation value
        if map_info['type'] == 'probability':
            # Use bilinear interpolation for probability maps
            value = _sample_bilinear(segmentation_map, y, x)
        else:
            # Use nearest neighbor for binary maps
            y_int, x_int = int(round(y)), int(round(x))
            if 0 <= y_int < h and 0 <= x_int < w:
                value = segmentation_map[y_int, x_int]
            else:
                value = 0
        
        # Check if still inside vessel
        if value > threshold:
            last_valid_step = step
        else:
            # Found vessel boundary - use last valid position
            right_dist = last_valid_step
            break
    else:
        # Reached max distance without finding boundary
        right_dist = last_valid_step
    
    return left_dist, right_dist


def _sample_bilinear(image: np.ndarray, y: float, x: float) -> float:
    """
    Sample image value using bilinear interpolation
    
    Args:
        image: Input image
        y, x: Sampling coordinates (can be fractional)
        
    Returns:
        Interpolated value
    """
    h, w = image.shape
    
    # Clamp coordinates to valid range
    y = max(0, min(h - 1, y))
    x = max(0, min(w - 1, x))
    
    # Get integer and fractional parts
    y0, x0 = int(y), int(x)
    y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
    
    dy, dx = y - y0, x - x0
    
    # Bilinear interpolation
    val00 = image[y0, x0]
    val01 = image[y0, x1]
    val10 = image[y1, x0]
    val11 = image[y1, x1]
    
    val = (1 - dy) * (1 - dx) * val00 + \
          (1 - dy) * dx * val01 + \
          dy * (1 - dx) * val10 + \
          dy * dx * val11
    
    return float(val)


def _measure_perpendicular_edges_subpixel(prob_map: np.ndarray, center: np.ndarray,
                                          perpendicular: np.ndarray,
                                          max_distance: int) -> Tuple[float, float]:
    """
    Measures vessel edge distances from a centerline point on a probability map 
    with sub-pixel accuracy using gradient analysis.
    """
    h, w = prob_map.shape
    cy, cx = center[0], center[1]

    # 1. Sample the probability profile along the perpendicular line
    search_dist = min(max_distance, h / 2, w / 2)
    n_samples = int(search_dist * 4) # Sample at quarter-pixel intervals for higher resolution
    if n_samples < 20: n_samples = 20 # Ensure enough samples
    distances = np.linspace(-search_dist, search_dist, n_samples)
    
    sample_points_y = cy + distances * perpendicular[0]
    sample_points_x = cx + distances * perpendicular[1]

    # Filter out-of-bounds points
    valid_indices = (sample_points_y >= 0) & (sample_points_y < h - 1) & \
                    (sample_points_x >= 0) & (sample_points_x < w - 1)
    
    if not np.any(valid_indices):
        return 0.0, 0.0

    # Get probability values using bilinear interpolation
    prob_profile = scipy.ndimage.map_coordinates(
        prob_map, [sample_points_y[valid_indices], sample_points_x[valid_indices]], 
        order=1, mode='constant', cval=0.0
    )
    distances = distances[valid_indices]

    # 2. Calculate a smooth gradient with better handling
    if len(prob_profile) < 5:
        return 0.0, 0.0
    
    # Use appropriate window size for savgol filter
    window_length = min(5, len(prob_profile) if len(prob_profile) % 2 == 1 else len(prob_profile) - 1)
    if window_length < 3:
        window_length = 3
    
    gradient = scipy.signal.savgol_filter(prob_profile, window_length=window_length, polyorder=2, deriv=1)

    # 3. Find edges by locating gradient peaks - improved algorithm
    center_idx = np.argmin(np.abs(distances))

    # Alternative: Use half-maximum method for more accurate Gaussian edge detection
    max_prob = np.max(prob_profile)
    half_max = max_prob * 0.5  # Half maximum threshold
    
    # --- Right Edge (positive direction) ---
    right_half_profile = prob_profile[center_idx:]
    right_half_distances = distances[center_idx:]
    right_dist = 0.0
    
    # Find where profile drops below half maximum
    below_half_indices = np.where(right_half_profile < half_max)[0]
    if len(below_half_indices) > 0:
        edge_idx = below_half_indices[0]
        if edge_idx > 0:
            # Linear interpolation for sub-pixel precision
            y1, y2 = right_half_profile[edge_idx-1], right_half_profile[edge_idx]
            x1, x2 = right_half_distances[edge_idx-1], right_half_distances[edge_idx]
            # Find exact point where y = half_max
            if abs(y2 - y1) > 1e-8:
                right_dist = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            else:
                right_dist = x1
        else:
            right_dist = right_half_distances[edge_idx]

    # --- Left Edge (negative direction) ---
    left_half_profile = prob_profile[:center_idx+1]
    left_half_distances = distances[:center_idx+1]
    left_dist = 0.0
    
    # Find where profile drops below half maximum (going backwards)
    below_half_indices = np.where(left_half_profile < half_max)[0]
    if len(below_half_indices) > 0:
        edge_idx = below_half_indices[-1]  # Last (closest to center) point below threshold
        if edge_idx < len(left_half_profile) - 1:
            # Linear interpolation for sub-pixel precision
            y1, y2 = left_half_profile[edge_idx], left_half_profile[edge_idx+1]
            x1, x2 = left_half_distances[edge_idx], left_half_distances[edge_idx+1]
            # Find exact point where y = half_max
            if abs(y2 - y1) > 1e-8:
                left_dist = abs(x1 + (half_max - y1) * (x2 - x1) / (y2 - y1))
            else:
                left_dist = abs(x1)
        else:
            left_dist = abs(left_half_distances[edge_idx])

    return left_dist, right_dist



def _find_subpixel_peak(x_coords: np.ndarray, y_coords: np.ndarray, peak_idx: int) -> float:
    """
    Finds the sub-pixel location of a peak in a 1D array using a quadratic fit.
    """
    if peak_idx <= 0 or peak_idx >= len(y_coords) - 1:
        # Peak is at the edge, return integer location
        return x_coords[peak_idx]

    try:
        # Get the three points around the peak
        x_points = x_coords[peak_idx-1 : peak_idx+2]
        y_points = y_coords[peak_idx-1 : peak_idx+2]
        
        # Fit a quadratic polynomial: y = ax^2 + bx + c
        coeffs = np.polyfit(x_points, y_points, 2)
        
        # The peak of the parabola is at x = -b / 2a
        if abs(coeffs[0]) > 1e-6: # Avoid division by zero for flat lines
            return -coeffs[1] / (2 * coeffs[0])
        else:
            return x_coords[peak_idx]
    except np.linalg.LinAlgError:
        # If fitting fails, return the integer peak
        return x_coords[peak_idx]



def _validate_inputs(mask: np.ndarray, centerline: np.ndarray) -> bool:
    """Validate input parameters"""
    if mask is None or centerline is None:
        logger.error("Mask or centerline is None")
        return False
    
    if len(centerline) == 0:
        logger.error("Empty centerline")
        return False
    
    # Check if centerline is within bounds
    max_y = centerline[:, 0].max()
    max_x = centerline[:, 1].max()
    
    if max_y >= mask.shape[0] or max_x >= mask.shape[1]:
        logger.warning(f"Centerline exceeds mask bounds: max_y={max_y}, max_x={max_x}, shape={mask.shape}")
    
    return True


def _calculate_perpendicular(centerline: np.ndarray, index: int) -> Optional[np.ndarray]:
    """Calculate perpendicular direction using adaptive window size"""
    n_points = len(centerline)
    
    if n_points < 2:
        return None
    
    # Adaptive window size based on centerline length
    window = min(3, (n_points - 1) // 2)
    
    if index == 0:
        # First point: use forward difference
        tangent = centerline[min(index + window, n_points - 1)] - centerline[index]
    elif index == n_points - 1:
        # Last point: use backward difference
        tangent = centerline[index] - centerline[max(index - window, 0)]
    else:
        # Middle points: use central difference with adaptive window
        window = min(window, min(index, n_points - 1 - index))
        tangent = centerline[index + window] - centerline[index - window]
    
    # Normalize tangent
    norm = np.linalg.norm(tangent)
    if norm < 1e-6:
        return None
    
    tangent = tangent / norm
    
    # Perpendicular is 90 degrees rotation
    # For tangent (dy, dx), perpendicular is (-dx, dy)
    perpendicular = np.array([-tangent[1], tangent[0]])
    
    return perpendicular


def _measure_perpendicular_diameter_exact(mask: np.ndarray, center: np.ndarray,
                                        perpendicular: np.ndarray,
                                        max_distance: int) -> float:
    """
    Measure diameter along perpendicular with exact last pixel inside mask
    """
    left_dist, right_dist = _measure_perpendicular_edges_exact(
        mask, center, perpendicular, max_distance
    )
    
    # Total diameter is the sum of both sides
    # No need to add 1 for center pixel as our edge detection is precise
    diameter = float(left_dist + right_dist)
    
    # Smart minimum diameter handling for stenosis detection
    if diameter < 0.3:  # Only apply minimum for very small measurements (was 1.0, then 0.5)
        cy, cx = int(round(center[0])), int(round(center[1]))
        h, w = mask.shape
        
        # Check if center is in mask
        if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
            # Center is in vessel, allow small diameter but prevent zero
            diameter = max(0.3, diameter)  # Minimum 0.3 pixels for enhanced stenosis detection
        else:
            diameter = 0.0  # Center not in vessel
    elif diameter < 0.0:
        diameter = 0.0  # Prevent negative diameters
    
    return diameter


def _measure_perpendicular_edges_exact(mask: np.ndarray, center: np.ndarray,
                                      perpendicular: np.ndarray,
                                      max_distance: int) -> Tuple[float, float]:
    """
    Measure exact distance to edges in both directions for binary mask
    Optimized for binary masks (0/1 values only)
    """
    cy, cx = center[0], center[1]
    h, w = mask.shape
    
    # Check if center point is within mask
    cy_int, cx_int = int(round(cy)), int(round(cx))
    if not (0 <= cy_int < h and 0 <= cx_int < w):
        return 0.0, 0.0
    
    # If center is not in mask, return 0
    if mask[cy_int, cx_int] == 0:
        return 0.0, 0.0
    
    # For binary mask, use fine steps for 3 decimal place accuracy
    step_size = 0.01
    
    # Left direction - find exact edge position
    left_dist = 0.0
    last_inside = 0.0
    first_outside = max_distance
    
    # Binary search for exact edge position
    for step in np.arange(step_size, max_distance, step_size):
        y = cy - perpendicular[0] * step
        x = cx - perpendicular[1] * step
        
        # Check bounds
        if not (0 <= x < w and 0 <= y < h):
            first_outside = min(first_outside, step)
            break
        
        # For binary mask, use nearest neighbor
        y_nearest = int(round(y))
        x_nearest = int(round(x))
        
        if 0 <= y_nearest < h and 0 <= x_nearest < w and mask[y_nearest, x_nearest] > 0:
            last_inside = step
        else:
            first_outside = min(first_outside, step)
            break
    
    # Refine edge position
    left_dist = last_inside
    if first_outside - last_inside > step_size:
        # Binary search for more precise edge
        left_edge = _binary_search_edge(mask, center, perpendicular, -1, last_inside, first_outside)
        left_dist = left_edge
    
    # Right direction - find exact edge position
    right_dist = 0.0
    last_inside = 0.0
    first_outside = max_distance
    
    for step in np.arange(step_size, max_distance, step_size):
        y = cy + perpendicular[0] * step
        x = cx + perpendicular[1] * step
        
        # Check bounds
        if not (0 <= x < w and 0 <= y < h):
            first_outside = min(first_outside, step)
            break
        
        # For binary mask, use nearest neighbor
        y_nearest = int(round(y))
        x_nearest = int(round(x))
        
        if 0 <= y_nearest < h and 0 <= x_nearest < w and mask[y_nearest, x_nearest] > 0:
            last_inside = step
        else:
            first_outside = min(first_outside, step)
            break
    
    # Refine edge position
    right_dist = last_inside
    if first_outside - last_inside > step_size:
        # Binary search for more precise edge
        right_edge = _binary_search_edge(mask, center, perpendicular, 1, last_inside, first_outside)
        right_dist = right_edge
    
    # CRITICAL FIX: Remove safety margin that causes overshooting
    # The edge detection should be accurate to the mask boundary
    # Safety margins cause diameter lines to extend beyond vessel walls
    # Instead, we ensure accurate edge detection within mask bounds
    
    return left_dist, right_dist




def _binary_search_edge(mask: np.ndarray, center: np.ndarray, 
                       perpendicular: np.ndarray, direction: int,
                       inside_dist: float, outside_dist: float,
                       tolerance: float = 0.05) -> float:
    """
    Binary search for precise edge location in binary mask
    
    Args:
        mask: Binary mask
        center: Center point (y, x)
        perpendicular: Perpendicular direction vector
        direction: -1 for left, 1 for right
        inside_dist: Known distance that's inside mask
        outside_dist: Known distance that's outside mask
        tolerance: Search tolerance in pixels
    
    Returns:
        Precise edge distance
    """
    # Input validation
    if mask is None or mask.size == 0:
        return inside_dist
    
    if not isinstance(center, np.ndarray) or center.size != 2:
        return inside_dist
    
    if not isinstance(perpendicular, np.ndarray) or perpendicular.size != 2:
        return inside_dist
    
    # Validate distances
    if inside_dist < 0 or outside_dist < 0 or inside_dist >= outside_dist:
        return inside_dist
    
    try:
        h, w = mask.shape
        cy, cx = float(center[0]), float(center[1])
    except (ValueError, AttributeError) as e:
        logger.error(f"Error in binary search edge: {e}")
        return inside_dist
    
    # Add maximum iteration limit to prevent infinite loops
    max_iterations = 50
    iteration = 0
    
    while outside_dist - inside_dist > tolerance and iteration < max_iterations:
        mid_dist = (inside_dist + outside_dist) / 2
        
        # Calculate position
        y = cy + direction * perpendicular[0] * mid_dist
        x = cx + direction * perpendicular[1] * mid_dist
        
        # Check bounds
        if not (0 <= x < w and 0 <= y < h):
            outside_dist = mid_dist
            iteration += 1
            continue
        
        # For binary mask, use nearest neighbor
        try:
            y_nearest = int(round(y))
            x_nearest = int(round(x))
            
            if 0 <= y_nearest < h and 0 <= x_nearest < w and mask[y_nearest, x_nearest] > 0:
                inside_dist = mid_dist
            else:
                outside_dist = mid_dist
        except (IndexError, ValueError) as e:
            logger.debug(f"Error accessing mask at ({x_nearest}, {y_nearest}): {e}")
            outside_dist = mid_dist
        
        iteration += 1
    
    if iteration >= max_iterations:
        logger.warning(f"Binary search reached maximum iterations ({max_iterations})")
    
    return inside_dist


def _calculate_dynamic_margin(mask: np.ndarray, center: np.ndarray, 
                            perpendicular: np.ndarray, direction: int,
                            edge_distance: float) -> float:
    """
    Calculate dynamic safety margin based on edge quality
    
    Args:
        mask: Binary mask
        center: Center point
        perpendicular: Perpendicular direction
        direction: -1 for left, 1 for right
        edge_distance: Distance to edge
    
    Returns:
        Safety margin in pixels
    """
    # Base margin
    base_margin = 0.25
    
    # Calculate edge position
    edge_y = center[0] + direction * perpendicular[0] * edge_distance
    edge_x = center[1] + direction * perpendicular[1] * edge_distance
    
    # Sample gradient around edge
    gradient_samples = []
    for offset in [-0.5, 0, 0.5]:
        y = edge_y + direction * perpendicular[0] * offset
        x = edge_x + direction * perpendicular[1] * offset
        
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            # Get interpolated value
            y_int, x_int = int(y), int(x)
            y_frac, x_frac = y - y_int, x - x_int
            
            if y_int + 1 < mask.shape[0] and x_int + 1 < mask.shape[1]:
                val = (1 - y_frac) * (1 - x_frac) * mask[y_int, x_int] + \
                      (1 - y_frac) * x_frac * mask[y_int, x_int + 1] + \
                      y_frac * (1 - x_frac) * mask[y_int + 1, x_int] + \
                      y_frac * x_frac * mask[y_int + 1, x_int + 1]
                gradient_samples.append(val)
    
    if len(gradient_samples) >= 2:
        # Calculate gradient steepness
        gradient = np.max(np.abs(np.diff(gradient_samples)))
        
        # Sharp edge (gradient > 0.8) -> smaller margin
        if gradient > 0.8:
            return base_margin * 0.4  # 0.1 pixel
        # Medium edge (gradient > 0.5)
        elif gradient > 0.5:
            return base_margin * 1.0  # 0.25 pixel
        # Soft edge -> larger margin
        else:
            return base_margin * 2.0  # 0.5 pixel
    
    # Default to base margin if calculation fails
    return base_margin


def _validate_diameter(diameter: float, index: int) -> float:
    """Validate and constrain diameter value"""
    # Maximum reasonable diameter for coronary arteries (40 pixels ≈ 12mm)
    MAX_DIAMETER = 40.0
    MIN_DIAMETER = 0.0  # Allow 0 for complete occlusions
    
    if diameter > MAX_DIAMETER:
        logger.warning(f"Point {index}: Capping diameter from {diameter:.1f} to {MAX_DIAMETER}")
        return MAX_DIAMETER
    elif diameter < MIN_DIAMETER:
        return MIN_DIAMETER
    
    return diameter


def _log_statistics(diameters: np.ndarray):
    """Log diameter statistics"""
    log_diameter_statistics(diameters, prefix="Diameter")


def smooth_diameter_profile(diameters: np.ndarray, window_size: int = 5, preserve_stenosis: bool = True) -> np.ndarray:
    """
    Apply stenosis-preserving smoothing to diameter profile
    
    Args:
        diameters: Raw diameter measurements
        window_size: Size of smoothing window (must be odd) - IGNORED if preserve_stenosis=True
        preserve_stenosis: Whether to preserve stenotic values (recommended: True)
        
    Returns:
        Smoothed diameter profile with preserved stenosis
    """
    if len(diameters) < 3:
        return diameters
    
    if not preserve_stenosis:
        # Legacy smoothing (NOT RECOMMENDED - destroys stenosis!)
        if window_size % 2 == 0:
            window_size += 1
        
        from scipy.ndimage import median_filter, gaussian_filter1d
        smoothed = median_filter(diameters, size=window_size)
        smoothed = gaussian_filter1d(smoothed, sigma=1.0)
        return smoothed
    
    # STENOSIS-PRESERVING SMOOTHING (NEW ALGORITHM)
    from scipy.ndimage import gaussian_filter1d
    import logging
    logger = logging.getLogger(__name__)
    
    original_profile = diameters.copy()
    
    # Identify stenotic regions before any smoothing
    valid_mask = original_profile > 0
    if np.sum(valid_mask) == 0:
        return original_profile
    
    valid_diameters = original_profile[valid_mask]
    mean_val = np.mean(valid_diameters)
    
    # Aggressive stenosis detection (50% reduction indicates stenosis)
    stenosis_threshold = mean_val * 0.5
    stenotic_mask = (original_profile < stenosis_threshold) & valid_mask
    stenotic_indices = np.where(stenotic_mask)[0]
    
    if len(stenotic_indices) > 0:
        # STRICT STENOSIS PRESERVATION
        final_smoothed = original_profile.copy()
        
        # Apply ultra-light smoothing only to non-stenotic regions
        non_stenotic_mask = ~stenotic_mask & valid_mask
        if np.sum(non_stenotic_mask) > 2:
            lightly_smoothed = gaussian_filter1d(original_profile, sigma=0.1)
            final_smoothed[non_stenotic_mask] = lightly_smoothed[non_stenotic_mask]
        
        logger.debug(f"Stenosis-preserving smoothing: {len(stenotic_indices)} points preserved")
    else:
        # No significant stenosis detected, apply conservative smoothing
        final_smoothed = gaussian_filter1d(original_profile, sigma=0.1)
        logger.debug("No stenosis detected - applied ultra-conservative smoothing")
    
    return final_smoothed