#!/usr/bin/env python3
"""
Gradient-based vessel diameter measurement
Alternative to segmentation-based approach using perpendicular intensity profiles
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from src.utils.diameter_utils import log_diameter_statistics

logger = logging.getLogger(__name__)


class GradientDiameterMeasurer:
    """
    Gradient-based vessel diameter measurement using perpendicular intensity profiles.

    Based on latest research (2023-2024) combining:
    - Perpendicular intensity profiles
    - First/second derivative edge detection
    - Zero-crossing detection
    - Multi-scale analysis
    """

    def __init__(
        self,
        profile_length: int = 60,
        smoothing_sigma: float = 1.0,
        min_vessel_width: float = 2.0,
        max_vessel_width: float = 40.0,
        subpixel_sampling: float = 1.0,
    ):
        """
        Initialize gradient-based diameter measurer.

        Args:
            profile_length: Length of perpendicular profile in pixels
            smoothing_sigma: Gaussian smoothing parameter
            min_vessel_width: Minimum expected vessel width in pixels
            max_vessel_width: Maximum expected vessel width in pixels
            subpixel_sampling: Sub-pixel sampling resolution (0.25 = 4x oversampling)
        """
        self.profile_length = profile_length
        self.smoothing_sigma = smoothing_sigma
        self.min_vessel_width = min_vessel_width
        self.max_vessel_width = max_vessel_width
        self.subpixel_sampling = subpixel_sampling

    def measure_diameters(
        self, image: np.ndarray, centerline: np.ndarray, method: str = "second_derivative"
    ) -> Dict[str, np.ndarray]:
        """
        Measure vessel diameters along centerline using gradient-based approach.

        Args:
            image: Input grayscale image
            centerline: Centerline points [(y, x), ...]
            method: Edge detection method ("first_derivative", "second_derivative", "hybrid")

        Returns:
            Dictionary with diameter measurements and edge positions
        """
        logger.info(f"Starting gradient-based diameter measurement with {method} method")

        if len(centerline) < 2:
            return self._empty_result(len(centerline))

        diameters = []
        left_edges = []
        right_edges = []
        perpendiculars = []

        for i, center in enumerate(centerline):
            # Calculate perpendicular direction
            perpendicular = self._calculate_perpendicular(centerline, i)
            if perpendicular is None:
                perpendicular = np.array([0, 1])  # Default horizontal

            # Extract perpendicular intensity profile
            profile, profile_coords = self._extract_perpendicular_profile(
                image, center, perpendicular
            )

            if profile is None:
                diameters.append(0.0)
                left_edges.append(0.0)
                right_edges.append(0.0)
                perpendiculars.append(perpendicular)
                continue

            # Detect edges using specified method
            if method == "first_derivative":
                left_dist, right_dist = self._detect_edges_first_derivative(profile)
            elif method == "second_derivative":
                left_dist, right_dist = self._detect_edges_second_derivative(profile)
            elif method == "hybrid":
                left_dist, right_dist = self._detect_edges_hybrid(profile)
            else:
                raise ValueError(f"Unknown method: {method}")

            diameter = left_dist + right_dist
            diameter = self._validate_diameter(diameter, i)

            diameters.append(diameter)
            left_edges.append(left_dist)
            right_edges.append(right_dist)
            perpendiculars.append(perpendicular)

        result = {
            "diameters": np.array(diameters),
            "left_edges": np.array(left_edges),
            "right_edges": np.array(right_edges),
            "perpendiculars": np.array(perpendiculars),
            "method": method,
        }

        self._log_statistics(result["diameters"], method)
        return result

    def _extract_perpendicular_profile(
        self, image: np.ndarray, center: np.ndarray, perpendicular: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract intensity profile perpendicular to centerline at given point.

        Args:
            image: Input image
            center: Center point (y, x)
            perpendicular: Perpendicular direction vector

        Returns:
            Tuple of (intensity_profile, coordinates) or (None, None) if invalid
        """
        cy, cx = center[0], center[1]
        h, w = image.shape

        # Generate sampling points along perpendicular line with configurable sub-pixel precision
        half_length = self.profile_length // 2
        distances = np.arange(
            -half_length, half_length + 1, self.subpixel_sampling
        )  # Enhanced sub-pixel sampling

        sample_y = cy + distances * perpendicular[0]
        sample_x = cx + distances * perpendicular[1]

        # Filter out-of-bounds points
        valid_mask = (sample_y >= 0) & (sample_y < h - 1) & (sample_x >= 0) & (sample_x < w - 1)

        if not np.any(valid_mask):
            return None, None

        valid_y = sample_y[valid_mask]
        valid_x = sample_x[valid_mask]

        # Sample intensity values using bilinear interpolation
        profile = ndimage.map_coordinates(
            image.astype(np.float32), [valid_y, valid_x], order=1, mode="constant", cval=0.0
        )

        coordinates = np.column_stack([valid_y, valid_x])

        return profile, coordinates

    def _detect_edges_first_derivative(self, profile: np.ndarray) -> Tuple[float, float]:
        """
        Edge detection using first derivative (gradient magnitude peaks).

        Args:
            profile: Intensity profile

        Returns:
            Tuple of (left_distance, right_distance) from center
        """
        if len(profile) < 5:
            return 0.0, 0.0

        # Smooth profile to reduce noise
        smoothed = gaussian_filter1d(profile, self.smoothing_sigma)

        # Calculate first derivative
        gradient = np.gradient(smoothed)

        # Find center index
        center_idx = len(profile) // 2

        # Look for negative peaks (dark-to-light transitions) on left
        # and positive peaks (light-to-dark transitions) on right
        left_candidates = []
        right_candidates = []

        # Left side: look for minimum gradient (vessel edge -> background)
        for i in range(1, center_idx - 1):
            if (
                gradient[i] < gradient[i - 1]
                and gradient[i] < gradient[i + 1]
                and gradient[i] < -0.1
            ):  # Threshold for significant edge
                left_candidates.append(i)

        # Right side: look for maximum gradient (vessel -> background)
        for i in range(center_idx + 1, len(gradient) - 1):
            if (
                gradient[i] > gradient[i - 1]
                and gradient[i] > gradient[i + 1]
                and gradient[i] > 0.1
            ):  # Threshold for significant edge
                right_candidates.append(i)

        # Select best candidates (closest to center with strong gradient)
        left_edge_idx = center_idx
        if left_candidates:
            # Choose closest to center with strongest gradient
            best_left = max(left_candidates, key=lambda x: abs(gradient[x]))
            left_edge_idx = best_left

        right_edge_idx = center_idx
        if right_candidates:
            best_right = max(right_candidates, key=lambda x: abs(gradient[x]))
            right_edge_idx = best_right

        # Convert to distances
        # No need to scale - indices directly represent pixel distances
        left_dist = abs(center_idx - left_edge_idx)
        right_dist = abs(right_edge_idx - center_idx)

        return left_dist, right_dist

    def _detect_edges_second_derivative(self, profile: np.ndarray) -> Tuple[float, float]:
        """
        Edge detection using second derivative zero-crossings.
        Most accurate method based on literature.

        Args:
            profile: Intensity profile

        Returns:
            Tuple of (left_distance, right_distance) from center
        """
        if len(profile) < 7:
            return 0.0, 0.0

        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(profile, self.smoothing_sigma)

        # Calculate first and second derivatives
        first_deriv = np.gradient(smoothed)
        second_deriv = np.gradient(first_deriv)

        center_idx = len(profile) // 2

        # Find zero-crossings in second derivative
        # Zero-crossings indicate inflection points (edges)

        # Left side: look for positive-to-negative zero-crossing
        left_edge_idx = center_idx
        for i in range(center_idx - 1, 0, -1):
            if (
                second_deriv[i - 1] > 0 and second_deriv[i] < 0 and abs(first_deriv[i]) > 0.1
            ):  # Significant gradient at zero-crossing
                # Sub-pixel refinement using linear interpolation
                if abs(second_deriv[i] - second_deriv[i - 1]) > 1e-6:
                    offset = second_deriv[i - 1] / (second_deriv[i - 1] - second_deriv[i])
                    left_edge_idx = i - 1 + offset
                else:
                    left_edge_idx = i
                break

        # Right side: look for negative-to-positive zero-crossing
        right_edge_idx = center_idx
        for i in range(center_idx + 1, len(second_deriv) - 1):
            if (
                second_deriv[i - 1] < 0 and second_deriv[i] > 0 and abs(first_deriv[i]) > 0.1
            ):  # Significant gradient at zero-crossing
                # Sub-pixel refinement
                if abs(second_deriv[i] - second_deriv[i - 1]) > 1e-6:
                    offset = -second_deriv[i - 1] / (second_deriv[i] - second_deriv[i - 1])
                    right_edge_idx = i - 1 + offset
                else:
                    right_edge_idx = i
                break

        # Convert to distances
        # No need to scale - indices directly represent pixel distances
        left_dist = abs(center_idx - left_edge_idx)
        right_dist = abs(right_edge_idx - center_idx)

        return left_dist, right_dist

    def _detect_edges_hybrid(self, profile: np.ndarray) -> Tuple[float, float]:
        """
        Hybrid method: Use second derivative for primary detection,
        first derivative for validation and fallback.

        Args:
            profile: Intensity profile

        Returns:
            Tuple of (left_distance, right_distance) from center
        """
        # Try second derivative first (most accurate)
        left_2nd, right_2nd = self._detect_edges_second_derivative(profile)

        # Get first derivative results for comparison
        left_1st, right_1st = self._detect_edges_first_derivative(profile)

        # Use second derivative if results are reasonable
        # Otherwise fallback to first derivative

        left_dist = left_2nd
        right_dist = right_2nd

        # Validation: if second derivative gives unrealistic results, use first derivative
        total_2nd = left_2nd + right_2nd
        total_1st = left_1st + right_1st

        if (
            total_2nd < self.min_vessel_width
            or total_2nd > self.max_vessel_width
            or total_2nd < 0.5 * total_1st
            or total_2nd > 2.0 * total_1st
        ):
            logger.debug(
                f"Second derivative failed validation, using first derivative: "
                f"2nd={total_2nd:.1f}, 1st={total_1st:.1f}"
            )
            left_dist = left_1st
            right_dist = right_1st

        return left_dist, right_dist

    def _calculate_perpendicular(self, centerline: np.ndarray, index: int) -> Optional[np.ndarray]:
        """Calculate perpendicular direction at given centerline point."""
        n_points = len(centerline)

        if n_points < 2:
            return None

        # Use adaptive window for tangent calculation
        window = min(3, (n_points - 1) // 2)

        if index == 0:
            tangent = centerline[min(index + window, n_points - 1)] - centerline[index]
        elif index == n_points - 1:
            tangent = centerline[index] - centerline[max(index - window, 0)]
        else:
            window = min(window, min(index, n_points - 1 - index))
            tangent = centerline[index + window] - centerline[index - window]

        # Normalize tangent
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            return None

        tangent = tangent / norm

        # Perpendicular is 90 degree rotation: (dy, dx) -> (-dx, dy)
        perpendicular = np.array([-tangent[1], tangent[0]])

        return perpendicular

    def _validate_diameter(self, diameter: float, index: int) -> float:
        """Validate and constrain diameter value."""
        if diameter > self.max_vessel_width:
            logger.warning(
                f"Point {index}: Capping diameter from {diameter:.1f} to {self.max_vessel_width}"
            )
            return self.max_vessel_width
        elif diameter < 0:
            return 0.0
        elif diameter < self.min_vessel_width:
            return max(diameter, 0.3)  # Minimum 0.3 pixel for stenosis detection

        return diameter

    def _empty_result(self, n_points: int) -> Dict[str, np.ndarray]:
        """Return empty result structure."""
        return {
            "diameters": np.zeros(n_points),
            "left_edges": np.zeros(n_points),
            "right_edges": np.zeros(n_points),
            "perpendiculars": np.zeros((n_points, 2)),
            "method": "none",
        }

    def _log_statistics(self, diameters: np.ndarray, method: str):
        """Log diameter measurement statistics."""
        log_diameter_statistics(diameters, prefix="Gradient-based", method=method)


# Convenience functions for integration
def measure_vessel_diameter_gradient(
    image: np.ndarray, centerline: np.ndarray, method: str = "second_derivative"
) -> np.ndarray:
    """
    Convenience function for gradient-based diameter measurement.

    Args:
        image: Input grayscale image
        centerline: Centerline points in (y, x) format
        method: Edge detection method

    Returns:
        Array of diameter measurements
    """
    measurer = GradientDiameterMeasurer()
    result = measurer.measure_diameters(image, centerline, method)
    return result["diameters"]


def measure_vessel_diameter_with_edges_gradient(
    image: np.ndarray, centerline: np.ndarray, method: str = "second_derivative"
) -> Dict[str, np.ndarray]:
    """
    Convenience function returning full gradient-based measurement results.

    Args:
        image: Input grayscale image
        centerline: Centerline points in (y, x) format
        method: Edge detection method

    Returns:
        Dictionary with diameters, edges, and perpendiculars
    """
    measurer = GradientDiameterMeasurer()
    return measurer.measure_diameters(image, centerline, method)
