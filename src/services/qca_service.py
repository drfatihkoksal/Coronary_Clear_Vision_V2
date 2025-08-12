"""
QCA (Quantitative Coronary Analysis) service for business logic.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class QCAService:
    """Service for handling QCA analysis operations."""

    def __init__(self):
        self.calibration_factor: Optional[float] = None

    def set_calibration(self, calibration_factor: float):
        """Set calibration factor for measurements."""
        self.calibration_factor = calibration_factor

    def analyze_stenosis(self,
                        vessel_mask: np.ndarray,
                        centerline: np.ndarray,
                        reference_segments: List[Tuple[int, int]]) -> Dict:
        """
        Analyze vessel stenosis.

        Args:
            vessel_mask: Binary vessel mask
            centerline: Vessel centerline points
            reference_segments: List of (start, end) indices for reference segments

        Returns:
            Dictionary containing stenosis analysis results
        """
        try:
            # Calculate diameter along centerline
            diameters = self._calculate_diameter_profile(vessel_mask, centerline)

            # Calculate reference diameter
            reference_diameter = self._calculate_reference_diameter(
                diameters, reference_segments
            )

            # Find minimal diameter
            min_diameter_idx = np.argmin(diameters)
            min_diameter = diameters[min_diameter_idx]

            # Calculate stenosis percentage
            stenosis_percent = ((reference_diameter - min_diameter) /
                               reference_diameter * 100)

            # Convert to mm if calibrated
            if self.calibration_factor:
                diameters_mm = diameters * self.calibration_factor
                reference_diameter_mm = reference_diameter * self.calibration_factor
                min_diameter_mm = min_diameter * self.calibration_factor
            else:
                diameters_mm = None
                reference_diameter_mm = None
                min_diameter_mm = None

            return {
                'success': True,
                'diameters_px': diameters,
                'diameters_mm': diameters_mm,
                'reference_diameter_px': reference_diameter,
                'reference_diameter_mm': reference_diameter_mm,
                'min_diameter_px': min_diameter,
                'min_diameter_mm': min_diameter_mm,
                'min_diameter_location': min_diameter_idx,
                'stenosis_percent': stenosis_percent,
                'centerline': centerline
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_diameter_profile(self,
                                   mask: np.ndarray,
                                   centerline: np.ndarray) -> np.ndarray:
        """
        Calculate vessel diameter at each centerline point.

        Args:
            mask: Binary vessel mask
            centerline: Vessel centerline points

        Returns:
            Array of diameters
        """
        diameters = []

        for i in range(len(centerline)):
            diameter = self._calculate_diameter_at_point(mask, centerline, i)
            diameters.append(diameter)

        return np.array(diameters)

    def _calculate_diameter_at_point(self,
                                    mask: np.ndarray,
                                    centerline: np.ndarray,
                                    point_idx: int) -> float:
        """
        Calculate vessel diameter at a specific centerline point.

        Args:
            mask: Binary vessel mask
            centerline: Vessel centerline points
            point_idx: Index of the point

        Returns:
            Diameter in pixels
        """
        if point_idx >= len(centerline):
            return 0

        point = centerline[point_idx]

        # Calculate local tangent
        if 0 < point_idx < len(centerline) - 1:
            tangent = centerline[point_idx + 1] - centerline[point_idx - 1]
        elif point_idx == 0:
            tangent = centerline[1] - centerline[0]
        else:
            tangent = centerline[-1] - centerline[-2]

        # Normalize
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm

        # Normal vector (perpendicular)
        normal = np.array([-tangent[1], tangent[0]])

        # Find vessel edges along normal
        edge1 = self._find_edge(mask, point, normal)
        edge2 = self._find_edge(mask, point, -normal)

        if edge1 is not None and edge2 is not None:
            diameter = np.linalg.norm(edge1 - edge2)
        else:
            diameter = 0

        return diameter

    def _find_edge(self,
                   mask: np.ndarray,
                   start_point: np.ndarray,
                   direction: np.ndarray,
                   max_distance: int = 50) -> Optional[np.ndarray]:
        """
        Find vessel edge along a direction.

        Args:
            mask: Binary vessel mask
            start_point: Starting point
            direction: Search direction (normalized)
            max_distance: Maximum search distance

        Returns:
            Edge point or None
        """
        for dist in range(1, max_distance):
            point = start_point + dist * direction

            # Check bounds
            if (0 <= point[0] < mask.shape[0] and
                0 <= point[1] < mask.shape[1]):

                # Check if we've left the vessel
                if not mask[int(point[0]), int(point[1])]:
                    # Return the last point inside the vessel
                    return start_point + (dist - 1) * direction
            else:
                break

        return None

    def _calculate_reference_diameter(self,
                                     diameters: np.ndarray,
                                     reference_segments: List[Tuple[int, int]]) -> float:
        """
        Calculate reference diameter from healthy segments.

        Args:
            diameters: Array of all diameters
            reference_segments: List of (start, end) indices

        Returns:
            Reference diameter
        """
        if not reference_segments:
            # Use the 95th percentile as reference
            return np.percentile(diameters, 95)

        reference_diameters = []
        for start, end in reference_segments:
            segment_diameters = diameters[start:end]
            if len(segment_diameters) > 0:
                reference_diameters.extend(segment_diameters)

        if reference_diameters:
            return np.mean(reference_diameters)
        else:
            return np.percentile(diameters, 95)

    def calculate_flow_metrics(self, analysis_results: Dict) -> Dict:
        """
        Calculate flow-related metrics from QCA analysis.

        Args:
            analysis_results: Results from analyze_stenosis

        Returns:
            Dictionary with flow metrics
        """
        if not analysis_results.get('success'):
            return {}

        stenosis = analysis_results['stenosis_percent']
        
        # Validate stenosis value
        stenosis = max(0.0, min(100.0, stenosis))

        # Simplified flow calculations
        # In reality, these would be more complex
        area_stenosis = (1 - (1 - stenosis/100)**2) * 100

        # Estimate flow reduction (simplified)
        if stenosis < 50:
            flow_reduction = stenosis * 0.5
        elif stenosis < 70:
            flow_reduction = 25 + (stenosis - 50) * 1.5
        else:
            flow_reduction = 55 + (stenosis - 70) * 2

        return {
            'area_stenosis_percent': area_stenosis,
            'estimated_flow_reduction_percent': min(flow_reduction, 100),
            'severity': self._classify_stenosis(stenosis)
        }

    def _classify_stenosis(self, stenosis_percent: float) -> str:
        """Classify stenosis severity."""
        if stenosis_percent < 25:
            return "Minimal"
        elif stenosis_percent < 50:
            return "Mild"
        elif stenosis_percent < 70:
            return "Moderate"
        elif stenosis_percent < 90:
            return "Severe"
        else:
            return "Critical"