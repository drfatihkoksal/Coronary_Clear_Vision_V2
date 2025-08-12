"""
Calibration service for handling calibration business logic.
Separates calibration calculations from UI concerns.
"""

import numpy as np
from typing import List, Tuple, Dict
from ..core.model_manager import ModelManager
from ..utils.subpixel_utils import subpixel_catheter_diameter


class CalibrationService:
    """Service for handling calibration operations."""

    def __init__(self):
        self.model_manager = ModelManager.instance()

    def calibrate_with_catheter(self,
                               image: np.ndarray,
                               user_points: List[Tuple[int, int]],
                               catheter_size: float,
                               output_threshold: float = 0.5) -> Dict:
        """
        Perform catheter-based calibration.

        Args:
            image: Input image array
            user_points: User-selected points for catheter detection
            catheter_size: Known catheter size in mm
            output_threshold: Threshold for catheter detection

        Returns:
            Dictionary containing calibration results
        """
        try:
            # Get segmentation model
            segmentation_model = self.model_manager.get_segmentation_model()

            # Perform catheter segmentation
            catheter_mask, confidence = segmentation_model.segment_catheter(
                image,
                user_points,
                output_threshold=output_threshold
            )

            if catheter_mask is None:
                raise ValueError("Catheter detection failed")

            # Calculate catheter diameter in pixels
            catheter_diameter_px = self._calculate_catheter_diameter(catheter_mask)

            if catheter_diameter_px == 0:
                raise ValueError("Could not measure catheter diameter")

            # Calculate calibration factor (mm/pixel)
            calibration_factor = catheter_size / catheter_diameter_px

            return {
                'success': True,
                'calibration_factor': calibration_factor,
                'catheter_diameter_px': catheter_diameter_px,
                'catheter_mask': catheter_mask,
                'confidence': confidence,
                'catheter_size_mm': catheter_size
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_catheter_diameter(self, mask: np.ndarray) -> float:
        """
        Calculate catheter diameter from segmentation mask with sub-pixel precision.

        Args:
            mask: Binary mask of catheter

        Returns:
            Diameter in pixels with sub-pixel precision
        """
        # Use sub-pixel precision catheter diameter calculation
        diameter = subpixel_catheter_diameter(mask, method='moments')
        
        if diameter == 0.0:
            # Fallback to gradient method if moments fail
            diameter = subpixel_catheter_diameter(mask, method='gradient')
        
        if diameter == 0.0:
            # Final fallback to contour method
            diameter = subpixel_catheter_diameter(mask, method='contours')
        
        return diameter

    def validate_calibration(self, calibration_factor: float) -> bool:
        """
        Validate if calibration factor is within reasonable range.

        Args:
            calibration_factor: Calibration factor in mm/pixel

        Returns:
            True if valid, False otherwise
        """
        # Typical calibration factors for coronary angiography
        # are between 0.1 and 0.5 mm/pixel
        return 0.05 <= calibration_factor <= 1.0

    def apply_calibration(self,
                         measurement_px: float,
                         calibration_factor: float) -> float:
        """
        Apply calibration to convert pixels to mm.

        Args:
            measurement_px: Measurement in pixels
            calibration_factor: Calibration factor in mm/pixel

        Returns:
            Measurement in mm
        """
        return measurement_px * calibration_factor