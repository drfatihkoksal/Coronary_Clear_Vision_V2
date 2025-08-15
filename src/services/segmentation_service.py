"""
Segmentation service for handling vessel segmentation business logic.
"""

import numpy as np
from typing import List, Tuple, Dict
import cv2
import logging
from ..core.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SegmentationService:
    """Service for handling vessel segmentation operations."""

    def __init__(self):
        self.model_manager = ModelManager.instance()

    def segment_vessel(
        self,
        image: np.ndarray,
        user_points: List[Tuple[int, int]],
        output_threshold: float = 0.5,
        use_curvature_resistant_centerline: bool = False,
    ) -> Dict:
        """
        Perform vessel segmentation.

        Args:
            image: Input image array
            user_points: User-selected points along vessel
            output_threshold: Threshold for vessel detection
            use_curvature_resistant_centerline: Whether to use curvature-resistant centerline extraction

        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Get segmentation model
            segmentation_model = self.model_manager.get_segmentation_model()

            if segmentation_model is None:
                raise ValueError("Segmentation model not loaded")

            logger.debug(f"Using segmentation model: {type(segmentation_model).__name__}")
            logger.debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")
            logger.debug(f"User points: {user_points}")

            # Perform vessel segmentation
            segmentation_result = segmentation_model.segment_vessel(
                image,
                user_points,
                use_curvature_resistant_centerline=use_curvature_resistant_centerline,
                use_light_mask_limiting=False,  # Always False - use moderate/tight limiting
            )

            # Check if segmentation was successful
            if not segmentation_result.get("success", False):
                error_msg = segmentation_result.get("error", "Unknown segmentation error")
                raise ValueError(f"Vessel segmentation failed: {error_msg}")

            # Extract mask and confidence from result
            vessel_mask = segmentation_result.get("mask")
            confidence = segmentation_result.get("confidence", 0.0)

            if vessel_mask is None:
                raise ValueError("Segmentation returned no mask")

            # Extract vessel properties
            vessel_properties = self._analyze_vessel(vessel_mask)

            # Get centerline from segmentation result if available
            centerline = segmentation_result.get("centerline")

            # NO CENTERLINE EXTRACTION - QCA will use tracked points
            if centerline is None:
                logger.info("No centerline from AngioPy - QCA will use tracked points instead")
                # Leave centerline as None

            # Get proximal and distal points from segmentation
            proximal_point = segmentation_result.get("proximal_point")
            distal_point = segmentation_result.get("distal_point")

            return {
                "success": True,
                "mask": vessel_mask,  # Also include as 'mask' for compatibility
                "vessel_mask": vessel_mask,
                "confidence": confidence,
                "centerline": centerline,
                "properties": vessel_properties,
                "proximal_point": proximal_point,
                "distal_point": distal_point,
            }

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Segmentation error: {e}")
            logger.debug(f"Full traceback: {error_details}")
            return {"success": False, "error": str(e), "details": error_details}

    def _analyze_vessel(self, mask: np.ndarray) -> Dict:
        """
        Analyze vessel properties from segmentation mask.

        Args:
            mask: Binary vessel mask

        Returns:
            Dictionary of vessel properties
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {}

        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Fit ellipse if possible
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center, (width, height), angle = ellipse
        else:
            center = (0, 0)
            width = height = angle = 0

        return {
            "area": area,
            "perimeter": perimeter,
            "center": center,
            "width": width,
            "height": height,
            "orientation": angle,
        }

    def _extract_centerline(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract vessel centerline using distance transform ridge.

        Args:
            mask: Binary vessel mask

        Returns:
            Array of centerline points
        """
        # Use distance transform to find vessel center
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

        # Threshold to get ridge (vessel center)
        # Use a higher threshold to stay in the center
        threshold = dist.max() * 0.1  # Very low threshold for maximum centerline coverage
        ridge_mask = (dist > threshold).astype(np.uint8)

        # Apply skeletonization to thin the ridge
        from skimage.morphology import skeletonize

        skeleton = skeletonize(ridge_mask > 0)

        # Extract centerline points
        centerline_points = np.column_stack(np.where(skeleton))

        logger.info(
            f"Distance transform centerline extraction found {len(centerline_points)} points"
        )

        # Sort points to form a path
        if len(centerline_points) > 0:
            # Simple ordering - can be improved
            centerline_points = self._order_centerline_points(centerline_points)
            logger.info(f"Ordered centerline has {len(centerline_points)} points")

        return centerline_points

    def _order_centerline_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order centerline points to form a continuous path.

        Args:
            points: Unordered centerline points

        Returns:
            Ordered centerline points
        """
        if len(points) < 2:
            return points

        # Simple nearest neighbor ordering
        ordered = [points[0]]
        remaining = list(points[1:])

        while remaining:
            last_point = ordered[-1]
            distances = [np.linalg.norm(p - last_point) for p in remaining]
            nearest_idx = np.argmin(distances)
            ordered.append(remaining.pop(nearest_idx))

        return np.array(ordered)

    def calculate_vessel_diameter(
        self, mask: np.ndarray, centerline: np.ndarray, point_index: int
    ) -> float:
        """
        Calculate vessel diameter at a specific centerline point.

        Args:
            mask: Binary vessel mask
            centerline: Vessel centerline points
            point_index: Index of centerline point

        Returns:
            Diameter in pixels
        """
        if point_index >= len(centerline):
            return 0

        # Get point and local tangent
        point = centerline[point_index]

        # Calculate tangent direction
        if point_index > 0 and point_index < len(centerline) - 1:
            tangent = centerline[point_index + 1] - centerline[point_index - 1]
        elif point_index == 0:
            tangent = centerline[1] - centerline[0]
        else:
            tangent = centerline[-1] - centerline[-2]

        # Normalize tangent
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 0:
            tangent = tangent / tangent_norm

        # Calculate normal (perpendicular) direction
        normal = np.array([-tangent[1], tangent[0]])

        # Sample along normal direction
        max_dist = 50  # Maximum search distance
        diameter = 0

        for dist in range(1, max_dist):
            # Check positive direction
            pos_point = point + dist * normal
            # Check negative direction
            neg_point = point - dist * normal

            # Check if points are within image bounds
            if (
                0 <= pos_point[0] < mask.shape[0]
                and 0 <= pos_point[1] < mask.shape[1]
                and 0 <= neg_point[0] < mask.shape[0]
                and 0 <= neg_point[1] < mask.shape[1]
            ):

                # Check if we've hit the vessel boundary
                if (
                    mask[int(pos_point[0]), int(pos_point[1])]
                    and mask[int(neg_point[0]), int(neg_point[1])]
                ):
                    diameter = 2 * dist
                else:
                    break

        return diameter
