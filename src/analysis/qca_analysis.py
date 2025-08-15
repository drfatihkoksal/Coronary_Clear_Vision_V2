"""
QCA (Quantitative Coronary Analysis) Module
Simplified and optimized for accurate diameter measurements
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
import traceback
from .minimum_cost_path import MinimumCostPathGenerator

# Setup logger
logger = logging.getLogger(__name__)


class QCAConstants:
    """Configuration parameters for QCA analysis"""

    # Calibration - catheter sizes in mm
    CATHETER_SIZES_MM = {"5F": 1.67, "6F": 2.00, "7F": 2.33, "8F": 2.67}

    # Ribbon Method Parameters
    RIBBON_WIDTH = 1  # Width of the ribbon for diameter measurement
    RIBBON_SEARCH_DISTANCE = 160  # Maximum distance to search for edges
    RIBBON_EDGE_THRESHOLD = 0.1  # Much lower threshold for better edge detection
    RIBBON_USE_ADAPTIVE_THRESHOLD = True  # Enable adaptive threshold for robustness
    RIBBON_USE_GRADIENT_EDGE = True  # Use gradient-based edge detection
    RIBBON_SUBPIXEL_PRECISION = True  # Enable sub-pixel precision

    # Diameter Processing
    DIAMETER_SMOOTHING_SIGMA = 0.5  # Minimal smoothing to preserve values

    # Stenosis Analysis
    EDGE_EXCLUSION_RATIO = 0.10  # Exclude 10% from each end for MLD search (was 5%)
    REFERENCE_DIAMETER_PERCENTILE = 90  # Use 90th percentile for reference diameter calculation
    STENOSIS_BOUNDARY_THRESHOLD = 0.9  # 90% of reference diameter for lesion boundaries
    MIN_DISTANCE_FROM_MLD = 10  # Minimum pixels between MLD and reference (was 8)
    MINIMUM_VESSEL_DIAMETER_MM = (
        0.05  # Minimum allowed vessel diameter in mm (reduced for stenosis detection)
    )
    MINIMUM_VESSEL_DIAMETER_PIXELS = (
        0.3  # Minimum in pixels to avoid noise (reduced for enhanced sensitivity)
    )


class QCAAnalysis:
    """Main class for Quantitative Coronary Analysis"""

    def __init__(self):
        """Initialize QCA analysis module"""
        self.calibration_factor = None  # mm/pixel
        self.last_calibration = None
        self.original_image = None  # Store original image for potential future use

    def calibrate_with_catheter(
        self,
        image: np.ndarray,
        catheter_points: Tuple[Tuple[int, int], Tuple[int, int]],
        catheter_size: str = "6F",
        use_subpixel: bool = True,
    ) -> bool:
        """
        Calibrate using catheter of known size with enhanced precision

        Args:
            image: Angiogram image
            catheter_points: Two points defining catheter diameter
            catheter_size: Catheter size (5F, 6F, 7F, 8F)
            use_subpixel: Whether to use sub-pixel precision

        Returns:
            True if calibration successful
        """
        try:
            # Get catheter size in mm
            catheter_mm = QCAConstants.CATHETER_SIZES_MM.get(catheter_size)
            if not catheter_mm:
                logger.error(f"Unknown catheter size: {catheter_size}")
                return False

            # Calculate pixel distance with sub-pixel precision if enabled
            p1, p2 = catheter_points

            if use_subpixel:
                # Enhanced sub-pixel measurement
                pixel_distance = self._measure_catheter_distance_subpixel(image, p1, p2)
            else:
                # Simple Euclidean distance
                pixel_distance = np.linalg.norm(np.array(p1) - np.array(p2))

            # Enhanced measurement validation
            validation_result = self._validate_catheter_measurement(
                pixel_distance, catheter_size, catheter_mm
            )
            if not validation_result["valid"]:
                logger.error(
                    f"Catheter measurement validation failed: {validation_result['reason']}"
                )
                return False

            # Calculate calibration factor
            self.calibration_factor = catheter_mm / pixel_distance

            # Validate calibration factor
            if not self._validate_calibration_factor(self.calibration_factor):
                logger.error(
                    f"Calibration factor out of range: {self.calibration_factor:.5f} mm/pixel"
                )
                return False

            # Store calibration info
            self.last_calibration = {
                "catheter_size": catheter_size,
                "catheter_mm": catheter_mm,
                "pixel_distance": pixel_distance,
                "mm_per_pixel": self.calibration_factor,
                "subpixel_used": use_subpixel,
                "timestamp": np.datetime64("now"),
                "confidence": self._calculate_calibration_confidence(pixel_distance),
            }

            logger.info(f"Calibration successful: {self.calibration_factor:.5f} mm/pixel")
            logger.info(f"Catheter {catheter_size} = {catheter_mm}mm = {pixel_distance:.2f} pixels")
            logger.info(f"Confidence: {self.last_calibration['confidence']:.1%}")

            return True

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False

    def _measure_catheter_distance_subpixel(
        self, image: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]
    ) -> float:
        """
        Measure catheter distance with sub-pixel precision
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Extract line profile between points
        num_samples = int(np.linalg.norm(np.array(p2) - np.array(p1)) * 2)
        x_values = np.linspace(p1[0], p2[0], num_samples)
        y_values = np.linspace(p1[1], p2[1], num_samples)

        # Use bilinear interpolation for sub-pixel sampling
        from scipy.ndimage import map_coordinates

        profile = map_coordinates(gray.astype(np.float32), [y_values, x_values], order=1)

        # Find edges using gradient
        gradient = np.gradient(profile)

        # Find edge positions with sub-pixel precision
        edge_threshold = np.std(gradient) * 0.5

        # Find first edge (rising)
        first_edge = None
        for i in range(1, len(gradient) - 1):
            if gradient[i] > edge_threshold:
                # Sub-pixel interpolation
                if gradient[i - 1] < gradient[i]:
                    t = (edge_threshold - gradient[i - 1]) / (gradient[i] - gradient[i - 1])
                    first_edge = i - 1 + t
                    break

        # Find second edge (falling)
        second_edge = None
        for i in range(len(gradient) - 2, 0, -1):
            if gradient[i] < -edge_threshold:
                # Sub-pixel interpolation
                if gradient[i + 1] > gradient[i]:
                    t = (-edge_threshold - gradient[i]) / (gradient[i + 1] - gradient[i])
                    second_edge = i + t
                    break

        # Calculate distance
        if first_edge is not None and second_edge is not None:
            # Scale back to original pixel units
            subpixel_distance = (second_edge - first_edge) * len(profile) / num_samples
            logger.info(f"Sub-pixel catheter measurement: {subpixel_distance:.3f} pixels")
            return subpixel_distance
        else:
            # Sub-pixel edge detection failed
            logger.error("Sub-pixel edge detection failed - cannot calibrate")
            return 0.0

    def _validate_catheter_measurement(
        self, pixel_distance: float, catheter_size: str, catheter_mm: float
    ) -> Dict:
        """
        Enhanced validation of catheter measurement with detailed feedback

        Args:
            pixel_distance: Measured distance in pixels
            catheter_size: Catheter size (e.g., '6F')
            catheter_mm: Catheter diameter in mm

        Returns:
            Validation result with detailed feedback
        """
        # Minimum pixel distance for reliable calibration
        min_pixels = 8
        if pixel_distance < min_pixels:
            return {
                "valid": False,
                "reason": f"Measurement too small: {pixel_distance:.1f} pixels (minimum: {min_pixels})",
                "suggestion": "Ensure you are measuring catheter diameter (width), not length",
            }

        # Calculate resulting calibration factor
        cal_factor = catheter_mm / pixel_distance

        # Expected ranges for different imaging systems
        # Typical coronary angiography: 0.1-0.5 mm/pixel
        # High resolution systems: 0.05-0.2 mm/pixel
        # Lower resolution: 0.2-0.8 mm/pixel
        min_factor, max_factor = 0.05, 1.0

        if cal_factor < min_factor:
            return {
                "valid": False,
                "reason": f"Calibration factor too small: {cal_factor:.4f} mm/pixel (min: {min_factor})",
                "suggestion": f"Catheter appears too large ({pixel_distance:.1f} pixels for {catheter_mm}mm). Check measurement accuracy.",
            }

        if cal_factor > max_factor:
            return {
                "valid": False,
                "reason": f"Calibration factor too large: {cal_factor:.4f} mm/pixel (max: {max_factor})",
                "suggestion": f"Catheter appears too small ({pixel_distance:.1f} pixels for {catheter_mm}mm). Measure actual catheter width.",
            }

        # Check for reasonable pixel count based on catheter size
        expected_pixels = self._estimate_expected_pixels(catheter_size)
        if abs(pixel_distance - expected_pixels["mean"]) > expected_pixels["tolerance"]:
            return {
                "valid": True,  # Warning, not error
                "reason": f'Measurement outside expected range: {pixel_distance:.1f} pixels (expected: {expected_pixels["mean"]:.1f}±{expected_pixels["tolerance"]:.1f})',
                "suggestion": "Double-check catheter measurement accuracy",
                "warning": True,
            }

        return {"valid": True, "reason": "Measurement validated successfully"}

    def _estimate_expected_pixels(self, catheter_size: str) -> Dict:
        """
        Estimate expected pixel count for catheter based on typical imaging parameters

        Args:
            catheter_size: Catheter size string

        Returns:
            Expected pixel range
        """
        # Typical ranges based on clinical experience
        expected_ranges = {
            "5F": {"mean": 12, "tolerance": 6},  # 1.67mm
            "6F": {"mean": 15, "tolerance": 7},  # 2.00mm
            "7F": {"mean": 18, "tolerance": 8},  # 2.33mm
            "8F": {"mean": 20, "tolerance": 9},  # 2.67mm
        }

        return expected_ranges.get(catheter_size, {"mean": 15, "tolerance": 8})

    def _validate_calibration_factor(self, factor: float) -> bool:
        """
        Validate if calibration factor is within reasonable range
        """
        # Typical ranges for coronary angiography
        # 0.1-0.5 mm/pixel is common, but allow wider range for different systems
        return 0.05 <= factor <= 1.0

    def _calculate_calibration_confidence(self, pixel_distance: float) -> float:
        """
        Calculate confidence score for calibration based on measurement quality
        """
        # Confidence based on pixel distance (more pixels = more accurate)
        if pixel_distance >= 50:
            return 0.95
        elif pixel_distance >= 30:
            return 0.85
        elif pixel_distance >= 20:
            return 0.75
        elif pixel_distance >= 10:
            return 0.65
        else:
            return 0.50

    def calculate_global_reference_diameter_from_frames(
        self,
        frames_data: List[Dict],
        use_gradient_method: bool = False,
        gradient_method: str = "second_derivative",
    ) -> Optional[float]:
        """
        Calculate global reference diameter from multiple frames

        Args:
            frames_data: List of frame data dicts with 'mask' and 'centerline'
            use_gradient_method: Whether to use gradient-based diameter measurement
            gradient_method: Gradient method to use

        Returns:
            Global reference diameter in mm (75th percentile)
        """
        try:
            logger.info("Calculating global reference diameter from multiple frames...")
            all_diameters_mm = []

            for i, frame_data in enumerate(frames_data):
                try:
                    mask = frame_data.get("mask")
                    centerline = frame_data.get("centerline")

                    if mask is None or centerline is None:
                        logger.warning(f"Frame {i}: Missing mask or centerline, skipping")
                        continue

                    centerline = np.array(centerline)

                    # Validate coordinates and centerline length
                    if len(centerline) < 10:
                        logger.warning(
                            f"Frame {i}: Centerline too short ({len(centerline)} points), skipping"
                        )
                        continue

                    # Validate coordinate system
                    centerline = self._validate_and_fix_coordinates(mask, centerline)

                    # Measure diameters for this frame
                    diameter_result = self._measure_diameters(
                        mask=mask,
                        centerline=centerline,
                        original_image=None,
                        use_gradient_method=use_gradient_method,
                        gradient_method=gradient_method,
                    )

                    if diameter_result is None:
                        logger.warning(f"Frame {i}: Failed to measure diameters, skipping")
                        continue

                    diameters_px = diameter_result["diameters"]
                    if np.all(diameters_px == 0):
                        logger.warning(f"Frame {i}: All diameters are zero, skipping")
                        continue

                    # Apply smoothing
                    diameters_px = self._apply_adaptive_smoothing(diameters_px)

                    # Convert to mm
                    calibration = self.calibration_factor if self.calibration_factor else 1.0
                    diameters_mm = diameters_px * calibration

                    # Add valid diameters to global collection
                    valid_diameters = diameters_mm[diameters_mm > 0]
                    all_diameters_mm.extend(valid_diameters)

                    logger.info(f"Frame {i}: Added {len(valid_diameters)} diameter measurements")

                except Exception as e:
                    logger.error(f"Error processing frame {i} for global reference: {e}")
                    continue

            if len(all_diameters_mm) == 0:
                logger.error("No valid diameter measurements found across all frames")
                return None

            # Calculate global reference diameter (75th percentile)
            global_reference_diameter = np.percentile(all_diameters_mm, 75)

            logger.info(f"GLOBAL REFERENCE CALCULATION COMPLETE:")
            logger.info(f"Total diameter measurements: {len(all_diameters_mm)}")
            logger.info(
                f"Diameter range: {np.min(all_diameters_mm):.2f} - {np.max(all_diameters_mm):.2f} mm"
            )
            logger.info(f"75th percentile (Global Reference): {global_reference_diameter:.2f} mm")

            return global_reference_diameter

        except Exception as e:
            logger.error(f"Failed to calculate global reference diameter: {e}")
            return None

    def analyze_from_angiopy(
        self,
        segmentation_result: Dict,
        original_image: Optional[np.ndarray] = None,
        proximal_point: Optional[Tuple[int, int]] = None,
        distal_point: Optional[Tuple[int, int]] = None,
        tracked_points: Optional[List[Tuple[float, float]]] = None,
        use_tracked_centerline: bool = False,
        use_gradient_method: bool = False,
        gradient_method: str = "second_derivative",
        global_reference_diameter: float = None,
    ) -> Dict:
        """
        Perform QCA using AngioPy segmentation results

        Args:
            segmentation_result: Dict with 'mask' and 'centerline'
            original_image: Original angiography image for centerline-based analysis
            proximal_point: Optional proximal reference point (not used)
            distal_point: Optional distal reference point (not used)
            tracked_points: List of (x, y) tracked points for custom centerline
            use_tracked_centerline: Whether to use tracked points instead of AngioPy centerline
            use_gradient_method: Whether to use gradient-based diameter measurement
            gradient_method: Gradient method ("first_derivative", "second_derivative", "hybrid")
            global_reference_diameter: Pre-calculated global reference diameter (optional)

        Returns:
            QCA analysis results
        """
        try:
            # Store original image if provided
            if original_image is not None:
                self.original_image = original_image
                logger.info("Original image provided for centerline-based analysis")

            # Get mask
            mask = segmentation_result.get("probability")
            if mask is None:
                mask = segmentation_result.get("mask")
            if mask is None:
                return {"success": False, "error": "No mask found in segmentation result"}

            # Determine centerline source
            logger.debug(f"QCA Analysis - use_tracked_centerline: {use_tracked_centerline}")
            logger.debug(
                f"QCA Analysis - tracked_points length: {len(tracked_points) if tracked_points else 0}"
            )

            if use_tracked_centerline and tracked_points:
                logger.info(f"Using tracked points centerline with {len(tracked_points)} points")

                # Generate centerline from tracked points using minimum cost path
                try:
                    cost_path_generator = MinimumCostPathGenerator()
                    centerline = cost_path_generator.generate_centerline_from_tracked_points(
                        mask=mask,
                        tracked_points=tracked_points,
                        smooth_factor=1.0,
                        use_vessel_guidance=True,
                    )

                    # Log quality metrics
                    quality_metrics = cost_path_generator.get_path_quality_metrics(centerline)
                    logger.info(f"Centerline quality metrics: {quality_metrics}")

                except Exception as e:
                    logger.error(f"Failed to generate centerline from tracked points: {e}")
                    return {"success": False, "error": f"Centerline generation failed: {str(e)}"}

            else:
                logger.info("Using AngioPy segmentation centerline")

                # Use AngioPy centerline
                centerline = segmentation_result.get("centerline")
                if centerline is None:
                    return {"success": False, "error": "No centerline found in segmentation result"}

                # Ensure centerline is numpy array
                centerline = np.array(centerline)

            logger.info(f"Starting QCA analysis with {len(centerline)} centerline points")
            logger.debug(
                f"analyze_from_angiopy - global_reference_diameter: {global_reference_diameter}"
            )

            # Perform analysis
            return self.analyze_vessel(
                mask=mask,
                centerline=centerline,
                use_gradient_method=use_gradient_method,
                gradient_method=gradient_method,
                global_reference_diameter=global_reference_diameter,
            )

        except Exception as e:
            logger.error(f"QCA analysis failed: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def analyze_multi_frame_vessel(
        self,
        frame_data: List[Dict],
        use_gradient_method: bool = False,
        gradient_method: str = "second_derivative",
    ) -> List[Dict]:
        """
        Analyze vessel across multiple frames with consistent global reference diameter

        Args:
            frame_data: List of dictionaries containing 'mask', 'centerline', and optionally 'original_image'
            use_gradient_method: Whether to use gradient-based diameter measurement
            gradient_method: Gradient method to use

        Returns:
            List of QCA results for each frame with consistent reference diameter
        """
        try:
            logger.info("=" * 60)
            logger.info(f"Starting Multi-Frame QCA Analysis ({len(frame_data)} frames)")
            logger.info("=" * 60)

            if len(frame_data) == 0:
                return []

            # Step 1: Calculate diameter measurements for all frames
            all_diameters_mm = []
            frame_diameter_data = []
            calibration = self.calibration_factor if self.calibration_factor else 1.0

            logger.info("Measuring diameters across all frames...")

            for i, frame_info in enumerate(frame_data):
                mask = frame_info["mask"]
                centerline = frame_info["centerline"]
                original_image = frame_info.get("original_image", None)

                # Validate and fix coordinates
                centerline = self._validate_and_fix_coordinates(mask, centerline)

                # Measure diameters for this frame
                diameter_result = self._measure_diameters(
                    mask=mask,
                    centerline=centerline,
                    original_image=original_image,
                    use_gradient_method=use_gradient_method,
                    gradient_method=gradient_method,
                )

                if diameter_result is None or np.all(diameter_result["diameters"] == 0):
                    logger.warning(f"Frame {i}: Failed to measure diameters, skipping")
                    continue

                diameters_px = diameter_result["diameters"]
                # Apply adaptive smoothing
                diameters_px = self._apply_adaptive_smoothing(diameters_px)

                # Convert to mm and collect all valid diameters
                diameters_mm = diameters_px * calibration
                valid_diameters = diameters_mm[diameters_mm > 0]

                if len(valid_diameters) > 0:
                    all_diameters_mm.extend(valid_diameters.tolist())
                    logger.info(
                        f"Frame {i}: Collected {len(valid_diameters)} valid diameter measurements"
                    )

                # Store frame data for later processing
                frame_diameter_data.append(
                    {
                        "frame_index": i,
                        "mask": mask,
                        "centerline": centerline,
                        "diameters_px": diameters_px,
                        "diameters_mm": diameters_mm,
                        "diameter_result": diameter_result,
                    }
                )

            if len(all_diameters_mm) == 0:
                logger.error("No valid diameter measurements found across all frames")
                return []

            # Step 2: Calculate global reference diameter (75th percentile of all diameters)
            global_reference_diameter = np.percentile(all_diameters_mm, 75)

            logger.info("=" * 50)
            logger.info(f"GLOBAL REFERENCE CALCULATION:")
            logger.info(f"Total diameter measurements: {len(all_diameters_mm)}")
            logger.info(
                f"Diameter range: {np.min(all_diameters_mm):.2f} - {np.max(all_diameters_mm):.2f} mm"
            )
            logger.info(f"75th percentile (Global Reference): {global_reference_diameter:.2f} mm")
            logger.info("=" * 50)

            # Step 3: Analyze each frame using the global reference diameter
            results = []

            for frame_data_item in frame_diameter_data:
                frame_index = frame_data_item["frame_index"]
                centerline = frame_data_item["centerline"]
                diameters_px = frame_data_item["diameters_px"]
                diameter_result = frame_data_item["diameter_result"]

                logger.info(
                    f"Processing frame {frame_index} with global reference {global_reference_diameter:.2f}mm"
                )

                # Calculate stenosis using global reference
                stenosis_results = self._calculate_stenosis(
                    centerline,
                    diameters_px,
                    calibration,
                    global_reference_diameter=global_reference_diameter,
                )

                if not stenosis_results["success"]:
                    logger.warning(
                        f"Frame {frame_index}: Stenosis calculation failed: {stenosis_results.get('error', 'Unknown')}"
                    )
                    continue

                # Prepare complete results for this frame
                frame_results = self._prepare_results(
                    centerline,
                    diameters_px,
                    calibration,
                    stenosis_results,
                    diameter_result["left_edges"],
                    diameter_result["right_edges"],
                    diameter_result["perpendiculars"],
                )

                # Add frame index and global reference info
                frame_results["frame_index"] = frame_index
                frame_results["global_reference_diameter"] = global_reference_diameter
                frame_results["total_diameter_measurements"] = len(all_diameters_mm)

                results.append(frame_results)

                logger.info(
                    f"Frame {frame_index}: {frame_results['percent_stenosis']:.1f}% stenosis (MLD: {frame_results['mld']:.3f}mm)"
                )

            logger.info("=" * 60)
            logger.info(f"Multi-Frame QCA Analysis Complete: {len(results)} frames processed")
            logger.info(f"Global Reference: {global_reference_diameter:.2f}mm used for all frames")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Multi-frame vessel analysis failed: {e}\n{traceback.format_exc()}")
            return []

    def analyze_vessel(
        self,
        mask: np.ndarray,
        centerline: np.ndarray,
        use_gradient_method: bool = False,
        gradient_method: str = "second_derivative",
        global_reference_diameter: float = None,
    ) -> Dict:
        """
        Main vessel analysis method

        Args:
            mask: Vessel segmentation mask
            centerline: Vessel centerline points (y, x)
            use_gradient_method: Whether to use gradient-based diameter measurement
            gradient_method: Gradient method to use
            global_reference_diameter: Pre-calculated global reference diameter (if None, will calculate from current frame)

        Returns:
            Complete QCA results
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting QCA Vessel Analysis")
            logger.debug(
                f"analyze_vessel - Received global_reference_diameter: {global_reference_diameter}"
            )
            logger.info("=" * 60)

            # Step 1: Validate inputs
            if len(centerline) < 10:
                return {"success": False, "error": "Centerline too short for analysis"}

            # Log mask and centerline coordinate system check
            logger.info(f"COORDINATE SYSTEM CHECK:")
            logger.info(f"Mask shape: {mask.shape}")
            logger.info(f"Centerline shape: {centerline.shape}")
            logger.info(
                f"Centerline bounds - y: [{centerline[:, 0].min():.1f}, {centerline[:, 0].max():.1f}], x: [{centerline[:, 1].min():.1f}, {centerline[:, 1].max():.1f}]"
            )

            # Validate coordinate system consistency
            centerline = self._validate_and_fix_coordinates(mask, centerline)

            # Normalize centerline to fixed number of points for consistency
            logger.info(f"Original centerline has {len(centerline)} points")
            centerline = self._normalize_centerline_length(centerline, target_points=100)
            logger.info(f"Using normalized centerline with {len(centerline)} points")

            # Step 2: Measure vessel diameters
            logger.info("Measuring vessel diameters...")
            logger.info(f"Mask shape for diameter measurement: {mask.shape}")
            logger.info(f"Centerline shape for diameter measurement: {centerline.shape}")
            logger.info(
                f"Centerline sample points: {centerline[:3] if len(centerline) > 3 else centerline}"
            )
            diameter_result = self._measure_diameters(
                mask=mask,
                centerline=centerline,
                original_image=getattr(self, "original_image", None),
                use_gradient_method=use_gradient_method,
                gradient_method=gradient_method,
            )

            if diameter_result is None or np.all(diameter_result["diameters"] == 0):
                return {"success": False, "error": "Failed to measure vessel diameters"}

            diameters_px = diameter_result["diameters"]
            left_edges = diameter_result["left_edges"]
            right_edges = diameter_result["right_edges"]
            perpendiculars = diameter_result["perpendiculars"]

            # Log diameter statistics
            valid_diameters = diameters_px[diameters_px > 0]
            logger.info(
                f"Diameter measurements: {len(valid_diameters)} valid of {len(diameters_px)} total"
            )
            logger.info(
                f"Diameter range: {np.min(valid_diameters):.1f} - {np.max(valid_diameters):.1f} pixels"
            )
            logger.info(f"Mean diameter: {np.mean(valid_diameters):.1f} pixels")

            # Debug: Log diameter values in mm for RWS investigation
            if self.calibration_factor:
                diameters_mm_debug = valid_diameters * self.calibration_factor
                logger.debug(
                    f"Diameter range in mm: {np.min(diameters_mm_debug):.2f} - {np.max(diameters_mm_debug):.2f} mm"
                )
                logger.debug(f"Mean diameter in mm: {np.mean(diameters_mm_debug):.2f} mm")
                logger.debug(f"Calibration factor: {self.calibration_factor:.5f} mm/pixel")

                # Warn if diameters seem unrealistic for coronary arteries
                if np.mean(diameters_mm_debug) > 5.0:
                    logger.warning(
                        f"Mean diameter {np.mean(diameters_mm_debug):.2f}mm is high for coronary arteries (normal: 2-4mm)"
                    )
                    logger.warning(f"Check calibration accuracy or segmentation width")

            # Step 3: Apply adaptive smoothing
            diameters_px = self._apply_adaptive_smoothing(diameters_px)
            logger.info("Applied adaptive diameter smoothing")

            # Step 4: Use provided global reference diameter or calculate it
            calibration = self.calibration_factor if self.calibration_factor else 1.0

            # Calculate frame-based reference diameter (always calculate for 1st phase)
            diameters_mm = diameters_px * calibration
            valid_indices = np.where(diameters_mm > 0)[0]

            if len(valid_indices) == 0:
                return {
                    "success": False,
                    "error": "No valid diameter measurements for reference calculation",
                }

            # Calculate frame-based reference (75th percentile of current frame)
            frame_reference_diameter = np.percentile(diameters_mm[valid_indices], 75)
            logger.info(
                f"[FRAME REF] Calculated frame-based reference diameter: {frame_reference_diameter:.2f}mm (75th percentile)"
            )

            if global_reference_diameter is not None:
                logger.info(
                    f"[GLOBAL REF] Using provided global reference diameter: {global_reference_diameter:.2f}mm"
                )
                final_global_ref = global_reference_diameter
            else:
                logger.info(
                    "[GLOBAL REF] No global reference provided - using frame-based reference for stenosis calculation"
                )
                final_global_ref = frame_reference_diameter

            # Step 5: Calculate stenosis using appropriate reference
            logger.info("Calculating stenosis parameters...")
            logger.debug(f"Using reference for stenosis calculation: {final_global_ref:.2f}mm")
            stenosis_results = self._calculate_stenosis(
                centerline, diameters_px, calibration, final_global_ref
            )

            # Add frame-based reference to results (for global reference calculation)
            stenosis_results["frame_reference_diameter"] = frame_reference_diameter

            if not stenosis_results["success"]:
                return stenosis_results

            # Step 5: Prepare complete results
            results = self._prepare_results(
                centerline,
                diameters_px,
                calibration,
                stenosis_results,
                left_edges,
                right_edges,
                perpendiculars,
            )

            logger.info(f"QCA Analysis Complete: {results['percent_stenosis']:.1f}% stenosis")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Vessel analysis failed: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def _validate_and_fix_coordinates(self, mask: np.ndarray, centerline: np.ndarray) -> np.ndarray:
        """
        Validate and fix coordinate system consistency between mask and centerline

        Args:
            mask: Segmentation mask
            centerline: Centerline points (y, x format)

        Returns:
            Fixed centerline coordinates
        """
        # Check bounds
        max_y, max_x = centerline[:, 0].max(), centerline[:, 1].max()
        min_y, min_x = centerline[:, 0].min(), centerline[:, 1].min()

        # Check if coordinates are within bounds with safety margin
        margin = 1.0
        if (
            max_y >= mask.shape[0] - margin
            or max_x >= mask.shape[1] - margin
            or min_y < margin
            or min_x < margin
        ):

            logger.error(f"COORDINATE MISMATCH DETECTED:")
            logger.error(f"Mask dimensions: {mask.shape[0]}x{mask.shape[1]}")
            logger.error(
                f"Centerline bounds: y[{min_y:.1f}, {max_y:.1f}], x[{min_x:.1f}, {max_x:.1f}]"
            )

            # Instead of scaling, clip to valid bounds
            logger.warning("Clipping centerline to mask bounds")
            centerline = centerline.copy().astype(np.float32)

            # Clip with safety margin
            centerline[:, 0] = np.clip(centerline[:, 0], margin, mask.shape[0] - margin - 1)
            centerline[:, 1] = np.clip(centerline[:, 1], margin, mask.shape[1] - margin - 1)

            logger.info(
                f"Clipped centerline bounds: y[{centerline[:, 0].min():.1f}, {centerline[:, 0].max():.1f}], x[{centerline[:, 1].min():.1f}, {centerline[:, 1].max():.1f}]"
            )

        return centerline

    def _normalize_centerline_length(
        self, centerline: np.ndarray, target_points: int = 100
    ) -> np.ndarray:
        """
        Normalize centerline to a fixed number of points for consistency

        Args:
            centerline: Original centerline points (y, x format)
            target_points: Target number of points (default 100)

        Returns:
            Normalized centerline with fixed number of points
        """
        try:
            from scipy.interpolate import interp1d

            # If centerline is already close to target, return as is
            if abs(len(centerline) - target_points) <= 5:
                logger.debug(
                    f"Centerline already has {len(centerline)} points, close to target {target_points}"
                )
                return centerline

            # Calculate cumulative distances along centerline
            distances = np.cumsum(np.sqrt(np.sum(np.diff(centerline, axis=0) ** 2, axis=1)))
            distances = np.insert(distances, 0, 0)

            # Total centerline length
            total_length = distances[-1]

            # Create target distances for uniform sampling
            target_distances = np.linspace(0, total_length, target_points)

            # Interpolate centerline coordinates
            interp_y = interp1d(
                distances,
                centerline[:, 0],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interp_x = interp1d(
                distances,
                centerline[:, 1],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Generate normalized centerline
            normalized = np.column_stack([interp_y(target_distances), interp_x(target_distances)])

            logger.info(f"Normalized centerline from {len(centerline)} to {target_points} points")
            logger.debug(f"Original length: {total_length:.1f} pixels, preserved in normalization")

            return normalized.astype(np.float32)

        except Exception as e:
            logger.warning(f"Centerline normalization failed: {e}, using original")
            return centerline

    def _apply_adaptive_smoothing(self, diameters_px: np.ndarray) -> np.ndarray:
        """
        Apply adaptive smoothing based on diameter profile characteristics

        Args:
            diameters_px: Raw diameter measurements in pixels

        Returns:
            Smoothed diameter profile
        """
        if len(diameters_px) < 5:
            return diameters_px

        # Analyze diameter profile characteristics
        valid_diameters = diameters_px[diameters_px > 0]
        if len(valid_diameters) == 0:
            return diameters_px

        # Calculate coefficient of variation
        mean_diameter = np.mean(valid_diameters)
        std_diameter = np.std(valid_diameters)
        cv = std_diameter / mean_diameter if mean_diameter > 0 else 0

        # Adaptive window size based on profile characteristics
        if cv < 0.1:  # Very smooth profile
            window_size = 3  # Minimal smoothing
        elif cv < 0.2:  # Moderately variable
            window_size = 5  # Standard smoothing
        else:  # Highly variable
            window_size = 7  # More aggressive smoothing

        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1

        logger.info(f"Adaptive smoothing: CV={cv:.3f}, window_size={window_size}")

        # Apply smoothing with stenosis preservation
        return self._smooth_with_stenosis_preservation(diameters_px, window_size)

    def _smooth_with_stenosis_preservation(
        self, diameters_px: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Apply conservative smoothing while strictly preserving stenotic values

        Args:
            diameters_px: Diameter measurements
            window_size: Smoothing window size

        Returns:
            Smoothed profile with preserved stenosis
        """
        from scipy.ndimage import gaussian_filter1d

        # CONSERVATIVE APPROACH: Start with raw data to preserve stenosis
        original_profile = diameters_px.copy()

        # Identify stenotic regions before any smoothing destroys them
        valid_mask = original_profile > 0
        if np.sum(valid_mask) == 0:
            return original_profile

        # Use raw data statistics for stenosis detection
        valid_diameters = original_profile[valid_mask]
        mean_val = np.mean(valid_diameters)

        # More aggressive stenosis detection (50% reduction threshold instead of 20%)
        stenosis_threshold = mean_val * 0.5  # 50% reduction indicates potential stenosis
        stenotic_mask = (original_profile < stenosis_threshold) & valid_mask
        stenotic_indices = np.where(stenotic_mask)[0]

        if len(stenotic_indices) > 0:
            # STRICT STENOSIS PRESERVATION: Keep original values for stenotic regions
            final_smoothed = original_profile.copy()

            # Apply very light smoothing only to non-stenotic regions
            non_stenotic_mask = ~stenotic_mask & valid_mask
            if np.sum(non_stenotic_mask) > 2:  # Need at least 3 points for smoothing
                # Ultra-light Gaussian smoothing (sigma=0.1 instead of 0.3-0.5)
                lightly_smoothed = gaussian_filter1d(original_profile, sigma=0.1)
                final_smoothed[non_stenotic_mask] = lightly_smoothed[non_stenotic_mask]

            logger.info(
                f"STENOSIS PRESERVED: {len(stenotic_indices)} points kept at original values"
            )
            logger.info(
                f"Stenosis threshold: {stenosis_threshold:.3f}mm (50% of mean {mean_val:.3f}mm)"
            )

            # Log preserved minimum
            preserved_min = np.min(final_smoothed[valid_mask])
            original_min = np.min(original_profile[valid_mask])
            logger.info(f"Minimum preserved: {original_min:.3f}mm -> {preserved_min:.3f}mm")
        else:
            # No significant stenosis detected, apply conservative smoothing
            final_smoothed = gaussian_filter1d(original_profile, sigma=0.1)  # Very light (was 0.5)
            logger.info(f"No stenosis detected - applied ultra-conservative smoothing (sigma=0.1)")

        return final_smoothed

    def _interpolate_centerline(
        self, centerline: np.ndarray, target_points: int = 50
    ) -> np.ndarray:
        """
        Interpolate centerline to have exactly target_points.
        This ensures anatomical correspondence across different segmentations
        by sampling at fixed percentages of vessel length (0%, 2%, 4%... 100%).

        Args:
            centerline: Original centerline points (N x 2)
            target_points: Desired number of points (50 = every 2% of length)

        Returns:
            Interpolated centerline with target_points
        """
        if len(centerline) >= target_points:
            # If we already have enough points, subsample
            indices = np.linspace(0, len(centerline) - 1, target_points, dtype=int)
            return centerline[indices]

        # Calculate cumulative distance along centerline
        distances = np.zeros(len(centerline))
        for i in range(1, len(centerline)):
            distances[i] = distances[i - 1] + np.linalg.norm(centerline[i] - centerline[i - 1])

        # Create evenly spaced points along the total distance
        total_distance = distances[-1]
        target_distances = np.linspace(0, total_distance, target_points)

        # Interpolate x and y coordinates separately
        from scipy.interpolate import interp1d

        # Create interpolation functions
        # centerline format is (y, x) so centerline[:, 0] is y and centerline[:, 1] is x
        fy = interp1d(distances, centerline[:, 0], kind="linear", fill_value="extrapolate")
        fx = interp1d(distances, centerline[:, 1], kind="linear", fill_value="extrapolate")

        # Generate interpolated points
        new_y = fy(target_distances)
        new_x = fx(target_distances)

        # Combine into new centerline
        # Keep same format as input centerline (y, x)
        interpolated_centerline = np.column_stack((new_y, new_x))

        # Debug: Check if interpolation produces valid coordinates
        logger.info(f"Interpolated centerline shape: {interpolated_centerline.shape}")
        logger.info(f"First few interpolated points: {interpolated_centerline[:3]}")

        return interpolated_centerline

    def _measure_diameters(
        self,
        mask: np.ndarray,
        centerline: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        use_gradient_method: bool = False,
        gradient_method: str = "second_derivative",
    ) -> Optional[Dict]:
        """
        Measure vessel diameters using either segmentation-based or gradient-based method.

        Args:
            mask: Segmentation mask
            centerline: Vessel centerline points
            original_image: Original grayscale image (needed for gradient method)
            use_gradient_method: If True, use gradient-based measurement instead of mask
            gradient_method: Gradient method ("first_derivative", "second_derivative", "hybrid")

        Returns:
            Dictionary with diameter measurements and edge information
        """
        try:
            if use_gradient_method and original_image is not None:
                logger.info(f"Using gradient-based diameter measurement ({gradient_method})")
                from .gradient_diameter_measurement import (
                    measure_vessel_diameter_with_edges_gradient,
                )

                result = measure_vessel_diameter_with_edges_gradient(
                    image=original_image, centerline=centerline, method=gradient_method
                )

                logger.info(
                    f"Gradient-based measurement completed with method: {result.get('method', 'unknown')}"
                )

            else:
                logger.info("Using segmentation-based diameter measurement with edge positions")
                from .vessel_diameter_accurate import measure_vessel_diameter_with_edges

                result = measure_vessel_diameter_with_edges(
                    segmentation_map=mask, centerline=centerline, max_search_distance=30
                )

            # Log statistics
            diameters = result["diameters"]
            non_zero = diameters[diameters > 0]
            if len(non_zero) > 0:
                method_name = "gradient" if use_gradient_method else "segmentation"
                logger.info(
                    f"{method_name}-based diameter range: {np.min(non_zero):.1f} - {np.max(non_zero):.1f} pixels"
                )
                logger.info(f"{method_name}-based mean diameter: {np.mean(non_zero):.1f} pixels")

            return result

        except Exception as e:
            logger.error(f"Diameter measurement failed: {e}")
            logger.error(traceback.format_exc())
            return None

    def _smooth_diameters(self, diameters: np.ndarray, smoothing_window: int = 3) -> np.ndarray:
        """
        Apply smoothing to diameter measurements while preserving MLD.

        Args:
            diameters: Array of diameter measurements
            smoothing_window: Window size for smoothing (must be odd)

        Returns:
            Smoothed diameter array
        """
        from scipy.ndimage import median_filter

        # Ensure window is odd
        if smoothing_window % 2 == 0:
            smoothing_window += 1

        # Find MLD location before smoothing
        valid_indices = np.where(diameters > 0)[0]
        if len(valid_indices) == 0:
            return diameters

        valid_diameters = diameters[valid_indices]
        mld_idx_in_valid = np.argmin(valid_diameters)
        mld_idx = valid_indices[mld_idx_in_valid]
        original_mld = diameters[mld_idx]

        # Apply median filter for robustness against outliers
        smoothed = median_filter(diameters, size=smoothing_window, mode="nearest")

        # Preserve MLD value to ensure stenosis calculation accuracy
        smoothed[mld_idx] = original_mld

        # Apply additional Gaussian smoothing for continuity
        from scipy.ndimage import gaussian_filter1d

        smoothed = gaussian_filter1d(smoothed, sigma=0.5)

        # Again preserve MLD after Gaussian smoothing
        smoothed[mld_idx] = original_mld

        # Ensure non-negative values
        smoothed = np.maximum(smoothed, 0)

        logger.debug(
            f"Diameter smoothing applied: window={smoothing_window}, preserved MLD at index {mld_idx}"
        )

        return smoothed

    def _calculate_stenosis(
        self,
        centerline: np.ndarray,
        diameters_px: np.ndarray,
        cal_factor: float,
        global_reference_diameter: float = None,
    ) -> Dict:
        """
        Calculate stenosis parameters from diameter profile

        Args:
            centerline: Vessel centerline points
            diameters_px: Diameter measurements in pixels
            cal_factor: Calibration factor (mm/pixel)
            global_reference_diameter: Pre-calculated global reference diameter for multi-frame analysis
        """

        # Convert to mm
        diameters_mm = diameters_px * cal_factor

        # Find valid diameter indices - include 0 values for complete occlusions
        # But exclude negative values (measurement errors)
        valid_indices = np.where(diameters_mm >= 0)[0]
        if len(valid_indices) == 0:
            return {"success": False, "error": "No valid diameter measurements"}

        # Apply edge exclusion for MLD search
        # Exclude percentage from each end based on vessel length
        edge_exclude = max(5, int(len(centerline) * QCAConstants.EDGE_EXCLUSION_RATIO))

        # Only apply if we have enough points
        if len(centerline) > 2 * edge_exclude + 5:
            search_indices = valid_indices[
                (valid_indices >= edge_exclude) & (valid_indices < len(centerline) - edge_exclude)
            ]
        else:
            # For very short vessels, use all valid indices
            search_indices = valid_indices

        if len(search_indices) == 0:
            search_indices = valid_indices  # Use all if too restrictive

        # Find MLD (Minimal Lumen Diameter)
        # Log diameter values in search region (debug level)
        search_diameters = diameters_mm[search_indices]
        logger.info(f"MLD search region: {len(search_indices)} indices")
        logger.info(
            f"Search diameter range: {np.min(search_diameters):.2f} - {np.max(search_diameters):.2f}mm"
        )

        # Find minimum
        min_search_idx = np.argmin(search_diameters)
        mld_idx = search_indices[min_search_idx]
        mld = diameters_mm[mld_idx]
        mld_location = centerline[mld_idx]

        logger.info(f"MLD found: {mld:.3f}mm at index {mld_idx}")
        logger.info(f"MLD diameter before smoothing check - investigating smoothing impact...")

        # Debug: Log MLD in pixels for RWS investigation
        mld_px = diameters_px[mld_idx]
        logger.debug(f"MLD in pixels: {mld_px:.1f} px")
        logger.debug(f"MLD in mm: {mld:.3f} mm (using calibration factor {cal_factor:.5f})")

        # Use provided reference diameter (could be frame-based or global)
        if global_reference_diameter is not None:
            reference_diameter = global_reference_diameter
            logger.info(f"Using provided reference diameter: {reference_diameter:.2f}mm")
        else:
            # Should not happen - reference should always be provided
            logger.error("No reference diameter provided! Using frame 75th percentile as fallback")
            reference_diameter = np.percentile(diameters_mm[valid_indices], 75)
            logger.warning(f"Fallback reference diameter: {reference_diameter:.2f}mm")

        # For compatibility, set individual references to None (not used anymore)
        prox_ref, prox_idx = None, None
        dist_ref, dist_idx = None, None

        # Validate measurements before calculating stenosis
        if reference_diameter <= 0:
            logger.warning("Invalid reference diameter: cannot calculate stenosis")
            return {"success": False, "error": "Invalid reference diameter"}

        if mld < 0:
            logger.warning("Invalid MLD: negative diameter detected")
            return {"success": False, "error": "Invalid MLD measurement"}

        # Check for measurement inconsistency
        if mld > reference_diameter:
            logger.warning(
                f"MLD ({mld:.3f}mm) > Reference ({reference_diameter:.2f}mm), using reference as upper bound"
            )
            # This suggests measurement error - cap stenosis at 0%
            percent_stenosis = 0.0
        else:
            # Calculate percent stenosis with proper bounds checking
            # Consider anything below 0.1mm as critical stenosis (>95%)
            if mld < 0.1:
                percent_stenosis = (reference_diameter - mld) / reference_diameter * 100
                percent_stenosis = max(95.0, min(100.0, percent_stenosis))
                logger.info(
                    f"Critical stenosis detected: MLD {mld:.3f}mm < 0.1mm, stenosis: {percent_stenosis:.1f}%"
                )
            else:
                percent_stenosis = (reference_diameter - mld) / reference_diameter * 100
                # Clamp stenosis to reasonable bounds (0-100%)
                percent_stenosis = max(0.0, min(100.0, percent_stenosis))

        # Log stenosis calculation details
        logger.debug(
            f"Stenosis calculation - MLD: {mld:.3f}mm, Reference: {reference_diameter:.3f}mm"
        )
        logger.debug(
            f"Stenosis formula: ({reference_diameter:.3f} - {mld:.3f}) / {reference_diameter:.3f} * 100"
        )
        logger.debug(f"Calculated stenosis: {percent_stenosis:.2f}%")

        # Find lesion boundaries
        lesion_length, prox_bound, dist_bound = self._find_lesion_boundaries(
            diameters_mm, mld_idx, reference_diameter
        )

        # Calculate areas with validation
        if mld >= 0:
            mla = np.pi * (mld / 2) ** 2
        else:
            logger.warning("Negative MLD for area calculation, setting MLA to 0")
            mla = 0.0

        reference_area = np.pi * (reference_diameter / 2) ** 2

        # Calculate area stenosis with bounds checking
        if reference_area > 0:
            percent_area_stenosis = (reference_area - mla) / reference_area * 100
            # Clamp area stenosis to reasonable bounds (0-100%)
            percent_area_stenosis = max(0.0, min(100.0, percent_area_stenosis))
        else:
            percent_area_stenosis = 0.0

        # Find stenosis boundaries (P and D points) at 75% of reference
        stenosis_threshold = 0.75 * reference_diameter

        # Find P point (proximal boundary)
        p_point = 0
        for i in range(mld_idx, -1, -1):
            if i in valid_indices and diameters_mm[i] >= stenosis_threshold:
                p_point = i
                break

        # Find D point (distal boundary)
        d_point = len(diameters_mm) - 1
        for i in range(mld_idx, len(diameters_mm)):
            if i in valid_indices and diameters_mm[i] >= stenosis_threshold:
                d_point = i
                break

        logger.info(f"Stenosis boundaries at 75%: P={p_point}, MLD={mld_idx}, D={d_point}")

        # Prepare stenosis boundaries info
        stenosis_boundaries = {
            "p_point": int(p_point),
            "d_point": int(d_point),
            "mld_point": int(mld_idx),
            "reference_diameter": float(reference_diameter),
            "threshold": float(stenosis_threshold),
            "p_diameter": float(diameters_mm[p_point]) if p_point < len(diameters_mm) else 0,
            "d_diameter": float(diameters_mm[d_point]) if d_point < len(diameters_mm) else 0,
        }

        return {
            "success": True,
            "mld": float(mld),
            "mld_index": int(mld_idx),
            "mld_location": mld_location.tolist(),
            "reference_diameter": float(reference_diameter),
            "proximal_reference": float(prox_ref) if prox_ref else None,
            "distal_reference": float(dist_ref) if dist_ref else None,
            "proximal_ref_index": int(prox_idx) if prox_idx is not None else None,
            "distal_ref_index": int(dist_idx) if dist_idx is not None else None,
            "percent_stenosis": float(percent_stenosis),
            "lesion_length": float(lesion_length),
            "proximal_boundary_index": int(prox_bound),
            "distal_boundary_index": int(dist_bound),
            "mla": float(mla),
            "reference_area": float(reference_area),
            "percent_area_stenosis": float(percent_area_stenosis),
            "stenosis_boundaries": stenosis_boundaries,
        }

    def _find_reference_diameter(
        self, diameters_mm: np.ndarray, valid_indices: np.ndarray, mld_idx: int, side: str
    ) -> Tuple[Optional[float], Optional[int]]:
        """Find reference diameter on proximal or distal side"""

        min_distance = QCAConstants.MIN_DISTANCE_FROM_MLD

        # Exclude percentage from each end from reference calculation
        edge_exclude = max(5, int(len(diameters_mm) * QCAConstants.EDGE_EXCLUSION_RATIO))

        if side == "proximal":
            # Look for reference before MLD, excluding first 5 measurements
            ref_indices = valid_indices[
                (valid_indices < mld_idx)
                & (valid_indices <= mld_idx - min_distance)
                & (valid_indices >= edge_exclude)
            ]
        else:
            # Look for reference after MLD, excluding last 5 measurements
            total_length = len(diameters_mm)
            ref_indices = valid_indices[
                (valid_indices > mld_idx)
                & (valid_indices >= mld_idx + min_distance)
                & (valid_indices < total_length - edge_exclude)
            ]

        if len(ref_indices) < 3:
            return None, None

        # Use percentile of diameters in reference region
        ref_diameters = diameters_mm[ref_indices]

        # Use adaptive percentile based on data distribution
        ref_std = np.std(ref_diameters)
        ref_mean = np.mean(ref_diameters)

        # Use more conservative percentile for stenotic vessels
        if ref_std / ref_mean > 0.15:  # Coefficient of variation > 15%
            percentile = 60  # More conservative for variable diameters
        else:
            percentile = QCAConstants.REFERENCE_DIAMETER_PERCENTILE

        ref_diameter = np.percentile(ref_diameters, percentile)

        # Find index of diameter closest to reference value
        ref_idx = ref_indices[np.argmin(np.abs(ref_diameters - ref_diameter))]

        return ref_diameter, ref_idx

    def _find_lesion_boundaries(
        self, diameters_mm: np.ndarray, mld_idx: int, reference_diameter: float
    ) -> Tuple[float, int, int]:
        """Find proximal and distal boundaries of stenotic lesion"""

        # Use 75% threshold as per requirements
        # P point: where diameter drops below 75% of reference
        # D point: where diameter recovers to 75% of reference
        threshold_factor = 0.75
        threshold_diameter = reference_diameter * threshold_factor

        # Find proximal boundary with smoothing
        prox_bound = mld_idx
        for i in range(mld_idx - 1, max(0, mld_idx - 50), -1):
            # Use moving average for stability
            if i >= 2:
                avg_diameter = np.mean(diameters_mm[i - 2 : i + 1])
            else:
                avg_diameter = diameters_mm[i]

            if avg_diameter >= threshold_diameter:
                prox_bound = i
                break

        # Find distal boundary with smoothing
        dist_bound = mld_idx
        for i in range(mld_idx + 1, min(len(diameters_mm), mld_idx + 50)):
            # Use moving average for stability
            if i < len(diameters_mm) - 2:
                avg_diameter = np.mean(diameters_mm[i : i + 3])
            else:
                avg_diameter = diameters_mm[i]

            if avg_diameter >= threshold_diameter:
                dist_bound = i
                break

        # Calculate lesion length in mm
        if self.calibration_factor:
            # Calculate physical distance along centerline
            lesion_length = (dist_bound - prox_bound) * self.calibration_factor
        else:
            lesion_length = float(dist_bound - prox_bound)

        return lesion_length, prox_bound, dist_bound

    def _prepare_results(
        self,
        centerline: np.ndarray,
        diameters_px: np.ndarray,
        cal_factor: float,
        stenosis_results: Dict,
        left_edges: np.ndarray,
        right_edges: np.ndarray,
        perpendiculars: np.ndarray,
    ) -> Dict:
        """Prepare complete QCA results"""

        # Calculate distances along centerline
        distances = np.zeros(len(centerline))
        if len(centerline) > 1:
            diffs = np.diff(centerline, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            distances[1:] = np.cumsum(segment_lengths)

            # Debug logging
            logger.info(f"Centerline shape: {centerline.shape}")
            logger.info(f"First 3 centerline points: {centerline[:3]}")
            logger.info(f"First 3 segment lengths: {segment_lengths[:3]}")
            logger.info(f"Total centerline length: {distances[-1]:.1f} pixels")

        # Convert to mm
        distances_mm = distances * cal_factor
        diameters_mm = diameters_px * cal_factor
        areas_mm2 = np.pi * (diameters_mm / 2) ** 2

        # Create profile data
        profile_data = {
            "distances": distances_mm.tolist(),
            "diameters": diameters_mm.tolist(),
            "areas": areas_mm2.tolist(),
            "unit": "mm",
        }

        # Combine all results
        results = stenosis_results.copy()
        results.update(
            {
                "centerline": centerline.tolist(),  # Add centerline for overlay drawing
                "diameters_pixels": diameters_px.tolist(),
                "diameters_mm": diameters_mm.tolist(),
                "calibration_factor": cal_factor,
                "profile_data": profile_data,
                # Edge information for accurate overlay drawing
                "left_edges": left_edges.tolist(),
                "right_edges": right_edges.tolist(),
                "perpendiculars": perpendiculars.tolist(),
            }
        )

        return results
