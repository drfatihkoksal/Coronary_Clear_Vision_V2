"""
Calibration Widget using AngioPy Segmentation Model
Uses the actual AngioPy AI model to segment catheter and calculate diameter
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QComboBox, QMessageBox,
                             QProgressBar, QCheckBox, QButtonGroup, QRadioButton)
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot, QTimer
import numpy as np
import cv2
from typing import Tuple, List
import logging
from datetime import datetime

from ..utils.subpixel_utils import subpixel_contour_width
from ..analysis.minimum_cost_path import MinimumCostPathGenerator
from ..analysis.vessel_diameter_accurate import measure_vessel_diameter_with_edges, smooth_diameter_profile
from ..analysis.qca_analysis import QCAConstants

logger = logging.getLogger(__name__)

class AngioPyCalibrationThread(QThread):
    """Worker thread for AngioPy-based catheter calibration"""
    progress = pyqtSignal(str, int)  # status, percentage
    finished = pyqtSignal(dict)  # result with mask
    error = pyqtSignal(str)
    warning = pyqtSignal(str)  # warning messages

    def __init__(self, angiopy_model, image: np.ndarray, user_points: List[Tuple[int, int]],
                 catheter_size_mm: float):
        super().__init__()
        self.angiopy_model = angiopy_model
        self.image = image
        self.user_points = user_points
        self.catheter_size_mm = catheter_size_mm

    def run(self):
        try:
            self.progress.emit("Running AngioPy segmentation...", 10)

            if len(self.user_points) < 2:
                raise ValueError("Need at least 2 points for calibration")

            # Use AngioPy to segment the catheter
            def progress_callback(status: str, percentage: int):
                # Map AngioPy progress to our progress
                mapped_percentage = 10 + int(percentage * 0.6)  # 10-70%
                self.progress.emit(status, mapped_percentage)

            # Run AngioPy segmentation
            segmentation_result = self.angiopy_model.segment_vessel(
                self.image,
                self.user_points,
                progress_callback=progress_callback
            )

            if not segmentation_result.get('success'):
                raise ValueError(segmentation_result.get('error', 'Segmentation failed'))

            # Use probability mask if available for better sub-pixel precision
            if 'probability' in segmentation_result and segmentation_result['probability'] is not None:
                mask = segmentation_result['probability']
                logger.info("Using probability mask for sub-pixel calibration")
            else:
                mask = segmentation_result.get('mask')
                logger.info("Using binary mask for calibration")
                
            if mask is None or np.sum(mask) == 0:
                raise ValueError("No vessel segmented")

            self.progress.emit("Analyzing catheter from segmentation...", 75)

            # Get the line between user points
            p1 = np.array(self.user_points[0])
            p2 = np.array(self.user_points[1])

            # Calculate perpendicular direction
            line_vec = p2 - p1
            line_length = np.linalg.norm(line_vec)
            if line_length < 10:
                raise ValueError("Points too close together")

            unit_vec = line_vec / line_length
            perp_vec = np.array([-unit_vec[1], unit_vec[0]])

            self.progress.emit("Measuring catheter width...", 85)

            # Measure catheter width at multiple points along the segmented vessel
            width_measurements = []
            num_samples = 30  # Increase samples to have enough after exclusion
            
            # Exclude 5 points from start and end
            start_offset = 5
            end_offset = 5

            for i in range(num_samples):
                t = i / (num_samples - 1)
                sample_point = p1 + t * line_vec
                x, y = int(sample_point[0]), int(sample_point[1])

                # Measure width using the segmentation mask
                width = self._measure_width_from_mask(mask, x, y, perp_vec)
                if width > 0:
                    width_measurements.append(width)

            if not width_measurements:
                # Try alternative measurement using contours
                width_measurements = self._measure_from_contours(mask, p1, p2, perp_vec)

            if not width_measurements:
                raise ValueError("Could not measure catheter width from segmentation")
            
            # Exclude 5 points from start and end if we have enough measurements
            min_required_measurements = 10
            if len(width_measurements) > (start_offset + end_offset + min_required_measurements):
                # Sort measurements by their position to ensure we exclude actual start/end points
                # Since measurements are taken in order, we can directly slice
                width_measurements = width_measurements[start_offset:-end_offset]
                logger.info(f"Excluded {start_offset} points from start and {end_offset} from end, "
                          f"using {len(width_measurements)} measurements")
            else:
                logger.warning(f"Not enough measurements ({len(width_measurements)}) to exclude edge points")
                # Emit warning to user
                self.warning.emit("Calibration points are too close. Please select points further apart for more accurate calibration.")

            self.progress.emit("Calculating calibration factor...", 95)

            # Calculate calibration factor using mean instead of median
            mean_width = np.mean(width_measurements)
            
            # Get vessel statistics from AngioPy result first
            thickness_stats = segmentation_result.get('thickness_stats', {})
            
            # For catheter calibration, we need to measure the catheter WIDTH
            # AngioPy gives us thickness measurements but they seem to be in a different scale
            
            logger.info(f"=== CATHETER CALIBRATION ===")
            logger.info(f"Width measurements from mask: {width_measurements[:5]}... (mean={mean_width:.2f})")
            logger.info(f"AngioPy thickness stats: {thickness_stats}")
            
            # The width measurements from AngioPy segmentation are too small
            # This is because AngioPy is trained on vessels, not catheters
            # Catheters appear differently in X-ray (solid vs hollow vessels)
            
            # We need to measure the actual catheter width in the 512x512 image
            # Method 1: Direct measurement from segmentation mask
            if mask is not None:
                # Measure catheter width using multiple methods for robustness
                
                # Method A: Distance transform (most accurate for solid objects)
                from scipy.ndimage import distance_transform_edt
                binary_mask = (mask > 0.5).astype(np.uint8)
                dist_transform = distance_transform_edt(binary_mask)
                
                # Find skeleton/centerline points
                from skimage.morphology import skeletonize
                skeleton = skeletonize(binary_mask)
                
                # Get maximum distance along skeleton (true radius)
                skeleton_distances = dist_transform * skeleton
                max_radius = np.max(skeleton_distances)
                catheter_width_dist = max_radius * 2
                
                # Method B: Use contours to find minimum enclosing rectangle
                import cv2
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest_contour)
                    width, height = rect[1]
                    catheter_width_rect = min(width, height)  # Use smaller dimension as width
                else:
                    catheter_width_rect = 0
                
                # Method C: Use the width measurements we already collected
                if width_measurements:
                    catheter_width_measured = np.median(width_measurements)
                else:
                    catheter_width_measured = 0
                
                logger.info(f"Width measurement methods:")
                logger.info(f"  - Distance transform: {catheter_width_dist:.1f} pixels")
                logger.info(f"  - Min area rectangle: {catheter_width_rect:.1f} pixels")
                logger.info(f"  - Direct measurements: {catheter_width_measured:.1f} pixels (median of {len(width_measurements)} points)")
                
                # Use the most reliable measurement
                # Prefer rectangle method for catheters as it's more robust
                if catheter_width_rect > 0:
                    catheter_width_pixels = catheter_width_rect
                    logger.info(f"Using rectangle method: {catheter_width_pixels:.1f} pixels")
                elif catheter_width_dist > 0:
                    catheter_width_pixels = catheter_width_dist
                    logger.info(f"Using distance transform: {catheter_width_pixels:.1f} pixels")
                else:
                    catheter_width_pixels = catheter_width_measured
                    logger.info(f"Using direct measurements: {catheter_width_pixels:.1f} pixels")
                
                # If this is still too small, we have a scaling issue
                if catheter_width_pixels < 15:  # Too small for a catheter in 512x512 image
                    logger.warning(f"Catheter width {catheter_width_pixels:.1f} pixels is too small for 512x512 image")
                    # Use the width measurements from the mask instead of fixed minimum
                    if width_measurements:
                        # Use the mean of actual measurements
                        catheter_width_pixels = mean_width
                        logger.info(f"Using mean width from measurements: {catheter_width_pixels:.1f} pixels")
                    else:
                        # Only use minimum as last resort
                        logger.error("No width measurements available, calibration may be inaccurate")
                        catheter_width_pixels = 15.0
            else:
                # Fallback to mean width with correction
                catheter_width_pixels = mean_width * 3.0
                logger.warning("No mask available, using corrected mean width")
            
            logger.info(f"Final catheter width: {catheter_width_pixels:.1f} pixels")
            logger.info(f"Catheter size: {self.catheter_size_mm}mm")
            
            # IMPORTANT: The measurements are in 512x512 space, need to scale back to original
            original_height, original_width = self.image.shape[:2]
            scale_factor = original_width / 512.0  # AngioPy resizes to 512x512
            
            # Convert catheter width from 512x512 space to original image space
            catheter_width_original = catheter_width_pixels * scale_factor
            
            logger.info(f"Scale factor from 512 to original ({original_width}x{original_height}): {scale_factor:.2f}")
            logger.info(f"Catheter width in original space: {catheter_width_original:.1f} pixels")
            
            pixels_per_mm = catheter_width_original / self.catheter_size_mm

            # Log calibration calculation details
            logger.info("=== FINAL CALIBRATION RESULTS ===")
            logger.info(f"Catheter width in 512x512: {catheter_width_pixels:.1f} pixels")
            logger.info(f"Catheter width in original: {catheter_width_original:.1f} pixels")
            logger.info(f"Catheter size (mm): {self.catheter_size_mm}")
            logger.info(f"Pixels per mm: {pixels_per_mm:.2f}")
            logger.info(f"Calibration factor (mm/pixel): {1.0 / pixels_per_mm:.4f}")
            
            # Create result
            result = {
                'success': True,
                'mask': mask,
                'boundaries': segmentation_result.get('boundaries'),
                'centerline': segmentation_result.get('centerline'),
                'catheter_size': f"{self.catheter_size_mm:.2f}mm",
                'catheter_diameter_mm': self.catheter_size_mm,
                'pixel_distance': catheter_width_original,
                'real_distance': self.catheter_size_mm,
                'factor': 1.0 / pixels_per_mm,  # mm per pixel
                'pixels_per_mm': pixels_per_mm,
                'num_measurements': len(width_measurements),
                'measurement_std': np.std(width_measurements) if len(width_measurements) > 1 else 0,
                'measurement_mean': mean_width,
                'angiopy_thickness': thickness_stats,
                'method': segmentation_result.get('method', 'angiopy'),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.progress.emit("Calibration complete!", 100)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _measure_width_from_mask(self, mask: np.ndarray, x: int, y: int,
                                perp_vec: np.ndarray) -> float:
        """Measure width from segmentation mask at a point with sub-pixel precision"""
        # Use sub-pixel precision measurement
        center_point = (float(x), float(y))
        # Check if mask contains probability values (float) or binary values (int)
        is_probability = mask.dtype in [np.float32, np.float64] and mask.max() <= 1.0
        
        width = subpixel_contour_width(mask, center_point, perp_vec, method='gradient', 
                                     is_probability=is_probability)
        
        if width == 0.0:
            # Fallback to moment-based method
            width = subpixel_contour_width(mask, center_point, perp_vec, method='moment',
                                         is_probability=is_probability)
        
        if width == 0.0:
            # Final fallback to integer-based method
            max_dist = 100
            left_edge = None
            right_edge = None

            for d in range(-max_dist, max_dist + 1):
                px = int(x + d * perp_vec[0])
                py = int(y + d * perp_vec[1])

                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
                    if mask[py, px] > 0:
                        if left_edge is None:
                            left_edge = d
                        right_edge = d

            if left_edge is not None and right_edge is not None:
                width = abs(right_edge - left_edge)
        
        return width

    def _measure_from_contours(self, mask: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                              perp_vec: np.ndarray) -> List[float]:
        """Alternative measurement using contours"""
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Sample measurements along the line
        measurements = []
        num_samples = 10

        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_point = p1 + t * (p2 - p1)

            # Find contour points perpendicular to the line
            distances = []
            for point in largest_contour:
                pt = point[0]
                # Project onto perpendicular line
                vec_to_pt = pt - sample_point
                perp_dist = np.dot(vec_to_pt, perp_vec)
                along_dist = abs(np.dot(vec_to_pt, p2 - p1) / np.linalg.norm(p2 - p1))

                # Only consider points close to the perpendicular line
                if along_dist < 5:
                    distances.append(perp_dist)

            if len(distances) >= 2:
                width = max(distances) - min(distances)
                if width > 0:
                    measurements.append(width)

        return measurements

class CalibrationAngioPyWidget(QWidget):
    """Calibration widget using AngioPy segmentation"""

    # Signals
    calibration_mode_changed = pyqtSignal(bool)
    calibration_completed = pyqtSignal(float, dict)  # factor, details
    overlay_settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration_mode = True  # Always in calibration mode when widget is active
        self.user_points = []
        self.current_result = None
        self.calibration_thread = None
        self.main_window = None
        self.angiopy_model = None
        self.show_mask = False  # Don't show segmentation for calibration
        self.show_points = True
        self.dicom_pixel_spacing = None  # Store DICOM metadata
        self.is_active = False  # Track if calibration panel is active
        
        # Calibration mode: 'angiopy' only
        self.calibration_method = 'angiopy'  # Default to AngioPy
        
        # Timer for clearing overlays
        self.overlay_clear_timer = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        layout.setSpacing(5)

        # Title with catheter selection in one line
        title_layout = QHBoxLayout()
        title = QLabel("Calibration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1976D2;")
        title_layout.addWidget(title)
        
        # Catheter selection
        title_layout.addWidget(QLabel("Catheter:"))
        self.size_combo = QComboBox()
        self.catheter_sizes = {
            "4F (1.33mm)": 1.33,
            "5F (1.67mm)": 1.67,
            "6F (2.0mm)": 2.0,
            "7F (2.33mm)": 2.33,
            "8F (2.67mm)": 2.67
        }
        self.size_combo.addItems(list(self.catheter_sizes.keys()))
        self.size_combo.setCurrentIndex(2)  # Default to 6F
        self.size_combo.setMaximumWidth(120)
        title_layout.addWidget(self.size_combo)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        method_layout.addWidget(method_label)
        
        # Radio buttons for method selection
        self.method_group = QButtonGroup()
        self.angiopy_radio = QRadioButton("AngioPy AI")
        self.angiopy_radio.setChecked(True)  # Default to AngioPy
        
        self.method_group.addButton(self.angiopy_radio, 0)
        method_layout.addWidget(self.angiopy_radio)
        method_layout.addStretch()
        layout.addLayout(method_layout)

        # Instructions (simplified)
        self.instructions = QLabel("Select two points along the catheter")
        self.instructions.setWordWrap(True)
        self.instructions.setStyleSheet("font-size: 13px; color: #333; padding: 5px 0;")
        layout.addWidget(self.instructions)

        # Point counter and action buttons
        action_layout = QHBoxLayout()
        self.point_count_label = QLabel("Points: 0/2")
        self.point_count_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        action_layout.addWidget(self.point_count_label)
        
        action_layout.addStretch()
        
        self.clear_button = QPushButton("Retry")
        self.clear_button.clicked.connect(self.force_clear_all)
        self.clear_button.setEnabled(False)
        self.clear_button.setFixedWidth(60)
        action_layout.addWidget(self.clear_button)
        
        layout.addLayout(action_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(15)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self.status_label)

        # Results (simplified)
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-size: 11px; color: #666; padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
        self.results_label.setVisible(False)
        layout.addWidget(self.results_label)

        # Current Calibration Values display
        cal_values_group = QGroupBox("Current Calibration Values")
        cal_values_group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; 
                margin-top: 10px; 
                border: 2px solid #1976D2;
                border-radius: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        cal_values_layout = QVBoxLayout()
        cal_values_layout.setSpacing(5)
        
        # Calibration values display label
        self.calibration_values_label = QLabel("No calibration applied yet")
        self.calibration_values_label.setStyleSheet("""
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            color: #333;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 8px;
        """)
        self.calibration_values_label.setWordWrap(True)
        cal_values_layout.addWidget(self.calibration_values_label)
        
        cal_values_group.setLayout(cal_values_layout)
        layout.addWidget(cal_values_group)
        self.cal_values_group = cal_values_group

        # Visualization controls
        viz_group = QGroupBox("Overlay Visualization")
        viz_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(5)

        # AngioPy overlays
        angiopy_layout = QHBoxLayout()
        angiopy_label = QLabel("AngioPy:")
        angiopy_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        angiopy_layout.addWidget(angiopy_label)
        
        self.show_angiopy_mask_cb = QCheckBox("Segmentation Mask")
        self.show_angiopy_mask_cb.setChecked(False)
        self.show_angiopy_mask_cb.toggled.connect(self._on_overlay_changed)
        angiopy_layout.addWidget(self.show_angiopy_mask_cb)
        angiopy_layout.addStretch()
        viz_layout.addLayout(angiopy_layout)

        viz_group.setLayout(viz_layout)
        viz_group.setVisible(False)  # Hidden until calibration completes
        layout.addWidget(viz_group)
        self.viz_group = viz_group

        layout.addStretch()
        self.setLayout(layout)
        
        # Removed unnecessary buttons and groups:
        # - Mode button (always in calibration mode when this panel is active)
        # - Display options (segmentation overlay not needed for calibration)
        # - Calibrate button (auto-runs after 2 points)
        # - Model status label (not needed for users)
        
    def _on_method_changed(self):
        """Handle method selection change"""
        if self.angiopy_radio.isChecked():
            self.calibration_method = 'angiopy'
            self.instructions.setText("Select two points along the catheter (AngioPy AI mode)")
        
        # Update instructions if calibration is active
        if self.calibration_mode and len(self.user_points) == 0:
            method_text = "AngioPy AI"
            self.instructions.setText(f"Select two points along the catheter ({method_text} mode)")
        
        logger.info(f"Calibration method changed to: {self.calibration_method}")
    
    def _on_overlay_changed(self):
        """Handle overlay visibility changes"""
        # Only show overlays if we have calibration results
        if not self.current_result:
            return
            
        # Gather current overlay settings
        overlay_settings = {
            'method': self.calibration_method,
            'show_angiopy_mask': self.show_angiopy_mask_cb.isChecked(),
            'result_data': self.current_result
        }
        
        # Emit signal to update overlays
        self.overlay_settings_changed.emit(overlay_settings)
        
        logger.info(f"Overlay settings changed: AngioPy mask={overlay_settings['show_angiopy_mask']}")

    def set_segmentation_model(self, model):
        """Set the segmentation model (called from main window)"""
        self.angiopy_model = model

    def set_main_window(self, main_window):
        """Set reference to main window"""
        self.main_window = main_window

    def toggle_mode(self):
        """Enable calibration mode (called when widget becomes active)"""
        self.calibration_mode = True
        self.is_active = True
        
        # Cancel overlay clear timer if it's running
        if hasattr(self, 'overlay_clear_timer') and self.overlay_clear_timer and self.overlay_clear_timer.isActive():
            self.overlay_clear_timer.stop()
        
        # Clear any existing calibration overlays when activating calibration mode
        self._clear_calibration_overlays()
        
        # Update instructions with current method
        method_text = "AngioPy AI"
        self.instructions.setText(f"Select two points along the catheter ({method_text} mode)")
        self.clear_button.setEnabled(True)
        self.clear_points()

        # Set viewer to calibration mode
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            self.main_window.viewer_widget.set_calibration_mode(True)

        # Check for existing tracking points
        self._check_and_import_tracking_points()
        
        self.calibration_mode_changed.emit(self.calibration_mode)
    
    def deactivate(self):
        """Deactivate calibration mode (called when switching to another panel)"""
        self.is_active = False
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            self.main_window.viewer_widget.set_calibration_mode(False)
        
        # Cancel overlay clear timer if it's running
        if hasattr(self, 'overlay_clear_timer') and self.overlay_clear_timer and self.overlay_clear_timer.isActive():
            self.overlay_clear_timer.stop()
        
        # Clear calibration overlays when switching away from calibration panel
        self._clear_calibration_overlays()

    def add_point(self, x: int, y: int):
        """Add a calibration point"""
        if not self.calibration_mode:
            return

        self.user_points.append((x, y))
        self.point_count_label.setText(f"Points: {len(self.user_points)}/2")

        if len(self.user_points) == 1:
            self.instructions.setText("Select the second point")
        elif len(self.user_points) == 2:
            self.instructions.setText("Calculating calibration...")
            # Auto-start calibration after 2 points
            self.perform_calibration()
        else:
            # Limit to 2 points
            self.user_points = self.user_points[:2]
            self.point_count_label.setText("Points: 2/2")

    def clear_points(self):
        """Clear all points"""
        self.user_points = []
        self.point_count_label.setText("Points: 0/2")
        self.results_label.setVisible(False)
        
        # Hide visualization controls only if we don't have a current result
        # This preserves the overlay controls after successful calibration
        if not self.current_result:
            self.viz_group.setVisible(False)
        
        # Don't clear current_result here - let it persist for overlay viewing
        
        if self.calibration_mode:
            method_text = "AngioPy AI"
            self.instructions.setText(f"Select two points along the catheter ({method_text} mode)")

        # Clear points in viewer but preserve overlays if we have results
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            self.main_window.viewer_widget.clear_calibration_points()
            # Only clear overlays if we don't have a current result
            if not self.current_result:
                self.overlay_settings_changed.emit({'clear_all': True})

    def force_clear_all(self):
        """Force clear everything including results and overlays"""
        self.user_points = []
        self.point_count_label.setText("Points: 0/2")
        self.results_label.setVisible(False)
        
        # Cancel overlay clear timer if it's running
        if hasattr(self, 'overlay_clear_timer') and self.overlay_clear_timer and self.overlay_clear_timer.isActive():
            self.overlay_clear_timer.stop()
        
        # Force hide visualization controls
        self.viz_group.setVisible(False)
        self.current_result = None
        
        if self.calibration_mode:
            method_text = "AngioPy AI"
            self.instructions.setText(f"Select two points along the catheter ({method_text} mode)")

        # Force clear everything in viewer
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            self.main_window.viewer_widget.clear_calibration_points()
            # Force clear all overlays
            self.overlay_settings_changed.emit({'clear_all': True})

    def perform_calibration(self):
        """Start calibration using selected method"""
        # Check for tracking points if no calibration points
        if len(self.user_points) < 2:
            self._check_and_import_tracking_points()
            if len(self.user_points) < 2:
                QMessageBox.warning(self, "Error", "Need 2 points for calibration")
                return

        # Check AngioPy model availability only if using AngioPy method
        if self.calibration_method == 'angiopy' and not self.angiopy_model:
            QMessageBox.warning(self, "Error", "AngioPy model not initialized")
            return

        # Get current image
        if not self.main_window or not hasattr(self.main_window, 'viewer_widget'):
            QMessageBox.warning(self, "Error", "No viewer found")
            return

        viewer = self.main_window.viewer_widget
        image = None

        if hasattr(viewer, 'get_current_frame'):
            image = viewer.get_current_frame()
        elif hasattr(viewer, 'current_frame'):
            image = viewer.current_frame

        if image is None:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        # Get catheter size
        catheter_size = self.catheter_sizes[self.size_combo.currentText()]

        # Disable UI
        self.clear_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)

        # Start AngioPy calibration thread (only method available)
        self.calibration_thread = AngioPyCalibrationThread(
            self.angiopy_model,
            image,
            self.user_points,
            catheter_size
        )

        self.calibration_thread.progress.connect(self.on_progress)
        self.calibration_thread.finished.connect(self.on_calibration_finished)
        self.calibration_thread.error.connect(self.on_calibration_error)
        self.calibration_thread.warning.connect(self.on_calibration_warning)

        self.calibration_thread.start()

    @pyqtSlot(str, int)
    def on_progress(self, status: str, percentage: int):
        """Update progress"""
        self.status_label.setText(status)
        self.progress_bar.setValue(percentage)

    @pyqtSlot(dict)
    def on_calibration_finished(self, result: dict):
        """Handle calibration completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.clear_button.setEnabled(True)

        if result.get('success'):
            self.current_result = result

            # Display simplified results with method info
            method_text = "AngioPy AI"
            self.results_label.setText(
                f"<b>Calibration Result ({method_text}):</b> {result['pixels_per_mm']:.1f} pixels/mm "
                f"({result['catheter_size']})"
            )
            self.results_label.setVisible(True)

            # Show confirmation dialog with method-specific details
            msg = QMessageBox(self)
            msg.setWindowTitle("Calibration Complete")
            
            # Create method-specific details text
            details_text = (
                f"Method: AngioPy AI\n"
                f"Measurements: {result.get('num_measurements', 0)} points\n"
                f"Width measurement: {result.get('measurement_mean', 0):.1f} pixels\n"
            )
            
            msg.setText(
                f"Calibration successful!\n\n"
                f"Catheter: {result['catheter_size']}\n"
                f"{details_text}"
                f"Scale: {result['pixels_per_mm']:.1f} pixels/mm\n"
                f"Factor: {result['factor']:.4f} mm/pixel\n\n"
                f"Do you want to use this calibration?"
            )
            
            # Add metadata option if available
            if self.dicom_pixel_spacing:
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | 
                                     QMessageBox.StandardButton.Retry | 
                                     QMessageBox.StandardButton.No)
                # Change No button text to "Use DICOM Metadata"
                no_button = msg.button(QMessageBox.StandardButton.No)
                if no_button:
                    no_button.setText("Use DICOM Metadata")
            else:
                msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Retry)
            
            msg.setDefaultButton(QMessageBox.StandardButton.Yes)
            
            response = msg.exec()
            
            if response == QMessageBox.StandardButton.Yes:
                # Apply calibration
                self.calibration_completed.emit(result['factor'], result)
                
                # Update instructions
                self.instructions.setText("Calibration applied successfully!")
                
                # Show visualization controls
                self.viz_group.setVisible(True)
                
                # Set initial overlay display based on method
                # Show AngioPy mask by default (only method available)
                self.show_angiopy_mask_cb.setChecked(True)
                
                # Trigger initial overlay display
                self._on_overlay_changed()
                
                # Update calibration values display
                self._update_calibration_values_display(result)
                
                # Disable calibration mode in viewer after a short delay
                QTimer.singleShot(1000, self._exit_calibration_mode)
                
                # Clear overlays after a longer timeout (10 seconds)
                self.overlay_clear_timer = QTimer()
                self.overlay_clear_timer.setSingleShot(True)
                self.overlay_clear_timer.timeout.connect(self._clear_calibration_overlays_timeout)
                self.overlay_clear_timer.start(10000)  # 10 seconds
            elif response == QMessageBox.StandardButton.Retry:
                # Retry - force clear everything
                self.force_clear_all()
            elif response == QMessageBox.StandardButton.No and self.dicom_pixel_spacing:
                # Use DICOM metadata
                self._use_dicom_metadata_calibration()

    @pyqtSlot(str)
    def on_calibration_warning(self, warning_msg: str):
        """Handle calibration warning"""
        # Show warning message but continue calibration
        QMessageBox.warning(self, "Calibration Warning", warning_msg)
    
    @pyqtSlot(str)
    def on_calibration_error(self, error_msg: str):
        """Handle calibration error"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.clear_button.setEnabled(True)

        msg = QMessageBox(self)
        msg.setWindowTitle("Calibration Failed")
        msg.setText(f"Calibration failed: {error_msg}\n\nWould you like to try again?")
        msg.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Retry)
        
        response = msg.exec()
        
        if response == QMessageBox.StandardButton.Retry:
            self.force_clear_all()
        else:
            # Check if we can use DICOM metadata
            if self.dicom_pixel_spacing:
                msg2 = QMessageBox(self)
                msg2.setWindowTitle("Use DICOM Metadata?")
                msg2.setText(
                    f"DICOM metadata contains pixel spacing information:\n"
                    f"Pixel spacing: {self.dicom_pixel_spacing:.4f} mm/pixel\n\n"
                    f"Would you like to use this for calibration?"
                )
                msg2.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg2.setDefaultButton(QMessageBox.StandardButton.Yes)
                
                if msg2.exec() == QMessageBox.StandardButton.Yes:
                    self._use_dicom_metadata_calibration()
                else:
                    # Exit calibration mode
                    self._exit_calibration_mode()
            else:
                # Exit calibration mode
                self._exit_calibration_mode()

    def _exit_calibration_mode(self):
        """Exit calibration mode and return to normal viewing"""
        self.calibration_mode = False
        self.is_active = False
        
        # Disable calibration mode in viewer
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            self.main_window.viewer_widget.set_calibration_mode(False)
        
        # Only clear overlays if no successful calibration result
        # Otherwise, let them remain visible until timeout or panel switch
        if not self.current_result:
            self._clear_calibration_overlays()
            self.clear_points()
        
        # Emit signal
        self.calibration_mode_changed.emit(False)
    
    def _use_dicom_metadata_calibration(self):
        """Use DICOM metadata for calibration"""
        if not self.dicom_pixel_spacing:
            return
        
        # Warn user about DICOM pixel spacing reliability
        warning_msg = (
            f"⚠️ WARNING: DICOM pixel spacing ({self.dicom_pixel_spacing:.3f} mm/pixel) "
            f"is often unreliable in angiography.\n\n"
            f"Expected coronary vessel diameters:\n"
            f"• With DICOM spacing: {30 * self.dicom_pixel_spacing:.1f}mm for 30-pixel vessel\n"
            f"• With catheter calibration: ~3mm for 30-pixel vessel\n\n"
            f"It is strongly recommended to use catheter calibration instead.\n\n"
            f"Do you still want to use DICOM metadata?"
        )
        
        reply = QMessageBox.warning(self, "DICOM Calibration Warning", warning_msg,
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Create calibration result from DICOM metadata
        result = {
            'success': True,
            'mask': None,
            'catheter_size': 'DICOM Metadata',
            'catheter_diameter_mm': 0,  # Unknown from metadata
            'pixel_distance': 0,
            'real_distance': 0,
            'factor': self.dicom_pixel_spacing,  # mm per pixel
            'pixels_per_mm': 1.0 / self.dicom_pixel_spacing,
            'num_measurements': 0,
            'measurement_std': 0,
            'method': 'dicom_metadata',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Display results
        self.results_label.setText(
            f"<b>Using DICOM Metadata:</b> {result['pixels_per_mm']:.1f} pixels/mm"
        )
        self.results_label.setVisible(True)
        
        # Apply calibration
        self.calibration_completed.emit(result['factor'], result)
        
        # Update calibration values display
        self._update_calibration_values_display(result)
        
        # Update instructions
        self.instructions.setText("DICOM metadata calibration applied!")
        
        # Exit calibration mode after a short delay
        QTimer.singleShot(1000, self._exit_calibration_mode)
        
        # Clear overlays after a longer timeout (10 seconds)
        self.overlay_clear_timer = QTimer()
        self.overlay_clear_timer.setSingleShot(True)
        self.overlay_clear_timer.timeout.connect(self._clear_calibration_overlays_timeout)
        self.overlay_clear_timer.start(10000)  # 10 seconds
    
    def set_dicom_pixel_spacing(self, pixel_spacing: float):
        """Set DICOM pixel spacing from main window"""
        self.dicom_pixel_spacing = pixel_spacing
        
        # Update instructions if we have metadata
        if pixel_spacing and self.calibration_mode:
            self.instructions.setText(
                f"Select two points along the catheter\n"
                f"(DICOM spacing available: {pixel_spacing:.4f} mm/pixel)"
            )

    def _check_and_import_tracking_points(self):
        """Check for tracking points in current frame and import them"""
        if not self.main_window or not hasattr(self.main_window, 'viewer_widget'):
            return

        viewer = self.main_window.viewer_widget

        # Check if viewer has overlay_item with frame_points
        if not hasattr(viewer, 'overlay_item') or not hasattr(viewer.overlay_item, 'frame_points'):
            return

        # Get current frame index
        current_frame = viewer.current_frame_index if hasattr(viewer, 'current_frame_index') else 0

        # Get tracking points for current frame
        tracking_points = viewer.overlay_item.frame_points.get(current_frame, [])

        # If we have at least 1 tracking point and no calibration points, use them (for single-point mode)
        # For calibration, we still need 2 points, so duplicate the single point if needed
        if len(tracking_points) >= 1 and len(self.user_points) == 0:
            # Clear existing points
            self.user_points = []

            # Add tracking points as calibration points
            if len(tracking_points) >= 2:
                # Use first two points
                for point in tracking_points[:2]:
                    self.user_points.append(tuple(point))
            else:
                # Single point mode - duplicate the point to create a line for calibration
                point = tracking_points[0]
                self.user_points.append(tuple(point))
                # Create a second point slightly offset for calibration line
                offset_point = (point[0] + 10, point[1])  # 10 pixel offset
                self.user_points.append(offset_point)
            
            # Update UI
            self.point_count_label.setText(f"Points: {len(self.user_points)}")
            self.instructions.setText("Tracking points imported. Ready for calibration!")
            self.calibrate_button.setEnabled(True)

            # Update viewer with imported points
            if hasattr(viewer, 'set_calibration_points'):
                viewer.set_calibration_points(self.user_points)

            logger.info(f"Imported {len(self.user_points)} tracking points for calibration")
    
    def _update_calibration_values_display(self, result: dict):
        """Update the current calibration values display with latest calibration result"""
        if not result or not result.get('success'):
            self.calibration_values_label.setText("No calibration applied yet")
            return
            
        # Get method information
        method = result.get('method', 'unknown')
        if method == 'dicom_metadata':
            method_text = "DICOM Metadata"
            method_color = "#FF9800"  # Orange
        elif hasattr(self, 'calibration_method'):
            if self.calibration_method == 'angiopy':
                method_text = "AngioPy AI"
                method_color = "#1976D2"  # Blue
        else:
            method_text = "Unknown"
            method_color = "#666"
        
        # Get timestamp or create current one
        timestamp = result.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Format calibration details
        catheter_info = result.get('catheter_size', 'N/A')
        pixels_per_mm = result.get('pixels_per_mm', 0)
        mm_per_pixel = result.get('factor', 0)
        num_measurements = result.get('num_measurements', 0)
        
        # Additional method-specific details
        extra_details = ""
        if method != 'dicom_metadata':
            measurement_mean = result.get('measurement_mean', 0)
            measurement_std = result.get('measurement_std', 0)
            measurement_median = result.get('measurement_median', 0)
            
            if measurement_mean > 0:
                extra_details += f"\nMean measurement: {measurement_mean:.1f} px"
            if measurement_std > 0:
                extra_details += f"\nStd deviation: {measurement_std:.1f} px"
            if measurement_median > 0 and measurement_median != measurement_mean:
                extra_details += f"\nMedian: {measurement_median:.1f} px"
        
        # Create formatted display text
        display_text = f"""
<b style="color: {method_color};">ACTIVE CALIBRATION</b>
<b>Method:</b> {method_text}
<b>Timestamp:</b> {timestamp}
<b>Catheter:</b> {catheter_info}

<b>Calibration Values:</b>
• Pixels per mm: {pixels_per_mm:.2f} px/mm
• mm per pixel: {mm_per_pixel:.5f} mm/px
• Measurements: {num_measurements} points{extra_details}
        """.strip()
        
        self.calibration_values_label.setText(display_text)
        
        # Log the update for debugging
        logger.info(f"Updated calibration values display: {method_text}, {pixels_per_mm:.2f} px/mm, {mm_per_pixel:.5f} mm/px")
    
    def _clear_calibration_overlays(self):
        """Clear all calibration overlays when not in calibration mode"""
        # Emit signal to clear all calibration overlays
        clear_signal = {
            'clear_all': True
        }
        self.overlay_settings_changed.emit(clear_signal)
        logger.info("Cleared all calibration overlays - calibration mode inactive")
    
    def _clear_calibration_overlays_timeout(self):
        """Clear calibration overlays after timeout"""
        # Clear overlays after timeout as long as we have results
        # If user has switched panels, deactivate() would have already cleared them
        if self.current_result:
            logger.info("Clearing calibration overlays after 10 second timeout")
            self._clear_calibration_overlays()