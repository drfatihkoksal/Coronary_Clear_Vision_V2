"""
Vessel Segmentation Widget - Clean Implementation
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QCheckBox,
    QProgressBar,
    QMessageBox,
    QFileDialog,
    QApplication,
)
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot
from typing import Dict
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# from ..analysis.traditional_segmentation import TraditionalSegmentation  # Removed - file deleted

logger = logging.getLogger(__name__)


# Legacy SegmentationThread kept for compatibility but deprecated
# Use UnifiedSegmentationProcessor instead
class SegmentationThread(QThread):
    """Worker thread for segmentation processing - DEPRECATED"""

    progress = pyqtSignal(str, int)  # status, percentage
    finished = pyqtSignal(dict)  # result
    error = pyqtSignal(str)

    def __init__(self, model, image, user_points):
        super().__init__()
        self.model = model
        self.image = image
        self.user_points = user_points

    def run(self):
        try:

            def progress_callback(status: str, percentage: int):
                self.progress.emit(status, percentage)

            result = self.model.segment_vessel(
                self.image, self.user_points, progress_callback=progress_callback
            )

            # Debug: Log result statistics
            if result.get("success") and result.get("centerline") is not None:
                centerline = result["centerline"]
                logger.info(f"SegmentationThread: Centerline has {len(centerline)} points")
                if len(centerline) <= 10:
                    logger.info(f"SegmentationThread: All centerline points: {centerline}")

            self.finished.emit(result)

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            self.error.emit(str(e))


class SegmentationWidget(QWidget):
    """Widget for vessel segmentation control"""

    # Signals
    segmentation_mode_changed = pyqtSignal(bool)
    segmentation_completed = pyqtSignal(dict)
    overlay_settings_changed = pyqtSignal(dict)
    qca_analysis_requested = pyqtSignal(dict)  # Emit segmentation data for QCA

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segmentation_model = None
        self.segmentation_mode = False
        self.user_points = []
        self.current_result = None
        self.segmentation_thread = None
        self.segmentation_history = []  # For undo functionality
        self.show_mask = False  # Default to hiding mask
        self.show_points = True
        self.total_frames = 0  # Total frames in DICOM
        self.main_window = None  # Reference to main window

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Mode control
        mode_group = QGroupBox("Segmentation Mode")
        mode_layout = QVBoxLayout()

        self.mode_button = QPushButton("Enable Segmentation Mode")
        self.mode_button.setCheckable(True)
        self.mode_button.clicked.connect(self.toggle_mode)
        mode_layout.addWidget(self.mode_button)

        self.instructions = QLabel("Click to enable segmentation mode")
        self.instructions.setWordWrap(True)
        mode_layout.addWidget(self.instructions)

        # Frame point status
        self.frame_status_label = QLabel("Frame status: Checking...")
        self.frame_status_label.setStyleSheet(
            "font-size: 11px; color: #666; background-color: #f0f0f0; padding: 5px; border-radius: 3px;"
        )
        self.frame_status_label.setWordWrap(True)
        mode_layout.addWidget(self.frame_status_label)

        # Point counter
        point_layout = QHBoxLayout()
        self.point_count_label = QLabel("Points: 0")
        point_layout.addWidget(self.point_count_label)

        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)
        self.clear_button.setEnabled(False)
        point_layout.addWidget(self.clear_button)

        mode_layout.addLayout(point_layout)

        # Segment button
        self.segment_button = QPushButton("Perform Segmentation")
        self.segment_button.clicked.connect(self.perform_segmentation)
        self.segment_button.setEnabled(False)
        mode_layout.addWidget(self.segment_button)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        mode_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        mode_layout.addWidget(self.status_label)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        # Checkboxes for show/hide
        self.show_mask_checkbox = QCheckBox("Show Segmentation Mask")
        self.show_mask_checkbox.setChecked(False)  # Default to unchecked
        self.show_mask_checkbox.clicked.connect(self.on_show_mask_changed)
        self.show_mask_checkbox.setEnabled(False)  # Disabled until segmentation is done
        viz_layout.addWidget(self.show_mask_checkbox)

        self.show_points_checkbox = QCheckBox("Show Click Points")
        self.show_points_checkbox.setChecked(True)
        self.show_points_checkbox.clicked.connect(self.on_show_points_changed)
        self.show_points_checkbox.setEnabled(False)  # Disabled until segmentation is done
        viz_layout.addWidget(self.show_points_checkbox)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        # Mask actions
        mask_action_layout = QHBoxLayout()

        self.clear_mask_button = QPushButton("Clear Mask")
        self.clear_mask_button.clicked.connect(self.clear_mask)
        self.clear_mask_button.setEnabled(False)
        mask_action_layout.addWidget(self.clear_mask_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_segmentation)
        self.undo_button.setEnabled(False)
        mask_action_layout.addWidget(self.undo_button)

        action_layout.addLayout(mask_action_layout)

        # Save/load actions
        save_action_layout = QHBoxLayout()

        self.save_button = QPushButton("Save Mask")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        save_action_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Mask")
        self.load_button.clicked.connect(self.load_mask)
        save_action_layout.addWidget(self.load_button)

        action_layout.addLayout(save_action_layout)

        # QCA Analysis button
        self.qca_button = QPushButton("Perform QCA Analysis")
        self.qca_button.clicked.connect(self.perform_qca_analysis)
        self.qca_button.setEnabled(False)
        self.qca_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """
        )
        action_layout.addWidget(self.qca_button)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        layout.addStretch()
        self.setLayout(layout)

    def set_segmentation_model(self, model):
        """Set the segmentation model"""
        self.segmentation_model = model
        logger.info("Segmentation model set")

    def set_main_window(self, main_window):
        """Set reference to main window for viewer access"""
        self.main_window = main_window
        logger.info("Main window reference set")

        # Update frame status when main window is set
        self.update_frame_status()

    def toggle_mode(self):
        """Toggle segmentation mode"""
        self.segmentation_mode = self.mode_button.isChecked()

        if self.segmentation_mode:
            self.mode_button.setText("Disable Segmentation Mode")
            self.instructions.setText(
                "Click vessel points (unlimited - 1, 2, 3, 4+ points supported)"
            )
            self.clear_button.setEnabled(True)

            # Check for existing tracking points in current frame
            self._check_and_import_tracking_points()

            # Update frame status
            self.update_frame_status()
        else:
            self.mode_button.setText("Enable Segmentation Mode")
            self.instructions.setText("Click to enable segmentation mode")
            self.clear_button.setEnabled(False)
            self.segment_button.setEnabled(False)
            # Don't clear points when disabling segmentation mode

        self.segmentation_mode_changed.emit(self.segmentation_mode)

    def add_point(self, x: int, y: int):
        """Add a user point"""
        if not self.segmentation_mode:
            return

        self.user_points.append((x, y))
        self.point_count_label.setText(f"Points: {len(self.user_points)}")

        if len(self.user_points) == 1:
            self.instructions.setText("Click distal stenosis point")
        elif len(self.user_points) == 2:
            self.instructions.setText("Click mid-vessel point or perform segmentation")
            self.segment_button.setEnabled(True)
        elif len(self.user_points) == 3:
            self.instructions.setText("3 points added. Ready for segmentation")
            self.segment_button.setEnabled(True)
        else:
            self.instructions.setText(f"{len(self.user_points)} points added")

        logger.info(f"Added point ({x}, {y}), total: {len(self.user_points)}")

        # Update frame status
        self.update_frame_status()

    def clear_points(self):
        """Clear all points"""
        self.user_points = []
        self.point_count_label.setText("Points: 0")
        self.segment_button.setEnabled(False)

        if self.segmentation_mode:
            self.instructions.setText("Click proximal and distal stenosis points")

        # Notify viewer to clear segmentation points
        if self.main_window and hasattr(self.main_window, "viewer_widget"):
            viewer = self.main_window.viewer_widget
            if hasattr(viewer, "clear_user_points"):
                viewer.clear_user_points()
            # Also clear segmentation-specific display if exists
            if hasattr(viewer, "clear_segmentation_points"):
                viewer.clear_segmentation_points()

        logger.info("Points cleared")

        # Update frame status
        self.update_frame_status()

    def perform_segmentation(self):
        """Start segmentation using traditional methods"""
        # Use traditional segmentation instead of requiring model
        if not hasattr(self, "traditional_segmenter"):
            self.traditional_segmenter = TraditionalSegmentation()

        # Get current image and frame index
        main_window = self.main_window or self.window()

        if not main_window or not hasattr(main_window, "viewer_widget"):
            QMessageBox.warning(self, "Error", "No viewer found")
            return

        viewer = main_window.viewer_widget
        image = None
        current_frame_idx = 0

        # Get current frame index
        if hasattr(viewer, "current_frame_idx"):
            current_frame_idx = viewer.current_frame_idx
        elif hasattr(viewer, "current_frame_index"):
            current_frame_idx = viewer.current_frame_index

        if hasattr(viewer, "get_current_frame"):
            image = viewer.get_current_frame()
        elif hasattr(viewer, "current_frame"):
            image = viewer.current_frame

        if image is None:
            QMessageBox.warning(self, "Error", "No image loaded")
            return

        # Check for tracking points if no segmentation points
        if len(self.user_points) < 2:
            self._check_and_import_tracking_points()
            if len(self.user_points) < 2:
                QMessageBox.warning(self, "Error", "At least 2 points required for segmentation")
                return

        # Clear any existing overlay before starting new segmentation
        # This prevents old markers from appearing in new segmentation
        if hasattr(viewer, "clear_overlays"):
            viewer.clear_overlays()
            # Force immediate update to ensure old overlay is gone
            if hasattr(viewer, "_batch_update"):
                viewer._batch_update()
            elif hasattr(viewer, "update"):
                viewer.update()

        # Also clear any segmentation-specific data
        if hasattr(viewer, "overlay_item") and hasattr(viewer.overlay_item, "segmentation_mask"):
            viewer.overlay_item.segmentation_mask = None
            viewer.overlay_item.update()

        # Process events to ensure UI is updated before starting thread
        QApplication.processEvents()

        # Disable UI
        self.segment_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)

        # Make sure segmentation model is available in the main window
        if hasattr(main_window, "segmentation_model"):
            main_window.segmentation_model = self.segmentation_model

        # Use original centerline length - no interpolation toggle

        # Get curvature-resistant centerline setting from main window
        use_curvature_resistant = False  # Default
        if hasattr(main_window, "curvature_resistant_checkbox"):
            main_window.curvature_resistant_checkbox.isChecked()

        # Use traditional segmentation directly
        try:
            self.on_progress("Starting traditional segmentation...", 10)

            # Convert image to proper format if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                segmentation_image = image
            elif len(image.shape) == 2:
                # Grayscale - convert to RGB for consistency
                segmentation_image = np.stack([image, image, image], axis=2)
            else:
                raise ValueError("Unsupported image format")

            self.on_progress("Processing image...", 50)

            # Perform traditional segmentation
            result = self.traditional_segmenter.segment_vessel(
                segmentation_image,
                user_points=self.user_points,
                method="adaptive",  # Use adaptive method by default
            )

            self.on_progress("Finalizing results...", 90)

            # Format result for compatibility
            formatted_result = {
                "success": result["success"],
                "mask": result["mask"],
                "centerline_points": self._extract_centerline_points(result["centerline"]),
                "method": result["method"],
                "frame_index": current_frame_idx,
            }

            self.on_progress("Complete!", 100)

            # Simulate thread completion
            QTimer.singleShot(100, lambda: self.on_segmentation_finished(formatted_result))

        except Exception as e:
            self.on_segmentation_error(str(e))
            return

    def _extract_centerline_points(self, centerline_mask):
        """Extract centerline points from binary mask"""
        if centerline_mask is None:
            return []

        # Find white pixels in centerline mask
        points = np.where(centerline_mask > 0)
        if len(points[0]) == 0:
            return []

        # Convert to (x, y) format
        centerline_points = [(int(x), int(y)) for y, x in zip(points[0], points[1])]

        # Sort points to create a connected path
        if len(centerline_points) > 1:
            # Simple ordering by distance (could be improved)
            ordered_points = [centerline_points[0]]
            remaining_points = centerline_points[1:]

            while remaining_points:
                current = ordered_points[-1]
                # Find closest remaining point
                distances = [
                    ((p[0] - current[0]) ** 2 + (p[1] - current[1]) ** 2) ** 0.5
                    for p in remaining_points
                ]
                min_idx = np.argmin(distances)
                closest_point = remaining_points.pop(min_idx)
                ordered_points.append(closest_point)

            return ordered_points

        return centerline_points

    @pyqtSlot(int, int, str)
    def _on_sequential_progress(self, current: int, total: int, message: str):
        """Adapter for sequential processor progress signal"""
        # Convert to percentage for UI
        percentage = int((current / total) * 100) if total > 0 else 0
        self.on_progress(message, percentage)

    @pyqtSlot(int, dict)
    def _on_sequential_segmentation_completed(self, frame_idx: int, result: dict):
        """Handle segmentation completion from sequential processor"""
        # For single frame, this is our final result
        self.on_segmentation_finished(result)

    @pyqtSlot(dict, dict)
    def _on_sequential_all_completed(self, seg_results: dict, qca_results: dict):
        """Handle all processing completion (for single frame, same as segmentation)"""
        # Already handled in segmentation_completed for single frame

    @pyqtSlot(int, str)
    def _on_sequential_error(self, frame_idx: int, error: str):
        """Handle error from sequential processor"""
        self.on_segmentation_error(error)

    @pyqtSlot(str, int)
    def on_progress(self, status: str, percentage: int):
        """Update progress"""
        self.status_label.setText(status)
        self.progress_bar.setValue(percentage)

    @pyqtSlot(dict)
    def on_segmentation_finished(self, result: dict):
        """Handle segmentation completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.segment_button.setEnabled(True)
        self.clear_button.setEnabled(True)

        if result.get("success"):
            # Save previous result to history if exists
            if self.current_result:
                self.segmentation_history.append(self.current_result)
                self.undo_button.setEnabled(True)

            self.current_result = result
            self.save_button.setEnabled(True)
            self.clear_mask_button.setEnabled(True)
            self.qca_button.setEnabled(True)  # Enable QCA button

            # Show checkboxes are enabled after first segmentation
            self.show_mask_checkbox.setEnabled(True)
            self.show_points_checkbox.setEnabled(True)

            # Emit completion signal if mask is shown
            if self.show_mask:
                self.segmentation_completed.emit(result)

            logger.info("Segmentation completed successfully")
        else:
            error_msg = result.get("error", "Segmentation failed")
            if "model not loaded" in error_msg.lower():
                QMessageBox.warning(
                    self,
                    "Model Required",
                    "Traditional segmentation is now being used.\n"
                    "The model will be downloaded automatically when you try again.",
                )
            else:
                QMessageBox.warning(self, "Failed", f"Segmentation failed: {error_msg}")

    @pyqtSlot(str)
    def on_segmentation_error(self, error: str):
        """Handle segmentation error"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.segment_button.setEnabled(True)
        self.clear_button.setEnabled(True)

        QMessageBox.critical(self, "Error", f"Segmentation error: {error}")

    def update_overlay_settings(self):
        """Update visualization settings (kept for compatibility)"""
        settings = {"enabled": True, "opacity": 1.0, "color": "Red", "contour_only": False}
        self.overlay_settings_changed.emit(settings)

    def get_overlay_settings(self) -> Dict:
        """Get visualization settings (kept for compatibility)"""
        return {"enabled": self.show_mask, "opacity": 1.0, "color": "Red", "contour_only": False}

    def save_result(self):
        """Save segmentation result"""
        if not self.current_result:
            return

        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Segmentation Result",
            "segmentation_result.npz",
            "NumPy files (*.npz);;PNG Image (*.png);;All Files (*.*)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".npz"):
                # Save as numpy file with all data
                np.savez_compressed(
                    file_path,
                    mask=self.current_result.get("mask"),
                    centerline=self.current_result.get("centerline"),
                    qca_results=self.current_result.get("qca_results"),
                    metadata={
                        "timestamp": self.current_result.get("timestamp"),
                        "method": self.current_result.get("method", "Traditional"),
                        "processing_time": self.current_result.get("processing_time", 0),
                    },
                )
                logger.info(f"Saved segmentation data to {file_path}")
            elif file_path.endswith(".png"):
                # Save mask as image
                mask = self.current_result.get("mask")
                if mask is not None:
                    # Convert boolean mask to uint8
                    mask_img = (mask * 255).astype(np.uint8)
                    import cv2

                    cv2.imwrite(file_path, mask_img)
                    logger.info(f"Saved segmentation mask to {file_path}")
            else:
                # Default to numpy format
                if not file_path.endswith(".npz"):
                    file_path += ".npz"
                np.savez_compressed(
                    file_path,
                    mask=self.current_result.get("mask"),
                    centerline=self.current_result.get("centerline"),
                    qca_results=self.current_result.get("qca_results"),
                )
                logger.info(f"Saved segmentation data to {file_path}")

            QMessageBox.information(self, "Success", f"Segmentation result saved to:\n{file_path}")

        except Exception as e:
            logger.error(f"Failed to save segmentation result: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save result: {str(e)}")

    def clear_result(self):
        """Clear current result (deprecated - use clear_mask)"""
        self.clear_mask()

    def on_show_mask_changed(self):
        """Handle show/hide mask checkbox"""
        self.show_mask = self.show_mask_checkbox.isChecked()
        if self.current_result:
            # Update visualization
            settings = self.get_overlay_settings()
            settings["enabled"] = self.show_mask
            self.overlay_settings_changed.emit(settings)
            # Re-emit the result to update display
            if self.show_mask:
                self.segmentation_completed.emit(self.current_result)

    def on_show_points_changed(self):
        """Handle show/hide points checkbox"""
        self.show_points = self.show_points_checkbox.isChecked()
        # Notify viewer to show/hide points
        main_window = self.main_window or self.window()

        if main_window and hasattr(main_window, "viewer_widget"):
            viewer = main_window.viewer_widget
            if self.show_points:
                # Re-add points to viewer
                for point in self.user_points:
                    viewer.add_segmentation_point(point[0], point[1])
            else:
                # Clear points from viewer
                viewer.clear_user_points()

    def clear_mask(self):
        """Clear the segmentation mask"""
        if self.current_result:
            # Save to history for undo
            self.segmentation_history.append(self.current_result)
            self.undo_button.setEnabled(True)

        self.current_result = None
        self.clear_mask_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.qca_button.setEnabled(False)

        # Clear from viewer
        self.overlay_settings_changed.emit({"enabled": False})

        # Notify viewer to restore original image
        main_window = self.main_window or self.window()

        if main_window and hasattr(main_window, "viewer_widget"):
            viewer = main_window.viewer_widget
            if hasattr(viewer, "clear_segmentation_graphics"):
                viewer.clear_segmentation_graphics()

    def clear_all_results(self):
        """Clear all segmentation results from all frames"""
        # Clear current result
        self.current_result = None
        self.clear_mask_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.qca_button.setEnabled(False)

        # Clear history
        self.segmentation_history.clear()
        self.undo_button.setEnabled(False)

        # Clear overlay
        self.overlay_settings_changed.emit({"enabled": False})

        # Notify viewer to update
        if self.main_window and hasattr(self.main_window, "viewer_widget"):
            self.main_window.viewer_widget._request_update()

    def undo_segmentation(self):
        """Undo last segmentation action"""
        if self.segmentation_history:
            # Restore previous result
            self.current_result = self.segmentation_history.pop()

            # Update UI
            self.clear_mask_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.qca_button.setEnabled(True)

            if not self.segmentation_history:
                self.undo_button.setEnabled(False)

            # Re-apply segmentation
            if self.show_mask:
                self.segmentation_completed.emit(self.current_result)

    def load_mask(self):
        """Load a segmentation mask from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Segmentation Mask",
            "",
            "NumPy files (*.npz *.npy);;Image files (*.png *.jpg *.bmp);;All Files (*.*)",
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".npz"):
                # Load numpy archive
                data = np.load(file_path, allow_pickle=True)
                mask = data.get("mask")
                centerline = data.get("centerline", None)
                qca_results = data.get("qca_results", None)
                if isinstance(qca_results, np.ndarray) and qca_results.shape == ():
                    qca_results = qca_results.item()
                metadata = data.get("metadata", {})
                if isinstance(metadata, np.ndarray) and metadata.shape == ():
                    metadata = metadata.item()

                # Create result dictionary
                self.current_result = {
                    "success": True,
                    "mask": mask,
                    "centerline": centerline,
                    "qca_results": qca_results,
                    "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                    "method": metadata.get("method", "Loaded"),
                    "processing_time": metadata.get("processing_time", 0),
                }

            elif file_path.endswith(".npy"):
                # Load single numpy array as mask
                mask = np.load(file_path)
                self.current_result = {
                    "success": True,
                    "mask": mask,
                    "centerline": None,
                    "qca_results": None,
                    "timestamp": datetime.now().isoformat(),
                    "method": "Loaded",
                    "processing_time": 0,
                }

            else:
                # Load image file as mask
                import cv2

                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Failed to load image file")

                # Convert to boolean mask
                mask = img > 127

                self.current_result = {
                    "success": True,
                    "mask": mask,
                    "centerline": None,
                    "qca_results": None,
                    "timestamp": datetime.now().isoformat(),
                    "method": "Loaded from image",
                    "processing_time": 0,
                }

            # Update UI
            self.show_mask_checkbox.setChecked(True)
            self.show_mask = True

            # Emit signal to update display
            self.segmentation_completed.emit(self.current_result)

            # Update status
            self.update_status(f"Loaded mask from: {Path(file_path).name}")
            logger.info(f"Successfully loaded segmentation mask from {file_path}")

            QMessageBox.information(
                self, "Success", f"Segmentation mask loaded successfully from:\n{file_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load segmentation mask: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load mask: {str(e)}")

    def set_frame_range(self, total_frames: int):
        """Set the total number of frames available"""
        self.total_frames = total_frames
        logger.info(f"Frame range set to {total_frames} frames")

    def perform_qca_analysis(self):
        """Perform QCA analysis using segmentation results"""
        if not self.current_result or not self.current_result.get("success"):
            QMessageBox.warning(self, "No Segmentation", "Please perform vessel segmentation first")
            return

        # Check if we have reference points (the user points used for segmentation)
        if hasattr(self, "user_points") and len(self.user_points) >= 1:
            # Add reference points to the result
            if len(self.user_points) >= 2:
                self.current_result["reference_points"] = {
                    "proximal": self.user_points[0],  # First point is proximal
                    "distal": self.user_points[-1],  # Last point is distal
                }
            else:
                # Single point - use same point for both proximal and distal
                self.current_result["reference_points"] = {
                    "proximal": self.user_points[0],
                    "distal": self.user_points[0],
                }
            logger.info(
                f"QCA will be limited to region between {self.user_points[0]} and {self.user_points[-1]}"
            )
        else:
            logger.info("No reference points available, QCA will analyze entire vessel")

        # Emit signal with segmentation data for QCA analysis
        # No need to check for centerline - QCA will generate its own
        self.qca_analysis_requested.emit(self.current_result)

        # Show message
        QMessageBox.information(
            self, "QCA Analysis", "QCA analysis started. Check the QCA panel for results."
        )

    def _check_and_import_tracking_points(self):
        """Check for tracking points in current frame and import them"""
        if not self.main_window or not hasattr(self.main_window, "viewer_widget"):
            return

        viewer = self.main_window.viewer_widget

        # Check if viewer has overlay_item with frame_points
        if not hasattr(viewer, "overlay_item") or not hasattr(viewer.overlay_item, "frame_points"):
            return

        # Get current frame index
        current_frame = viewer.current_frame_index if hasattr(viewer, "current_frame_index") else 0

        # Get tracking points for current frame
        tracking_points = viewer.overlay_item.frame_points.get(current_frame, [])

        # If we have at least 1 tracking point and no segmentation points, use all of them
        if len(tracking_points) >= 1 and len(self.user_points) == 0:
            # Clear existing points
            self.user_points = []

            # Add all tracking points as segmentation points (no limit)
            for point in tracking_points:
                self.user_points.append(tuple(point))

            # Update UI
            self.point_count_label.setText(f"Points: {len(self.user_points)}")
            self.instructions.setText("Tracking points imported. Ready for segmentation")
            self.segment_button.setEnabled(True)

            # Also update viewer to show these points as segmentation points
            if hasattr(viewer, "set_segmentation_points"):
                viewer.set_segmentation_points(self.user_points)

            logger.info(f"Imported {len(self.user_points)} tracking points for segmentation")

        # Update frame status after import
        self.update_frame_status()

    def on_frame_changed(self, new_frame_index: int):
        """Called when frame changes - clear points and check for tracking points"""
        # Clear current segmentation points when frame changes
        self.user_points = []
        self.point_count_label.setText("Points: 0")

        # Clear viewer's segmentation display
        if self.main_window and hasattr(self.main_window, "viewer_widget"):
            viewer = self.main_window.viewer_widget
            if hasattr(viewer, "clear_segmentation_points"):
                viewer.clear_segmentation_points()

        # Check for tracking points in new frame
        if self.segmentation_mode:
            self._check_and_import_tracking_points()

        # Update frame status
        self.update_frame_status(new_frame_index)

    def update_frame_status(self, frame_index: int = None):
        """Update the frame point status display"""
        if not self.main_window or not hasattr(self.main_window, "viewer_widget"):
            self.frame_status_label.setText("Frame status: No viewer available")
            return

        viewer = self.main_window.viewer_widget

        # Get current frame index if not provided
        if frame_index is None:
            frame_index = (
                viewer.current_frame_index if hasattr(viewer, "current_frame_index") else 0
            )

        # Check for tracking points
        tracking_points = []
        if hasattr(viewer, "overlay_item") and hasattr(viewer.overlay_item, "frame_points"):
            all_frame_points = viewer.overlay_item.frame_points
            tracking_points = all_frame_points.get(frame_index, [])

        # Build status text
        status_parts = []

        # Frame info
        status_parts.append(f"Frame {frame_index + 1}")

        # Tracking points
        if tracking_points:
            status_parts.append(f"{len(tracking_points)} tracking pts")

        # Segmentation points (current widget points)
        if self.user_points:
            status_parts.append(f"{len(self.user_points)} seg pts")

        # Total and readiness
        total_available = len(tracking_points) + len(self.user_points)
        if total_available == 0:
            status_parts.append("No points")
            color = "#d32f2f"  # Red
        elif total_available >= 2:
            status_parts.append("✓ Ready")
            color = "#388e3c"  # Green
        else:
            status_parts.append("Need 1 more pt")
            color = "#f57c00"  # Orange

        # Update label
        status_text = " | ".join(status_parts)
        self.frame_status_label.setText(f"Status: {status_text}")
        self.frame_status_label.setStyleSheet(
            f"font-size: 11px; color: {color}; background-color: #f0f0f0; padding: 5px; border-radius: 3px; font-weight: bold;"
        )

        # Update segmentation button state based on available points
        if self.segmentation_mode:
            # Enable button if we have at least 2 points total (tracking + segmentation)
            if total_available >= 2:
                self.segment_button.setEnabled(True)
                self.instructions.setText("Ready for segmentation - Click 'Perform Segmentation'")
            else:
                # Only disable if we don't have enough points from any source
                if len(self.user_points) < 2:
                    self.segment_button.setEnabled(False)
                    if total_available == 1:
                        self.instructions.setText("Add 1 more point or use tracking points")
                    else:
                        self.instructions.setText("Add at least 2 points for segmentation")
