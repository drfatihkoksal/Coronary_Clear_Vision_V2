"""Analysis coordination and workflow management"""

from PyQt6.QtCore import pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QDialog, QVBoxLayout, QLabel, QPushButton
import logging
import traceback
from typing import Dict, Optional

from ...core.sequential_processor import SequentialSegmentationQCAProcessor

logger = logging.getLogger(__name__)


class AnalysisCoordinator(QObject):
    """Coordinates analysis workflows including segmentation, QCA, and sequential processing"""
    
    # Signals for communication with main window
    analysis_started = pyqtSignal(str)  # analysis_type
    analysis_completed = pyqtSignal(str, dict)  # analysis_type, results
    analysis_error = pyqtSignal(str, str)  # analysis_type, error_message
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    
    def __init__(self, main_window):
        """
        Initialize analysis coordinator
        
        Args:
            main_window: Reference to the main window
        """
        super().__init__()
        self.main_window = main_window
        self.sequential_processor = None
        self.current_analysis = None
        self.progress_dialog = None
        
    def start_calibration(self):
        """Start calibration workflow"""
        try:
            # Check if we have a loaded DICOM
            if not self.main_window.dicom_parser:
                QMessageBox.warning(
                    self.main_window,
                    "No DICOM Loaded",
                    "Please load a DICOM file first."
                )
                return
                
            # Switch to calibration mode
            self.main_window.activity_bar.set_mode("calibration")
            self.current_analysis = "calibration"
            self.analysis_started.emit("calibration")
            
            logger.info("Started calibration workflow")
            
        except Exception as e:
            logger.error(f"Error starting calibration: {e}")
            self.analysis_error.emit("calibration", str(e))
            
    def start_segmentation(self, points: list = None):
        """
        Start vessel segmentation
        
        Args:
            points: Optional list of user points for segmentation
        """
        try:
            if not self.main_window.dicom_parser:
                QMessageBox.warning(
                    self.main_window,
                    "No DICOM Loaded",
                    "Please load a DICOM file first."
                )
                return
                
            self.current_analysis = "segmentation"
            self.analysis_started.emit("segmentation")
            
            # Switch to segmentation mode if not already there
            if self.main_window.activity_bar.current_mode != "segmentation":
                self.main_window.activity_bar.set_mode("segmentation")
                
            # Start segmentation in the segmentation widget
            if hasattr(self.main_window, 'segmentation_widget'):
                self.main_window.segmentation_widget.start_segmentation()
                
            logger.info("Started segmentation workflow")
            
        except Exception as e:
            logger.error(f"Error starting segmentation: {e}")
            self.analysis_error.emit("segmentation", str(e))
            
    def start_qca_analysis(self, segmentation_result: dict = None):
        """
        Start QCA analysis
        
        Args:
            segmentation_result: Optional segmentation result to use for QCA
        """
        try:
            if not self.main_window.dicom_parser:
                QMessageBox.warning(
                    self.main_window,
                    "No DICOM Loaded",
                    "Please load a DICOM file first."
                )
                return
                
            # Check calibration
            if not hasattr(self.main_window, 'calibration_factor') or not self.main_window.calibration_factor:
                QMessageBox.warning(
                    self.main_window,
                    "Calibration Required",
                    "Please calibrate the measurements first."
                )
                return
                
            self.current_analysis = "qca"
            self.analysis_started.emit("qca")
            
            # Switch to QCA mode
            self.main_window.activity_bar.set_mode("qca")
            
            # Start QCA analysis
            if hasattr(self.main_window, 'qca_widget'):
                if segmentation_result:
                    self.main_window.qca_widget.start_qca_from_segmentation(segmentation_result)
                else:
                    self.main_window.qca_widget.start_qca_analysis()
                    
            logger.info("Started QCA analysis workflow")
            
        except Exception as e:
            logger.error(f"Error starting QCA analysis: {e}")
            self.analysis_error.emit("qca", str(e))
            
    def start_sequential_processing(self, start_frame: int = None, end_frame: int = None):
        """
        Start sequential processing across multiple frames
        
        Args:
            start_frame: Starting frame index (default: current frame)
            end_frame: Ending frame index (default: last frame)
        """
        try:
            if not self.main_window.dicom_parser:
                QMessageBox.warning(
                    self.main_window,
                    "No DICOM Loaded",
                    "Please load a DICOM file first."
                )
                return
                
            # Check calibration
            if not hasattr(self.main_window, 'calibration_factor') or not self.main_window.calibration_factor:
                QMessageBox.warning(
                    self.main_window,
                    "Calibration Required",
                    "Please calibrate the measurements first."
                )
                return
                
            # Get frame range
            if start_frame is None:
                start_frame = self.main_window.viewer_widget.current_frame_index
            if end_frame is None:
                end_frame = self.main_window.dicom_parser.num_frames - 1
                
            # Validate range
            if start_frame >= end_frame:
                QMessageBox.warning(
                    self.main_window,
                    "Invalid Range",
                    "End frame must be greater than start frame."
                )
                return
                
            self.current_analysis = "sequential"
            self.analysis_started.emit("sequential")
            
            # Create progress dialog
            total_frames = end_frame - start_frame + 1
            self._create_progress_dialog("Sequential Processing", total_frames)
            
            # Initialize sequential processor
            self.sequential_processor = SequentialSegmentationQCAProcessor(
                dicom_parser=self.main_window.dicom_parser,
                calibration_factor=self.main_window.calibration_factor
            )
            
            # Connect signals
            self.sequential_processor.frame_started.connect(self._on_sequential_frame_started)
            self.sequential_processor.segmentation_completed.connect(self._on_sequential_segmentation)
            self.sequential_processor.qca_completed.connect(self._on_sequential_qca)
            self.sequential_processor.frame_completed.connect(self._on_sequential_frame_completed)
            self.sequential_processor.all_completed.connect(self._on_sequential_all_completed)
            self.sequential_processor.error_occurred.connect(self._on_sequential_error)
            
            # Start processing
            self.sequential_processor.process_range(start_frame, end_frame)
            
            logger.info(f"Started sequential processing: frames {start_frame} to {end_frame}")
            
        except Exception as e:
            logger.error(f"Error starting sequential processing: {e}")
            self.analysis_error.emit("sequential", str(e))
            
    def start_tracking_workflow(self, start_frame: int = None, end_frame: int = None):
        """
        Start tracking workflow across frames
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
        """
        try:
            if not self.main_window.dicom_parser:
                QMessageBox.warning(
                    self.main_window,
                    "No DICOM Loaded",
                    "Please load a DICOM file first."
                )
                return
                
            # Check if we have points to track
            if not hasattr(self.main_window.viewer_widget, 'overlay_item') or \
               not self.main_window.viewer_widget.overlay_item.frame_points:
                QMessageBox.warning(
                    self.main_window,
                    "No Points to Track",
                    "Please add some tracking points first."
                )
                return
                
            self.current_analysis = "tracking"
            self.analysis_started.emit("tracking")
            
            # Get current frame points
            current_frame = self.main_window.viewer_widget.current_frame_index
            points = self.main_window.viewer_widget.overlay_item.frame_points.get(current_frame, [])
            
            if not points:
                QMessageBox.warning(
                    self.main_window,
                    "No Points in Current Frame",
                    "Please add tracking points in the current frame."
                )
                return
                
            # Start tracking
            if start_frame is None:
                start_frame = current_frame
            if end_frame is None:
                end_frame = self.main_window.dicom_parser.num_frames - 1
                
            total_frames = abs(end_frame - start_frame) + 1
            self._create_progress_dialog("Tracking Points", total_frames)
            
            # Start tracking in viewer
            self.main_window.viewer_widget.track_all_frames(
                progress_callback=self._update_tracking_progress,
                start_frame=start_frame,
                end_frame=end_frame
            )
            
            logger.info(f"Started tracking workflow: frames {start_frame} to {end_frame}")
            
        except Exception as e:
            logger.error(f"Error starting tracking: {e}")
            self.analysis_error.emit("tracking", str(e))
            
    def _create_progress_dialog(self, title: str, total_steps: int):
        """Create progress dialog for long-running operations"""
        self.progress_dialog = QProgressDialog(
            f"Processing...", "Cancel", 0, total_steps, self.main_window
        )
        self.progress_dialog.setWindowTitle(title)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.show()
        
    def _update_progress(self, current: int, total: int, message: str = ""):
        """Update progress dialog"""
        if self.progress_dialog:
            self.progress_dialog.setValue(current)
            if message:
                self.progress_dialog.setLabelText(message)
            self.progress_dialog.show()
            
        # Emit signal for external listeners
        self.progress_updated.emit(current, total, message)
        
    def _update_tracking_progress(self, current: int, total: int):
        """Callback for tracking progress updates"""
        self._update_progress(current, total, f"Tracking frame {current}/{total}")
        
    def _close_progress_dialog(self):
        """Close progress dialog"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
    def _on_sequential_frame_started(self, frame_idx: int, current: int, total: int):
        """Handle sequential processing frame started"""
        self._update_progress(current, total, f"Processing frame {frame_idx+1}")
        
    def _on_sequential_segmentation(self, frame_idx: int, result: dict):
        """Handle sequential segmentation completed"""
        logger.info(f"Segmentation completed for frame {frame_idx}")
        
    def _on_sequential_qca(self, frame_idx: int, result: dict):
        """Handle sequential QCA completed"""
        logger.info(f"QCA completed for frame {frame_idx}")
        
    def _on_sequential_frame_completed(self, frame_idx: int, seg_result: dict, qca_result: dict):
        """Handle sequential frame processing completed"""
        logger.info(f"Frame {frame_idx} processing completed")
        
    def _on_sequential_all_completed(self, seg_results: dict, qca_results: dict):
        """Handle sequential processing all completed"""
        self._close_progress_dialog()
        
        # Show results summary
        self._show_sequential_results_summary(seg_results, qca_results)
        
        self.analysis_completed.emit("sequential", {
            'segmentation_results': seg_results,
            'qca_results': qca_results
        })
        
        logger.info("Sequential processing completed successfully")
        
    def _on_sequential_error(self, frame_idx: int, error_msg: str):
        """Handle sequential processing error"""
        self._close_progress_dialog()
        
        QMessageBox.critical(
            self.main_window,
            "Sequential Processing Error",
            f"Error processing frame {frame_idx}:\n{error_msg}"
        )
        
        self.analysis_error.emit("sequential", f"Frame {frame_idx}: {error_msg}")
        
    def _show_sequential_results_summary(self, seg_results: dict, qca_results: dict):
        """Show summary of sequential processing results"""
        # Create summary dialog
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Sequential Processing Results")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Summary statistics
        seg_count = len([r for r in seg_results.values() if r.get('success')])
        qca_count = len([r for r in qca_results.values() if r.get('success')])
        total_frames = len(seg_results)
        
        summary_text = f"""
        Sequential Processing Complete
        
        Total Frames Processed: {total_frames}
        Successful Segmentations: {seg_count}
        Successful QCA Analyses: {qca_count}
        
        Success Rate: {(qca_count/total_frames)*100:.1f}%
        """
        
        label = QLabel(summary_text)
        layout.addWidget(label)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def cancel_current_analysis(self):
        """Cancel currently running analysis"""
        if self.sequential_processor:
            self.sequential_processor.stop_processing()
            self.sequential_processor = None
            
        self._close_progress_dialog()
        
        if self.current_analysis:
            logger.info(f"Cancelled {self.current_analysis} analysis")
            self.current_analysis = None
            
    def get_current_analysis(self) -> Optional[str]:
        """Get currently running analysis type"""
        return self.current_analysis
        
    def is_analysis_running(self) -> bool:
        """Check if any analysis is currently running"""
        return self.current_analysis is not None