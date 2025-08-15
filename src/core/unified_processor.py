"""
Unified processor for single frame and batch operations.
Provides a common interface where single frame is treated as batch with size 1.
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np

logger = logging.getLogger(__name__)


class ProcessorResult:
    """Container for processing results"""

    def __init__(
        self, frame_index: int, success: bool, data: Dict[str, Any], error: Optional[str] = None
    ):
        self.frame_index = frame_index
        self.success = success
        self.data = data
        self.error = error


class BaseProcessor(QThread):
    """Base processor for unified single/batch operations"""

    # Common signals
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    frame_processed = pyqtSignal(int, dict)  # frame_index, result
    processing_completed = pyqtSignal(dict)  # all results
    error_occurred = pyqtSignal(str)

    def __init__(self, frames: List[Tuple[int, np.ndarray]], stop_on_error: bool = True):
        """
        Initialize processor.

        Args:
            frames: List of (frame_index, frame_data) tuples
            stop_on_error: Whether to stop processing on first error (single frame behavior)
        """
        super().__init__()
        self.frames = frames
        self.stop_on_error = stop_on_error
        self._stop_requested = False
        self.results = {}

        # For single frame compatibility
        self.is_single_frame = len(frames) == 1

    def stop(self):
        """Request to stop processing"""
        self._stop_requested = True

    def process_frame(
        self, frame_index: int, frame_data: np.ndarray, progress_callback: Optional[Callable] = None
    ) -> ProcessorResult:
        """
        Process a single frame. Must be implemented by subclasses.

        Args:
            frame_index: Index of the frame
            frame_data: Frame image data
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessorResult containing the processing outcome
        """
        raise NotImplementedError("Subclasses must implement process_frame method")

    def run(self):
        """Main processing loop"""
        total_frames = len(self.frames)
        successful_count = 0

        try:
            for i, (frame_index, frame_data) in enumerate(self.frames):
                if self._stop_requested:
                    break

                # Update progress
                progress_msg = (
                    f"Processing frame {frame_index + 1}"
                    if not self.is_single_frame
                    else "Processing..."
                )
                self.progress_updated.emit(i + 1, total_frames, progress_msg)

                # Create progress callback for single frame operations
                def frame_progress_callback(status: str, percentage: int):
                    if self.is_single_frame:
                        # For single frame, emit detailed progress
                        self.progress_updated.emit(percentage, 100, status)

                # Process frame
                try:
                    result = self.process_frame(
                        frame_index,
                        frame_data,
                        frame_progress_callback if self.is_single_frame else None,
                    )

                    if result.success:
                        successful_count += 1
                        self.results[frame_index] = result.data
                        self.frame_processed.emit(frame_index, result.data)
                    else:
                        logger.warning(f"Frame {frame_index} processing failed: {result.error}")
                        if self.stop_on_error:
                            self.error_occurred.emit(result.error or "Processing failed")
                            break

                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}")
                    if self.stop_on_error:
                        self.error_occurred.emit(str(e))
                        break

            # Emit completion
            if not self._stop_requested:
                logger.info(
                    f"Processing completed: {successful_count}/{total_frames} frames successful"
                )
                self.processing_completed.emit(self.results)

        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.error_occurred.emit(str(e))


class UnifiedSegmentationProcessor(BaseProcessor):
    """Unified processor for segmentation operations"""

    def __init__(
        self,
        frames: List[Tuple[int, np.ndarray]],
        segmentation_service,
        frame_points: Dict[int, List[Tuple[int, int]]],
        stop_on_error: bool = True,
    ):
        """
        Initialize segmentation processor.

        Args:
            frames: List of (frame_index, frame_data) tuples
            segmentation_service: Service for performing segmentation
            frame_points: Dictionary mapping frame indices to reference points
            stop_on_error: Whether to stop on first error
        """
        super().__init__(frames, stop_on_error)
        self.segmentation_service = segmentation_service
        self.frame_points = frame_points
        self.last_successful_points = None

    def process_frame(
        self, frame_index: int, frame_data: np.ndarray, progress_callback: Optional[Callable] = None
    ) -> ProcessorResult:
        """Process segmentation for a single frame"""

        # Get points for this frame
        points = self.frame_points.get(frame_index, [])

        # If no points, try to use last successful points (for batch mode)
        if not points and self.last_successful_points and not self.is_single_frame:
            points = self.last_successful_points
            logger.debug(f"Frame {frame_index}: Using previous frame points")

        if len(points) < 2:
            return ProcessorResult(
                frame_index, False, {}, f"Insufficient points for frame {frame_index}"
            )

        # Perform segmentation
        if progress_callback:
            # For single frame, we can pass the callback through
            # This would require modifying segment_vessel to accept it
            result = self.segmentation_service.segment_vessel(frame_data, points)
        else:
            result = self.segmentation_service.segment_vessel(frame_data, points)

        if result["success"]:
            # Store points for potential reuse
            self.last_successful_points = points

            # Add frame metadata
            result["frame_index"] = frame_index
            result["reference_points"] = points

            return ProcessorResult(frame_index, True, result)
        else:
            return ProcessorResult(frame_index, False, {}, result.get("error", "Unknown error"))


class UnifiedQCAProcessor(BaseProcessor):
    """Unified processor for QCA operations"""

    def __init__(
        self,
        frames: List[Tuple[int, Dict]],
        qca_analyzer,
        calibration_factor: Optional[float] = None,
        stop_on_error: bool = True,
    ):
        """
        Initialize QCA processor.

        Args:
            frames: List of (frame_index, segmentation_result) tuples
            qca_analyzer: QCA analysis instance
            calibration_factor: Calibration factor for measurements
            stop_on_error: Whether to stop on first error
        """
        # For QCA, we pass segmentation results as "frame data"
        super().__init__(frames, stop_on_error)
        self.qca_analyzer = qca_analyzer

        # Set calibration
        if calibration_factor:
            self.qca_analyzer.calibration_factor = calibration_factor
            self.qca_analyzer.last_calibration = {
                "catheter_size": "Unknown",
                "pixel_distance": 1.0,
                "mm_per_pixel": calibration_factor,
            }
            logger.info(f"QCA calibration set to {calibration_factor:.6f} mm/pixel")

    def process_frame(
        self,
        frame_index: int,
        segmentation_result: Dict,
        progress_callback: Optional[Callable] = None,
    ) -> ProcessorResult:
        """Process QCA analysis for a single frame"""

        logger.debug(f"UnifiedQCAProcessor: Processing frame {frame_index}")
        logger.debug(f"Segmentation result keys: {list(segmentation_result.keys())}")

        # Get reference points if available
        proximal_point = segmentation_result.get("proximal_point")
        distal_point = segmentation_result.get("distal_point")

        # If not explicitly set, try to extract from reference_points
        if proximal_point is None or distal_point is None:
            reference_points = segmentation_result.get("reference_points", [])
            if len(reference_points) >= 2:
                # Use first point as proximal, last as distal
                proximal_point = reference_points[0] if proximal_point is None else proximal_point
                distal_point = reference_points[-1] if distal_point is None else distal_point

        if progress_callback:
            progress_callback("Analyzing vessel measurements...", 50)

        # Perform QCA analysis
        result = self.qca_analyzer.analyze_from_angiopy(
            segmentation_result,
            original_image=frame,  # Pass original image for centerline-based analysis
            proximal_point=proximal_point,
            distal_point=distal_point,
        )

        if progress_callback:
            progress_callback("Analysis complete", 100)

        if result.get("success"):
            # Add frame metadata
            result["frame_index"] = frame_index
            return ProcessorResult(frame_index, True, result)
        else:
            return ProcessorResult(
                frame_index, False, {}, result.get("error", "QCA analysis failed")
            )


def create_segmentation_processor(
    dicom_parser,
    viewer_widget,
    frame_indices: List[int],
    segmentation_service,
    single_frame: bool = False,
) -> UnifiedSegmentationProcessor:
    """
    Factory function to create segmentation processor.

    Args:
        dicom_parser: DICOM parser instance
        viewer_widget: Viewer widget for getting frame points
        frame_indices: List of frame indices to process
        segmentation_service: Segmentation service instance
        single_frame: Whether this is single frame processing

    Returns:
        Configured UnifiedSegmentationProcessor
    """
    # Prepare frames
    frames = []
    for idx in frame_indices:
        frame = dicom_parser.get_frame(idx)
        if frame is not None:
            frames.append((idx, frame))

    # Get frame points
    frame_points = {}
    if hasattr(viewer_widget, "overlay_item"):
        frame_points = viewer_widget.overlay_item.frame_points.copy()

    # For single frame, add current points if available
    if single_frame and frame_indices:
        current_frame = frame_indices[0]
        if hasattr(viewer_widget, "user_points"):
            frame_points[current_frame] = viewer_widget.user_points

    return UnifiedSegmentationProcessor(
        frames=frames,
        segmentation_service=segmentation_service,
        frame_points=frame_points,
        stop_on_error=single_frame,  # Single frame stops on error
    )


def create_qca_processor(
    segmentation_results: Dict[int, Dict],
    qca_analyzer,
    calibration_factor: Optional[float] = None,
    single_frame: bool = False,
) -> UnifiedQCAProcessor:
    """
    Factory function to create QCA processor.

    Args:
        segmentation_results: Dictionary of frame_index -> segmentation result
        qca_analyzer: QCA analyzer instance
        calibration_factor: Calibration factor
        single_frame: Whether this is single frame processing

    Returns:
        Configured UnifiedQCAProcessor
    """
    # Convert results to frame list
    frames = [(idx, result) for idx, result in segmentation_results.items()]

    return UnifiedQCAProcessor(
        frames=frames,
        qca_analyzer=qca_analyzer,
        calibration_factor=calibration_factor,
        stop_on_error=single_frame,
    )
