"""
Sequential processor for frame-by-frame segmentation and QCA analysis.
Processes frames with tracking points one by one.
"""

import logging
from typing import Dict, List, Optional, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np

from src.services.segmentation_service import SegmentationService
from src.analysis.qca_analysis import QCAAnalysis

logger = logging.getLogger(__name__)


class SequentialSegmentationQCAProcessor(QThread):
    """
    Process segmentation and QCA sequentially for frames.
    Handles both tracked frames (multiple frames with points) and single frame operations.
    Each frame is fully processed before moving to the next.
    """
    
    # Signals
    frame_started = pyqtSignal(int, int, int)  # current_frame, current_index, total_frames
    segmentation_completed = pyqtSignal(int, dict)  # frame_index, segmentation_result
    qca_completed = pyqtSignal(int, dict)  # frame_index, qca_result
    frame_completed = pyqtSignal(int, dict, dict)  # frame_index, seg_result, qca_result
    all_completed = pyqtSignal(dict, dict)  # all_seg_results, all_qca_results
    error_occurred = pyqtSignal(int, str)  # frame_index, error_message
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    
    def __init__(self, dicom_parser, viewer_widget, tracked_frames: Optional[List[int]] = None,
                 single_frame_data: Optional[Dict] = None,
                 calibration_factor: Optional[float] = None,
                 main_window=None,
                 use_curvature_resistant_centerline: bool = False):
        """
        Initialize sequential processor.
        
        Args:
            dicom_parser: DICOM parser instance
            viewer_widget: Viewer widget for getting tracking points
            tracked_frames: List of frame indices that have tracking points (for multi-frame processing)
            single_frame_data: Dict with 'frame_index' and 'points' for single frame processing
            calibration_factor: Calibration factor for QCA analysis
            use_curvature_resistant_centerline: Whether to use curvature-resistant centerline extraction
        """
        super().__init__()
        self.dicom_parser = dicom_parser
        self.viewer_widget = viewer_widget
        self.calibration_factor = calibration_factor
        self.use_curvature_resistant_centerline = use_curvature_resistant_centerline
        self.main_window = main_window
        
        # Determine processing mode
        if single_frame_data:
            # Single frame mode with provided points
            self.tracked_frames = [single_frame_data['frame_index']]
            self.single_frame_points = {single_frame_data['frame_index']: single_frame_data['points']}
            self.is_single_frame_mode = True
        elif tracked_frames:
            # Multi-frame tracked mode
            self.tracked_frames = sorted(tracked_frames)  # Ensure frames are in order
            self.single_frame_points = None
            self.is_single_frame_mode = False
        else:
            raise ValueError("Either tracked_frames or single_frame_data must be provided")
        
        # Services
        self.segmentation_service = SegmentationService()
        
        # If main window has segmentation model, use it
        if main_window and hasattr(main_window, 'segmentation_model'):
            self.segmentation_service.model_manager._segmentation_model = main_window.segmentation_model
            
        self.qca_analyzer = QCAAnalysis()
        if calibration_factor:
            self.qca_analyzer.calibration_factor = calibration_factor
            
        # Results storage
        self.segmentation_results = {}
        self.qca_results = {}
        
        # Control
        self._stop_requested = False
        
    def stop(self):
        """Request to stop processing"""
        self._stop_requested = True
        
    def run(self):
        """Main processing loop - sequential frame processing"""
        print(f"🚀 [TRACKED DEBUG] SEQUENTIAL PROCESSOR STARTING!")
        print(f"🚀 [TRACKED DEBUG] Processing {len(self.tracked_frames)} tracked frames")
        logger.warning(f"🚀 [TRACKED DEBUG] SEQUENTIAL PROCESSOR STARTING with {len(self.tracked_frames)} frames!")
        
        total_frames = len(self.tracked_frames)
        
        try:
            for i, frame_idx in enumerate(self.tracked_frames):
                if self._stop_requested:
                    logger.info("Processing stopped by user request")
                    break
                    
                # Signal frame start
                self.frame_started.emit(frame_idx, i + 1, total_frames)
                self.progress_updated.emit(i + 1, total_frames, f"Processing frame {frame_idx + 1}")
                
                try:
                    # Step 1: Get frame data
                    frame = self.dicom_parser.get_frame(frame_idx)
                    if frame is None:
                        raise ValueError(f"Failed to get frame {frame_idx}")
                    
                    # Step 2: Get tracking points for this frame
                    tracking_points = self._get_tracking_points(frame_idx)
                    print(f"🔍 [TRACKED DEBUG] Frame {frame_idx}: Retrieved {len(tracking_points)} tracking points: {tracking_points}")
                    logger.warning(f"🔍 [TRACKED DEBUG] Frame {frame_idx}: Retrieved {len(tracking_points)} tracking points: {tracking_points}")
                    if len(tracking_points) < 1:
                        raise ValueError(f"Insufficient tracking points for frame {frame_idx}")
                    
                    # Step 3: Perform segmentation
                    logger.info(f"Frame {frame_idx}: Starting segmentation with {len(tracking_points)} points")
                    seg_result = self.segmentation_service.segment_vessel(
                        frame, tracking_points, 
                        use_curvature_resistant_centerline=self.use_curvature_resistant_centerline)
                    
                    if not seg_result.get('success'):
                        raise ValueError(f"Segmentation failed: {seg_result.get('error', 'Unknown error')}")
                    
                    # Add frame metadata
                    seg_result['frame_index'] = frame_idx
                    seg_result['reference_points'] = tracking_points
                    
                    # Store and emit segmentation result
                    self.segmentation_results[frame_idx] = seg_result
                    self.segmentation_completed.emit(frame_idx, seg_result)
                    
                    # Step 4: Perform QCA analysis if calibration is available
                    qca_result = {'success': False, 'error': 'No calibration factor'}
                    
                    if self.calibration_factor:
                        logger.info(f"Frame {frame_idx}: Starting QCA analysis")
                        
                        # Extract proximal and distal points
                        proximal_point = tracking_points[0] if len(tracking_points) > 0 else None
                        distal_point = tracking_points[-1] if len(tracking_points) > 1 else None
                        
                        # Prepare tracked points for centerline generation
                        tracked_points_for_centerline = tracking_points if len(tracking_points) >= 2 else None
                        use_tracked_centerline = tracked_points_for_centerline is not None
                        
                        print(f"🎯 [TRACKED DEBUG] Frame {frame_idx}: tracked_points_for_centerline = {tracked_points_for_centerline}")
                        print(f"🎯 [TRACKED DEBUG] Frame {frame_idx}: use_tracked_centerline = {use_tracked_centerline}")
                        logger.warning(f"🎯 [TRACKED DEBUG] Frame {frame_idx}: use_tracked_centerline = {use_tracked_centerline}")
                        
                        if use_tracked_centerline:
                            print(f"✅ Frame {frame_idx}: Using tracked centerline with {len(tracked_points_for_centerline)} points")
                            logger.warning(f"✅ Frame {frame_idx}: Using tracked centerline with {len(tracked_points_for_centerline)} points")
                        else:
                            print(f"❌ Frame {frame_idx}: Using AngioPy centerline (insufficient tracked points: {len(tracking_points)})")
                            logger.warning(f"❌ Frame {frame_idx}: Using AngioPy centerline (insufficient tracked points: {len(tracking_points)})")
                        
                        # Perform QCA (with gradient method option)
                        # Use segmentation-based method for more stable results
                        use_gradient_method = False  # Use segmentation mask for diameter measurement
                        gradient_method = 'second_derivative'  # Not used when use_gradient_method=False
                        
                        qca_result = self.qca_analyzer.analyze_from_angiopy(
                            seg_result,
                            original_image=frame,  # Pass original frame for centerline-based analysis
                            proximal_point=proximal_point,
                            distal_point=distal_point,
                            tracked_points=tracked_points_for_centerline,
                            use_tracked_centerline=use_tracked_centerline,
                            use_gradient_method=use_gradient_method,
                            gradient_method=gradient_method
                        )
                        
                        if qca_result.get('success'):
                            qca_result['frame_index'] = frame_idx
                            qca_result['display_frame_id'] = frame_idx + 1  # UI shows 1-based
                            
                            # Add cardiac phase info if main window has it
                            if hasattr(self, 'main_window') and self.main_window:
                                if hasattr(self.main_window, 'get_cardiac_phase_for_frame'):
                                    phase_info = self.main_window.get_cardiac_phase_for_frame(frame_idx)
                                    if phase_info:
                                        qca_result['cardiac_phase'] = phase_info.get('phase', '')
                                        qca_result['frame_type'] = phase_info.get('type', '')
                            
                            self.qca_results[frame_idx] = qca_result
                            self.qca_completed.emit(frame_idx, qca_result)
                        else:
                            logger.warning(f"Frame {frame_idx}: QCA failed - {qca_result.get('error')}")
                    else:
                        logger.info(f"Frame {frame_idx}: Skipping QCA - no calibration")
                    
                    # Emit frame completion
                    self.frame_completed.emit(frame_idx, seg_result, qca_result)
                    
                    # Update progress
                    percent_complete = int(((i + 1) / total_frames) * 100)
                    logger.info(f"Frame {frame_idx}: Completed ({percent_complete}% overall)")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Frame {frame_idx}: Processing failed - {error_msg}")
                    self.error_occurred.emit(frame_idx, error_msg)
                    # Continue with next frame instead of stopping
                    continue
            
            # Emit final completion signal
            logger.info(f"Sequential processing completed: {len(self.segmentation_results)} segmentations, "
                       f"{len(self.qca_results)} QCA analyses")
            logger.info("[SIGNAL DEBUG] About to emit all_completed signal...")
            self.all_completed.emit(self.segmentation_results, self.qca_results)
            logger.info("[SIGNAL DEBUG] all_completed signal emitted!")
            
        except Exception as e:
            logger.error(f"Sequential processing error: {e}")
            self.error_occurred.emit(-1, str(e))
            
    def _get_tracking_points(self, frame_idx: int) -> List[Tuple[int, int]]:
        """Get tracking points for a specific frame"""
        points = []
        
        logger.info(f"[TRACKED DEBUG] _get_tracking_points called for frame {frame_idx}")
        logger.info(f"[TRACKED DEBUG] is_single_frame_mode: {self.is_single_frame_mode}")
        logger.info(f"[TRACKED DEBUG] has viewer_widget: {hasattr(self, 'viewer_widget')}")
        
        # For single frame mode, use provided points
        if self.is_single_frame_mode and self.single_frame_points:
            points = self.single_frame_points.get(frame_idx, [])
            logger.info(f"[TRACKED DEBUG] Single frame mode - points: {points}")
        else:
            # Check overlay item for frame points
            if hasattr(self.viewer_widget, 'overlay_item') and hasattr(self.viewer_widget.overlay_item, 'frame_points'):
                frame_points = self.viewer_widget.overlay_item.frame_points.get(frame_idx, [])
                points = frame_points
                logger.info(f"[TRACKED DEBUG] Overlay item frame_points: {frame_points}")
            
            # Also check tracking data if available
            if not points and hasattr(self.viewer_widget, 'tracking_data'):
                tracking_data = self.viewer_widget.tracking_data.get(frame_idx, {})
                if 'points' in tracking_data:
                    points = tracking_data['points']
                    logger.info(f"[TRACKED DEBUG] Tracking data points: {points}")
                
        logger.info(f"[TRACKED DEBUG] Final points for frame {frame_idx}: {points}")
        return points


def create_sequential_processor(main_window, start_frame: int, end_frame: int, 
                               calibration_factor: Optional[float] = None,
                               use_curvature_resistant_centerline: bool = False) -> Optional[SequentialSegmentationQCAProcessor]:
    """
    Factory function to create a sequential processor for tracked frames.
    
    Args:
        main_window: Main window instance
        start_frame: Start frame index
        end_frame: End frame index
        calibration_factor: Optional calibration factor to use (if None, will check main_window)
        
    Returns:
        Configured SequentialSegmentationQCAProcessor or None if no tracked frames
    """
    # Get frames with tracking points
    tracked_frames = []
    
    if hasattr(main_window.viewer_widget, 'overlay_item') and hasattr(main_window.viewer_widget.overlay_item, 'frame_points'):
        frame_points = main_window.viewer_widget.overlay_item.frame_points
        tracked_frames = [
            frame_idx for frame_idx in range(start_frame, end_frame + 1)
            if frame_idx in frame_points and len(frame_points[frame_idx]) >= 1
        ]
    
    if not tracked_frames:
        logger.warning(f"No tracked frames found in range {start_frame}-{end_frame}")
        return None
    
    logger.info(f"Found {len(tracked_frames)} tracked frames for sequential processing")
    
    # Use provided calibration factor or get from main window
    if calibration_factor is None:
        calibration_factor = getattr(main_window, 'calibration_factor', None)
        
        # Fallback to DICOM pixel spacing if no user calibration
        if calibration_factor is None and hasattr(main_window, 'dicom_parser'):
            calibration_factor = main_window.dicom_parser.pixel_spacing
            if calibration_factor:
                logger.info(f"Using DICOM pixel spacing as calibration: {calibration_factor:.5f} mm/pixel")
    
    return SequentialSegmentationQCAProcessor(
        dicom_parser=main_window.dicom_parser,
        viewer_widget=main_window.viewer_widget,
        tracked_frames=tracked_frames,
        calibration_factor=calibration_factor,
        main_window=main_window,
        use_curvature_resistant_centerline=use_curvature_resistant_centerline
    )


def create_single_frame_processor(main_window, frame_index: int, points: List[Tuple[int, int]],
                                 calibration_factor: Optional[float] = None,
                                 use_curvature_resistant_centerline: bool = False) -> SequentialSegmentationQCAProcessor:
    """
    Factory function to create a processor for single frame with given points.
    
    Args:
        main_window: Main window instance
        frame_index: Frame index to process
        points: Points for segmentation (at least 2 points)
        calibration_factor: Optional calibration factor to use (if None, will check main_window)
        use_curvature_resistant_centerline: Whether to use curvature-resistant centerline extraction
        
    Returns:
        Configured SequentialSegmentationQCAProcessor for single frame
    """
    if len(points) < 2:
        raise ValueError("At least 2 points are required for segmentation")
    
    # Use provided calibration factor or get from main window
    if calibration_factor is None:
        calibration_factor = getattr(main_window, 'calibration_factor', None)
        
        # Fallback to DICOM pixel spacing if no user calibration
        if calibration_factor is None and hasattr(main_window, 'dicom_parser'):
            calibration_factor = main_window.dicom_parser.pixel_spacing
            if calibration_factor:
                logger.info(f"Using DICOM pixel spacing as calibration: {calibration_factor:.5f} mm/pixel")
    
    single_frame_data = {
        'frame_index': frame_index,
        'points': points
    }
    
    return SequentialSegmentationQCAProcessor(
        dicom_parser=main_window.dicom_parser,
        viewer_widget=main_window.viewer_widget,
        single_frame_data=single_frame_data,
        calibration_factor=calibration_factor,
        main_window=main_window,
        use_curvature_resistant_centerline=use_curvature_resistant_centerline
    )