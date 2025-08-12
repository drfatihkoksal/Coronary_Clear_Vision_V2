"""
Threading utilities for background processing
Handles heavy computations without blocking UI
"""

from PyQt6.QtCore import QThread, pyqtSignal, QObject, QMutex, QMutexLocker
from typing import Callable
import numpy as np
import traceback


class WorkerSignals(QObject):
    """Signals for worker thread communication"""
    started = pyqtSignal()
    progress = pyqtSignal(int, int)  # current, total
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class Worker(QThread):
    """Generic worker thread for background processing"""

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._is_running = True
        self._mutex = QMutex()

    def run(self):
        """Execute the worker function"""
        self.signals.started.emit()

        try:
            # Add progress callback to kwargs if function supports it
            if 'progress_callback' in self.func.__code__.co_varnames:
                self.kwargs['progress_callback'] = self._progress_callback

            result = self.func(*self.args, **self.kwargs)
            self.signals.result.emit({'success': True, 'data': result})

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
            self.signals.result.emit({'success': False, 'error': error_msg})

        finally:
            self.signals.finished.emit()

    def _progress_callback(self, current: int, total: int) -> bool:
        """Internal progress callback"""
        with QMutexLocker(self._mutex):
            if not self._is_running:
                return False
        self.signals.progress.emit(current, total)
        return True

    def stop(self):
        """Stop the worker thread"""
        with QMutexLocker(self._mutex):
            self._is_running = False
        self.wait()


class SegmentationWorker(Worker):
    """Specialized worker for vessel segmentation"""

    def __init__(self, segmenter, frame: np.ndarray, points: list,
                 frame_index: int, settings: dict):
        def segment_task():
            return segmenter.segment_vessel(frame, points, frame_index, settings)

        super().__init__(segment_task)
        self.frame_index = frame_index


class QCAWorker(Worker):
    """Specialized worker for QCA analysis"""

    def __init__(self, qca_analyzer, frame: np.ndarray, vessel_mask: np.ndarray,
                 proximal_point: tuple, distal_point: tuple):
        def qca_task():
            return qca_analyzer.analyze_stenosis(
                frame, vessel_mask, proximal_point, distal_point
            )

        super().__init__(qca_task)


class TrackingWorker(Worker):
    """Specialized worker for point tracking"""

    def __init__(self, viewer, start_frame: int, end_frame: int, points: list):
        def tracking_task(progress_callback=None):
            tracked_frames = {}
            total_frames = abs(end_frame - start_frame) + 1

            # Determine direction
            step = 1 if end_frame > start_frame else -1

            current_points = points.copy()
            prev_frame_idx = start_frame

            for i, frame_idx in enumerate(range(start_frame + step, end_frame + step, step)):
                if progress_callback and not progress_callback(i, total_frames):
                    break

                # Get frames
                prev_frame = viewer.dicom_parser.get_frame(prev_frame_idx)
                curr_frame = viewer.dicom_parser.get_frame(frame_idx)

                if prev_frame is None or curr_frame is None:
                    break

                # Track points
                tracked_points = viewer._track_points_between_frames(
                    prev_frame, curr_frame, current_points
                )

                if tracked_points:
                    tracked_frames[frame_idx] = tracked_points
                    current_points = tracked_points
                    prev_frame_idx = frame_idx
                else:
                    break

            return tracked_frames

        super().__init__(tracking_task)


class BatchTrackingWorker(Worker):
    """Worker for tracking points through all frames"""

    def __init__(self, viewer, current_frame_idx: int, num_frames: int,
                 initial_points: list):
        def batch_tracking_task(progress_callback=None):
            all_tracked = {current_frame_idx: initial_points}

            # Track forward
            if current_frame_idx < num_frames - 1:
                forward_worker = TrackingWorker(
                    viewer, current_frame_idx, num_frames - 1, initial_points
                )
                forward_result = forward_worker.func(
                    lambda c, t: progress_callback(c, t * 2) if progress_callback else True
                )
                all_tracked.update(forward_result)

            # Track backward
            if current_frame_idx > 0:
                backward_worker = TrackingWorker(
                    viewer, current_frame_idx, 0, initial_points
                )
                backward_result = backward_worker.func(
                    lambda c, t: progress_callback(t + c, t * 2) if progress_callback else True
                )
                all_tracked.update(backward_result)

            return all_tracked

        super().__init__(batch_tracking_task)


class FrameProcessor(QThread):
    """Background frame processor with caching"""

    # Signals
    frame_ready = pyqtSignal(int, np.ndarray)  # frame_index, processed_frame

    def __init__(self, dicom_parser, cache_size: int = 10):
        super().__init__()
        self.dicom_parser = dicom_parser
        self.cache_size = cache_size
        self._frame_queue = []
        self._cache = {}
        self._mutex = QMutex()
        self._running = True

    def add_frame_request(self, frame_index: int, window_center: float,
                         window_width: float):
        """Add frame to processing queue"""
        with QMutexLocker(self._mutex):
            request = (frame_index, window_center, window_width)
            if request not in self._frame_queue:
                self._frame_queue.append(request)

    def run(self):
        """Process frames in background"""
        while self._running:
            request = None

            with QMutexLocker(self._mutex):
                if self._frame_queue:
                    request = self._frame_queue.pop(0)

            if request:
                frame_index, wc, ww = request
                cache_key = (frame_index, wc, ww)

                # Check cache
                if cache_key in self._cache:
                    self.frame_ready.emit(frame_index, self._cache[cache_key])
                else:
                    # Process frame
                    frame = self.dicom_parser.get_frame(frame_index)
                    if frame is not None:
                        processed = self.dicom_parser.apply_window_level(frame, wc, ww)

                        # Update cache
                        with QMutexLocker(self._mutex):
                            self._cache[cache_key] = processed
                            # Limit cache size
                            if len(self._cache) > self.cache_size:
                                oldest = next(iter(self._cache))
                                del self._cache[oldest]

                        self.frame_ready.emit(frame_index, processed)
            else:
                self.msleep(10)  # Small delay when idle

    def stop(self):
        """Stop the processor"""
        self._running = False
        self.wait()

    def clear_cache(self):
        """Clear the frame cache"""
        with QMutexLocker(self._mutex):
            self._cache.clear()


class WorkerPool:
    """Manages a pool of worker threads"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.active_workers = []
        self.pending_tasks = []

    def submit(self, worker: Worker):
        """Submit a worker to the pool"""
        # Clean up finished workers
        self.active_workers = [w for w in self.active_workers if w.isRunning()]

        if len(self.active_workers) < self.max_workers:
            # Start immediately
            worker.signals.finished.connect(lambda: self._worker_finished(worker))
            self.active_workers.append(worker)
            worker.start()
        else:
            # Queue for later
            self.pending_tasks.append(worker)

    def _worker_finished(self, worker: Worker):
        """Handle worker completion"""
        if worker in self.active_workers:
            self.active_workers.remove(worker)

        # Start next pending task
        if self.pending_tasks:
            next_worker = self.pending_tasks.pop(0)
            self.submit(next_worker)

    def stop_all(self):
        """Stop all workers"""
        for worker in self.active_workers:
            worker.stop()
        self.pending_tasks.clear()