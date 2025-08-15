"""Video playback control for DICOM sequences"""

from PyQt6.QtCore import QTimer
import logging

logger = logging.getLogger(__name__)


class PlaybackController:
    """Handles video playback functionality for DICOM sequences"""

    def __init__(self, main_window):
        """
        Initialize playback controller

        Args:
            main_window: Reference to the main window
        """
        self.main_window = main_window
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self._next_frame_auto)
        self.is_playing = False

    def start_auto_playback(self):
        """Start automatic playback with configured FPS"""
        if not self.main_window.dicom_parser:
            return

        # Get FPS from DICOM or default
        if hasattr(self.main_window.dicom_parser, "fps") and self.main_window.dicom_parser.fps > 0:
            fps = self.main_window.dicom_parser.fps
        else:
            fps = 15  # Default FPS

        # Calculate interval in milliseconds
        interval = int(1000 / fps)
        self.auto_play_timer.setInterval(interval)
        self.auto_play_timer.start()
        self.is_playing = True

        # Update UI
        self._update_play_button()
        logger.info(f"Started playback at {fps} FPS")

    def stop_auto_playback(self):
        """Stop automatic playback"""
        self.auto_play_timer.stop()
        self.is_playing = False
        self._update_play_button()
        logger.info("Stopped playback")

    def toggle_play(self):
        """Toggle between play and pause"""
        if self.is_playing:
            self.stop_auto_playback()
        else:
            self.start_auto_playback()

    def next_frame(self):
        """Go to next frame"""
        if not self.main_window.dicom_parser:
            return

        current = self.main_window.frame_slider.value()
        if current < self.main_window.dicom_parser.num_frames - 1:
            self.main_window.frame_slider.setValue(current + 1)

    def previous_frame(self):
        """Go to previous frame"""
        if not self.main_window.dicom_parser:
            return

        current = self.main_window.frame_slider.value()
        if current > 0:
            self.main_window.frame_slider.setValue(current - 1)

    def next_projection(self):
        """Navigate to next projection/series"""
        if not hasattr(self.main_window, "current_projection_index"):
            return

        if (
            self.main_window.current_projection_index
            < len(self.main_window.available_projections) - 1
        ):
            self.main_window.current_projection_index += 1
            self.main_window._load_projection_at_index(self.main_window.current_projection_index)

    def previous_projection(self):
        """Navigate to previous projection/series"""
        if not hasattr(self.main_window, "current_projection_index"):
            return

        if self.main_window.current_projection_index > 0:
            self.main_window.current_projection_index -= 1
            self.main_window._load_projection_at_index(self.main_window.current_projection_index)

    def navigate_to_frame(self, frame_index: int):
        """Navigate to specific frame"""
        if not self.main_window.dicom_parser:
            return

        if 0 <= frame_index < self.main_window.dicom_parser.num_frames:
            self.main_window.frame_slider.setValue(frame_index)

    def set_frame_range(self, start_frame: int, end_frame: int):
        """Set playback range limits"""
        if not self.main_window.dicom_parser:
            return

        # Store navigation range
        self.main_window.navigation_range = (start_frame, end_frame)

        # Update slider range
        self.main_window.frame_slider.setMinimum(start_frame)
        self.main_window.frame_slider.setMaximum(end_frame)

        # Navigate to start if outside range
        current = self.main_window.frame_slider.value()
        if current < start_frame or current > end_frame:
            self.main_window.frame_slider.setValue(start_frame)

        # Update UI
        self.main_window.navigation_label.setText(f"Range: {start_frame+1}-{end_frame+1}")
        self.main_window.navigation_label.setVisible(True)

        logger.info(f"Set navigation range: frames {start_frame} to {end_frame}")

    def clear_frame_range(self):
        """Clear playback range limits"""
        if not self.main_window.dicom_parser:
            return

        # Clear range
        if hasattr(self.main_window, "navigation_range"):
            delattr(self.main_window, "navigation_range")

        # Reset slider range
        self.main_window.frame_slider.setMinimum(0)
        self.main_window.frame_slider.setMaximum(self.main_window.dicom_parser.num_frames - 1)

        # Hide label
        self.main_window.navigation_label.setVisible(False)

        logger.info("Cleared navigation range")

    def _next_frame_auto(self):
        """Internal method for auto-playback timer"""
        current = self.main_window.frame_slider.value()
        max_frame = self.main_window.frame_slider.maximum()

        if current < max_frame:
            self.main_window.frame_slider.setValue(current + 1)
        else:
            # Loop back to beginning or range start
            min_frame = self.main_window.frame_slider.minimum()
            self.main_window.frame_slider.setValue(min_frame)

    def _update_play_button(self):
        """Update play button icon/text"""
        if hasattr(self.main_window, "play_button"):
            if self.is_playing:
                self.main_window.play_button.setText("⏸")  # Pause icon
            else:
                self.main_window.play_button.setText("▶")  # Play icon

    def get_current_frame(self) -> int:
        """Get current frame index"""
        return self.main_window.frame_slider.value() if self.main_window.frame_slider else 0

    def get_total_frames(self) -> int:
        """Get total number of frames"""
        return self.main_window.dicom_parser.num_frames if self.main_window.dicom_parser else 0

    def is_at_start(self) -> bool:
        """Check if at first frame"""
        return self.get_current_frame() == self.main_window.frame_slider.minimum()

    def is_at_end(self) -> bool:
        """Check if at last frame"""
        return self.get_current_frame() == self.main_window.frame_slider.maximum()
