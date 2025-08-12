"""
Base class for analysis widgets with common functionality.
Implements common patterns for threading, progress tracking, and error handling.
"""

from typing import Optional, Type, Dict, Any
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QProgressBar, QGroupBox)
from PyQt6.QtCore import pyqtSignal, QThread
from ..core.threading import BaseThread


class BaseAnalysisWidget(QWidget):
    """Base class for analysis widgets with common functionality."""

    # Common signals
    analysis_started = pyqtSignal()
    analysis_completed = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    progress_updated = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_thread: Optional[QThread] = None
        self.main_window = None

        # Common UI elements
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_analysis)

        self._init_base_ui()

    def _init_base_ui(self):
        """Initialize common UI elements."""
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        # Status and progress
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        progress_layout.addLayout(button_layout)

        progress_group.setLayout(progress_layout)

        # Add to main layout (to be completed by subclasses)
        self.progress_group = progress_group

    def set_main_window(self, main_window):
        """Set reference to main window."""
        self.main_window = main_window

    def start_analysis(self, thread_class: Type[BaseThread], **kwargs):
        """
        Start analysis with a worker thread.

        Args:
            thread_class: The thread class to instantiate
            **kwargs: Arguments to pass to the thread
        """
        if self.current_thread and self.current_thread.isRunning():
            self.show_error("Another analysis is already running")
            return

        # Create and configure thread
        self.current_thread = thread_class(**kwargs)

        # Connect signals
        self.current_thread.progress.connect(self.on_progress)
        self.current_thread.finished.connect(self.on_analysis_finished)
        self.current_thread.error.connect(self.on_error)

        # Update UI state
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting analysis...")

        # Emit signal and start
        self.analysis_started.emit()
        self.current_thread.start()

    def cancel_analysis(self):
        """Cancel the current analysis."""
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.terminate()
            self.current_thread.wait()
            self.on_analysis_cancelled()

    def on_progress(self, status: str, percentage: int):
        """Handle progress updates from worker thread."""
        self.status_label.setText(status)
        self.progress_bar.setValue(percentage)
        self.progress_updated.emit(status, percentage)

    def on_analysis_finished(self, result: Dict[str, Any]):
        """Handle successful analysis completion."""
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Analysis completed")
        self.analysis_completed.emit(result)
        self.current_thread = None

    def on_error(self, error_msg: str):
        """Handle analysis errors."""
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Error: {error_msg}")
        self.analysis_error.emit(error_msg)
        self.show_error(error_msg)
        self.current_thread = None

    def on_analysis_cancelled(self):
        """Handle analysis cancellation."""
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis cancelled")
        self.current_thread = None

    def show_error(self, message: str):
        """Show error message to user."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Analysis Error", message)

    def get_current_frame(self):
        """Get current frame from main window viewer."""
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            return self.main_window.viewer_widget.get_current_frame()
        return None

    def get_current_frame_index(self):
        """Get current frame index from main window."""
        if self.main_window:
            return self.main_window.current_frame_index
        return 0

    def update_viewer_overlay(self, overlay_data: Any):
        """Update viewer widget overlay."""
        if self.main_window and hasattr(self.main_window, 'viewer_widget'):
            # To be implemented based on specific overlay needs
            pass