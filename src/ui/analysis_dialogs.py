"""
Analysis Dialog Windows
Popup dialogs for QCA, Segmentation, and Calibration analysis
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox
from PyQt6.QtCore import Qt, pyqtSignal
import logging

logger = logging.getLogger(__name__)


class CalibrationControlDialog(QDialog):
    """Dialog window for calibration control panel"""

    # Signals
    calibration_confirmed = pyqtSignal(float, dict)
    calibration_reset = pyqtSignal()
    calibration_loaded = pyqtSignal(float, dict)
    calibration_cancelled = pyqtSignal()
    calibration_deleted = pyqtSignal()

    def __init__(self, calibration_control_panel, parent=None):
        super().__init__(parent)
        self.calibration_control_panel = calibration_control_panel
        self.init_ui()

    def init_ui(self):
        """Initialize dialog UI"""
        self.setWindowTitle("Calibration Control")
        self.setModal(False)
        # Set window flags for non-blocking dialog
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        # Don't steal focus from parent
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Set size - optimized for HD 720p+ screens
        self.resize(320, 450)
        self.setMinimumSize(300, 400)
        self.setMaximumWidth(400)

        # Layout
        layout = QVBoxLayout()

        # Add calibration control panel
        layout.addWidget(self.calibration_control_panel)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Connect signals
        self.calibration_control_panel.calibration_confirmed.connect(
            self.calibration_confirmed.emit
        )
        self.calibration_control_panel.calibration_reset.connect(self.calibration_reset.emit)
        self.calibration_control_panel.calibration_loaded.connect(self.calibration_loaded.emit)
        self.calibration_control_panel.calibration_cancelled.connect(
            self.calibration_cancelled.emit
        )
        self.calibration_control_panel.calibration_deleted.connect(self.calibration_deleted.emit)

    def keyPressEvent(self, event):
        """Pass navigation keys to parent window for playback control"""
        # Pass navigation keys to parent for playback control
        if event.key() in [
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Space,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
        ]:
            if self.parent():
                # Pass event to parent and don't consume it
                self.parent().keyPressEvent(event)
                return  # Don't call super, let parent handle it
        # For other keys, handle normally
        super().keyPressEvent(event)

    def show(self):
        """Show dialog and position it on the right side"""
        super().show()

        # Automatically activate calibration mode when dialog opens
        # The calibration is already activated by the main window when opening the dialog
        # Just ensure the dialog is ready to receive calibration data

        # Position to the right of parent window
        if self.parent():
            parent_rect = self.parent().geometry()
            screen = self.parent().screen()
            if screen:
                screen_rect = screen.availableGeometry()
                # Calculate position on the right side
                x = parent_rect.right() + 20
                y = parent_rect.top() + 50

                # Make sure dialog stays within screen bounds
                if x + self.width() > screen_rect.right():
                    x = screen_rect.right() - self.width() - 10
                if y + self.height() > screen_rect.bottom():
                    y = screen_rect.bottom() - self.height() - 10

                self.move(x, y)

        # Keep parent window active for navigation
        if self.parent():
            # Use a timer to return focus after dialog is fully shown
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(100, lambda: self.parent().setFocus())

    def set_calibration_info(self, calibration_factor, details):
        """Set calibration information"""
        self.calibration_control_panel.set_calibration_info(calibration_factor, details)

    def reset(self):
        """Reset calibration control panel"""
        self.calibration_control_panel.reset()

    def update_calibration_info(self, calibration_factor, details):
        """Update calibration information"""
        self.calibration_control_panel.update_calibration_info(calibration_factor, details)


class SegmentationDialog(QDialog):
    """Dialog window for vessel segmentation"""

    # Signals
    segmentation_mode_changed = pyqtSignal(bool)
    segmentation_completed = pyqtSignal(dict)
    overlay_settings_changed = pyqtSignal(dict)
    qca_analysis_requested = pyqtSignal(dict)

    def __init__(self, segmentation_widget, parent=None):
        super().__init__(parent)
        self.segmentation_widget = segmentation_widget
        self.init_ui()

    def init_ui(self):
        """Initialize dialog UI"""
        self.setWindowTitle("Vessel Segmentation")
        self.setModal(False)  # Non-modal to allow interaction with main window
        # Set window flags for non-blocking dialog
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        # Don't steal focus from parent
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Set size - optimized for HD 720p+ screens
        self.resize(360, 520)
        self.setMinimumSize(320, 480)
        self.setMaximumWidth(420)

        # Layout
        layout = QVBoxLayout()

        # Add segmentation widget
        layout.addWidget(self.segmentation_widget)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Connect signals
        self.segmentation_widget.segmentation_mode_changed.connect(
            self.segmentation_mode_changed.emit
        )
        self.segmentation_widget.segmentation_completed.connect(self.segmentation_completed.emit)
        self.segmentation_widget.overlay_settings_changed.connect(
            self.overlay_settings_changed.emit
        )
        self.segmentation_widget.qca_analysis_requested.connect(self.qca_analysis_requested.emit)

    def keyPressEvent(self, event):
        """Pass navigation keys to parent window for playback control"""
        # Pass navigation keys to parent for playback control
        if event.key() in [
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Space,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
        ]:
            if self.parent():
                # Pass event to parent and don't consume it
                self.parent().keyPressEvent(event)
                return  # Don't call super, let parent handle it
        # For other keys, handle normally
        super().keyPressEvent(event)

    def show(self):
        """Show dialog and position it on the right side"""
        # First show the dialog
        super().show()

        # Check for existing points in current frame
        self._check_current_frame_points()

        # Automatically activate segmentation mode when dialog opens
        if hasattr(self.segmentation_widget, "mode_button"):
            if not self.segmentation_widget.mode_button.isChecked():
                self.segmentation_widget.mode_button.setChecked(True)
                self.segmentation_widget.toggle_mode()
            # If already checked, ensure the mode is properly set
            elif not self.segmentation_widget.segmentation_mode:
                self.segmentation_widget.segmentation_mode = True
                self.segmentation_widget.segmentation_mode_changed.emit(True)
                # Update frame status when mode is set directly
                self.segmentation_widget.update_frame_status()

        # Position to the right of parent window
        if self.parent():
            parent_rect = self.parent().geometry()
            screen = self.parent().screen()
            if screen:
                screen_rect = screen.availableGeometry()
                # Calculate position on the right side
                x = parent_rect.right() + 20
                y = parent_rect.top() + 50

                # Make sure dialog stays within screen bounds
                if x + self.width() > screen_rect.right():
                    x = screen_rect.right() - self.width() - 10
                if y + self.height() > screen_rect.bottom():
                    y = screen_rect.bottom() - self.height() - 10

                self.move(x, y)

        # Keep parent window active for navigation
        if self.parent():
            # Use a timer to return focus after dialog is fully shown
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(100, lambda: self.parent().setFocus())

        logger.info(f"Segmentation dialog positioned at ({x}, {y})")

    def _check_current_frame_points(self):
        """Check for existing points (tracking or GT) in current frame"""
        if not self.parent() or not hasattr(self.parent(), "viewer_widget"):
            return

        viewer = self.parent().viewer_widget
        current_frame = viewer.current_frame_index if hasattr(viewer, "current_frame_index") else 0

        # Check for tracking points
        tracking_points = []
        if hasattr(viewer, "overlay_item") and hasattr(viewer.overlay_item, "frame_points"):
            tracking_points = viewer.overlay_item.frame_points.get(current_frame, [])

        # Check for segmentation points
        segmentation_points = []
        if hasattr(self.segmentation_widget, "user_points"):
            segmentation_points = self.segmentation_widget.user_points

        # Update status in segmentation widget
        total_points = len(tracking_points) + len(segmentation_points)

        if tracking_points and not segmentation_points:
            # Only tracking points exist
            status_text = f"Found {len(tracking_points)} tracking point(s) in current frame"
            if len(tracking_points) >= 1:
                status_text += " - Ready for segmentation"
            else:
                status_text += " - Need at least 1 point"
        elif segmentation_points:
            # Segmentation points exist
            status_text = f"{len(segmentation_points)} segmentation point(s) set"
        elif total_points == 0:
            # No points
            status_text = "No points found in current frame"
        else:
            # Mixed points
            status_text = f"Found {len(tracking_points)} tracking + {len(segmentation_points)} segmentation points"

        # Update UI to show point status
        if hasattr(self.segmentation_widget, "instructions"):
            # Save original instruction text
            original_text = self.segmentation_widget.instructions.text()

            # Show status temporarily
            self.segmentation_widget.instructions.setText(
                f"<b>Frame {current_frame + 1}:</b> {status_text}"
            )
            self.segmentation_widget.instructions.setStyleSheet(
                "font-size: 12px; color: #1976D2; font-weight: bold;"
            )

            # Create a timer to restore original text after 3 seconds
            from PyQt6.QtCore import QTimer

            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._restore_instructions(original_text))
            timer.start(3000)

        logger.info(
            f"Point check - Frame {current_frame}: {len(tracking_points)} tracking, {len(segmentation_points)} segmentation points"
        )

    def _restore_instructions(self, original_text):
        """Restore original instruction text"""
        if hasattr(self.segmentation_widget, "instructions"):
            self.segmentation_widget.instructions.setText(original_text)
            self.segmentation_widget.instructions.setStyleSheet("font-size: 12px; color: #666;")


class QCADialog(QDialog):
    """Dialog window for QCA analysis"""

    # Signals
    qca_started = pyqtSignal()
    qca_completed = pyqtSignal(dict)
    overlay_changed = pyqtSignal(bool, dict)
    calibration_requested = pyqtSignal()

    def __init__(self, qca_widget, parent=None):
        super().__init__(parent)
        self.qca_widget = qca_widget
        self.init_ui()

    def init_ui(self):
        """Initialize dialog UI"""
        self.setWindowTitle("QCA Analysis")
        self.setModal(False)
        # Set window flags for non-blocking dialog
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        # Don't steal focus from parent
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # Set size - optimized for HD 720p+ screens
        self.resize(700, 600)
        self.setMinimumSize(600, 550)

        # Layout
        layout = QVBoxLayout()

        # Add QCA widget
        layout.addWidget(self.qca_widget)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Connect signals
        self.qca_widget.qca_started.connect(self.qca_started.emit)
        self.qca_widget.qca_completed.connect(self.qca_completed.emit)
        self.qca_widget.overlay_changed.connect(self.overlay_changed.emit)
        self.qca_widget.calibration_requested.connect(self.calibration_requested.emit)

    def keyPressEvent(self, event):
        """Pass navigation keys to parent window for playback control"""
        # Pass navigation keys to parent for playback control
        if event.key() in [
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Space,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
        ]:
            if self.parent():
                # Pass event to parent and don't consume it
                self.parent().keyPressEvent(event)
                return  # Don't call super, let parent handle it
        # For other keys, handle normally
        super().keyPressEvent(event)

    def show(self):
        """Show dialog and position it"""
        super().show()

        # Position to the right of parent window
        if self.parent():
            parent_rect = self.parent().geometry()
            screen = self.parent().screen()
            if screen:
                screen_rect = screen.availableGeometry()
                # Calculate position on the right side
                x = parent_rect.right() + 20
                y = parent_rect.top() + 50

                # Make sure dialog stays within screen bounds
                if x + self.width() > screen_rect.right():
                    x = screen_rect.right() - self.width() - 10
                if y + self.height() > screen_rect.bottom():
                    y = screen_rect.bottom() - self.height() - 10

                self.move(x, y)

        # Keep parent window active for navigation
        if self.parent():
            # Use a timer to return focus after dialog is fully shown
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(100, lambda: self.parent().setFocus())

        logger.info(f"QCA dialog positioned")

    def start_analysis(self, segmentation_result=None):
        """Start QCA analysis"""
        self.qca_widget.start_analysis(segmentation_result)
