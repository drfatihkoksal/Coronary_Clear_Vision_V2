"""
Tracking Control Widget for CoTracker3 Fine-tuning
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.cotracker3_optimized_adapter import get_cotracker3_adapter


class TrackingControlWidget(QWidget):
    """Widget for controlling CoTracker3 parameters in real-time."""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracker = get_cotracker3_adapter()
        self.init_ui()

    def init_ui(self):
        """Initialize UI controls."""
        layout = QVBoxLayout()

        # Preset selector
        preset_group = QGroupBox("Tracking Preset")
        preset_layout = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Angiography", "High Accuracy", "Fast", "Custom"])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Confidence threshold
        conf_group = QGroupBox("Confidence Settings")
        conf_layout = QVBoxLayout()

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.tracker.confidence_threshold * 100))
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)

        self.conf_label = QLabel(f"Confidence Threshold: {self.tracker.confidence_threshold:.2f}")
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)

        # Motion threshold
        motion_group = QGroupBox("Motion Settings")
        motion_layout = QVBoxLayout()

        self.motion_spin = QDoubleSpinBox()
        self.motion_spin.setRange(10.0, 200.0)
        self.motion_spin.setValue(self.tracker.motion_threshold)
        self.motion_spin.setSuffix(" pixels")
        self.motion_spin.valueChanged.connect(self.on_motion_changed)

        motion_layout.addWidget(QLabel("Max Motion Between Frames:"))
        motion_layout.addWidget(self.motion_spin)
        motion_group.setLayout(motion_layout)
        layout.addWidget(motion_group)

        # Enhancement options
        enhance_group = QGroupBox("Image Enhancement")
        enhance_layout = QVBoxLayout()

        self.enhance_check = QCheckBox("Enable Contrast Enhancement")
        self.enhance_check.setChecked(self.tracker.enhance_contrast)
        self.enhance_check.toggled.connect(self.on_enhance_changed)

        self.kalman_check = QCheckBox("Enable Kalman Filtering")
        self.kalman_check.setChecked(self.tracker.use_kalman)
        self.kalman_check.toggled.connect(self.on_kalman_changed)

        enhance_layout.addWidget(self.enhance_check)
        enhance_layout.addWidget(self.kalman_check)
        enhance_group.setLayout(enhance_layout)
        layout.addWidget(enhance_group)

        # Window size
        window_group = QGroupBox("Processing Window")
        window_layout = QHBoxLayout()

        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 16)
        self.window_spin.setValue(self.tracker.window_size)
        self.window_spin.valueChanged.connect(self.on_window_changed)

        window_layout.addWidget(QLabel("Window Size:"))
        window_layout.addWidget(self.window_spin)
        window_layout.addWidget(QLabel("frames"))
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)

        # Reset button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_settings)
        layout.addWidget(self.reset_btn)

        layout.addStretch()
        self.setLayout(layout)

        # Set compact size
        self.setMaximumWidth(300)

    def on_preset_changed(self, preset_name):
        """Handle preset change."""
        if preset_name == "Angiography":
            self.apply_angiography_preset()
        elif preset_name == "High Accuracy":
            self.apply_high_accuracy_preset()
        elif preset_name == "Fast":
            self.apply_fast_preset()
        # Custom does nothing - keeps current settings

    def apply_angiography_preset(self):
        """Apply angiography preset."""
        self.conf_slider.setValue(10)  # 0.1
        self.motion_spin.setValue(100.0)
        self.enhance_check.setChecked(True)
        self.kalman_check.setChecked(True)
        self.window_spin.setValue(6)

    def apply_high_accuracy_preset(self):
        """Apply high accuracy preset."""
        self.conf_slider.setValue(30)  # 0.3
        self.motion_spin.setValue(50.0)
        self.enhance_check.setChecked(True)
        self.kalman_check.setChecked(True)
        self.window_spin.setValue(12)

    def apply_fast_preset(self):
        """Apply fast preset."""
        self.conf_slider.setValue(20)  # 0.2
        self.motion_spin.setValue(150.0)
        self.enhance_check.setChecked(False)
        self.kalman_check.setChecked(False)
        self.window_spin.setValue(4)

    def on_confidence_changed(self, value):
        """Handle confidence threshold change."""
        conf = value / 100.0
        self.tracker.set_confidence_threshold(conf)
        self.conf_label.setText(f"Confidence Threshold: {conf:.2f}")
        self.preset_combo.setCurrentText("Custom")
        self.settings_changed.emit()

    def on_motion_changed(self, value):
        """Handle motion threshold change."""
        self.tracker.set_motion_threshold(value)
        self.preset_combo.setCurrentText("Custom")
        self.settings_changed.emit()

    def on_enhance_changed(self, checked):
        """Handle enhancement toggle."""
        self.tracker.enhance_contrast = checked
        self.preset_combo.setCurrentText("Custom")
        self.settings_changed.emit()

    def on_kalman_changed(self, checked):
        """Handle Kalman filter toggle."""
        self.tracker.use_kalman = checked
        self.preset_combo.setCurrentText("Custom")
        self.settings_changed.emit()

    def on_window_changed(self, value):
        """Handle window size change."""
        self.tracker.window_size = value
        self.preset_combo.setCurrentText("Custom")
        self.settings_changed.emit()

    def reset_settings(self):
        """Reset to default angiography settings."""
        self.preset_combo.setCurrentText("Angiography")
        self.apply_angiography_preset()
