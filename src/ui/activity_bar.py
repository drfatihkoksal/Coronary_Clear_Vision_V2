"""
VSCode-style Activity Bar for mode selection
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QButtonGroup
from PyQt6.QtCore import pyqtSignal


class ActivityButton(QPushButton):
    """Icon button for activity bar"""

    def __init__(self, icon_text: str, tooltip: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 48)
        self.setText(icon_text)  # Using text as icon placeholder
        self.setToolTip(tooltip)
        self.setCheckable(True)

        # Style
        self.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                color: #333;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.05);
                color: #000;
            }
            QPushButton:checked {
                background-color: rgba(0, 122, 204, 0.1);
                color: #007ACC;
                border-left: 3px solid #007ACC;
            }
        """
        )


class ActivityBar(QWidget):
    """VSCode-style activity bar"""

    # Signals
    mode_changed = pyqtSignal(str)  # Emits mode name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(50)
        self.setStyleSheet("background-color: #f3f3f3; border-right: 1px solid #e0e0e0;")

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Button group for exclusive selection
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Create buttons
        self.buttons = {}

        # Calibration (default active)
        calib_btn = ActivityButton("📏", "Calibration")
        self.buttons["calibration"] = calib_btn
        layout.addWidget(calib_btn)

        # Tracking
        track_btn = ActivityButton("📍", "Tracking")
        self.buttons["tracking"] = track_btn
        layout.addWidget(track_btn)

        # Segmentation
        seg_btn = ActivityButton("🔍", "Segmentation")
        self.buttons["segmentation"] = seg_btn
        layout.addWidget(seg_btn)

        # QCA Analysis
        qca_btn = ActivityButton("📊", "QCA Analysis")
        self.buttons["qca"] = qca_btn
        layout.addWidget(qca_btn)

        # Batch Processing
        batch_btn = ActivityButton("⚡", "Batch Processing")
        self.buttons["batch"] = batch_btn
        layout.addWidget(batch_btn)

        # Add buttons to group
        for name, btn in self.buttons.items():
            self.button_group.addButton(btn)
            btn.clicked.connect(lambda checked, n=name: self.on_button_clicked(n))

        # Spacer
        layout.addStretch()

        # Export (bottom)
        export_btn = ActivityButton("💾", "Export")
        self.buttons["export"] = export_btn
        layout.addWidget(export_btn)
        self.button_group.addButton(export_btn)
        export_btn.clicked.connect(lambda: self.on_button_clicked("export"))

        # Set calibration as default
        calib_btn.setChecked(True)

    def on_button_clicked(self, mode: str):
        """Handle button click"""
        self.mode_changed.emit(mode)

    def set_active_mode(self, mode: str):
        """Set active mode programmatically"""
        if mode in self.buttons:
            self.buttons[mode].setChecked(True)
