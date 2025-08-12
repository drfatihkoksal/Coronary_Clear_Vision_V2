"""
Calibration Dialog for catheter-based calibration
Allows selecting catheter size and provides calibration instructions
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                            QComboBox, QPushButton, QGroupBox, QDialogButtonBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont

class CalibrationDialog(QDialog):
    """Dialog for catheter calibration settings"""

    calibration_started = pyqtSignal(str)  # Emits catheter size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Catheter Calibration")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Instructions
        instructions_group = QGroupBox("Calibration Instructions")
        instructions_layout = QVBoxLayout()

        instructions_text = """
1. Select the catheter size being used in the procedure
2. Click 'Start Calibration' to enter calibration mode
3. Click two points along the catheter's centerline
4. The system will automatically segment the catheter between these points
5. The catheter width will be measured perpendicular to its centerline
6. The calibration factor will be calculated as mm/pixel ratio
        """

        instructions_label = QLabel(instructions_text.strip())
        instructions_label.setWordWrap(True)
        instructions_layout.addWidget(instructions_label)

        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)

        # Catheter size selection
        size_group = QGroupBox("Catheter Size Selection")
        size_layout = QVBoxLayout()

        # Size selector
        size_selector_layout = QHBoxLayout()
        size_selector_layout.addWidget(QLabel("Catheter Size:"))

        self.size_combo = QComboBox()
        # Common catheter sizes in French (1 French = 0.33mm)
        catheter_sizes = [
            ("4F", "4 French (1.33 mm)"),
            ("5F", "5 French (1.67 mm)"),
            ("6F", "6 French (2.00 mm)"),
            ("7F", "7 French (2.33 mm)"),
            ("8F", "8 French (2.67 mm)")
        ]

        for size_code, size_description in catheter_sizes:
            self.size_combo.addItem(size_description, size_code)

        # Default to 6F (most common)
        self.size_combo.setCurrentIndex(2)

        size_selector_layout.addWidget(self.size_combo)
        size_selector_layout.addStretch()

        size_layout.addLayout(size_selector_layout)

        # Visual representation
        self.visual_label = QLabel()
        self.visual_label.setMinimumHeight(100)
        self.visual_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.update_visual()

        self.size_combo.currentIndexChanged.connect(self.update_visual)

        size_layout.addWidget(QLabel("Visual Reference:"))
        size_layout.addWidget(self.visual_label)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Dialog buttons
        button_box = QDialogButtonBox()

        self.start_button = QPushButton("Start Calibration")
        self.start_button.clicked.connect(self.start_calibration)
        button_box.addButton(self.start_button, QDialogButtonBox.ButtonRole.AcceptRole)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_box.addButton(self.cancel_button, QDialogButtonBox.ButtonRole.RejectRole)

        layout.addWidget(button_box)

        self.setLayout(layout)

    def update_visual(self):
        """Update visual representation of catheter size"""
        # Create pixmap for visual representation
        pixmap = QPixmap(400, 80)
        pixmap.fill(Qt.GlobalColor.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get current catheter size
        size_code = self.size_combo.currentData()
        if size_code:
            # Calculate diameter in mm
            french_to_mm = 0.33333
            french_size = int(size_code[:-1])  # Remove 'F'
            diameter_mm = french_size * french_to_mm

            # Draw catheter representation
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(100, 100, 100))

            # Scale for visualization (10 pixels per mm)
            scale = 10
            catheter_height = diameter_mm * scale
            catheter_width = 300

            y_center = 40
            y_top = y_center - catheter_height / 2

            painter.drawRect(50, int(y_top), catheter_width, int(catheter_height))

            # Draw dimension lines
            painter.setPen(QColor(255, 0, 0))
            painter.drawLine(30, int(y_top), 30, int(y_top + catheter_height))
            painter.drawLine(25, int(y_top), 35, int(y_top))
            painter.drawLine(25, int(y_top + catheter_height), 35, int(y_top + catheter_height))

            # Draw dimension text
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(5, int(y_center + 5), f"{diameter_mm:.2f}mm")

            # Draw catheter label
            painter.setPen(QColor(0, 0, 0))
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(160, int(y_center + 5), f"{size_code} Catheter")

        painter.end()

        self.visual_label.setPixmap(pixmap)

    def start_calibration(self):
        """Start the calibration process"""
        size_code = self.size_combo.currentData()
        self.calibration_started.emit(size_code)
        self.accept()

    def get_catheter_size(self) -> str:
        """Get selected catheter size"""
        return self.size_combo.currentData()

    def get_catheter_diameter_mm(self) -> float:
        """Get catheter diameter in millimeters"""
        size_code = self.size_combo.currentData()
        if size_code:
            french_size = int(size_code[:-1])
            # 1 French = 0.33333... mm (1/3 mm)
            return round(french_size / 3.0, 2)
        return 2.0  # Default to 6F