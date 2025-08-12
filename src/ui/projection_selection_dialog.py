"""
Projection Selection Dialog
Dialog for selecting DICOM projections with metadata display
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QListWidgetItem,
                             QLabel, QGroupBox, QDialogButtonBox, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ProjectionItem:
    """Data class for projection information"""
    def __init__(self, file_path: str, metadata: Dict[str, Any]):
        self.file_path = file_path
        self.metadata = metadata
        self.projection_name = self._format_projection_name()

    def _format_projection_name(self) -> str:
        """Format projection name from metadata"""
        # Try to get angles from positioner angles
        primary_angle = self.metadata.get('positioner_primary_angle')
        secondary_angle = self.metadata.get('positioner_secondary_angle')

        if primary_angle is not None and secondary_angle is not None:
            # Format as LAO/RAO and CRA/CAU
            lr_view = 'LAO' if primary_angle > 0 else 'RAO'
            cc_view = 'CRA' if secondary_angle > 0 else 'CAU'
            return f"{lr_view} {abs(primary_angle):.1f}° {cc_view} {abs(secondary_angle):.1f}°"

        # Try view position
        view_position = self.metadata.get('view_position')
        if view_position:
            return view_position

        # Try series description
        series_desc = self.metadata.get('series_description', '')
        if series_desc:
            return series_desc

        # Default to filename
        import os
        return os.path.basename(self.file_path)

class ProjectionSelectionDialog(QDialog):
    """Dialog for selecting DICOM projections"""

    projection_selected = pyqtSignal(str)  # Emits file path

    def __init__(self, projections: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.projections = [ProjectionItem(p['file_path'], p['metadata']) for p in projections]
        self.selected_projection = None
        self.setup_ui()
        self.populate_list()

    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Select Projection")
        self.setModal(True)
        self.resize(800, 500)  # Optimized for HD 720p+ displays
        self.setMinimumSize(700, 450)

        # Main layout
        layout = QVBoxLayout()

        # Title
        title = QLabel("Select DICOM Projection")
        title.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        # Create splitter for list and details
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Projection list
        list_group = QGroupBox("Available Projections")
        list_layout = QVBoxLayout()

        self.projection_list = QListWidget()
        self.projection_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.projection_list.itemDoubleClicked.connect(self.on_double_click)
        list_layout.addWidget(self.projection_list)

        list_group.setLayout(list_layout)
        splitter.addWidget(list_group)

        # Right side - Details
        details_group = QGroupBox("Projection Details")
        details_layout = QVBoxLayout()

        self.details_label = QLabel("Select a projection to view details")
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.details_label.setStyleSheet("padding: 10px;")
        details_layout.addWidget(self.details_label)

        details_group.setLayout(details_layout)
        splitter.addWidget(details_group)

        # Set splitter sizes
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        self.ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setEnabled(False)

        layout.addWidget(button_box)

        self.setLayout(layout)

    def populate_list(self):
        """Populate the projection list"""
        for projection in self.projections:
            item = QListWidgetItem(projection.projection_name)
            item.setData(Qt.ItemDataRole.UserRole, projection)
            self.projection_list.addItem(item)

        # Select first item if available
        if self.projection_list.count() > 0:
            self.projection_list.setCurrentRow(0)

    def on_selection_changed(self):
        """Handle selection change"""
        current_item = self.projection_list.currentItem()
        if current_item:
            projection = current_item.data(Qt.ItemDataRole.UserRole)
            self.selected_projection = projection
            self.ok_button.setEnabled(True)
            self.update_details(projection)
        else:
            self.selected_projection = None
            self.ok_button.setEnabled(False)
            self.details_label.setText("Select a projection to view details")

    def update_details(self, projection: ProjectionItem):
        """Update the details panel"""
        details = []

        # Add projection name
        details.append(f"<b>Projection:</b> {projection.projection_name}")

        # Add angle information
        primary_angle = projection.metadata.get('positioner_primary_angle')
        secondary_angle = projection.metadata.get('positioner_secondary_angle')

        if primary_angle is not None:
            details.append(f"<b>Primary Angle:</b> {primary_angle:.1f}°")

        if secondary_angle is not None:
            details.append(f"<b>Secondary Angle:</b> {secondary_angle:.1f}°")

        # Add other metadata
        if projection.metadata.get('patient_name'):
            details.append(f"<b>Patient:</b> {projection.metadata['patient_name']}")

        if projection.metadata.get('study_date'):
            details.append(f"<b>Study Date:</b> {projection.metadata['study_date']}")

        if projection.metadata.get('series_description'):
            details.append(f"<b>Series:</b> {projection.metadata['series_description']}")

        if projection.metadata.get('rows') and projection.metadata.get('columns'):
            details.append(f"<b>Dimensions:</b> {projection.metadata['rows']}x{projection.metadata['columns']}")

        if projection.metadata.get('num_frames'):
            details.append(f"<b>Frames:</b> {projection.metadata['num_frames']}")

        # Update label
        self.details_label.setText("<br>".join(details))

    def on_double_click(self, item):
        """Handle double click - accept dialog"""
        if item:
            self.accept()

    def accept(self):
        """Accept the dialog and emit signal"""
        if self.selected_projection:
            self.projection_selected.emit(self.selected_projection.file_path)
        super().accept()

    def get_selected_file_path(self) -> Optional[str]:
        """Get the selected file path"""
        if self.selected_projection:
            return self.selected_projection.file_path
        return None