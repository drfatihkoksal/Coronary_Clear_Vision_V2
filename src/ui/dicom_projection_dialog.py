"""
Projection Selection Dialog
Shows available projections/series from a DICOM folder
"""

import logging
from typing import List, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialogButtonBox,
    QGroupBox, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..core.dicom_folder_loader import DicomSeries

logger = logging.getLogger(__name__)


class ProjectionSelectionDialog(QDialog):
    """Dialog for selecting DICOM projection/series from a study"""
    
    def __init__(self, series_list: List[DicomSeries], study_info: dict, parent=None):
        super().__init__(parent)
        self.series_list = series_list
        self.study_info = study_info
        self.selected_series: Optional[DicomSeries] = None
        
        self.setWindowTitle("Select Projection/Series")
        self.setModal(True)
        self.resize(900, 600)
        
        self.setup_ui()
        self.populate_data()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Study information
        study_group = QGroupBox("Study Information")
        study_layout = QVBoxLayout()
        
        study_text = f"""
        <b>Patient:</b> {self.study_info.get('patient_name', 'Unknown')}<br>
        <b>Patient ID:</b> {self.study_info.get('patient_id', 'Unknown')}<br>
        <b>Study Date:</b> {self.study_info.get('study_date', 'Unknown')}<br>
        <b>Study Description:</b> {self.study_info.get('study_description', 'Unknown')}<br>
        <b>Number of Series:</b> {self.study_info.get('num_series', 0)}
        """
        
        study_label = QLabel(study_text)
        study_label.setWordWrap(True)
        study_layout.addWidget(study_label)
        study_group.setLayout(study_layout)
        layout.addWidget(study_group)
        
        # Series table
        series_group = QGroupBox("Available Series/Projections")
        series_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Series #", "Description", "Frames", "LAO/RAO", "CRAN/CAUD", "View"
        ])
        
        # Enable selection
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # Connect double-click
        self.table.itemDoubleClicked.connect(self.on_double_click)
        
        series_layout.addWidget(self.table)
        series_group.setLayout(series_layout)
        layout.addWidget(series_group)
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def populate_data(self):
        """Populate the series table"""
        self.table.setRowCount(len(self.series_list))
        
        for row, series in enumerate(self.series_list):
            # Series number
            self.table.setItem(row, 0, QTableWidgetItem(str(series.series_number)))
            
            # Description
            self.table.setItem(row, 1, QTableWidgetItem(series.series_description))
            
            # Number of frames
            frames_text = str(series.num_frames) if series.num_frames else str(series.num_instances)
            self.table.setItem(row, 2, QTableWidgetItem(frames_text))
            
            # LAO/RAO angle
            lao_rao = f"{series.primary_angle:.1f}°" if series.primary_angle is not None else "-"
            self.table.setItem(row, 3, QTableWidgetItem(lao_rao))
            
            # CRAN/CAUD angle
            cran_caud = f"{series.secondary_angle:.1f}°" if series.secondary_angle is not None else "-"
            self.table.setItem(row, 4, QTableWidgetItem(cran_caud))
            
            # View position
            view = series.view_position if series.view_position else "-"
            self.table.setItem(row, 5, QTableWidgetItem(view))
            
        # Adjust column widths
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Select first row by default
        if self.series_list:
            self.table.selectRow(0)
            
    def on_double_click(self):
        """Handle double-click on a row"""
        self.accept()
        
    def get_selected_series(self) -> Optional[DicomSeries]:
        """Get the selected series"""
        current_row = self.table.currentRow()
        if 0 <= current_row < len(self.series_list):
            return self.series_list[current_row]
        return None
        
    def accept(self):
        """Override accept to store selected series"""
        self.selected_series = self.get_selected_series()
        super().accept()