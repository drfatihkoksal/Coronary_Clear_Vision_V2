"""
QCA (Quantitative Coronary Analysis) UI Widget - Compact Version
Smart layout that displays all elements without scrollbars
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QMessageBox,
    QSplitter,
    QFileDialog,
    QDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer
from PyQt6.QtGui import QColor
import numpy as np
import cv2
from typing import Dict, Optional
import logging
from .qca_graph_widget import QCAGraphWidget
from ..analysis.qca_report import QCAReportGenerator

logger = logging.getLogger(__name__)


class QCAThread(QThread):
    """Worker thread for QCA analysis"""

    progress = pyqtSignal(str)  # status message
    finished = pyqtSignal(dict)  # results
    error = pyqtSignal(str)

    def __init__(
        self,
        qca_analyzer,
        mask=None,
        calibration_factor=None,
        segmentation_result=None,
        proximal_point=None,
        distal_point=None,
        original_image=None,
        tracked_points=None,
        use_tracked_centerline=False,
        global_reference_diameter=None,
    ):
        super().__init__()
        self.qca_analyzer = qca_analyzer
        self.mask = mask
        self.calibration_factor = calibration_factor
        self.segmentation_result = segmentation_result  # For AngioPy data
        self.proximal_point = proximal_point  # Reference point (x, y)
        self.distal_point = distal_point  # Reference point (x, y)
        self.original_image = original_image  # Original image for centerline-based analysis
        self.tracked_points = tracked_points  # List of (x, y) tracked points
        self.use_tracked_centerline = use_tracked_centerline  # Whether to use tracked centerline
        self.global_reference_diameter = (
            global_reference_diameter  # Pre-calculated global reference diameter
        )

    def run(self):
        try:
            # Set calibration factor on analyzer if available
            if self.calibration_factor is not None:
                self.qca_analyzer.calibration_factor = self.calibration_factor
                logger.info(
                    f"QCAThread: Set calibration factor to {self.calibration_factor:.5f} mm/pixel"
                )
            else:
                logger.warning("QCAThread: No calibration factor provided")

            if self.segmentation_result:
                # Use AngioPy segmentation data directly
                self.progress.emit("Analyzing AngioPy segmentation data...")

                # Pass reference points, original image, and tracked points if available
                logger.info(f"[TRACKED DEBUG] QCAThread - tracked_points: {self.tracked_points}")
                logger.info(
                    f"[TRACKED DEBUG] QCAThread - use_tracked_centerline: {self.use_tracked_centerline}"
                )
                logger.info(
                    f"[GLOBAL REF DEBUG] QCAThread - global_reference_diameter: {self.global_reference_diameter}"
                )

                result = self.qca_analyzer.analyze_from_angiopy(
                    self.segmentation_result,
                    original_image=self.original_image,
                    proximal_point=self.proximal_point,
                    distal_point=self.distal_point,
                    tracked_points=self.tracked_points,
                    use_tracked_centerline=self.use_tracked_centerline,
                    global_reference_diameter=self.global_reference_diameter,
                )

                if result.get("success"):
                    self.progress.emit("Analysis complete")
                    self.finished.emit(result)
                else:
                    self.error.emit(result.get("error", "Analysis failed"))
            else:
                # Original flow using mask
                self.progress.emit("Analyzing vessel mask...")

                # Use unified mask analysis
                result = self.qca_analyzer.analyze_mask(self.mask)

                if result.get("success"):
                    result["source"] = "direct_mask"
                    self.progress.emit("Analysis complete")
                    self.finished.emit(result)
                else:
                    self.error.emit(result.get("error", "Analysis failed"))

        except Exception as e:
            self.error.emit(str(e))


class QCAWidget(QWidget):
    """Widget for QCA controls and results display with compact layout"""

    # Signals
    calibration_requested = pyqtSignal()  # Request calibration mode
    qca_started = pyqtSignal()
    qca_completed = pyqtSignal(dict)  # results
    overlay_changed = pyqtSignal(bool, dict)  # enabled, settings

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qca_analyzer = None
        self.current_results = None
        self.qca_thread = None
        self.graph_widget = None
        self.calibration_factor = None
        self.current_frame_index = None
        self.batch_results = {}  # Store all sequential results
        self.main_window = None  # Will be set by main window

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface with smart compact layout"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(3)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # Top section: Status and Controls (Horizontal layout)
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setSpacing(10)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Status info (left side)
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setSpacing(2)
        status_layout.setContentsMargins(0, 0, 0, 0)

        # Combine calibration and frame info in one line
        self.status_label = QLabel("Calibration: Not available")
        self.status_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #495057;")
        status_layout.addWidget(self.status_label)

        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet("color: #007bff; font-size: 11px;")
        status_layout.addWidget(self.frame_label)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-size: 10px; color: #6c757d;")
        status_layout.addWidget(self.progress_label)

        top_layout.addWidget(status_widget)
        top_layout.addStretch()

        # Control buttons (right side)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)

        self.analyze_btn = QPushButton("RWS Analyze")
        self.analyze_btn.setFixedHeight(32)
        self.analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 0 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:pressed { background-color: #1e7e34; }
            QPushButton:disabled { background-color: #e9ecef; color: #6c757d; }
        """
        )
        self.analyze_btn.clicked.connect(self.perform_rws_analysis)
        button_layout.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("Export")
        self.export_btn.setFixedHeight(32)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007bff;
                color: white;
                font-size: 12px;
                padding: 0 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0069d9; }
            QPushButton:pressed { background-color: #0056b3; }
            QPushButton:disabled { background-color: #e9ecef; color: #6c757d; }
        """
        )
        self.export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(self.export_btn)

        self.report_btn = QPushButton("Report")
        self.report_btn.setFixedHeight(32)
        self.report_btn.setEnabled(False)
        self.report_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #17a2b8;
                color: white;
                font-size: 12px;
                padding: 0 15px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #138496; }
            QPushButton:pressed { background-color: #117a8b; }
            QPushButton:disabled { background-color: #e9ecef; color: #6c757d; }
        """
        )
        self.report_btn.clicked.connect(self.generate_report)
        button_layout.addWidget(self.report_btn)

        top_layout.addLayout(button_layout)
        main_layout.addWidget(top_section)

        # Update calibration display
        self.calib_status_label = self.status_label  # For compatibility

        # Main vertical layout
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(5)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Top section: Analysis and Summary side by side
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Measurements section
        measure_widget = QWidget()
        measure_widget.setStyleSheet(
            """
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """
        )
        measure_layout = QVBoxLayout(measure_widget)
        measure_layout.setSpacing(5)
        measure_layout.setContentsMargins(10, 10, 10, 10)

        # Measurements header
        measure_header = QWidget()
        measure_header_layout = QHBoxLayout(measure_header)
        measure_header_layout.setContentsMargins(0, 0, 0, 0)

        measure_label = QLabel("Measurements")
        measure_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50;")
        measure_header_layout.addWidget(measure_label)
        measure_header_layout.addStretch()

        measure_layout.addWidget(measure_header)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        # Make columns responsive
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)  # Parameter column stretches
        header.setSectionResizeMode(
            1, header.ResizeMode.ResizeToContents
        )  # Value column fits content
        header.setMinimumSectionSize(80)  # Minimum width for value column
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet(
            """
            QTableWidget {
                font-size: 11px;
                background-color: white;
                gridline-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 3px;
            }
            QTableWidget::item {
                padding: 4px;
                color: #495057;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                color: #495057;
                font-size: 11px;
                font-weight: bold;
                padding: 4px;
                border: none;
                border-bottom: 2px solid #dee2e6;
            }
            QTableWidget::item:alternate {
                background-color: #f8f9fa;
            }
        """
        )

        # Compact measurement list
        measurements = [
            ("Reference Diameter", "- mm"),
            ("MLD", "- mm"),
            ("Stenosis %", "- %"),
            ("Area Stenosis %", "- %"),
            ("Lesion Length", "- mm"),
            ("MLA", "- mm²"),
            ("RVA", "- mm²"),
        ]

        self.results_table.setRowCount(len(measurements))
        for i, (param, value) in enumerate(measurements):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(i, 0, param_item)
            value_item = QTableWidgetItem(value)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.results_table.setItem(i, 1, value_item)

        # Set increased height for measurements table
        header_height = self.results_table.horizontalHeader().height()
        row_height = 25  # Increased row height
        table_height = header_height + (row_height * len(measurements)) + 10
        self.results_table.setFixedHeight(table_height)
        self.results_table.verticalHeader().setDefaultSectionSize(row_height)

        measure_layout.addWidget(self.results_table)

        # Add spacing after table
        measure_layout.addSpacing(15)

        # Visualization options
        vis_widget = QWidget()
        vis_widget.setStyleSheet("background-color: transparent; border: none;")
        vis_layout = QVBoxLayout(vis_widget)
        vis_layout.setSpacing(3)
        vis_layout.setContentsMargins(0, 0, 0, 0)

        vis_label = QLabel("Display Options")
        vis_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        vis_layout.addWidget(vis_label)

        # Checkboxes in horizontal layout
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(5)

        self.show_centerline_checkbox = QCheckBox("Centerline")
        self.show_centerline_checkbox.setChecked(True)
        self.show_centerline_checkbox.setStyleSheet("font-size: 10px; color: #495057;")
        self.show_centerline_checkbox.toggled.connect(self.update_visualization)
        checkbox_layout.addWidget(self.show_centerline_checkbox)

        self.show_stenosis_checkbox = QCheckBox("Stenosis")
        self.show_stenosis_checkbox.setChecked(True)
        self.show_stenosis_checkbox.setStyleSheet("font-size: 10px; color: #495057;")
        self.show_stenosis_checkbox.toggled.connect(self.update_visualization)
        checkbox_layout.addWidget(self.show_stenosis_checkbox)

        self.show_diameter_checkbox = QCheckBox("Diameters")
        self.show_diameter_checkbox.setChecked(True)
        self.show_diameter_checkbox.setStyleSheet("font-size: 10px; color: #495057;")
        self.show_diameter_checkbox.toggled.connect(self.update_visualization)
        checkbox_layout.addWidget(self.show_diameter_checkbox)

        vis_layout.addLayout(checkbox_layout)

        measure_layout.addWidget(vis_widget)

        # Overlay Controls (moved from main window)
        overlay_widget = QWidget()
        overlay_widget.setStyleSheet("background-color: transparent; border: none;")
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setSpacing(3)
        overlay_layout.setContentsMargins(0, 10, 0, 0)

        overlay_label = QLabel("Overlay Controls")
        overlay_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        overlay_layout.addWidget(overlay_label)

        # Show/Hide overlays
        self.show_points_cb = QCheckBox("Show Points")
        self.show_points_cb.setChecked(True)
        self.show_points_cb.setStyleSheet("font-size: 10px; color: #495057;")
        overlay_layout.addWidget(self.show_points_cb)

        self.show_segmentation_cb = QCheckBox("Show Segmentation")
        self.show_segmentation_cb.setChecked(True)
        self.show_segmentation_cb.setStyleSheet("font-size: 10px; color: #495057;")
        overlay_layout.addWidget(self.show_segmentation_cb)

        self.show_qca_cb = QCheckBox("Show QCA Results")
        self.show_qca_cb.setChecked(True)
        self.show_qca_cb.setStyleSheet("font-size: 10px; color: #495057;")
        overlay_layout.addWidget(self.show_qca_cb)

        # Clear buttons
        clear_btn_style = """
            QPushButton {
                font-size: 10px;
                padding: 3px 8px;
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """

        self.clear_points_btn = QPushButton("Clear Current Points")
        self.clear_points_btn.setStyleSheet(clear_btn_style)
        overlay_layout.addWidget(self.clear_points_btn)

        self.clear_all_frame_points_btn = QPushButton("Clear All Frame Points")
        self.clear_all_frame_points_btn.setStyleSheet(clear_btn_style)
        overlay_layout.addWidget(self.clear_all_frame_points_btn)

        self.clear_overlays_btn = QPushButton("Clear Current Overlays")
        self.clear_overlays_btn.setStyleSheet(clear_btn_style)
        overlay_layout.addWidget(self.clear_overlays_btn)

        self.clear_all_frames_overlays_btn = QPushButton("Clear All Frames Overlays")
        self.clear_all_frames_overlays_btn.setStyleSheet(clear_btn_style)
        overlay_layout.addWidget(self.clear_all_frames_overlays_btn)

        measure_layout.addWidget(overlay_widget)
        measure_layout.addStretch()

        # Right: Sequential Summary
        summary_widget = QWidget()
        summary_widget.setStyleSheet(
            """
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """
        )
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setSpacing(5)
        summary_layout.setContentsMargins(10, 10, 10, 10)

        # Summary header
        summary_header = QWidget()
        header_layout = QHBoxLayout(summary_header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        summary_label = QLabel("Sequential Analysis Summary")
        summary_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2c3e50;")
        header_layout.addWidget(summary_label)
        header_layout.addStretch()

        self.export_summary_btn = QPushButton("Export CSV")
        self.export_summary_btn.setFixedHeight(28)
        self.export_summary_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 11px;
                padding: 2px 12px;
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """
        )
        self.export_summary_btn.clicked.connect(self.export_summary)
        header_layout.addWidget(self.export_summary_btn)

        summary_layout.addWidget(summary_header)

        # Summary table with increased height
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(5)
        self.summary_table.setHorizontalHeaderLabels(
            ["Frame", "Beat", "Cardiac Phase", "Stenosis %", "MLD (mm)"]
        )
        # Set column widths - double cardiac phase, reduce others to 2/3
        summary_header = self.summary_table.horizontalHeader()
        summary_header.setSectionResizeMode(0, summary_header.ResizeMode.Fixed)  # Frame
        summary_header.setSectionResizeMode(1, summary_header.ResizeMode.Fixed)  # Beat
        summary_header.setSectionResizeMode(
            2, summary_header.ResizeMode.Stretch
        )  # Cardiac Phase stretches
        summary_header.setSectionResizeMode(3, summary_header.ResizeMode.Fixed)  # Stenosis %
        summary_header.setSectionResizeMode(4, summary_header.ResizeMode.Fixed)  # MLD

        # Set specific widths
        self.summary_table.setColumnWidth(0, 50)  # Frame - reduced to 2/3 (was ~75)
        self.summary_table.setColumnWidth(1, 40)  # Beat - reduced to 2/3 (was ~60)
        self.summary_table.setColumnWidth(3, 70)  # Stenosis % - reduced to 2/3 (was ~105)
        self.summary_table.setColumnWidth(4, 60)  # MLD - reduced to 2/3 (was ~90)
        # Cardiac Phase will stretch to fill remaining space (doubled)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.summary_table.setStyleSheet(
            """
            QTableWidget {
                font-size: 11px;
                background-color: white;
                gridline-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-radius: 3px;
            }
            QTableWidget::item {
                padding: 4px;
                color: #495057;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                color: #495057;
                font-size: 11px;
                font-weight: bold;
                padding: 4px;
                border: none;
                border-bottom: 2px solid #dee2e6;
            }
            QTableWidget::item:alternate {
                background-color: #f8f9fa;
            }
            QTableWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
        """
        )
        self.summary_table.itemSelectionChanged.connect(self.on_summary_row_selected)

        # Set minimum height for summary table (much larger)
        summary_header_height = self.summary_table.horizontalHeader().height()
        summary_row_height = 24
        summary_table_height = summary_header_height + (summary_row_height * 15)  # Show 15 rows
        self.summary_table.setMinimumHeight(summary_table_height)
        self.summary_table.verticalHeader().setDefaultSectionSize(summary_row_height)

        summary_layout.addWidget(self.summary_table)

        # Initially hide summary
        self.sequential_widget = summary_widget
        self.sequential_widget.setVisible(False)

        # Add to top splitter (swapped order)
        top_splitter.addWidget(summary_widget)
        top_splitter.addWidget(measure_widget)
        # Set stretch factors: summary table 4/3, measurements table 2/3
        # Using ratio 4:2 = 2:1 for the stretch factors
        top_splitter.setStretchFactor(0, 4)  # Summary table gets 4 parts
        top_splitter.setStretchFactor(1, 2)  # Measurements table gets 2 parts

        content_layout.addWidget(top_splitter)

        # Bottom: Graph full width
        graph_widget = QWidget()
        graph_widget.setStyleSheet(
            """
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """
        )
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(10, 10, 10, 10)

        graph_label = QLabel("Diameter Profile")
        graph_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #2c3e50; padding: 2px;"
        )
        graph_layout.addWidget(graph_label)

        self.graph_widget = QCAGraphWidget()
        self.graph_widget.setMinimumHeight(250)
        graph_layout.addWidget(self.graph_widget)

        content_layout.addWidget(graph_widget)

        # Set stretch factors for vertical layout
        content_layout.setStretchFactor(top_splitter, 2)
        content_layout.setStretchFactor(graph_widget, 1)

        main_layout.addWidget(content_widget)

        # Color combo for compatibility
        self.color_combo = QComboBox()
        self.color_combo.addItems(["None", "By Diameter", "By Stenosis"])
        self.color_combo.setVisible(False)  # Hidden but available for compatibility

        self.setLayout(main_layout)

        # Ensure widget expands to fill available space
        from PyQt6.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def display_frame_results(self, frame_index: int, qca_result: dict):
        """Display QCA results for a specific frame from sequential analysis"""
        self.current_frame_index = frame_index

        # Update frame indicator (UI shows 1-based frame numbers)
        self.frame_label.setText(f"Frame {frame_index + 1} Analysis")

        if qca_result and qca_result.get("success"):
            # Store current results
            self.current_results = qca_result

            # Store in batch results
            self.batch_results[frame_index] = qca_result

            # Update results table
            self.update_results_table(qca_result)

            # Update graph if profile data exists
            if "profile_data" in qca_result and self.graph_widget:
                self.graph_widget.set_data(qca_result["profile_data"], qca_result)
                self.graph_widget.update()

            # Enable export buttons
            self.export_btn.setEnabled(True)
            self.report_btn.setEnabled(True)

            # Update visualization
            self.update_visualization()
        else:
            # Clear results if analysis failed
            self.clear_results()
            if qca_result:
                error_msg = qca_result.get("error", "Analysis failed")
                self.progress_label.setText(f"Frame {frame_index}: {error_msg}")

    def clear_results(self):
        """Clear all QCA results from display"""
        # Reset table to default values
        for i in range(self.results_table.rowCount()):
            self.results_table.item(i, 1).setText("- mm" if i < 7 else "-")

        # Clear graph
        if self.graph_widget:
            self.graph_widget.clear_data()

        # Clear frame indicator
        self.frame_label.setText("")

        # Disable export buttons
        self.export_btn.setEnabled(False)
        self.report_btn.setEnabled(False)

        # Clear current results
        self.current_results = None
        self.current_frame_index = None

    def update_sequential_summary(self, all_results: dict):
        """Update the summary table with all sequential analysis results"""
        self.batch_results = all_results
        # Show the sequential summary section
        self.sequential_widget.setVisible(True)

        # Clear and populate table
        self.summary_table.setRowCount(len(all_results))

        # Cardiac phase mapping
        cardiac_phases = {
            "d2": "End-diastole",
            "s1": "Early-systole",
            "s2": "End-systole",
            "d1": "Mid-diastole",
        }

        # First pass: collect all MLD values to find min and max
        mld_values = []
        for frame_idx in all_results.keys():
            result = all_results[frame_idx]
            if result.get("success"):
                mld = result.get("mld", 0)
                if mld > 0:  # Only consider valid MLD values
                    mld_values.append((frame_idx, mld))

        # Find min and max MLD frames
        min_mld_frame = None
        max_mld_frame = None
        if mld_values:
            min_mld_frame = min(mld_values, key=lambda x: x[1])[0]
            max_mld_frame = max(mld_values, key=lambda x: x[1])[0]

        row = 0
        for frame_idx in sorted(all_results.keys()):
            result = all_results[frame_idx]
            if result.get("success"):
                # Frame number (UI shows 1-based)
                self.summary_table.setItem(row, 0, QTableWidgetItem(str(frame_idx + 1)))

                # Beat number
                beat_number = result.get("beat_number", "-")
                beat_text = str(beat_number) if beat_number != "-" else "-"
                self.summary_table.setItem(row, 1, QTableWidgetItem(beat_text))

                # Cardiac phase
                phase = result.get("cardiac_phase", "")
                if not phase:
                    # Try alternative keys
                    phase = result.get("frame_type", "")
                    if not phase:
                        phase = result.get("phase", "")

                # Debug logging
                logger.info(
                    f"Frame {frame_idx}: cardiac_phase='{result.get('cardiac_phase', '')}', "
                    f"frame_type='{result.get('frame_type', '')}', phase='{result.get('phase', '')}', "
                    f"final phase='{phase}'"
                )

                # Map phase codes to readable text
                phase_text = cardiac_phases.get(phase, phase) if phase else "-"
                self.summary_table.setItem(row, 2, QTableWidgetItem(phase_text))

                # Stenosis % - no background colors
                stenosis = result.get("percent_stenosis", 0)
                stenosis_item = QTableWidgetItem(f"{stenosis:.1f}")
                self.summary_table.setItem(row, 3, stenosis_item)

                # MLD - only highlight min and max
                mld = result.get("mld", 0)
                mld_item = QTableWidgetItem(f"{mld:.2f}")

                # Only apply background to min/max MLD values
                if frame_idx == min_mld_frame:
                    # Minimum MLD (most stenotic) - light red background
                    mld_item.setBackground(QColor(255, 230, 230))
                    mld_item.setForeground(QColor(180, 40, 40))
                elif frame_idx == max_mld_frame:
                    # Maximum MLD (least stenotic) - light green background
                    mld_item.setBackground(QColor(230, 255, 230))
                    mld_item.setForeground(QColor(40, 120, 40))

                self.summary_table.setItem(row, 4, mld_item)

                row += 1

        self.summary_table.setRowCount(row)  # Adjust for failed analyses

    def on_summary_row_selected(self):
        """Handle selection of a row in the summary table"""
        selected_items = self.summary_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            frame_str = self.summary_table.item(row, 0).text()
            frame_idx = int(frame_str) - 1  # Convert from 1-based UI to 0-based index

            # Navigate to the selected frame through main window
            main_window = self.window()
            if hasattr(main_window, "viewer_widget") and hasattr(
                main_window.viewer_widget, "navigate_to_frame"
            ):
                main_window.viewer_widget.navigate_to_frame(frame_idx)
            elif hasattr(main_window, "frame_slider"):
                main_window.frame_slider.setValue(frame_idx)

            # Display the results for that frame
            if frame_idx in self.batch_results:
                self.display_frame_results(frame_idx, self.batch_results[frame_idx])

    def export_summary(self):
        """Export the summary table as CSV"""
        if not self.batch_results:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export QCA Summary", "qca_summary.csv", "CSV Files (*.csv)"
        )

        if file_path:
            import csv

            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Write headers
                headers = []
                for col in range(self.summary_table.columnCount()):
                    headers.append(self.summary_table.horizontalHeaderItem(col).text())
                writer.writerow(headers)

                # Write data
                for row in range(self.summary_table.rowCount()):
                    row_data = []
                    for col in range(self.summary_table.columnCount()):
                        item = self.summary_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

            QMessageBox.information(self, "Export Complete", f"Summary exported to {file_path}")

    def set_qca_analyzer(self, analyzer):
        """Set the QCA analyzer"""
        self.qca_analyzer = analyzer

    def set_main_window(self, main_window):
        """Set reference to main window for overlay controls"""
        self.main_window = main_window

        # Connect overlay controls to main window functions
        if self.main_window:
            self.show_points_cb.toggled.connect(self.main_window.toggle_points_visibility)
            self.show_segmentation_cb.toggled.connect(
                self.main_window.toggle_segmentation_visibility
            )
            self.show_qca_cb.toggled.connect(self.main_window.toggle_qca_visibility)

            # Connect clear buttons
            self.clear_points_btn.clicked.connect(self.main_window.clear_current_points)
            self.clear_all_frame_points_btn.clicked.connect(self.main_window.clear_all_frame_points)
            self.clear_overlays_btn.clicked.connect(self.main_window.clear_all_overlays)
            self.clear_all_frames_overlays_btn.clicked.connect(
                self.main_window.clear_all_frames_overlays
            )

    def rws_analyze(self):
        """Perform Enhanced RWS (Radial Wall Strain) analysis - redirects to enhanced method"""
        logger.info("Legacy RWS method called - redirecting to Enhanced RWS Analysis")

        try:
            # Check calibration first
            if not self.calibration_factor:
                QMessageBox.warning(self, "No Calibration", "Please perform calibration first.")
                return

            # Redirect to enhanced RWS analysis
            self.perform_rws_analysis()

        except Exception as e:
            logger.error(f"RWS analysis error: {e}")
            QMessageBox.critical(self, "RWS Analysis Error", str(e))
            self.status_label.setText("RWS analysis error")

    def _show_rws_results_dialog(self, results):
        """Show RWS analysis results in a comprehensive dialog"""
        from .rws_results_dialog import RWSResultsDialog

        # Get sequential QCA results for the dialog
        qca_results = self.sequential_results if hasattr(self, "sequential_results") else {}

        # Create and show the dialog
        dialog = RWSResultsDialog(results, qca_results, self)

        # Connect export signals
        dialog.pdf_export_requested.connect(lambda: self._export_rws_pdf())
        dialog.excel_export_requested.connect(lambda: self._export_rws_excel())
        dialog.database_save_requested.connect(lambda: self._save_rws_to_database())

        dialog.exec()

    def _export_rws_pdf(self):
        """Export RWS results to PDF"""
        if not hasattr(self, "rws_analyzer") or not self.rws_analyzer:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save RWS Report", "rws_report.pdf", "PDF Files (*.pdf)"
        )

        if file_path:
            if self.rws_analyzer.generate_pdf_report(file_path):
                QMessageBox.information(self, "Success", f"RWS report saved to {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to generate PDF report")

    def _export_rws_excel(self):
        """Export RWS results to Excel"""
        if not hasattr(self, "rws_analyzer") or not self.rws_analyzer:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export RWS Data", "rws_data.xlsx", "Excel Files (*.xlsx)"
        )

        if file_path:
            if self.rws_analyzer.export_to_excel(file_path):
                QMessageBox.information(self, "Success", f"RWS data exported to {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to export to Excel")

    def _save_rws_to_database(self):
        """Save RWS results to database"""
        if not hasattr(self, "rws_analyzer") or not self.rws_analyzer:
            return

        try:
            # Import database service
            from ..services.database_service import DatabaseService
            from ..database.models import CoronaryVessel

            # Initialize database service
            db_service = DatabaseService()

            # Set patient information (get from main window if available)
            patient_name = getattr(self.parent(), "current_patient_name", "Unknown Patient")
            medical_record_number = getattr(self.parent(), "current_mrn", None)

            try:
                patient_id = db_service.set_current_patient(
                    patient_name=patient_name, medical_record_number=medical_record_number
                )

                # Set study information
                study_id = db_service.set_current_study(
                    study_description="Coronary Angiography - RWS Analysis"
                )

            except Exception as e:
                # If patient/study setup fails, use defaults
                logger.warning(f"Failed to setup patient/study: {e}")
                db_service.set_current_patient("Unknown Patient")
                db_service.set_current_study()

            # Get RWS results data
            db_record = self.rws_analyzer.get_database_record()
            if not db_record:
                QMessageBox.warning(self, "No Data", "No RWS data to save")
                return

            # Parse vessel from results or use default
            vessel_name = db_record.get("vessel", "UNKNOWN").upper()
            try:
                vessel = CoronaryVessel(vessel_name)
            except ValueError:
                vessel = CoronaryVessel.UNKNOWN

            # Extract RWS data
            rws_percentage = db_record.get("rws_percentage", 0.0)
            mld_min_mm = db_record.get("mld_min_mm", 0.0)
            mld_max_mm = db_record.get("mld_max_mm", 0.0)
            min_frame = db_record.get("min_frame_number", 0)
            max_frame = db_record.get("max_frame_number", 0)
            frame_numbers = db_record.get("frame_numbers", [min_frame, max_frame])

            # Save calibration data if available
            if hasattr(self, "calibration_factor") and self.calibration_factor:
                db_service.save_calibration_data(
                    calibration_factor=self.calibration_factor,
                    method="manual",
                    confidence_score=1.0,
                )

            # Save RWS analysis
            analysis_id = db_service.save_rws_analysis(
                vessel=vessel,
                rws_percentage=rws_percentage,
                mld_min_mm=mld_min_mm,
                mld_max_mm=mld_max_mm,
                min_frame_number=min_frame,
                max_frame_number=max_frame,
                frame_numbers=frame_numbers,
                cardiac_phase_min=db_record.get("cardiac_phase_min"),
                cardiac_phase_max=db_record.get("cardiac_phase_max"),
                raw_diameter_data=db_record.get("raw_diameter_data"),
                clinical_interpretation=db_record.get("clinical_interpretation"),
                projection_angle=db_record.get("projection_angle"),
                operator="User",
            )

            QMessageBox.information(
                self,
                "Success",
                f"RWS results saved to database\n"
                f"Analysis ID: {analysis_id[:8]}...\n"
                f"Patient: {patient_name}\n"
                f"Vessel: {vessel.value}\n"
                f"RWS: {rws_percentage:.1f}%",
            )

            logger.info(f"RWS data saved to database: {analysis_id}")

        except Exception as e:
            logger.error(f"Database save error: {e}")
            QMessageBox.critical(
                self, "Database Error", f"Failed to save RWS data to database:\n{str(e)}"
            )

    def analyze_vessel(
        self, segmentation_result=None, tracked_points=None, use_tracked_centerline=False
    ):
        """Start vessel analysis"""
        # Check calibration first
        if not self.calibration_factor:
            QMessageBox.warning(
                self,
                "No Calibration",
                "Please perform calibration first using the Calibration tool.",
            )
            return

        logger.info(f"=== STARTING VESSEL ANALYSIS ===")
        logger.info(f"Current calibration factor: {self.calibration_factor:.5f} mm/pixel")
        logger.info(
            f"QCA analyzer calibration: {self.qca_analyzer.calibration_factor:.5f} mm/pixel"
        )

        if not self.qca_analyzer:
            from ..analysis.qca_analysis import QCAAnalysis

            self.qca_analyzer = QCAAnalysis()
            logger.info("QCA Analyzer created")

        # Set calibration factor
        if self.calibration_factor:
            self.qca_analyzer.calibration_factor = self.calibration_factor
            logger.info(
                f"analyze_vessel: Set calibration factor to {self.calibration_factor:.5f} mm/pixel"
            )
        else:
            logger.warning("analyze_vessel: No calibration factor available")

        if segmentation_result:
            # Direct analysis from AngioPy data
            mask = segmentation_result.get("mask")
            if mask is None:
                QMessageBox.warning(self, "No Mask", "No segmentation mask available")
                return
        else:
            # Get current frame as mask (for manual QCA without segmentation)
            try:
                main_window = self.window()
                if hasattr(main_window, "viewer_widget"):
                    current_frame = main_window.viewer_widget.get_current_frame()
                    if current_frame is not None:
                        # Convert to grayscale if needed
                        if len(current_frame.shape) == 3:
                            mask = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                        else:
                            mask = current_frame.copy()

                        # Ask user to confirm they want to use raw image
                        reply = QMessageBox.question(
                            self,
                            "No Segmentation",
                            "No vessel segmentation found.\n"
                            "Do you want to perform QCA on the current frame directly?\n"
                            "(This may produce less accurate results)",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        )

                        if reply != QMessageBox.StandardButton.Yes:
                            return

                        logger.info("Using raw frame for QCA analysis")
                    else:
                        QMessageBox.warning(self, "No Image", "No image available for analysis")
                        return
                else:
                    QMessageBox.warning(self, "Error", "Cannot access viewer widget")
                    return
            except Exception as e:
                logger.error(f"Failed to get current frame: {e}")
                QMessageBox.warning(self, "Error", f"Failed to get current frame: {str(e)}")
                return

        # Update progress
        self.progress_label.setText("Analyzing...")
        self.analyze_btn.setEnabled(False)

        # No longer using reference points
        proximal_point = None
        distal_point = None

        # Create a simple QCA thread for analysis
        logger.info(f"Starting QCA analysis with calibration_factor={self.calibration_factor}")

        # Prepare segmentation result for QCA
        if segmentation_result:
            # Use existing segmentation result
            seg_data = segmentation_result.copy()
            seg_data["proximal_point"] = proximal_point
            seg_data["distal_point"] = distal_point
        else:
            # Create minimal segmentation result for direct mask analysis
            seg_data = {
                "mask": mask,
                "vessel_mask": mask,
                "success": True,
                "proximal_point": proximal_point,
                "distal_point": distal_point,
            }

        # Calculate global reference diameter from existing sequential results if available
        logger.info("=== GLOBAL REFERENCE CALCULATION START ===")
        global_reference_diameter = self._calculate_global_reference_from_sequential_results()
        logger.info(f"=== GLOBAL REFERENCE RESULT: {global_reference_diameter} ===")

        # Create QCA thread with segmentation data
        self.qca_thread = QCAThread(
            qca_analyzer=self.qca_analyzer,
            segmentation_result=seg_data,
            proximal_point=proximal_point,
            distal_point=distal_point,
            tracked_points=tracked_points,
            use_tracked_centerline=use_tracked_centerline,
            global_reference_diameter=global_reference_diameter,
        )

        # Connect signals
        self.qca_thread.progress.connect(lambda msg, pct: self.progress_label.setText(msg))
        self.qca_thread.finished.connect(self.analysis_finished)
        self.qca_thread.error.connect(self.analysis_error)

        self.qca_started.emit()
        self.qca_thread.start()

    def _calculate_global_reference_from_sequential_results(self) -> Optional[float]:
        """Calculate global reference diameter from existing sequential QCA results"""
        try:
            logger.info(
                "[GLOBAL REF DEBUG] Starting global reference calculation from sequential results..."
            )
            main_window = self.window()
            logger.info(f"[GLOBAL REF DEBUG] main_window: {main_window}")

            if not main_window:
                logger.warning("[GLOBAL REF DEBUG] No main window found!")
                return None

            logger.info(
                f"[GLOBAL REF DEBUG] main_window has sequential_qca_results: {hasattr(main_window, 'sequential_qca_results')}"
            )

            if not hasattr(main_window, "sequential_qca_results"):
                logger.info("[GLOBAL REF DEBUG] No sequential_qca_results attribute on main window")
                return None

            sequential_results = main_window.sequential_qca_results
            logger.info(f"[GLOBAL REF DEBUG] sequential_results type: {type(sequential_results)}")
            logger.info(
                f"[GLOBAL REF DEBUG] sequential_results length: {len(sequential_results) if sequential_results else 0}"
            )

            if not sequential_results:
                logger.info("[GLOBAL REF DEBUG] Sequential QCA results are empty")
                return None

            # Collect all diameter measurements from sequential results
            all_diameters_mm = []
            for frame_idx, result in sequential_results.items():
                if result.get("success") and "diameter_profile_mm" in result:
                    diameter_profile = result["diameter_profile_mm"]
                    if diameter_profile is not None and len(diameter_profile) > 0:
                        # Add valid diameters (> 0) to global collection
                        valid_diameters = [d for d in diameter_profile if d > 0]
                        all_diameters_mm.extend(valid_diameters)

            if len(all_diameters_mm) == 0:
                logger.warning("No valid diameter measurements found in sequential results")
                return None

            # Calculate global reference diameter (75th percentile)
            global_reference_diameter = np.percentile(all_diameters_mm, 75)

            logger.info(f"GLOBAL REFERENCE from Sequential Results:")
            logger.info(f"Total diameter measurements: {len(all_diameters_mm)}")
            logger.info(
                f"Diameter range: {np.min(all_diameters_mm):.2f} - {np.max(all_diameters_mm):.2f} mm"
            )
            logger.info(f"75th percentile (Global Reference): {global_reference_diameter:.2f} mm")

            return global_reference_diameter

        except Exception as e:
            logger.error(f"Failed to calculate global reference from sequential results: {e}")
            return None

    def set_calibration(self, calibration_factor: float, details: dict = None):
        """Set calibration from main window"""
        logger.info(f"=== QCA WIDGET SET_CALIBRATION ===")
        logger.info(f"Received calibration factor: {calibration_factor:.5f} mm/pixel")

        if calibration_factor:
            self.calibration_factor = calibration_factor
            catheter_size = details.get("catheter_size", "Unknown") if details else "Unknown"
            self.calib_status_label.setText(
                f"Calibration: {calibration_factor:.5f} mm/pixel ({catheter_size})"
            )
            self.calib_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.calibration_factor = None
            self.calib_status_label.setText("Calibration: Not available")
            self.calib_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")

    def start_analysis(self, segmentation_result=None):
        """Start QCA analysis

        Args:
            segmentation_result: Optional AngioPy segmentation result dict
        """
        # Check calibration
        if not hasattr(self, "calibration_factor") or self.calibration_factor is None:
            QMessageBox.warning(
                self,
                "No Calibration",
                "Please perform calibration first using the Calibration tool.",
            )
            return

        if not self.qca_analyzer:
            from ..analysis.qca_analysis import QCAAnalysis

            self.qca_analyzer = QCAAnalysis()
            logger.info("QCA Analyzer created")

        # Set calibration factor
        if self.calibration_factor:
            self.qca_analyzer.calibration_factor = self.calibration_factor
            logger.info(
                f"analyze_vessel: Set calibration factor to {self.calibration_factor:.5f} mm/pixel"
            )
        else:
            logger.warning("analyze_vessel: No calibration factor available")

        if segmentation_result:
            # Direct analysis from AngioPy data
            mask = segmentation_result.get("mask")
            if mask is None:
                QMessageBox.warning(self, "No Mask", "No segmentation mask available")
                return
        else:
            # Get current frame as mask (for manual QCA without segmentation)
            try:
                main_window = self.window()
                if hasattr(main_window, "viewer_widget"):
                    current_frame = main_window.viewer_widget.get_current_frame()
                    if current_frame is not None:
                        # Convert to grayscale if needed
                        if len(current_frame.shape) == 3:
                            mask = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                        else:
                            mask = current_frame.copy()

                        # Ask user to confirm they want to use raw image
                        reply = QMessageBox.question(
                            self,
                            "No Segmentation",
                            "No vessel segmentation found.\n"
                            "Do you want to perform QCA on the current frame directly?\n"
                            "(This may produce less accurate results)",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        )

                        if reply != QMessageBox.StandardButton.Yes:
                            return

                        logger.info("Using raw frame for QCA analysis")
                    else:
                        QMessageBox.warning(self, "No Image", "No image available for analysis")
                        return
                else:
                    QMessageBox.warning(self, "Error", "Cannot access viewer widget")
                    return
            except Exception as e:
                logger.error(f"Failed to get current frame: {e}")
                QMessageBox.warning(self, "Error", f"Failed to get current frame: {str(e)}")
                return

        # Update progress
        self.progress_label.setText("Analyzing...")
        self.analyze_btn.setEnabled(False)

        # No longer using reference points
        proximal_point = None
        distal_point = None

        # Create a simple QCA thread for analysis
        logger.info(f"Starting QCA analysis with calibration_factor={self.calibration_factor}")

        # Prepare segmentation result for QCA
        if segmentation_result:
            # Use existing segmentation result
            seg_data = segmentation_result.copy()
            seg_data["proximal_point"] = proximal_point
            seg_data["distal_point"] = distal_point
        else:
            # Create minimal segmentation result for direct mask analysis
            seg_data = {
                "mask": mask,
                "vessel_mask": mask,
                "success": True,
                "proximal_point": proximal_point,
                "distal_point": distal_point,
            }

        # Calculate global reference diameter from existing sequential results if available
        logger.info("=== GLOBAL REFERENCE CALCULATION START ===")
        global_reference_diameter = self._calculate_global_reference_from_sequential_results()
        logger.info(f"=== GLOBAL REFERENCE RESULT: {global_reference_diameter} ===")

        # Create QCA thread with segmentation data
        self.qca_thread = QCAThread(
            qca_analyzer=self.qca_analyzer,
            segmentation_result=seg_data,
            proximal_point=proximal_point,
            distal_point=distal_point,
            tracked_points=tracked_points,
            use_tracked_centerline=use_tracked_centerline,
            global_reference_diameter=global_reference_diameter,
        )

        # Connect signals
        self.qca_thread.progress.connect(lambda msg, pct: self.progress_label.setText(msg))
        self.qca_thread.finished.connect(self.analysis_finished)
        self.qca_thread.error.connect(self.analysis_error)

        self.qca_started.emit()
        self.qca_thread.start()

    # Removed unified processor adapters - now using QCAThread directly

    def update_progress(self, message):
        """Update progress message"""
        self.progress_label.setText(message)

    @pyqtSlot(dict)
    def analysis_finished(self, results):
        """Handle analysis completion"""
        self.current_results = results
        self.progress_label.setText("")
        self.analyze_btn.setEnabled(True)

        # Debug logging for results
        logger.info("=" * 80)
        logger.info("QCA ANALYSIS RESULTS RECEIVED IN UI")
        logger.info("=" * 80)
        logger.info(f"Results keys: {list(results.keys())}")
        logger.info(f"Success: {results.get('success', 'NOT FOUND')}")
        logger.info(f"MLD: {results.get('mld', 'NOT FOUND')} mm")
        logger.info(f"Reference Diameter: {results.get('reference_diameter', 'NOT FOUND')} mm")
        logger.info(f"Percent Stenosis: {results.get('percent_stenosis', 'NOT FOUND')} %")
        if results.get("diameters") is not None:
            diameters = results.get("diameters")
            logger.info(f"Diameters array length: {len(diameters)}")
            valid_diameters = [d for d in diameters if d > 0]
            logger.info(f"Valid diameter measurements: {len(valid_diameters)}")
            if valid_diameters:
                logger.info(
                    f"Diameter range: {min(valid_diameters):.2f} - {max(valid_diameters):.2f} pixels"
                )
        logger.info("=" * 80)

        if results.get("success", False):
            # Update results table
            self.update_results_table(results)

            # Enable export
            self.export_btn.setEnabled(True)
            self.report_btn.setEnabled(True)

            # Emit completion signal
            self.qca_completed.emit(results)

            # Update visualization
            self.update_visualization()
        else:
            QMessageBox.warning(
                self,
                "Analysis Failed",
                "Failed to analyze vessel. Please check the segmentation quality.",
            )

    @pyqtSlot(str)
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress_label.setText("")
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"Error: {error_msg}")

    def update_results_table(self, results):
        """Update the results table with measurements"""
        # Clear any previous frame indicator
        if hasattr(self, "frame_label"):
            self.frame_label.setText("")

        # Update compact table values
        row_mapping = {
            0: ("reference_diameter", "{:.2f} mm"),
            1: ("mld", "{:.2f} mm"),
            2: ("percent_stenosis", "{:.1f} %"),
            3: ("percent_area_stenosis", "{:.1f} %"),
            4: ("lesion_length", "{:.1f} mm"),
            5: ("mla", "{:.2f} mm²"),
            6: ("reference_area", "{:.2f} mm²"),
        }

        for row, (key, format_str) in row_mapping.items():
            value = results.get(key)
            if value is not None and key != "mld_location":
                text = format_str.format(value)
            elif key == "mld_location" and value is not None:
                text = f"({int(value[1])}, {int(value[0])})"
            else:
                text = "-"

            item = self.results_table.item(row, 1)
            if item:
                item.setText(text)

                # Color code stenosis severity
                if key in ["percent_stenosis", "percent_area_stenosis"] and value is not None:
                    if value < 50:
                        item.setBackground(QColor(220, 240, 220))  # Light green
                        item.setForeground(QColor(40, 120, 40))  # Dark green text
                    elif value < 70:
                        item.setBackground(QColor(255, 243, 205))  # Light yellow
                        item.setForeground(QColor(133, 100, 4))  # Dark yellow text
                    else:
                        item.setBackground(QColor(255, 230, 230))  # Light red
                        item.setForeground(QColor(180, 40, 40))  # Dark red text

        # Update graph
        if self.graph_widget and results.get("profile_data"):
            self.graph_widget.set_data(results["profile_data"], results)

    def update_visualization(self):
        """Update visualization settings"""
        if not self.current_results:
            return

        settings = {
            "show_centerline": self.show_centerline_checkbox.isChecked(),
            "show_diameter": self.show_diameter_checkbox.isChecked(),
            "show_stenosis": self.show_stenosis_checkbox.isChecked(),
            "color_mode": self.color_combo.currentText(),
            "results": self.current_results,
        }

        enabled = any(
            [settings["show_centerline"], settings["show_diameter"], settings["show_stenosis"]]
        )

        self.overlay_changed.emit(enabled, settings)

    def clear_results(self):
        """Clear all QCA results"""
        self.current_results = None

        # Reset results table
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 1)
            if item:
                item.setText("-")
                item.setBackground(QColor())

        # Clear graph
        if self.graph_widget:
            self.graph_widget.clear_data()

        # Update visualization
        self.overlay_changed.emit(False, {})

        # Update UI
        self.export_btn.setEnabled(False)
        self.report_btn.setEnabled(False)

    def export_results(self):
        """Export QCA results"""
        if not self.current_results:
            return

        # Emit signal to parent to handle export
        if hasattr(self.parent(), "export_qca_results"):
            self.parent().export_qca_results(self.current_results)

    def generate_report(self):
        """Generate comprehensive QCA report"""
        if not self.current_results or not self.current_results.get("success"):
            QMessageBox.warning(self, "No Results", "Please perform QCA analysis first")
            return

        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save QCA Report", "QCA_Report.pdf", "PDF Files (*.pdf);;All Files (*)"
        )

        if not file_path:
            return

        # Gather patient information
        patient_info = self._get_patient_info()

        # Get current angiogram image
        angiogram_image = self._get_current_image()

        # Generate report
        report_generator = QCAReportGenerator()
        success = report_generator.generate_report(
            self.current_results, patient_info, file_path, angiogram_image, with_overlay=True
        )

        if success:
            QMessageBox.information(self, "Success", f"QCA report saved to:\n{file_path}")
        else:
            QMessageBox.critical(self, "Error", "Failed to generate report")

    def _get_patient_info(self) -> Dict:
        """Get patient information from DICOM or defaults"""
        # Try to get from main window's DICOM parser
        patient_info = {
            "patient_id": "Unknown",
            "patient_name": "Unknown",
            "study_date": "Unknown",
            "birth_date": "Unknown",
            "accession_number": "Unknown",
            "referring_physician": "Unknown",
        }

        try:
            main_window = self.window()
            if hasattr(main_window, "dicom_parser") and main_window.dicom_parser.has_data():
                dicom_parser = main_window.dicom_parser
                patient_info.update(
                    {
                        "patient_id": dicom_parser.metadata.get("patient_id", "Unknown"),
                        "patient_name": str(dicom_parser.metadata.get("patient_name", "Unknown")),
                        "study_date": dicom_parser.metadata.get("study_date", "Unknown"),
                        "birth_date": dicom_parser.metadata.get("patient_birth_date", "Unknown"),
                        "accession_number": dicom_parser.metadata.get(
                            "accession_number", "Unknown"
                        ),
                        "referring_physician": dicom_parser.metadata.get(
                            "referring_physician_name", "Unknown"
                        ),
                    }
                )
        except Exception as e:
            logger.error(f"Failed to get patient info: {e}")

        return patient_info

    def get_overlay_settings(self) -> Dict:
        """Get current overlay settings"""
        return {
            "show_centerline": self.show_centerline_checkbox.isChecked(),
            "show_diameter": self.show_diameter_checkbox.isChecked(),
            "show_stenosis": self.show_stenosis_checkbox.isChecked(),
            "color_mode": self.color_combo.currentText(),
        }

    @property
    def last_results(self):
        """Compatibility property for report generation"""
        return self.current_results

    def _get_current_image(self) -> Optional[np.ndarray]:
        """Get current angiogram image"""
        try:
            main_window = self.window()
            if hasattr(main_window, "viewer_widget"):
                # Get current frame from viewer
                pixmap = main_window.viewer_widget.get_current_pixmap()
                if pixmap:
                    # Convert QPixmap to numpy array
                    image = pixmap.toImage()
                    width = image.width()
                    height = image.height()

                    # Convert to numpy array
                    ptr = image.bits()
                    ptr.setsize(height * width * 4)
                    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

                    # Convert BGRA to RGB
                    return arr[:, :, [2, 1, 0]]

            # Alternative: get from DICOM parser
            if hasattr(main_window, "dicom_parser") and hasattr(main_window, "frame_slider"):
                frame_index = main_window.frame_slider.value()
                return main_window.dicom_parser.get_frame(frame_index)

        except Exception as e:
            logger.error(f"Failed to get current image: {e}")

    def perform_rws_analysis(self):
        """Perform Enhanced RWS analysis on sequential QCA results"""
        logger.info("=== STARTING ENHANCED RWS ANALYSIS ===")
        logger.info(f"Current calibration factor: {self.calibration_factor:.5f} mm/pixel")

        try:
            # Get QCA results from main window
            main_window = self.parent()
            while main_window and not hasattr(main_window, "sequential_qca_results"):
                main_window = main_window.parent()

            if (
                not main_window
                or not hasattr(main_window, "sequential_qca_results")
                or not main_window.sequential_qca_results
            ):
                QMessageBox.warning(
                    self,
                    "No Sequential Data",
                    "Please perform sequential QCA analysis first.\n"
                    "RWS analysis requires QCA data from multiple cardiac phases.",
                )
                return

            # Directly use Enhanced RWS analysis (skip the choice dialog)
            logger.info("Using Enhanced RWS analysis with motion artifact detection")
            self._perform_enhanced_rws_analysis(main_window)

        except Exception as e:
            logger.error(f"Error in Enhanced RWS analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Enhanced RWS analysis failed: {str(e)}")

    def _perform_standard_rws_analysis(self, main_window):
        """Perform standard MLD-based RWS analysis"""
        # Import RWS helper
        from .qca_widget_rws import QCAWidgetRWS

        # Validate results
        is_valid, error_msg = QCAWidgetRWS.validate_qca_results_for_rws(
            main_window.sequential_qca_results
        )
        if not is_valid:
            QMessageBox.warning(self, "Invalid Data", error_msg)
            return

        # Calculate RWS
        rws_results = QCAWidgetRWS.calculate_rws_from_qca_results(
            main_window.sequential_qca_results
        )

        if not rws_results:
            QMessageBox.critical(self, "Error", "Failed to calculate RWS")
            return

        # Add calibration factor to results
        rws_results["calibration_factor"] = self.calibration_factor

        # Show results dialog
        from ..ui.rws_results_dialog import RWSResultsDialog

        dialog = RWSResultsDialog(rws_results, main_window.sequential_qca_results, self)

        # Connect export signals if main window has handlers
        if hasattr(main_window, "export_rws_to_pdf"):
            dialog.pdf_export_requested.connect(lambda: main_window.export_rws_to_pdf(rws_results))
        if hasattr(main_window, "export_rws_to_excel"):
            dialog.excel_export_requested.connect(
                lambda: main_window.export_rws_to_excel(rws_results)
            )

        dialog.exec()

    def _perform_enhanced_rws_analysis(self, main_window):
        """Perform enhanced RWS analysis with motion artifact detection"""
        try:
            # Get cardiac phases from viewer widget
            cardiac_phases = None
            if hasattr(main_window, "viewer_widget") and hasattr(
                main_window.viewer_widget, "cardiac_phases"
            ):
                cardiac_phases = main_window.viewer_widget.cardiac_phases

            # Enhanced RWS can work without cardiac phases (using motion detection only)
            if not cardiac_phases:
                logger.info(
                    "Cardiac phase data not available - Enhanced RWS will use motion detection only"
                )
                # Create empty cardiac phases dict and continue automatically
                cardiac_phases = {"phases": [], "frame_phases": []}

            # Get frame timestamps
            frame_timestamps = None
            if hasattr(main_window, "viewer_widget") and hasattr(
                main_window.viewer_widget, "frame_timestamps"
            ):
                frame_timestamps = main_window.viewer_widget.frame_timestamps
            elif hasattr(main_window, "frame_timestamps"):
                frame_timestamps = main_window.frame_timestamps

            if frame_timestamps is None:
                QMessageBox.warning(
                    self, "Missing Data", "Frame timestamps are required for enhanced RWS analysis."
                )
                return

            # Get frames
            frames = []
            # Convert keys to integers and sort
            frame_indices = []
            for key in main_window.sequential_qca_results.keys():
                try:
                    idx = int(key) if isinstance(key, str) else key
                    frame_indices.append(idx)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid frame key: {key}")

            frame_indices = sorted(frame_indices)
            logger.info(f"Processing frames: {frame_indices}")

            for idx in frame_indices:
                if hasattr(main_window, "dicom_parser"):
                    frame = main_window.dicom_parser.get_frame(idx)
                    if frame is not None:
                        # Convert to grayscale if needed
                        if len(frame.shape) == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        frames.append(frame)
                    else:
                        logger.warning(f"Could not get frame {idx}")

            if len(frames) < 2:
                QMessageBox.warning(
                    self, "Insufficient Data", "Need at least 2 frames for enhanced RWS analysis."
                )
                return

            # Add frame_phases to cardiac_phases if available
            if hasattr(main_window, "dicom_parser") and main_window.dicom_parser.has_data():
                total_frames = main_window.dicom_parser.num_frames
                frame_rate = main_window.dicom_parser.get_frame_rate() or 30.0  # Default to 30 fps

                # Use CardiacPhaseDetector to map phases to frames
                from ..core.cardiac_phase_detector import CardiacPhaseDetector

                # Get ECG sampling rate if available
                ecg_sampling_rate = 1000  # Default
                if hasattr(main_window, "ekg_parser") and hasattr(
                    main_window.ekg_parser, "sampling_rate"
                ):
                    ecg_sampling_rate = main_window.ekg_parser.sampling_rate
                detector = CardiacPhaseDetector(ecg_sampling_rate)

                if "phases" in cardiac_phases and cardiac_phases["phases"]:
                    frame_phases = detector.map_phases_to_frames(
                        cardiac_phases["phases"], total_frames, frame_rate, start_time=0.0
                    )
                    cardiac_phases["frame_phases"] = frame_phases
                    logger.info(f"Mapped {len(frame_phases)} phase transitions to frames")

            # Create enhanced RWS widget
            from .rws_enhanced_widget import EnhancedRWSWidget

            # Create dialog to host the widget
            dialog = QDialog(self)
            dialog.setWindowTitle("Enhanced RWS Analysis")
            dialog.resize(1200, 800)

            layout = QVBoxLayout(dialog)
            rws_widget = EnhancedRWSWidget(dialog)
            layout.addWidget(rws_widget)

            # Connect analysis request
            def start_analysis():
                logger.info("Starting enhanced RWS analysis with consistent global reference...")

                # Step 1: Use existing QCA results directly (RWS is independent of segmentation)
                qca_results_int = {}

                # Convert sequential QCA results to integer keys and validate
                for key, result in main_window.sequential_qca_results.items():
                    try:
                        frame_idx = int(key) if isinstance(key, str) else key
                        if frame_idx in frame_indices:
                            qca_results_int[frame_idx] = result
                            logger.info(
                                f"Frame {frame_idx}: Using existing QCA - {result['percent_stenosis']:.1f}% stenosis, "
                                f"MLD: {result['mld']:.2f}mm"
                            )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid QCA result key: {key} - {e}")

                if len(qca_results_int) == 0:
                    QMessageBox.critical(
                        self,
                        "No QCA Data",
                        "No QCA results found for enhanced RWS analysis.\n"
                        "Please perform QCA analysis first.",
                    )
                    return

                logger.info(
                    f"Using {len(qca_results_int)} frames with existing QCA results for RWS analysis"
                )

                # Step 2: Start RWS analysis with existing QCA results
                rws_widget.start_analysis(
                    frames,
                    qca_results_int,
                    cardiac_phases,
                    frame_timestamps,
                    self.calibration_factor,
                )

            rws_widget.analysis_requested.connect(start_analysis)

            # Auto-start analysis immediately after dialog opens
            QTimer.singleShot(100, start_analysis)  # Small delay to ensure dialog is fully loaded

            dialog.exec()

            # Store enhanced RWS results in main window for Excel export access
            if hasattr(rws_widget, "analysis_results") and rws_widget.analysis_results:
                logger.info("Storing Enhanced RWS results in main window for Excel export")
                main_window.rws_enhanced_results = rws_widget.analysis_results
                # Also create a reference to the widget itself for compatibility
                main_window.rws_enhanced_widget = rws_widget

        except Exception as e:
            logger.error(f"Error in enhanced RWS analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Enhanced RWS analysis failed: {str(e)}")
