"""
Enhanced RWS Analysis Widget
Displays results from the enhanced RWS analysis methodology
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg
import numpy as np
import logging
from typing import Dict

from ..analysis.rws_enhanced_analysis import EnhancedRWSAnalysis

logger = logging.getLogger(__name__)


class RWSAnalysisThread(QThread):
    """Worker thread for enhanced RWS analysis"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, frames, qca_results, cardiac_phases, frame_timestamps, calibration_factor):
        super().__init__()
        self.frames = frames
        self.qca_results = qca_results
        self.cardiac_phases = cardiac_phases
        self.frame_timestamps = frame_timestamps
        self.calibration_factor = calibration_factor

    def run(self):
        try:
            analyzer = EnhancedRWSAnalysis()

            self.progress.emit(20, "Detecting motion artifacts...")

            results = analyzer.analyze_cardiac_cycle(
                self.frames,
                self.qca_results,
                self.cardiac_phases,
                self.frame_timestamps,
                self.calibration_factor,
            )

            self.finished.emit(results)

        except Exception as e:
            logger.error(f"Error in RWS analysis thread: {str(e)}")
            self.error.emit(str(e))


class EnhancedRWSWidget(QWidget):
    """Widget for enhanced RWS analysis and visualization"""

    analysis_requested = pyqtSignal()
    stenosis_boundaries_updated = pyqtSignal(dict)  # Signal to update overlay with P/D points

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis_results = None
        self.analysis_thread = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()

        title_label = QLabel("Enhanced RWS Analysis")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.analyze_button = QPushButton("Analyze RWS")
        self.analyze_button.clicked.connect(self.on_analyze_clicked)
        header_layout.addWidget(self.analyze_button)

        layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # Tab widget for results
        self.tab_widget = QTabWidget()

        # Summary tab
        self.summary_widget = self.create_summary_tab()
        self.tab_widget.addTab(self.summary_widget, "Summary")

        # RWS Profile tab
        self.profile_widget = self.create_profile_tab()
        self.tab_widget.addTab(self.profile_widget, "RWS Profile")

        # Segment Analysis tab
        self.segment_widget = self.create_segment_tab()
        self.tab_widget.addTab(self.segment_widget, "Segment Analysis")

        # Motion Quality tab
        self.motion_widget = self.create_motion_tab()
        self.tab_widget.addTab(self.motion_widget, "Motion Quality")

        # Frame-MLD Profile tab - shows frame-by-frame MLD variability
        self.frame_mld_widget = self.create_frame_mld_tab()
        self.tab_widget.addTab(self.frame_mld_widget, "Frame-MLD Profile")

        layout.addWidget(self.tab_widget)

        # Initially hide tabs
        self.tab_widget.setVisible(False)

    def create_summary_tab(self) -> QWidget:
        """Create summary results tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Risk level group
        risk_group = QGroupBox("Risk Assessment")
        risk_layout = QVBoxLayout()

        self.risk_label = QLabel("Risk Level: -")
        risk_font = QFont()
        risk_font.setPointSize(16)
        risk_font.setBold(True)
        self.risk_label.setFont(risk_font)
        risk_layout.addWidget(self.risk_label)

        self.max_rws_label = QLabel("Maximum RWS: -")
        max_rws_font = QFont()
        max_rws_font.setPointSize(14)
        self.max_rws_label.setFont(max_rws_font)
        risk_layout.addWidget(self.max_rws_label)

        self.interpretation_text = QTextEdit()
        self.interpretation_text.setReadOnly(True)
        self.interpretation_text.setMaximumHeight(80)
        risk_layout.addWidget(self.interpretation_text)

        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)

        # Key measurements group
        measurements_group = QGroupBox("Key Measurements")
        measurements_layout = QVBoxLayout()

        self.measurements_table = QTableWidget(7, 2)
        self.measurements_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.measurements_table.horizontalHeader().setStretchLastSection(True)
        self.measurements_table.verticalHeader().setVisible(False)

        measurements_layout.addWidget(self.measurements_table)
        measurements_group.setLayout(measurements_layout)
        layout.addWidget(measurements_group)

        return widget

    def create_profile_tab(self) -> QWidget:
        """Create RWS profile visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # RWS profile plot
        self.rws_plot = pg.PlotWidget(title="RWS Profile Along Vessel")
        self.rws_plot.setLabel("left", "RWS (%)")
        self.rws_plot.setLabel("bottom", "Position along vessel")
        self.rws_plot.showGrid(x=True, y=True, alpha=0.3)

        # Add threshold lines
        self.rws_plot.addLine(y=10, pen=pg.mkPen("g", width=2, style=Qt.PenStyle.DashLine))
        self.rws_plot.addLine(y=12, pen=pg.mkPen("y", width=2, style=Qt.PenStyle.DashLine))
        self.rws_plot.addLine(y=15, pen=pg.mkPen("r", width=2, style=Qt.PenStyle.DashLine))

        layout.addWidget(self.rws_plot)

        # Diameter variation plot
        self.diameter_plot = pg.PlotWidget(title="Diameter Variation Across Cardiac Phases")
        self.diameter_plot.setLabel("left", "Diameter (mm)")
        self.diameter_plot.setLabel("bottom", "Position along vessel")
        self.diameter_plot.showGrid(x=True, y=True, alpha=0.3)

        layout.addWidget(self.diameter_plot)

        return widget

    def create_segment_tab(self) -> QWidget:
        """Create segment analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.segment_table = QTableWidget(3, 6)
        self.segment_table.setHorizontalHeaderLabels(
            [
                "Segment",
                "RWS (%)",
                "Max Diameter (mm)",
                "Max Frame",
                "Min Diameter (mm)",
                "Min Frame",
            ]
        )
        self.segment_table.setVerticalHeaderLabels(["Proximal", "Throat", "Distal"])
        self.segment_table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.segment_table)

        # Segment visualization
        self.segment_plot = pg.PlotWidget(title="RWS by Vessel Segment")
        self.segment_plot.setLabel("left", "RWS (%)")
        self.segment_plot.showGrid(x=True, y=True, alpha=0.3)

        layout.addWidget(self.segment_plot)

        return widget

    def create_motion_tab(self) -> QWidget:
        """Create motion quality assessment tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Motion quality summary
        self.motion_quality_label = QLabel("Overall Motion Quality: -")
        motion_font = QFont()
        motion_font.setPointSize(12)
        motion_font.setBold(True)
        self.motion_quality_label.setFont(motion_font)
        layout.addWidget(self.motion_quality_label)

        # Motion metrics table
        self.motion_table = QTableWidget(6, 2)
        self.motion_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.motion_table.horizontalHeader().setStretchLastSection(True)
        self.motion_table.verticalHeader().setVisible(False)

        layout.addWidget(self.motion_table)

        # Selected frames info
        self.selected_frames_text = QTextEdit()
        self.selected_frames_text.setReadOnly(True)
        self.selected_frames_text.setMaximumHeight(100)
        layout.addWidget(QLabel("Selected Frames:"))
        layout.addWidget(self.selected_frames_text)

        return widget

    def create_frame_mld_tab(self) -> QWidget:
        """Create Frame-MLD Profile tab to show frame-by-frame MLD variability"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Frame-by-frame MLD plot
        self.frame_mld_plot = pg.PlotWidget(title="Frame-by-Frame MLD Variability")
        self.frame_mld_plot.setLabel("left", "MLD (mm)")
        self.frame_mld_plot.setLabel("bottom", "Frame Number")
        self.frame_mld_plot.showGrid(x=True, y=True, alpha=0.3)

        # Add legend
        self.frame_mld_plot.addLegend()

        # Set background
        self.frame_mld_plot.setBackground("w")

        layout.addWidget(self.frame_mld_plot)

        # Statistics group
        stats_group = QGroupBox("MLD Statistics")
        stats_layout = QVBoxLayout()

        # Create statistics table
        self.mld_stats_table = QTableWidget(8, 2)
        self.mld_stats_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self.mld_stats_table.horizontalHeader().setStretchLastSection(True)
        self.mld_stats_table.verticalHeader().setVisible(False)

        # Set default statistics rows
        stats_items = [
            ("Minimum MLD", "-"),
            ("Maximum MLD", "-"),
            ("Mean MLD", "-"),
            ("Median MLD", "-"),
            ("Std Deviation", "-"),
            ("MLD Range", "-"),
            ("Coefficient of Variation", "-"),
            ("Total Frames Analyzed", "-"),
        ]

        for row, (stat, value) in enumerate(stats_items):
            self.mld_stats_table.setItem(row, 0, QTableWidgetItem(stat))
            self.mld_stats_table.setItem(row, 1, QTableWidgetItem(value))

        stats_layout.addWidget(self.mld_stats_table)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Frame details table
        details_group = QGroupBox("Frame-by-Frame MLD Values")
        details_layout = QVBoxLayout()

        self.frame_mld_table = QTableWidget()
        self.frame_mld_table.setColumnCount(4)
        self.frame_mld_table.setHorizontalHeaderLabels(["Frame", "MLD (mm)", "Phase", "Outlier"])
        self.frame_mld_table.horizontalHeader().setStretchLastSection(True)

        details_layout.addWidget(self.frame_mld_table)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        return widget

    @pyqtSlot()
    def on_analyze_clicked(self):
        """Handle analyze button click"""
        self.analysis_requested.emit()

    def start_analysis(
        self, frames, qca_results, cardiac_phases, frame_timestamps, calibration_factor
    ):
        """Start enhanced RWS analysis"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            QMessageBox.warning(self, "Analysis Running", "Analysis is already in progress")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.analyze_button.setEnabled(False)
        self.tab_widget.setVisible(False)

        # Create and start thread
        self.analysis_thread = RWSAnalysisThread(
            frames, qca_results, cardiac_phases, frame_timestamps, calibration_factor
        )

        self.analysis_thread.progress.connect(self.on_progress)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)

        self.analysis_thread.start()

    @pyqtSlot(int, str)
    def on_progress(self, value: int, message: str):
        """Update progress"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    @pyqtSlot(dict)
    def on_analysis_finished(self, results: Dict):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_button.setEnabled(True)

        if results.get("success", False):
            self.analysis_results = results
            self.update_results_display()
            self.tab_widget.setVisible(True)

            # Emit stenosis boundaries for overlay
            stenosis_rws = results.get("stenosis_rws", {})
            if "stenosis_boundaries" in stenosis_rws:
                boundaries_data = {
                    "stenosis_boundaries": stenosis_rws["stenosis_boundaries"],
                    "centerline": stenosis_rws.get("centerline", []),
                }
                self.stenosis_boundaries_updated.emit(boundaries_data)
        else:
            QMessageBox.warning(self, "Analysis Failed", results.get("error", "Unknown error"))

    @pyqtSlot(str)
    def on_analysis_error(self, error: str):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_button.setEnabled(True)

        QMessageBox.critical(self, "Analysis Error", f"Error: {error}")

    def update_results_display(self):
        """Update all result displays"""
        if not self.analysis_results:
            return

        # Update summary tab
        self.update_summary_display()

        # Update profile tab
        self.update_profile_display()

        # Update segment tab
        self.update_segment_display()

        # Update motion tab
        self.update_motion_display()

        # Update Frame-MLD Profile tab
        self.update_frame_mld_display()

    def update_summary_display(self):
        """Update summary tab"""
        stenosis_rws = self.analysis_results.get("stenosis_rws", {})
        rws_percentage = stenosis_rws.get("rws_stenosis", 0)
        risk_level = self.analysis_results["risk_level"]

        # Update risk label with color
        self.risk_label.setText(f"Risk Level: {risk_level}")
        if risk_level == "LOW":
            self.risk_label.setStyleSheet("color: green;")
        elif risk_level == "MODERATE":
            self.risk_label.setStyleSheet("color: orange;")
        elif risk_level == "HIGH":
            self.risk_label.setStyleSheet("color: red;")
        else:
            self.risk_label.setStyleSheet("color: darkred;")

        # Update stenosis RWS
        self.max_rws_label.setText(f"Stenosis RWS: {rws_percentage}%")

        # Update interpretation
        self.interpretation_text.setText(self.analysis_results["clinical_interpretation"])

        # Get selected frames for frame number lookup
        selected_frames = self.analysis_results.get("selected_frames", {})

        # Get point-specific RWS results for better summary display
        point_rws_results = stenosis_rws.get("point_rws_results", {})

        if point_rws_results and "stenosis" in point_rws_results:
            # Use stenosis point data for summary (most accurate)
            stenosis_data = point_rws_results["stenosis"]
            max_mld_phase = stenosis_data.get("max_phase", "unknown")
            min_mld_phase = stenosis_data.get("min_phase", "unknown")
            max_mld = stenosis_data.get("max_diameter", 0.0)
            min_mld = stenosis_data.get("min_diameter", 0.0)
            diameter_change = stenosis_data.get("diameter_change", 0.0)
        else:
            # Fallback to old MLD values
            max_mld_phase = stenosis_rws.get("max_mld_phase", "unknown")
            min_mld_phase = stenosis_rws.get("min_mld_phase", "unknown")
            max_mld = stenosis_rws.get("max_mld", 0.0)
            min_mld = stenosis_rws.get("min_mld", 0.0)
            diameter_change = stenosis_rws.get("diameter_change", 0.0)

        # Get frame numbers from selected_frames
        max_mld_frame = selected_frames.get(max_mld_phase, "N/A")
        min_mld_frame = selected_frames.get(min_mld_phase, "N/A")

        # Format combined display: phase name + frame number
        if isinstance(max_mld_frame, int):
            max_display = f"{max_mld_phase} - Frame {max_mld_frame + 1}"
        else:
            max_display = f"{max_mld_phase} - {max_mld_frame}"

        if isinstance(min_mld_frame, int):
            min_display = f"{min_mld_phase} - Frame {min_mld_frame + 1}"
        else:
            min_display = f"{min_mld_phase} - {min_mld_frame}"

        # Update measurements table with accurate stenosis point data
        measurements = [
            ("Max MLD", f"{max_mld:.2f} mm ({max_display})"),
            ("Min MLD", f"{min_mld:.2f} mm ({min_display})"),
            ("MLD Change", f"{diameter_change:.2f} mm"),
            ("RWS at Stenosis", f"{rws_percentage:.1f}%"),
            ("Phases Analyzed", f"{self.analysis_results['num_phases_analyzed']}"),
        ]

        # Add point-specific summary if available
        if point_rws_results:
            proximal_rws = point_rws_results.get("proximal", {}).get("rws", 0.0)
            distal_rws = point_rws_results.get("distal", {}).get("rws", 0.0)
            measurements.extend(
                [("Proximal RWS", f"{proximal_rws:.1f}%"), ("Distal RWS", f"{distal_rws:.1f}%")]
            )

        # Ensure table has enough rows
        if len(measurements) > self.measurements_table.rowCount():
            self.measurements_table.setRowCount(len(measurements))

        for i, (param, value) in enumerate(measurements):
            self.measurements_table.setItem(i, 0, QTableWidgetItem(param))
            self.measurements_table.setItem(i, 1, QTableWidgetItem(value))

    def update_profile_display(self):
        """Update profile plots"""
        # Clear plots
        self.rws_plot.clear()
        self.diameter_plot.clear()

        stenosis_rws = self.analysis_results.get("stenosis_rws", {})

        # Plot RWS profile using P, MLD, and D points
        point_rws_results = stenosis_rws.get("point_rws_results", {})
        stenosis_boundaries = stenosis_rws.get("stenosis_boundaries", {})
        calibration = self.analysis_results.get("calibration_factor", 1.0)

        if point_rws_results and stenosis_boundaries:
            self.rws_plot.clear()
            self.rws_plot.setTitle("RWS Profile at Anatomical Points")

            # Get positions and RWS values
            positions = []
            rws_values = []
            point_names = []
            colors = ["r", "b", "g"]  # Proximal=red, Stenosis=blue, Distal=green

            # Extract data for P, MLD, D points
            point_order = ["proximal", "stenosis", "distal"]
            position_keys = ["p_point", "mld_point", "d_point"]
            display_names = ["P-point", "MLD-point", "D-point"]

            for i, (point_key, pos_key, display_name) in enumerate(
                zip(point_order, position_keys, display_names)
            ):
                if point_key in point_rws_results and pos_key in stenosis_boundaries:
                    pos_pixels = stenosis_boundaries[pos_key]
                    pos_mm = pos_pixels * calibration
                    rws_val = point_rws_results[point_key]["rws"]

                    positions.append(pos_mm)
                    rws_values.append(rws_val)
                    point_names.append(display_name)

            if positions and rws_values:
                # Plot points
                for i, (pos, rws, name, color) in enumerate(
                    zip(positions, rws_values, point_names, colors)
                ):
                    self.rws_plot.plot(
                        [pos],
                        [rws],
                        pen=None,
                        symbol="o",
                        symbolSize=12,
                        symbolBrush=color,
                        name=name,
                    )

                # Connect points with lines
                if len(positions) > 1:
                    self.rws_plot.plot(
                        positions,
                        rws_values,
                        pen=pg.mkPen("gray", width=2, style=Qt.PenStyle.DashLine),
                    )

                # Add threshold lines
                self.rws_plot.addLine(y=10, pen=pg.mkPen("g", width=1, style=Qt.PenStyle.DashLine))
                self.rws_plot.addLine(y=12, pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine))
                self.rws_plot.addLine(y=15, pen=pg.mkPen("r", width=1, style=Qt.PenStyle.DashLine))

                # Add labels for each point
                for pos, rws, name in zip(positions, rws_values, point_names):
                    label = pg.TextItem(f"{name}\n{rws:.1f}%", anchor=(0.5, 1.5), color="black")
                    label.setPos(pos, rws)
                    self.rws_plot.addItem(label)

                # Set axis ranges
                min_pos = min(positions) - 1
                max_pos = max(positions) + 1
                max_rws = max(max(rws_values), 20)

                self.rws_plot.setXRange(min_pos, max_pos)
                self.rws_plot.setYRange(0, max_rws * 1.2)

                # Add legend
                self.rws_plot.addLegend()

            else:
                # No valid point data
                error_text = pg.TextItem(
                    "No anatomical point data available\nPlease check QCA analysis",
                    color="r",
                    anchor=(0.5, 0.5),
                )
                error_text.setPos(0.5, 12.5)
                self.rws_plot.addItem(error_text)
                self.rws_plot.setYRange(0, 25)
        else:
            # Fallback to old single-point display
            self.rws_plot.clear()
            self.rws_plot.setTitle("⚠️ RWS Profile - Limited Data")

            # Get MLD-based RWS value for display
            mld_based_rws = stenosis_rws.get("rws_stenosis", 0)
            max_mld = stenosis_rws.get("max_mld", 0)
            min_mld = stenosis_rws.get("min_mld", 0)

            # Show single point representing MLD-based RWS
            if mld_based_rws > 0:
                self.rws_plot.plot(
                    [0.5], [mld_based_rws], pen=None, symbol="o", symbolSize=15, symbolBrush="b"
                )

                # Add threshold lines
                self.rws_plot.addLine(y=10, pen=pg.mkPen("g", width=2, style=Qt.PenStyle.DashLine))
                self.rws_plot.addLine(y=12, pen=pg.mkPen("y", width=2, style=Qt.PenStyle.DashLine))
                self.rws_plot.addLine(y=15, pen=pg.mkPen("r", width=2, style=Qt.PenStyle.DashLine))

                # Add text annotation
                info_text = pg.TextItem(
                    f"MLD-based RWS: {mld_based_rws:.1f}%\n"
                    + f"Max MLD: {max_mld:.2f}mm\n"
                    + f"Min MLD: {min_mld:.2f}mm",
                    color="b",
                    anchor=(0, 0.5),
                )
                info_text.setPos(0.1, mld_based_rws)
                self.rws_plot.addItem(info_text)

                self.rws_plot.setYRange(0, max(25, mld_based_rws * 1.5))
            else:
                # Show error message
                error_text = pg.TextItem(
                    "No valid RWS data available\nPlease check QCA analysis",
                    color="r",
                    anchor=(0.5, 0.5),
                )
                error_text.setPos(0.5, 12.5)
                self.rws_plot.addItem(error_text)
                self.rws_plot.setYRange(0, 25)

        # Set axis labels
        self.rws_plot.setLabel("bottom", "Position along vessel (mm)")
        self.rws_plot.setLabel("left", "RWS (%)")

        # Plot diameter profiles for different cardiac phases
        diameter_profiles = stenosis_rws.get("diameter_profiles", {})
        if diameter_profiles:
            calibration = self.analysis_results.get("calibration_factor", 1.0)

            # Different colors for each phase
            phase_colors = ["b", "r", "g", "y", "m", "c"]

            for i, (phase_name, diameters) in enumerate(diameter_profiles.items()):
                positions_mm = np.arange(len(diameters)) * calibration
                color = phase_colors[i % len(phase_colors)]

                self.diameter_plot.plot(
                    positions_mm, diameters, pen=pg.mkPen(color, width=2), name=phase_name
                )

            # Mark stenosis position
            stenosis_pos = stenosis_rws.get("stenosis_position", 0)
            stenosis_line = pg.InfiniteLine(
                pos=stenosis_pos * calibration,
                angle=90,
                pen=pg.mkPen("w", width=2, style=Qt.PenStyle.DashLine),
            )
            self.diameter_plot.addItem(stenosis_line)

            self.diameter_plot.setLabel("bottom", "Position along vessel (mm)")
            self.diameter_plot.addLegend()

        # Y-range is set above in the if-else blocks

    def update_segment_display(self):
        """Update segment analysis display"""
        stenosis_rws = self.analysis_results.get("stenosis_rws", {})
        stenosis_boundaries = stenosis_rws.get("stenosis_boundaries", {})
        stenosis_rws.get("rws_profile", [])
        stenosis_rws.get("diameter_profiles", {})

        # Clear table first
        self.segment_table.clearContents()

        # Get anatomical point positions
        p_point = stenosis_boundaries.get("p_point", 0)
        mld_point = stenosis_boundaries.get("mld_point", 0)
        d_point = stenosis_boundaries.get("d_point", 0)

        # Prepare anatomical points data
        anatomical_points = [
            ("Proximal", p_point, "P Point"),
            ("Throat", mld_point, "Stenosis"),
            ("Distal", d_point, "D Point"),
        ]

        # FIXED: USE POINT-SPECIFIC RWS CALCULATIONS
        # Get point-specific RWS results from enhanced analysis
        point_rws_results = stenosis_rws.get("point_rws_results", {})
        selected_frames = self.analysis_results.get("selected_frames", {})

        # Point mapping for table rows
        point_mapping = {"Proximal": "proximal", "Throat": "stenosis", "Distal": "distal"}

        # Update table with point-specific values
        for row, (region_name, point_position, point_description) in enumerate(anatomical_points):
            if row >= self.segment_table.rowCount():
                self.segment_table.setRowCount(row + 1)

            # Set region name
            self.segment_table.setItem(row, 0, QTableWidgetItem(region_name))

            # Get point-specific data
            point_key = point_mapping.get(region_name, "stenosis")
            point_data = point_rws_results.get(point_key, {})

            if point_data:
                # FIXED: Use actual point-specific RWS calculations
                rws_value = point_data.get("rws", 0.0)
                max_diameter = point_data.get("max_diameter", 0.0)
                min_diameter = point_data.get("min_diameter", 0.0)
                max_phase = point_data.get("max_phase", "N/A")
                min_phase = point_data.get("min_phase", "N/A")

                # Get frame numbers from phases
                max_frame = selected_frames.get(max_phase, "N/A")
                min_frame = selected_frames.get(min_phase, "N/A")

                # Format phase + frame display
                if isinstance(max_frame, int):
                    max_display = f"{max_phase} (F{max_frame + 1})"
                else:
                    max_display = str(max_phase)

                if isinstance(min_frame, int):
                    min_display = f"{min_phase} (F{min_frame + 1})"
                else:
                    min_display = str(min_phase)

                # Update table with point-specific values - DIFFERENT FOR EACH ROW!
                self.segment_table.setItem(row, 1, QTableWidgetItem(f"{rws_value:.1f}"))
                self.segment_table.setItem(row, 2, QTableWidgetItem(f"{max_diameter:.2f}"))
                self.segment_table.setItem(row, 3, QTableWidgetItem(max_display))
                self.segment_table.setItem(row, 4, QTableWidgetItem(f"{min_diameter:.2f}"))
                self.segment_table.setItem(row, 5, QTableWidgetItem(min_display))

            else:
                # Fallback: Use MLD values for missing point data
                mld_values = stenosis_rws.get("mld_values", {})
                if mld_values:
                    max_mld_phase = stenosis_rws.get("max_mld_phase", "N/A")
                    min_mld_phase = stenosis_rws.get("min_mld_phase", "N/A")
                    max_mld = stenosis_rws.get("max_mld", 0.0)
                    min_mld = stenosis_rws.get("min_mld", 0.0)
                    rws_fallback = stenosis_rws.get("rws_stenosis", 0.0)

                    self.segment_table.setItem(row, 1, QTableWidgetItem(f"{rws_fallback:.1f}"))
                    self.segment_table.setItem(row, 2, QTableWidgetItem(f"{max_mld:.2f}"))
                    self.segment_table.setItem(row, 3, QTableWidgetItem(str(max_mld_phase)))
                    self.segment_table.setItem(row, 4, QTableWidgetItem(f"{min_mld:.2f}"))
                    self.segment_table.setItem(row, 5, QTableWidgetItem(str(min_mld_phase)))
                else:
                    # No data at all
                    self.segment_table.setItem(row, 1, QTableWidgetItem("N/A"))
                    self.segment_table.setItem(row, 2, QTableWidgetItem("N/A"))
                    self.segment_table.setItem(row, 3, QTableWidgetItem("N/A"))
                    self.segment_table.setItem(row, 4, QTableWidgetItem("N/A"))
                    self.segment_table.setItem(row, 5, QTableWidgetItem("N/A"))

        # Update segment plot to show point-specific RWS values
        self.segment_plot.clear()
        self.segment_plot.setTitle("Point-Specific RWS Analysis")
        self.segment_plot.setLabel("left", "RWS (%)")

        # Show bars for each anatomical point with different RWS values
        if point_rws_results:
            # Extract RWS values for each point
            x_positions = []
            rws_values = []
            point_names = []
            colors = ["r", "b", "g"]  # Proximal=red, Stenosis=blue, Distal=green

            point_order = ["proximal", "stenosis", "distal"]
            display_names = ["Proximal", "Stenosis", "Distal"]

            for i, (point_key, display_name) in enumerate(zip(point_order, display_names)):
                if point_key in point_rws_results:
                    rws_val = point_rws_results[point_key]["rws"]
                    x_positions.append(i)
                    rws_values.append(rws_val)
                    point_names.append(display_name)

            if x_positions and rws_values:
                # Create bar graph with different colors for each point
                for i, (x, rws, name, color) in enumerate(
                    zip(x_positions, rws_values, point_names, colors)
                ):
                    bargraph = pg.BarGraphItem(
                        x=[x], height=[rws], width=0.6, brush=color, name=name
                    )
                    self.segment_plot.addItem(bargraph)

                # Add threshold lines
                self.segment_plot.addLine(
                    y=10, pen=pg.mkPen("g", width=1, style=Qt.PenStyle.DashLine)
                )
                self.segment_plot.addLine(
                    y=12, pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine)
                )
                self.segment_plot.addLine(
                    y=15, pen=pg.mkPen("r", width=1, style=Qt.PenStyle.DashLine)
                )

                # Set x-axis labels
                ax = self.segment_plot.getAxis("bottom")
                ticks = [(i, name) for i, name in enumerate(point_names)]
                ax.setTicks([ticks])

                # Add text labels above bars
                for x, rws, name in zip(x_positions, rws_values, point_names):
                    rws_text = pg.TextItem(f"{rws:.1f}%", color="w", anchor=(0.5, 1.2))
                    rws_text.setPos(x, rws + max(rws_values) * 0.05)
                    self.segment_plot.addItem(rws_text)

                # Set appropriate Y range
                max_rws = max(max(rws_values), 20)
                self.segment_plot.setYRange(0, max_rws * 1.3)

                # Add legend
                self.segment_plot.addLegend()

            else:
                self.segment_plot.setTitle("No Point-Specific RWS Data")
                self.segment_plot.setYRange(0, 25)
        else:
            # Fallback to single MLD bar if no point data
            stenosis_rws_val = stenosis_rws.get("rws_stenosis", 0.0)
            if stenosis_rws_val > 0:
                bargraph = pg.BarGraphItem(
                    x=[0], height=[stenosis_rws_val], width=0.6, brush="b", name="Stenosis RWS"
                )
                self.segment_plot.addItem(bargraph)

                ax = self.segment_plot.getAxis("bottom")
                ax.setTicks([[(0, "Stenosis")]])

                self.segment_plot.setYRange(0, max(25, stenosis_rws_val * 1.3))
            else:
                self.segment_plot.setTitle("No Valid RWS Data")
                self.segment_plot.setYRange(0, 25)

    def update_motion_display(self):
        """Update motion quality display"""
        motion_quality = self.analysis_results["motion_quality"]

        # Update overall quality
        overall = motion_quality.get("overall_quality", "Unknown")
        self.motion_quality_label.setText(f"Overall Motion Quality: {overall}")

        # Color code quality
        if overall == "Excellent":
            self.motion_quality_label.setStyleSheet("color: green;")
        elif overall == "Good":
            self.motion_quality_label.setStyleSheet("color: blue;")
        elif overall == "Fair":
            self.motion_quality_label.setStyleSheet("color: orange;")
        else:
            self.motion_quality_label.setStyleSheet("color: red;")

        # Update metrics table
        metrics = [
            ("Mean Motion Score", f"{motion_quality.get('mean_motion', 0):.3f}"),
            ("Std Motion Score", f"{motion_quality.get('std_motion', 0):.3f}"),
            ("Min Motion Score", f"{motion_quality.get('min_motion', 0):.3f}"),
            ("Max Motion Score", f"{motion_quality.get('max_motion', 0):.3f}"),
            ("Low Motion Frames", f"{motion_quality.get('num_low_motion_frames', 0)}"),
            ("Low Motion %", f"{motion_quality.get('percent_low_motion', 0):.1f}%"),
        ]

        for i, (metric, value) in enumerate(metrics):
            self.motion_table.setItem(i, 0, QTableWidgetItem(metric))
            self.motion_table.setItem(i, 1, QTableWidgetItem(value))

        # Update selected frames with MLD values
        selected_frames = self.analysis_results.get("selected_frames", {})
        stenosis_rws = self.analysis_results.get("stenosis_rws", {})
        mld_values = stenosis_rws.get("mld_values", {})

        frames_text = "Cardiac Phase → Frame Index → MLD Value:\n"
        for phase, frame_idx in selected_frames.items():
            # Get MLD value for this phase
            mld_value = mld_values.get(phase, None)

            if mld_value is not None:
                frames_text += f"{phase}: Frame {frame_idx + 1} → {mld_value:.2f} mm\n"
            else:
                frames_text += f"{phase}: Frame {frame_idx + 1} → N/A\n"

        self.selected_frames_text.setText(frames_text)

    def update_frame_mld_display(self):
        """Update Frame-MLD Profile tab with frame-by-frame MLD variability"""
        if not self.analysis_results:
            return

        # Get stenosis RWS data which contains MLD values
        stenosis_rws = self.analysis_results.get("stenosis_rws", {})

        # Try to get frame-by-frame MLD data from different possible sources
        frame_mld_data = {}

        # Option 1: From mld_info_by_frame (if available)
        if "mld_info_by_frame" in stenosis_rws:
            frame_mld_data = stenosis_rws["mld_info_by_frame"]
        # Option 2: From selected_frames and mld_values
        elif "mld_values" in stenosis_rws:
            selected_frames = self.analysis_results.get("selected_frames", {})
            mld_values_by_phase = stenosis_rws.get("mld_values", {})

            # Convert phase-based MLD to frame-based
            for phase, frame_idx in selected_frames.items():
                if phase in mld_values_by_phase:
                    frame_mld_data[frame_idx] = {
                        "mld_value": mld_values_by_phase[phase],
                        "phase": phase,
                    }

        if not frame_mld_data:
            # No MLD data available
            self.frame_mld_plot.clear()
            self.frame_mld_plot.setTitle("No MLD Data Available")
            return

        # Sort frames by index
        sorted_frames = sorted(frame_mld_data.keys())

        # Extract MLD values
        frame_numbers = []
        mld_values = []
        phases = []

        for frame_idx in sorted_frames:
            frame_data = frame_mld_data[frame_idx]
            frame_numbers.append(frame_idx + 1)  # Convert to 1-based for display

            if isinstance(frame_data, dict):
                mld_values.append(frame_data.get("mld_value", 0))
                phases.append(frame_data.get("phase", "Unknown"))
            else:
                # If it's just a number
                mld_values.append(float(frame_data))
                phases.append("Unknown")

        # Calculate statistics
        mld_array = np.array(mld_values)
        min_mld = np.min(mld_array)
        max_mld = np.max(mld_array)
        mean_mld = np.mean(mld_array)
        median_mld = np.median(mld_array)
        std_mld = np.std(mld_array)
        range_mld = max_mld - min_mld
        cv_mld = (std_mld / mean_mld * 100) if mean_mld > 0 else 0

        # Update plot
        self.frame_mld_plot.clear()
        self.frame_mld_plot.setTitle("Frame-by-Frame MLD Variability")

        # Plot MLD values
        pen = pg.mkPen(color="b", width=2)
        self.frame_mld_plot.plot(
            frame_numbers,
            mld_values,
            pen=pen,
            symbol="o",
            symbolSize=8,
            symbolBrush="b",
            name="MLD",
        )

        # Add mean line
        mean_line = pg.InfiniteLine(
            pos=mean_mld, angle=0, pen=pg.mkPen("g", width=2, style=Qt.PenStyle.DashLine)
        )
        self.frame_mld_plot.addItem(mean_line)

        # Add min/max lines
        min_line = pg.InfiniteLine(
            pos=min_mld, angle=0, pen=pg.mkPen("r", width=1, style=Qt.PenStyle.DotLine)
        )
        max_line = pg.InfiniteLine(
            pos=max_mld, angle=0, pen=pg.mkPen("r", width=1, style=Qt.PenStyle.DotLine)
        )
        self.frame_mld_plot.addItem(min_line)
        self.frame_mld_plot.addItem(max_line)

        # Highlight min and max points
        min_idx = np.argmin(mld_array)
        max_idx = np.argmax(mld_array)

        self.frame_mld_plot.plot(
            [frame_numbers[min_idx]],
            [mld_values[min_idx]],
            pen=None,
            symbol="o",
            symbolSize=12,
            symbolBrush="r",
            name=f"Min MLD ({min_mld:.3f}mm)",
        )

        self.frame_mld_plot.plot(
            [frame_numbers[max_idx]],
            [mld_values[max_idx]],
            pen=None,
            symbol="o",
            symbolSize=12,
            symbolBrush="g",
            name=f"Max MLD ({max_mld:.3f}mm)",
        )

        # Update statistics table
        stats = [
            ("Minimum MLD", f"{min_mld:.3f} mm (Frame {frame_numbers[min_idx]})"),
            ("Maximum MLD", f"{max_mld:.3f} mm (Frame {frame_numbers[max_idx]})"),
            ("Mean MLD", f"{mean_mld:.3f} mm"),
            ("Median MLD", f"{median_mld:.3f} mm"),
            ("Std Deviation", f"{std_mld:.3f} mm"),
            ("MLD Range", f"{range_mld:.3f} mm"),
            ("Coefficient of Variation", f"{cv_mld:.1f}%"),
            ("Total Frames Analyzed", str(len(frame_numbers))),
        ]

        for row, (stat, value) in enumerate(stats):
            self.mld_stats_table.setItem(row, 0, QTableWidgetItem(stat))
            self.mld_stats_table.setItem(row, 1, QTableWidgetItem(value))

        # Update frame details table
        self.frame_mld_table.setRowCount(len(frame_numbers))

        # Check for outliers (using Hampel filter results if available)
        outlier_info = stenosis_rws.get("outlier_frames", [])

        for row, (frame_num, mld_val, phase) in enumerate(zip(frame_numbers, mld_values, phases)):
            # Frame number
            self.frame_mld_table.setItem(row, 0, QTableWidgetItem(str(frame_num)))

            # MLD value
            mld_item = QTableWidgetItem(f"{mld_val:.3f}")
            # Highlight min/max values
            if row == min_idx:
                mld_item.setBackground(QColor(255, 200, 200))  # Light red for min
            elif row == max_idx:
                mld_item.setBackground(QColor(200, 255, 200))  # Light green for max
            self.frame_mld_table.setItem(row, 1, mld_item)

            # Phase
            self.frame_mld_table.setItem(row, 2, QTableWidgetItem(phase))

            # Outlier status
            is_outlier = (frame_num - 1) in outlier_info  # Convert back to 0-based
            outlier_text = "Yes" if is_outlier else "No"
            outlier_item = QTableWidgetItem(outlier_text)
            if is_outlier:
                outlier_item.setForeground(QColor(255, 0, 0))  # Red text for outliers
            self.frame_mld_table.setItem(row, 3, outlier_item)

        # Resize columns to content
        self.frame_mld_table.resizeColumnsToContents()
