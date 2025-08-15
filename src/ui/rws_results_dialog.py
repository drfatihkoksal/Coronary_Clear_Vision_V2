"""
RWS Analysis Results Dialog
Shows comprehensive results including frame data and visualizations
"""

import logging
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QDialogButtonBox,
    QGroupBox,
    QGridLayout,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class RWSResultsDialog(QDialog):
    """Dialog to display comprehensive RWS analysis results"""

    pdf_export_requested = pyqtSignal()
    excel_export_requested = pyqtSignal()
    database_save_requested = pyqtSignal()

    def __init__(self, results: Dict, qca_results: Dict, parent=None):
        super().__init__(parent)
        self.results = results
        self.qca_results = qca_results
        self.setWindowTitle("RWS Analysis Results")
        self.setModal(True)
        self.resize(1200, 800)

        self.setup_ui()
        self.populate_data()

    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)

        # Header with summary
        header_widget = self.create_header_widget()
        layout.addWidget(header_widget)

        # Tab widget for detailed views
        self.tab_widget = QTabWidget()

        # Summary tab
        summary_tab = self.create_summary_tab()
        self.tab_widget.addTab(summary_tab, "Summary")

        # Frame Analysis tab
        frame_tab = self.create_frame_analysis_tab()
        self.tab_widget.addTab(frame_tab, "Frame Analysis")

        # Visualizations tab
        viz_tab = self.create_visualizations_tab()
        self.tab_widget.addTab(viz_tab, "Visualizations")

        # Detailed Data tab
        data_tab = self.create_detailed_data_tab()
        self.tab_widget.addTab(data_tab, "Detailed Data")

        layout.addWidget(self.tab_widget)

        # Bottom buttons
        button_box = self.create_button_box()
        layout.addWidget(button_box)

    def create_header_widget(self) -> QWidget:
        """Create header widget with main RWS result"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Main RWS value
        rws_value = self.results.get("rws_at_mld", 0)
        rws_label = QLabel(
            f"RWS at MLD: {rws_value}%" if rws_value is not None else "RWS at MLD: N/A"
        )
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        rws_label.setFont(font)
        rws_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Color code based on value
        if rws_value is not None and rws_value > 12:
            rws_label.setStyleSheet(
                "color: #ff4444; background-color: #ffeeee; padding: 10px; border-radius: 5px;"
            )
        elif rws_value is not None:
            rws_label.setStyleSheet(
                "color: #44ff44; background-color: #eeffee; padding: 10px; border-radius: 5px;"
            )
        else:
            rws_label.setStyleSheet(
                "color: #666666; background-color: #f0f0f0; padding: 10px; border-radius: 5px;"
            )

        layout.addWidget(rws_label)

        # Clinical interpretation
        interpretation = QLabel()
        if rws_value is not None and rws_value > 12:
            interpretation.setText("⚠️ HIGH RWS - Indicates potential plaque vulnerability")
            interpretation.setStyleSheet("color: #cc0000; font-size: 14px; padding: 5px;")
        elif rws_value is not None:
            interpretation.setText("✓ NORMAL RWS - Indicates stable plaque characteristics")
            interpretation.setStyleSheet("color: #00cc00; font-size: 14px; padding: 5px;")
        else:
            interpretation.setText("Unable to determine - insufficient data")
            interpretation.setStyleSheet("color: #666666; font-size: 14px; padding: 5px;")
        interpretation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(interpretation)

        return widget

    def create_summary_tab(self) -> QWidget:
        """Create summary tab with key metrics"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create info groups
        info_layout = QHBoxLayout()

        # MLD Information
        mld_group = QGroupBox("MLD Information")
        mld_layout = QGridLayout()

        min_frame = self.results.get("mld_min_frame", 0)
        max_frame = self.results.get("mld_max_frame", 0)

        mld_min_val = self.results.get("mld_min_value")
        mld_max_val = self.results.get("mld_max_value")

        mld_data = [
            ("Minimum MLD:", f"{mld_min_val:.3f} mm" if mld_min_val is not None else "N/A"),
            ("Min MLD Frame:", f"Frame {min_frame + 1}" if min_frame is not None else "N/A"),
            ("Maximum MLD:", f"{mld_max_val:.3f} mm" if mld_max_val is not None else "N/A"),
            ("Max MLD Frame:", f"Frame {max_frame + 1}" if max_frame is not None else "N/A"),
            (
                "MLD Variation:",
                (
                    f"{mld_max_val - mld_min_val:.3f} mm"
                    if mld_max_val is not None and mld_min_val is not None
                    else "N/A"
                ),
            ),
        ]

        for i, (label, value) in enumerate(mld_data):
            mld_layout.addWidget(QLabel(f"<b>{label}</b>"), i, 0)
            mld_layout.addWidget(QLabel(value), i, 1)

        mld_group.setLayout(mld_layout)
        info_layout.addWidget(mld_group)

        # Analysis Information
        analysis_group = QGroupBox("Analysis Information")
        analysis_layout = QGridLayout()

        cal_factor = self.results.get("calibration_factor")

        analysis_data = [
            ("Frames Analyzed:", str(self.results.get("num_frames_analyzed", 0))),
            ("Beat Frames:", f"{len(self.results.get('beat_frames', []))} frames"),
            (
                "Calibration Factor:",
                f"{cal_factor:.4f} mm/pixel" if cal_factor is not None else "N/A",
            ),
            (
                "Analysis Time:",
                datetime.fromisoformat(
                    self.results.get("timestamp", datetime.now().isoformat())
                ).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ]

        for i, (label, value) in enumerate(analysis_data):
            analysis_layout.addWidget(QLabel(f"<b>{label}</b>"), i, 0)
            analysis_layout.addWidget(QLabel(value), i, 1)

        analysis_group.setLayout(analysis_layout)
        info_layout.addWidget(analysis_group)

        layout.addLayout(info_layout)

        # RWS Calculation Details
        calc_group = QGroupBox("RWS Calculation")
        calc_layout = QVBoxLayout()

        # Safe formatting with None checks
        mld_max = self.results.get("mld_max_value")
        mld_min = self.results.get("mld_min_value")
        rws_val = self.results.get("rws_at_mld")

        mld_max_str = f"{mld_max:.2f}" if mld_max is not None else "N/A"
        mld_min_str = f"{mld_min:.2f}" if mld_min is not None else "N/A"
        rws_str = f"{rws_val:.1f}" if rws_val is not None else "N/A"

        calc_text = f"""
        <p><b>Formula:</b> RWS = (MLD<sub>max</sub> - MLD<sub>min</sub>) / MLD<sub>max</sub> × 100%</p>
        <p><b>Calculation:</b> ({mld_max_str} - {mld_min_str}) / {mld_max_str} × 100%</p>
        <p><b>Result:</b> {rws_str}%</p>
        """

        calc_label = QLabel(calc_text)
        calc_label.setWordWrap(True)
        calc_layout.addWidget(calc_label)
        calc_group.setLayout(calc_layout)
        layout.addWidget(calc_group)

        layout.addStretch()
        return widget

    def create_frame_analysis_tab(self) -> QWidget:
        """Create frame analysis tab showing min and max MLD frames"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create horizontal layout for min and max frames
        frames_layout = QHBoxLayout()

        # Minimum MLD Frame
        min_frame_widget = self.create_frame_detail_widget(
            "Minimum MLD Frame",
            self.results.get("mld_min_frame"),
            self.results.get("mld_min_value"),
            "#ffeeee",
        )
        frames_layout.addWidget(min_frame_widget)

        # Maximum MLD Frame
        max_frame_widget = self.create_frame_detail_widget(
            "Maximum MLD Frame",
            self.results.get("mld_max_frame"),
            self.results.get("mld_max_value"),
            "#eeffee",
        )
        frames_layout.addWidget(max_frame_widget)

        layout.addLayout(frames_layout)

        # MLD progression table
        mld_table_group = QGroupBox("MLD Values Across All Frames")
        mld_table_layout = QVBoxLayout()

        self.mld_table = QTableWidget()
        self.mld_table.setColumnCount(3)
        self.mld_table.setHorizontalHeaderLabels(["Frame", "MLD (mm)", "Position (index)"])

        # Populate MLD table
        mld_info = self.results.get("mld_values_by_frame", {})
        self.mld_table.setRowCount(len(mld_info))

        for row, (frame_idx, info) in enumerate(sorted(mld_info.items())):
            self.mld_table.setItem(row, 0, QTableWidgetItem(str(frame_idx + 1)))
            mld_val = info.get("mld_value")
            mld_val_str = f"{mld_val:.2f}" if mld_val is not None else "N/A"
            self.mld_table.setItem(row, 1, QTableWidgetItem(mld_val_str))
            self.mld_table.setItem(row, 2, QTableWidgetItem(str(info.get("mld_index", "N/A"))))

            # Highlight min and max rows
            if frame_idx == self.results.get("mld_min_frame"):
                for col in range(3):
                    self.mld_table.item(row, col).setBackground(Qt.GlobalColor.red)
                    self.mld_table.item(row, col).setForeground(Qt.GlobalColor.white)
            elif frame_idx == self.results.get("mld_max_frame"):
                for col in range(3):
                    self.mld_table.item(row, col).setBackground(Qt.GlobalColor.green)
                    self.mld_table.item(row, col).setForeground(Qt.GlobalColor.white)

        self.mld_table.horizontalHeader().setStretchLastSection(True)
        mld_table_layout.addWidget(self.mld_table)
        mld_table_group.setLayout(mld_table_layout)
        layout.addWidget(mld_table_group)

        return widget

    def create_frame_detail_widget(
        self, title: str, frame_idx: Optional[int], mld_value: Optional[float], bg_color: str
    ) -> QGroupBox:
        """Create detailed widget for a specific frame"""
        group = QGroupBox(title)
        group.setStyleSheet(f"QGroupBox {{ background-color: {bg_color}; }}")
        layout = QVBoxLayout()

        if frame_idx is not None and frame_idx in self.qca_results:
            qca_data = self.qca_results[frame_idx]

            # Frame info
            prox_diam = qca_data.get("proximal_diameter")
            dist_diam = qca_data.get("distal_diameter")
            stenosis_pct = qca_data.get("percent_diameter_stenosis")
            lesion_len = qca_data.get("lesion_length")

            info_text = f"""
            <p><b>Frame Number:</b> {frame_idx + 1}</p>
            <p><b>MLD Value:</b> {f'{mld_value:.3f} mm' if mld_value is not None else 'N/A'}</p>
            <p><b>MLD Position:</b> {qca_data.get('mld_index', 'N/A')}</p>
            <p><b>Proximal Diameter:</b> {f'{prox_diam:.2f} mm' if prox_diam is not None else 'N/A'}</p>
            <p><b>Distal Diameter:</b> {f'{dist_diam:.2f} mm' if dist_diam is not None else 'N/A'}</p>
            <p><b>Stenosis %:</b> {f'{stenosis_pct:.1f}%' if stenosis_pct is not None else 'N/A'}</p>
            <p><b>Lesion Length:</b> {f'{lesion_len:.2f} mm' if lesion_len is not None else 'N/A'}</p>
            """

            info_label = QLabel(info_text)
            info_label.setWordWrap(True)
            layout.addWidget(info_label)

            # Add mini diameter profile plot if available
            if "diameters_mm" in qca_data or "diameter_profile" in qca_data:
                fig = Figure(figsize=(4, 3))
                ax = fig.add_subplot(111)

                # Get diameter profile
                diameters = qca_data.get("diameters_mm", qca_data.get("diameter_profile", []))
                if diameters is not None and len(diameters) > 0:
                    positions = np.arange(len(diameters))
                    ax.plot(positions, diameters, "b-", linewidth=2)

                    # Mark MLD position
                    mld_idx = qca_data.get("mld_index")
                    if mld_idx is not None and 0 <= mld_idx < len(diameters):
                        ax.plot(mld_idx, diameters[mld_idx], "ro", markersize=8)
                        ax.annotate(
                            f"MLD: {diameters[mld_idx]:.3f}mm",
                            xy=(mld_idx, diameters[mld_idx]),
                            xytext=(5, 5),
                            textcoords="offset points",
                        )

                    ax.set_xlabel("Position (pixels)")
                    ax.set_ylabel("Diameter (mm)")
                    ax.set_title(f"Diameter Profile - Frame {frame_idx + 1}")
                    ax.grid(True, alpha=0.3)

                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
        else:
            layout.addWidget(QLabel("Frame data not available"))

        group.setLayout(layout)
        return group

    def create_visualizations_tab(self) -> QWidget:
        """Create visualizations tab with graphs"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create scroll area for multiple plots
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 1. MLD variation across frames
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)

        mld_info = self.results.get("mld_values_by_frame", {})
        if mld_info:
            frame_indices = sorted(mld_info.keys())
            mld_values = [mld_info[idx]["mld_value"] for idx in frame_indices]
            frame_numbers = [idx + 1 for idx in frame_indices]  # 1-based for display

            ax1.plot(frame_numbers, mld_values, "b-", linewidth=2, marker="o", markersize=6)

            # Highlight min and max points
            min_frame = self.results.get("mld_min_frame")
            max_frame = self.results.get("mld_max_frame")

            if min_frame is not None and min_frame in mld_info:
                min_idx = frame_indices.index(min_frame)
                ax1.plot(frame_numbers[min_idx], mld_values[min_idx], "ro", markersize=12)
                ax1.annotate(
                    f"Min: {mld_values[min_idx]:.2f}mm",
                    xy=(frame_numbers[min_idx], mld_values[min_idx]),
                    xytext=(10, -20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            if max_frame is not None and max_frame in mld_info:
                max_idx = frame_indices.index(max_frame)
                ax1.plot(frame_numbers[max_idx], mld_values[max_idx], "go", markersize=12)
                ax1.annotate(
                    f"Max: {mld_values[max_idx]:.2f}mm",
                    xy=(frame_numbers[max_idx], mld_values[max_idx]),
                    xytext=(10, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="green", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            ax1.set_xlabel("Frame Number")
            ax1.set_ylabel("MLD Diameter (mm)")
            rws_val = self.results.get("rws_at_mld")
            title_text = (
                f"MLD Variation Across Cardiac Cycle - RWS = {rws_val}%"
                if rws_val is not None
                else "MLD Variation Across Cardiac Cycle - RWS = N/A"
            )
            ax1.set_title(title_text)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(min(frame_numbers) - 1, max(frame_numbers) + 1)

        canvas1 = FigureCanvas(fig1)
        scroll_layout.addWidget(canvas1)

        # 2. Diameter profiles comparison
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)

        diameter_profiles = self.results.get("diameter_profiles", {})
        min_frame = self.results.get("mld_min_frame")
        max_frame = self.results.get("mld_max_frame")

        if diameter_profiles and min_frame in diameter_profiles and max_frame in diameter_profiles:
            # Plot min and max frame profiles
            min_profile = diameter_profiles[min_frame]
            max_profile = diameter_profiles[max_frame]

            # Create separate position arrays for each profile since they might have different lengths
            min_positions = np.arange(len(min_profile))
            max_positions = np.arange(len(max_profile))

            ax2.plot(
                min_positions,
                min_profile,
                "r-",
                linewidth=2,
                label=f"Min MLD (Frame {min_frame + 1})",
            )
            ax2.plot(
                max_positions,
                max_profile,
                "g-",
                linewidth=2,
                label=f"Max MLD (Frame {max_frame + 1})",
            )

            # Mark MLD positions
            if min_frame in mld_info and mld_info[min_frame].get("mld_index") is not None:
                mld_idx = mld_info[min_frame]["mld_index"]
                ax2.plot(mld_idx, min_profile[mld_idx], "ro", markersize=10)

            if max_frame in mld_info and mld_info[max_frame].get("mld_index") is not None:
                mld_idx = mld_info[max_frame]["mld_index"]
                ax2.plot(mld_idx, max_profile[mld_idx], "go", markersize=10)

            ax2.set_xlabel("Position along vessel (pixels)")
            ax2.set_ylabel("Diameter (mm)")
            ax2.set_title("Diameter Profile Comparison: Min vs Max MLD Frames")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        canvas2 = FigureCanvas(fig2)
        scroll_layout.addWidget(canvas2)

        # Set scroll widget
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        return widget

    def create_detailed_data_tab(self) -> QWidget:
        """Create tab with detailed numerical data"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create text editor with detailed results
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        # Format detailed results
        detailed_text = "=== RWS ANALYSIS DETAILED RESULTS ===\n\n"

        # Basic metrics
        detailed_text += "SUMMARY METRICS:\n"
        rws_val = self.results.get("rws_at_mld")
        mld_min = self.results.get("mld_min_value")
        mld_max = self.results.get("mld_max_value")

        detailed_text += f"  RWS at MLD: {f'{rws_val}%' if rws_val is not None else 'N/A'}\n"
        detailed_text += f"  MLD Range: {f'{mld_min:.3f}' if mld_min is not None else 'N/A'} - {f'{mld_max:.3f}' if mld_max is not None else 'N/A'} mm\n"
        detailed_text += f"  MLD Variation: {f'{mld_max - mld_min:.3f}' if mld_max is not None and mld_min is not None else 'N/A'} mm\n\n"

        # Frame details
        detailed_text += "FRAME ANALYSIS:\n"
        detailed_text += f"  Total Frames Analyzed: {self.results.get('num_frames_analyzed', 0)}\n"
        detailed_text += f"  Beat Frames: {self.results.get('beat_frames', [])}\n"
        detailed_text += f"  Min MLD Frame: {self.results.get('mld_min_frame', 'N/A')}\n"
        detailed_text += f"  Max MLD Frame: {self.results.get('mld_max_frame', 'N/A')}\n\n"

        # MLD details by frame
        detailed_text += "MLD VALUES BY FRAME:\n"
        mld_info = self.results.get("mld_values_by_frame", {})
        for frame_idx in sorted(mld_info.keys()):
            info = mld_info[frame_idx]
            detailed_text += f"  Frame {frame_idx + 1}: MLD = {info['mld_value']:.3f} mm"
            if info.get("mld_index") is not None:
                detailed_text += f" at position {info['mld_index']}"
            if frame_idx == self.results.get("mld_min_frame"):
                detailed_text += " [MINIMUM]"
            elif frame_idx == self.results.get("mld_max_frame"):
                detailed_text += " [MAXIMUM]"
            detailed_text += "\n"

        # Calibration info
        detailed_text += f"\nCALIBRATION:\n"
        cal_factor = self.results.get("calibration_factor")
        detailed_text += (
            f"  Factor: {f'{cal_factor:.4f} mm/pixel' if cal_factor is not None else 'N/A'}\n"
        )

        # Timestamp
        detailed_text += f"\nANALYSIS TIMESTAMP: {self.results.get('timestamp', 'N/A')}\n"

        text_edit.setPlainText(detailed_text)
        text_edit.setFont(QFont("Courier", 10))
        layout.addWidget(text_edit)

        return widget

    def create_button_box(self) -> QWidget:
        """Create button box with export options"""
        button_box = QDialogButtonBox()

        # Export buttons
        pdf_btn = QPushButton("Generate PDF Report")
        pdf_btn.clicked.connect(self.pdf_export_requested.emit)
        button_box.addButton(pdf_btn, QDialogButtonBox.ButtonRole.ActionRole)

        excel_btn = QPushButton("Export to Excel")
        excel_btn.clicked.connect(self.excel_export_requested.emit)
        button_box.addButton(excel_btn, QDialogButtonBox.ButtonRole.ActionRole)

        db_btn = QPushButton("Save to Database")
        db_btn.clicked.connect(self.database_save_requested.emit)
        button_box.addButton(db_btn, QDialogButtonBox.ButtonRole.ActionRole)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_box.addButton(close_btn, QDialogButtonBox.ButtonRole.AcceptRole)

        return button_box

    def populate_data(self):
        """Populate dialog with analysis results"""
        # Data population is done in individual create methods
