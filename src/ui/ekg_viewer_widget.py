"""
EKG Viewer Widget
MVP Phase 2: ECG signal visualization and interaction
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolTip, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QAction, QCursor
import numpy as np
from typing import Optional, Tuple


class EKGViewerWidget(QWidget):
    """Widget for displaying and interacting with EKG data"""

    # Signals
    time_clicked = pyqtSignal(float)  # Emitted when user clicks on EKG with time value
    r_peak_clicked = pyqtSignal(int)  # Emitted when an R-peak is detected/clicked

    def __init__(self):
        super().__init__()
        self.ekg_data: Optional[np.ndarray] = None
        self.time_data: Optional[np.ndarray] = None
        self.sampling_rate: float = 1000.0  # Default 1000 Hz
        self.current_time_marker: Optional[pg.InfiniteLine] = None
        self.r_peaks: Optional[np.ndarray] = None
        self.cardiac_phases: Optional[dict] = None
        self.phase_markers = []  # Store phase marker items
        self.show_phases = False
        self.phase_hover_enabled = False
        self.current_hover_phase = None
        self.editable_phase_markers = {}  # Store editable phase markers
        self.selected_phase = None

        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)  # Reduce spacing

        # Control panel at top
        control_panel = QWidget()
        control_panel.setMaximumHeight(40)  # Increased height
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # Title label
        title_label = QLabel("ECG Signal")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        control_layout.addWidget(title_label)

        # Info label
        self.info_label = QLabel("No EKG data loaded")
        self.info_label.setStyleSheet("font-size: 12px;")
        control_layout.addWidget(self.info_label)

        # Quality indicator
        self.quality_label = QLabel("")
        self.quality_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        control_layout.addWidget(self.quality_label)

        control_layout.addStretch()

        # Toggle phases button
        self.toggle_phases_button = QPushButton("Show Phases")
        self.toggle_phases_button.clicked.connect(self.toggle_cardiac_phases)
        self.toggle_phases_button.setEnabled(False)
        self.toggle_phases_button.setMinimumHeight(30)
        self.toggle_phases_button.setStyleSheet(
            "QPushButton { font-size: 12px; padding: 5px 10px; }"
        )
        control_layout.addWidget(self.toggle_phases_button)

        # Clear markers button
        self.clear_button = QPushButton("Clear Markers")
        self.clear_button.clicked.connect(self.clear_markers)
        self.clear_button.setEnabled(False)
        self.clear_button.setMinimumHeight(30)
        self.clear_button.setStyleSheet("QPushButton { font-size: 12px; padding: 5px 10px; }")
        control_layout.addWidget(self.clear_button)

        layout.addWidget(control_panel)

        # Create plot widget with custom configuration
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Configure plot with minimal margins
        self.plot_widget.setLabel("left", "mV", units="")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.plotItem.setContentsMargins(0, 0, 0, 0)

        # Increase axis label font sizes
        font = self.plot_widget.getAxis("left").label.font()
        font.setPointSize(12)
        self.plot_widget.getAxis("left").label.setFont(font)
        self.plot_widget.getAxis("left").setWidth(60)  # More space for labels
        self.plot_widget.getAxis("left").setTickFont(font)

        self.plot_widget.getAxis("bottom").label.setFont(font)
        self.plot_widget.getAxis("bottom").setHeight(40)  # More space for labels
        self.plot_widget.getAxis("bottom").setTickFont(font)

        # Enable mouse interaction
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)

        # Add crosshair
        self.vline = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("b", width=1, style=Qt.PenStyle.DashLine)
        )
        self.hline = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("b", width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self.vline)
        self.plot_widget.addItem(self.hline)

        # Connect proxy for mouse move
        self.proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_moved
        )

        layout.addWidget(self.plot_widget)

    def set_ekg_data(
        self,
        data: np.ndarray,
        sampling_rate: float = 1000.0,
        r_peaks: Optional[np.ndarray] = None,
        quality_metrics: Optional[dict] = None,
    ):
        """Set EKG data for display"""
        self.ekg_data = data
        self.sampling_rate = sampling_rate
        self.r_peaks = r_peaks

        # Create time axis
        self.time_data = np.arange(len(data)) / sampling_rate

        # Update quality indicator
        if quality_metrics:
            self.update_quality_indicator(quality_metrics)

        # Clear previous plot
        self.plot_widget.clear()

        # Plot EKG data with thicker line
        self.ekg_curve = self.plot_widget.plot(
            self.time_data, self.ekg_data, pen=pg.mkPen("k", width=2)
        )

        # Plot R-peaks if available - make them editable
        self.r_peak_items = []
        if self.r_peaks is not None and len(self.r_peaks) > 0:
            peak_times = self.r_peaks / sampling_rate
            peak_values = self.ekg_data[self.r_peaks]

            # Create individual scatter plot items for each R-peak
            for i, (time, value) in enumerate(zip(peak_times, peak_values)):
                scatter = pg.ScatterPlotItem(
                    x=[time],
                    y=[value],
                    pen=None,
                    symbol="o",
                    symbolBrush="r",
                    symbolSize=12,
                    symbolPen=pg.mkPen("darkred", width=2),
                )
                self.plot_widget.addItem(scatter)
                self.r_peak_items.append(scatter)

            # Don't emit signals for all peaks on initial load
            # Only emit when user interacts or when playing

        # Re-add crosshair lines
        self.plot_widget.addItem(self.vline)
        self.plot_widget.addItem(self.hline)

        # Update info
        duration = len(data) / sampling_rate
        self.info_label.setText(f"Duration: {duration:.2f}s | Rate: {sampling_rate:.0f}Hz")

        # Enable controls
        self.clear_button.setEnabled(True)

        # Auto-range
        self.plot_widget.autoRange()

    def update_quality_indicator(self, quality_metrics: dict):
        """Update quality indicator based on metrics"""
        quality_score = quality_metrics.get("quality_score", 0)
        message = quality_metrics.get("message", "")

        # Set color based on quality
        if quality_score >= 0.8:
            color = "#4CAF50"  # Green
            symbol = "✓"
        elif quality_score >= 0.6:
            color = "#FF9800"  # Orange
            symbol = "!"
        else:
            color = "#F44336"  # Red
            symbol = "✗"

        # Update label
        self.quality_label.setText(f"{symbol} Quality: {quality_score:.2f}")
        self.quality_label.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {color};")
        self.quality_label.setToolTip(message)

    def on_mouse_moved(self, evt):
        """Handle mouse movement for crosshair and phase hover"""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.vline.setPos(mouse_point.x())
            self.hline.setPos(mouse_point.y())

            # Check if hovering over a phase
            if self.show_phases and self.cardiac_phases:
                self._check_phase_hover(mouse_point.x())

    def on_mouse_clicked(self, evt):
        """Handle mouse click events"""
        if evt.button() == Qt.MouseButton.LeftButton:
            pos = evt.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                time_clicked = mouse_point.x()

                # Ensure time is within valid range
                if self.time_data is not None:
                    if 0 <= time_clicked <= self.time_data[-1]:
                        # Add or update time marker
                        self.set_time_marker(time_clicked)
                        # Emit signal
                        self.time_clicked.emit(time_clicked)

        elif evt.button() == Qt.MouseButton.RightButton:
            pos = evt.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                self.show_context_menu(evt.screenPos(), mouse_point)

    def set_time_marker(self, time: float):
        """Set a vertical marker at the specified time"""
        # Remove previous marker if exists
        if self.current_time_marker is not None:
            self.plot_widget.removeItem(self.current_time_marker)

        # Create new marker
        self.current_time_marker = pg.InfiniteLine(
            pos=time, angle=90, pen=pg.mkPen("r", width=2), movable=False
        )
        self.plot_widget.addItem(self.current_time_marker)

    def clear_markers(self):
        """Clear all markers"""
        if self.current_time_marker is not None:
            self.plot_widget.removeItem(self.current_time_marker)
            self.current_time_marker = None

        # Also clear phase markers if any
        if self.show_phases:
            self._clear_phase_markers()
            self.show_phases = False
            self.toggle_phases_button.setText("Show Phases")

        # Clear phase data
        self.cardiac_phases = None
        self.toggle_phases_button.setEnabled(False)
        self.editable_phase_markers.clear()

    def get_visible_range(self) -> Tuple[float, float]:
        """Get the currently visible time range"""
        vb = self.plot_widget.plotItem.vb
        xrange = vb.viewRange()[0]
        return xrange[0], xrange[1]

    def set_visible_range(self, start: float, end: float):
        """Set the visible time range"""
        self.plot_widget.setXRange(start, end)

    def set_cardiac_phases(self, phases: dict):
        """Set cardiac phase data"""
        self.cardiac_phases = phases
        self.toggle_phases_button.setEnabled(True)

        # Update button text based on current state
        if self.show_phases:
            self._display_cardiac_phases()

    def toggle_cardiac_phases(self):
        """Toggle display of cardiac phases"""
        self.show_phases = not self.show_phases

        if self.show_phases:
            self.toggle_phases_button.setText("Hide Phases")
            self._display_cardiac_phases()
        else:
            self.toggle_phases_button.setText("Show Phases")
            self._clear_phase_markers()

    def _display_cardiac_phases(self):
        """Display cardiac phase markers on the ECG"""
        if not self.cardiac_phases or "phases" not in self.cardiac_phases:
            return

        # Clear existing phase markers
        self._clear_phase_markers()

        # Define colors for each phase
        phase_colors = {
            "D2": {"color": "#4CAF50", "name": "End-diastole"},  # Green
            "S1": {"color": "#2196F3", "name": "Early-systole"},  # Blue
            "S2": {"color": "#FF9800", "name": "End-systole"},  # Orange
            "D1": {"color": "#9C27B0", "name": "Mid-diastole"},  # Purple
        }

        # Plot phase markers for each cardiac cycle
        for cycle_idx, cycle in enumerate(self.cardiac_phases["phases"]):
            for phase_key, phase_info in phase_colors.items():
                if phase_key in cycle:
                    phase_data = cycle[phase_key]

                    # Add vertical line at phase time - now movable with thick handle
                    phase_line = pg.InfiniteLine(
                        pos=phase_data["time"],
                        angle=90,
                        pen=pg.mkPen(phase_info["color"], width=6, style=Qt.PenStyle.DashLine),
                        movable=True,
                        hoverPen=pg.mkPen(phase_info["color"], width=8),
                    )

                    # Store phase info in the line object
                    phase_line.phase_key = phase_key
                    phase_line.cycle_idx = cycle_idx

                    # Connect drag signal
                    phase_line.sigPositionChanged.connect(self._on_phase_moved)

                    self.plot_widget.addItem(phase_line)
                    self.phase_markers.append(phase_line)

                    # Store in editable markers dict
                    marker_key = f"{cycle_idx}_{phase_key}"
                    self.editable_phase_markers[marker_key] = phase_line

                    # Add phase label at top of plot
                    if cycle_idx == 0:  # Only label first cycle to avoid clutter
                        label = pg.TextItem(phase_key, color=phase_info["color"], anchor=(0.5, 1))
                        label.setPos(phase_data["time"], self._get_plot_top())
                        self.plot_widget.addItem(label)
                        self.phase_markers.append(label)

        # Add phase regions (systole/diastole shading)
        for cycle_idx, cycle in enumerate(self.cardiac_phases["phases"]):
            if "S1" in cycle and "S2" in cycle:
                # Systole region (S1 to S2)
                systole_region = pg.LinearRegionItem(
                    [cycle["S1"]["time"], cycle["S2"]["time"]],
                    brush=pg.mkBrush("#FF5252", alpha=15),
                    movable=False,
                    pen=pg.mkPen(None),
                )
                systole_region.setZValue(-10)  # Put behind other elements
                self.plot_widget.addItem(systole_region)
                self.phase_markers.append(systole_region)

            if "S2" in cycle and cycle_idx < len(self.cardiac_phases["phases"]) - 1:
                # Diastole region (S2 to next D2)
                next_cycle = self.cardiac_phases["phases"][cycle_idx + 1]
                if "D2" in next_cycle:
                    diastole_region = pg.LinearRegionItem(
                        [cycle["S2"]["time"], next_cycle["D2"]["time"]],
                        brush=pg.mkBrush("#448AFF", alpha=10),
                        movable=False,
                        pen=pg.mkPen(None),
                    )
                    diastole_region.setZValue(-10)  # Put behind other elements
                    self.plot_widget.addItem(diastole_region)
                    self.phase_markers.append(diastole_region)

    def _clear_phase_markers(self):
        """Clear all phase markers from plot"""
        for marker in self.phase_markers:
            self.plot_widget.removeItem(marker)
        self.phase_markers.clear()
        self.editable_phase_markers.clear()

    def _get_plot_top(self):
        """Get the top y-value of the current plot range"""
        if self.ekg_data is not None:
            return np.max(self.ekg_data) * 1.1
        return 1.0

    def _check_phase_hover(self, mouse_time: float):
        """Check if mouse is hovering over a phase marker"""
        if not self.cardiac_phases or "phases" not in self.cardiac_phases:
            return

        # Define phase info (transition descriptions)
        phase_info = {
            "D2": "D2→S1: End-diastole phase - Ventricular filling complete",
            "S1": "S1→S2: Early-systole phase - Ventricular contraction begins",
            "S2": "S2→D1: End-systole phase - Ventricular contraction ends",
            "D1": "D1→D2: Mid-diastole phase - Ventricular relaxation and filling",
        }

        hover_tolerance = 0.01  # 10ms tolerance
        found_phase = None

        # Check all phases in all cycles
        for cycle in self.cardiac_phases["phases"]:
            for phase_key in ["D2", "S1", "S2", "D1"]:
                if phase_key in cycle:
                    phase_time = cycle[phase_key]["time"]
                    if abs(mouse_time - phase_time) < hover_tolerance:
                        found_phase = phase_key
                        break
            if found_phase:
                break

        # Show/hide tooltip
        if found_phase and found_phase != self.current_hover_phase:
            self.current_hover_phase = found_phase

            # Get phase statistics
            stats = self.cardiac_phases.get("statistics", {})
            info_text = phase_info.get(found_phase, "")

            # Add timing info
            if found_phase == "D2":
                offset = stats.get("D2_offset_ms_mean", 0)
                info_text += f"\nOffset: {offset:.1f}ms before R-peak"
            elif found_phase == "S2":
                offset = stats.get("S2_offset_ms_mean", 0)
                info_text += f"\nOffset: {offset:.1f}ms after R-peak"
            elif found_phase == "D1":
                offset = stats.get("D1_offset_ms_mean", 0)
                info_text += f"\nOffset: {offset:.1f}ms after R-peak"

            # Show tooltip at cursor position
            QToolTip.showText(QCursor.pos(), info_text)

        elif not found_phase and self.current_hover_phase:
            self.current_hover_phase = None
            QToolTip.hideText()

    def reset(self):
        """Reset widget state for new file"""
        # Clear all data
        self.ekg_data = None
        self.time_data = None
        self.r_peaks = None
        self.cardiac_phases = None

        # Clear plot
        self.plot_widget.clear()

        # Re-add crosshair lines
        self.plot_widget.addItem(self.vline)
        self.plot_widget.addItem(self.hline)

        # Clear markers
        self.clear_markers()

        # Reset UI elements
        self.info_label.setText("No EKG data loaded")
        self.quality_label.setText("")
        self.clear_button.setEnabled(False)
        self.toggle_phases_button.setEnabled(False)
        self.show_phases = False
        self.toggle_phases_button.setText("Show Phases")

    def _on_phase_moved(self, line):
        """Handle phase marker movement"""
        if hasattr(line, "phase_key") and hasattr(line, "cycle_idx"):
            new_time = line.pos()[0]
            phase_key = line.phase_key
            cycle_idx = line.cycle_idx

            # Update the phase data
            if self.cardiac_phases and cycle_idx < len(self.cardiac_phases["phases"]):
                cycle = self.cardiac_phases["phases"][cycle_idx]
                if phase_key in cycle:
                    # Convert time to index
                    new_index = int(new_time * self.sampling_rate)
                    cycle[phase_key]["time"] = new_time
                    cycle[phase_key]["index"] = new_index

                    # Recalculate statistics would go here

    def show_context_menu(self, screen_pos, plot_pos):
        """Show context menu for phase operations"""
        menu = QMenu(self)

        # Check if we're near an existing phase
        near_phase = self._find_nearest_phase(plot_pos.x())

        if near_phase:
            delete_action = QAction(f"Delete {near_phase['phase_key']} marker", self)
            delete_action.triggered.connect(lambda: self._delete_phase(near_phase))
            menu.addAction(delete_action)

        # Check if we're near an R-peak
        near_r_peak = self._find_nearest_r_peak(plot_pos.x())
        if near_r_peak is not None:
            delete_r_action = QAction(f"Delete R-peak", self)
            delete_r_action.triggered.connect(lambda: self._delete_r_peak(near_r_peak))
            menu.addAction(delete_r_action)

        # Add new phase marker options
        menu.addSeparator()
        add_menu = menu.addMenu("Add Phase Marker")

        phases = ["D2", "S1", "S2", "D1"]
        for phase in phases:
            action = QAction(f"Add {phase}", self)
            action.triggered.connect(
                lambda checked, p=phase: self._add_phase_at_time(p, plot_pos.x())
            )
            add_menu.addAction(action)

        # Add R-peak option
        menu.addSeparator()
        add_r_action = QAction("Add R-peak", self)
        add_r_action.triggered.connect(lambda: self._add_r_peak_at_time(plot_pos.x()))
        menu.addAction(add_r_action)

        # Delete all markers
        if self.phase_markers or (self.r_peaks is not None and len(self.r_peaks) > 0):
            menu.addSeparator()
            clear_action = QAction("Clear All Markers", self)
            clear_action.triggered.connect(self.clear_markers)
            menu.addAction(clear_action)

        # Convert to QPoint
        menu.exec(QPoint(int(screen_pos.x()), int(screen_pos.y())))

    def _find_nearest_phase(self, time, tolerance=0.02):
        """Find the nearest phase marker to given time"""
        if not self.cardiac_phases:
            return None

        for cycle_idx, cycle in enumerate(self.cardiac_phases["phases"]):
            for phase_key in ["D2", "S1", "S2", "D1"]:
                if phase_key in cycle:
                    phase_time = cycle[phase_key]["time"]
                    if abs(phase_time - time) < tolerance:
                        return {"phase_key": phase_key, "cycle_idx": cycle_idx, "time": phase_time}
        return None

    def _delete_phase(self, phase_info):
        """Delete a phase marker"""
        marker_key = f"{phase_info['cycle_idx']}_{phase_info['phase_key']}"
        if marker_key in self.editable_phase_markers:
            marker = self.editable_phase_markers[marker_key]
            self.plot_widget.removeItem(marker)
            del self.editable_phase_markers[marker_key]

            # Remove from phase markers list
            if marker in self.phase_markers:
                self.phase_markers.remove(marker)

            # Update phase data
            if phase_info["cycle_idx"] < len(self.cardiac_phases["phases"]):
                cycle = self.cardiac_phases["phases"][phase_info["cycle_idx"]]
                if phase_info["phase_key"] in cycle:
                    del cycle[phase_info["phase_key"]]

    def _add_phase_at_time(self, phase_key, time):
        """Add a new phase marker at given time"""
        if not self.cardiac_phases:
            return

        # Find which cycle this time belongs to
        cycle_idx = 0
        for i, cycle in enumerate(self.cardiac_phases["phases"]):
            if "S1" in cycle and time >= cycle["S1"]["time"]:
                cycle_idx = i

        # Add to phase data
        if cycle_idx < len(self.cardiac_phases["phases"]):
            cycle = self.cardiac_phases["phases"][cycle_idx]
            cycle[phase_key] = {
                "index": int(time * self.sampling_rate),
                "time": time,
                "phase_name": phase_key,
            }

            # Refresh display
            self._display_cardiac_phases()

    def _add_r_peak_at_time(self, time):
        """Add a new R-peak at given time"""
        if self.ekg_data is None:
            return

        index = int(time * self.sampling_rate)
        if 0 <= index < len(self.ekg_data):
            # Add to R-peaks array
            if self.r_peaks is None:
                self.r_peaks = np.array([index])
            else:
                self.r_peaks = np.sort(np.append(self.r_peaks, index))

            # Refresh display
            self.set_ekg_data(self.ekg_data, self.sampling_rate, self.r_peaks)

    def _find_nearest_r_peak(self, time, tolerance=0.02):
        """Find the nearest R-peak to given time"""
        if self.r_peaks is None or len(self.r_peaks) == 0:
            return None

        peak_times = self.r_peaks / self.sampling_rate
        for i, peak_time in enumerate(peak_times):
            if abs(peak_time - time) < tolerance:
                return i
        return None

    def _delete_r_peak(self, peak_idx):
        """Delete an R-peak"""
        if self.r_peaks is not None and 0 <= peak_idx < len(self.r_peaks):
            self.r_peaks = np.delete(self.r_peaks, peak_idx)

            # Refresh display and recalculate phases
            self.set_ekg_data(self.ekg_data, self.sampling_rate, self.r_peaks)

            # Recalculate cardiac phases if they exist
            if self.cardiac_phases is not None:
                from ..core.cardiac_phase_detector import CardiacPhaseDetector

                phase_detector = CardiacPhaseDetector(self.sampling_rate)
                self.cardiac_phases = phase_detector.detect_phases(self.ekg_data, self.r_peaks)
                if self.show_phases:
                    self._display_cardiac_phases()
