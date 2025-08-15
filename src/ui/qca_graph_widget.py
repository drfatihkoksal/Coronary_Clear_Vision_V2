"""
QCA Graph Widget for diameter and area profile visualization
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath
import numpy as np
from typing import Dict


class QCAGraphWidget(QWidget):
    """Widget for displaying QCA diameter and area graphs"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.diameter_data = None
        self.area_data = None
        self.distances = None
        self.mld_index = None
        self.proximal_ref_index = None
        self.distal_ref_index = None
        self.unit = "mm"

        # Graph settings
        self.margin = 30
        self.grid_color = QColor(100, 100, 100)
        self.diameter_color = QColor(255, 255, 0)  # Yellow
        self.area_color = QColor(255, 255, 0)  # Yellow
        self.reference_color = QColor(0, 255, 0)  # Green
        self.mld_color = QColor(255, 0, 0)  # Red

        self.setMinimumHeight(300)
        self.setMaximumHeight(600)

    def set_data(self, profile_data: Dict, stenosis_results: Dict):
        """Set graph data from QCA analysis results"""
        if profile_data:
            self.distances = np.array(profile_data.get("distances", []))
            self.diameter_data = np.array(profile_data.get("diameters", []))
            self.area_data = np.array(profile_data.get("areas", []))
            self.unit = profile_data.get("unit", "mm")

        if stenosis_results:
            self.mld_index = stenosis_results.get("mld_index")
            self.proximal_ref_index = stenosis_results.get("proximal_ref_index")
            self.distal_ref_index = stenosis_results.get("distal_ref_index")

        self.update()

    def clear_data(self):
        """Clear all graph data"""
        self.diameter_data = None
        self.area_data = None
        self.distances = None
        self.mld_index = None
        self.proximal_ref_index = None
        self.distal_ref_index = None
        self.update()

    def paintEvent(self, event):
        """Paint the graphs"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.diameter_data is None or len(self.diameter_data) == 0:
            # Show placeholder text
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No QCA data available")
            return

        # Split view for diameter and area graphs
        graph_height = (self.height() - 3 * self.margin) // 2

        # Draw diameter graph
        diameter_rect = QRectF(
            self.margin, self.margin, self.width() - 2 * self.margin, graph_height
        )
        self._draw_graph(
            painter,
            diameter_rect,
            self.diameter_data,
            "Diameter",
            f"{self.unit}",
            self.diameter_color,
        )

        # Draw area graph
        area_rect = QRectF(
            self.margin,
            graph_height + 2 * self.margin,
            self.width() - 2 * self.margin,
            graph_height,
        )
        self._draw_graph(
            painter, area_rect, self.area_data, "Area", f"{self.unit}²", self.area_color
        )

    def _draw_graph(
        self,
        painter: QPainter,
        rect: QRectF,
        data: np.ndarray,
        title: str,
        unit: str,
        color: QColor,
    ):
        """Draw a single graph"""
        # Draw border
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.drawRect(rect)

        # Draw title
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QPen(Qt.GlobalColor.white))
        title_rect = QRectF(rect.x(), rect.y() - 20, rect.width(), 20)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft, f"{title} ({unit})")

        # Calculate scaling
        if len(data) == 0:
            return

        # Always start from 0 for better visualization
        min_val = 0.0
        max_val = np.max(data) * 1.1  # Add 10% margin at top
        value_range = max_val - min_val

        if value_range == 0:
            return

        # Draw grid
        painter.setPen(QPen(self.grid_color, 0.5, Qt.PenStyle.DotLine))

        # Horizontal grid lines (5 lines)
        for i in range(5):
            y = int(rect.bottom() - (i / 4) * rect.height())
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)

            # Value labels
            value = min_val + (i / 4) * value_range
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(
                int(rect.left() - 35),
                int(y - 5),
                30,
                10,
                Qt.AlignmentFlag.AlignRight,
                f"{value:.1f}",
            )
            painter.setPen(QPen(self.grid_color, 0.5, Qt.PenStyle.DotLine))

        # Vertical grid lines (distance markers)
        if self.distances is not None and len(self.distances) > 0:
            max_dist = self.distances[-1]
            for i in range(0, int(max_dist) + 1, 5):  # Every 5mm
                if max_dist > 0:
                    x = int(rect.left() + (i / max_dist) * rect.width())
                    painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))

                    # Distance labels
                    painter.setPen(QPen(Qt.GlobalColor.white, 1))
                    painter.drawText(
                        int(x - 10),
                        int(rect.bottom() + 5),
                        20,
                        15,
                        Qt.AlignmentFlag.AlignCenter,
                        str(i),
                    )
                    painter.setPen(QPen(self.grid_color, 0.5, Qt.PenStyle.DotLine))

        # Draw reference lines if available
        if self.proximal_ref_index is not None and self.distal_ref_index is not None:
            painter.setPen(QPen(self.reference_color, 1, Qt.PenStyle.DashLine))

            # Calculate reference value as average of proximal and distal
            if self.proximal_ref_index < len(data) and self.distal_ref_index < len(data):
                ref_value = (data[self.proximal_ref_index] + data[self.distal_ref_index]) / 2
                y_ref = rect.bottom() - ((ref_value - min_val) / value_range) * rect.height()
                painter.drawLine(int(rect.left()), int(y_ref), int(rect.right()), int(y_ref))

        # Draw data curve
        painter.setPen(QPen(color, 2))

        path = QPainterPath()
        for i in range(len(data)):
            x = rect.left() + (i / (len(data) - 1)) * rect.width() if len(data) > 1 else rect.left()
            y = rect.bottom() - ((data[i] - min_val) / value_range) * rect.height()

            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.drawPath(path)

        # Mark special points
        if self.mld_index is not None and self.mld_index < len(data):
            # MLD point
            x_mld = (
                rect.left() + (self.mld_index / (len(data) - 1)) * rect.width()
                if len(data) > 1
                else rect.left()
            )
            y_mld = rect.bottom() - ((data[self.mld_index] - min_val) / value_range) * rect.height()

            painter.setPen(QPen(self.mld_color, 3))
            painter.drawEllipse(int(x_mld - 3), int(y_mld - 3), 6, 6)

        # Skip drawing reference diameter markers (P and D) - removed as per requirements
        # The reference diameters are now shown only in the numeric display
