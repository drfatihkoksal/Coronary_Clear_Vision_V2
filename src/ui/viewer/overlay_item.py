"""Overlay graphics item for DICOM viewer"""

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QFont, QImage, QPainter,
                         QPainterPath, QPen)
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem
import logging

logger = logging.getLogger(__name__)


class OverlayItem(QGraphicsItem):
    """Custom graphics item for rendering overlays on DICOM images"""

    def __init__(self, viewer=None):
        super().__init__()
        self.viewer = viewer  # Reference to viewer widget
        self.segmentation_mask = None
        self.segmentation_settings = {}
        self.segmentation_centerline = None
        self.segmentation_boundaries = None
        self.qca_results = None
        self.qca_settings = {}
        self.user_points = []
        self.frame_points = {}  # Store points per frame: {frame_index: [(x,y), ...]}
        self.current_frame_index = 0

        # Calibration dialog specific items
        self.calibration_points = []  # Two points for calibration
        self.calibration_mask = None  # Segmentation mask for calibration
        self.calibration_centerline = None  # Centerline for automatic calibration
        self.calibration_diameters = None  # Diameter measurements along centerline
        self.calibration_left_edges = None  # Left edge distances from centerline
        self.calibration_right_edges = None  # Right edge distances from centerline  
        self.calibration_perpendiculars = None  # Perpendicular vectors at each point
        self.show_calibration_centerline = False  # Control centerline visibility
        self.show_calibration_diameters = False  # Control diameter measurements visibility

        # Visibility flags
        self.show_points = True
        self.show_segmentation = True
        self.show_qca = True

        # Make sure this item doesn't accept mouse events
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self) -> QRectF:
        """Return bounding rect of the overlay"""
        # Since we're now a child of pixmap_item, use parent's bounds
        parent = self.parentItem()
        if parent and isinstance(parent, QGraphicsPixmapItem) and parent.pixmap():
            return QRectF(0, 0, parent.pixmap().width(), parent.pixmap().height())
        
        # Use segmentation mask bounds if available
        if self.segmentation_mask is not None:
            h, w = self.segmentation_mask.shape
            return QRectF(0, 0, w, h)

        # Default large rect
        return QRectF(0, 0, 1000, 1000)

    def paint(self, painter: QPainter, option, widget):
        """Paint all overlays"""
        # Draw calibration items first (background)
        if self.calibration_mask is not None and self.show_segmentation:
            self.paint_calibration_mask(painter)
        
        # Draw automatic calibration centerline if available
        if self.calibration_centerline is not None and self.show_calibration_centerline:
            self.paint_calibration_centerline(painter)
        
        # Draw automatic calibration diameter measurements if available
        if (self.calibration_centerline is not None and 
            self.calibration_diameters is not None and 
            self.show_calibration_diameters):
            self.paint_calibration_diameters(painter)

        # Draw user points if any exist for current frame
        if self.show_points and self.current_frame_index in self.frame_points and self.frame_points[self.current_frame_index]:
            # Check if we're in calibration mode by looking at calibration_points
            if hasattr(self, 'calibration_points') and self.calibration_points:
                # Use calibration painting style
                self.paint_calibration_points(painter)
            else:
                # Use regular segmentation painting style
                self.paint_user_points(painter)

        # Draw segmentation overlay
        if self.segmentation_mask is not None and self.segmentation_settings.get('enabled', False) and self.show_segmentation:
            self.paint_segmentation(painter)

        # Draw QCA overlay
        if self.show_qca and self.qca_results and self.qca_settings.get('enabled', False):
            self.paint_qca(painter)

    def paint_segmentation(self, painter: QPainter):
        """Paint segmentation overlay"""
        if self.segmentation_mask is None:
            return

        # Save painter state to restore later
        painter.save()
        
        # Ensure segmentation_settings is a dictionary
        if not isinstance(self.segmentation_settings, dict):
            self.segmentation_settings = {}

        opacity = self.segmentation_settings.get('opacity', 0.5)
        color_name = self.segmentation_settings.get('color', 'Red')
        
        # Debug: Check color_name type
        if not isinstance(color_name, str):
            logger.warning(f"Invalid color type: {type(color_name)}, value: {color_name}. Using default 'Red'")
            # Handle list color format [R, G, B]
            if isinstance(color_name, list) and len(color_name) == 3:
                # Convert RGB list to color name
                if color_name == [255, 0, 0]:
                    color_name = 'Red'
                elif color_name == [0, 255, 0]:
                    color_name = 'Green'
                elif color_name == [0, 0, 255]:
                    color_name = 'Blue'
                else:
                    color_name = 'Red'  # Default
            else:
                color_name = 'Red'
            
        contour_only = self.segmentation_settings.get('contour_only', False)

        # Color mapping
        color_map = {
            'Red': QColor(255, 0, 0),
            'Green': QColor(0, 255, 0),
            'Blue': QColor(0, 0, 255),
            'Yellow': QColor(255, 255, 0),
            'Cyan': QColor(0, 255, 255),
            'Magenta': QColor(255, 0, 255),
            'PistachioGreen': QColor(147, 197, 114)
        }
        color = color_map.get(color_name, QColor(255, 0, 0))

        if contour_only:
            # Draw only contours
            vessel_mask = (self.segmentation_mask == 1) | (self.segmentation_mask == 255)
            if np.any(vessel_mask):
                contours, _ = cv2.findContours(vessel_mask.astype(np.uint8),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                pen = QPen(color, 2)
                pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                for contour in contours:
                    path = QPainterPath()
                    points = contour.squeeze()
                    if len(points) > 2:
                        path.moveTo(points[0][0], points[0][1])
                        for point in points[1:]:
                            path.lineTo(point[0], point[1])
                        path.closeSubpath()
                        painter.drawPath(path)
        else:
            # Draw filled overlay
            h, w = self.segmentation_mask.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Check if mask has multiple values
            unique_values = np.unique(self.segmentation_mask)
            if len(unique_values) > 2:  # More than just 0 and 255/1
                # Handle multi-level mask
                vessel_mask = (self.segmentation_mask == 1) | (self.segmentation_mask == 255)
                overlay[vessel_mask] = [color.red(), color.green(), color.blue(), int(255 * opacity)]
            else:
                # Single color mask
                overlay[self.segmentation_mask > 0] = [color.red(), color.green(), color.blue(), int(255 * opacity)]

            qimage = QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            painter.drawImage(0, 0, qimage)

        # Draw centerline if available
        if hasattr(self, 'segmentation_centerline') and self.segmentation_centerline is not None:
            pen = QPen(QColor(255, 255, 0), 2)  # Yellow centerline
            painter.setPen(pen)

            points = self.segmentation_centerline
            if len(points) > 1:
                path = QPainterPath()
                # Centerline is in (y,x) format, convert to (x,y) for drawing
                path.moveTo(points[0][1], points[0][0])
                for point in points[1:]:
                    path.lineTo(point[1], point[0])
                painter.drawPath(path)
        
        # Restore painter state
        painter.restore()

    def paint_qca(self, painter: QPainter):
        """Paint QCA overlay including stenosis markers and diameter measurements"""
        settings = self.qca_settings
        results = self.qca_results

        if not results or not results.get('success'):
            return

        # Draw stenosis markers
        if settings.get('show_stenosis', True):
            self._paint_stenosis_markers(painter, results)
            
        # Draw MLD location
        if 'mld_location' in results and results['mld_location'] is not None:
            self._paint_mld_location(painter, results)

        # Draw diameter measurements along vessel if enabled
        if settings.get('show_diameter', False):
            self._paint_diameter_measurements(painter, results)

    def _paint_stenosis_markers(self, painter: QPainter, results: dict):
        """Paint P, D, and MLD markers for stenosis boundaries"""
        if 'stenosis_boundaries' not in results:
            return
            
        boundaries = results['stenosis_boundaries']
        centerline = results.get('centerline', [])
        
        if not boundaries or len(centerline) == 0:
            return
            
        # P point (proximal boundary) - green marker
        p_idx = boundaries.get('p_point', 0)
        if 0 <= p_idx < len(centerline):
            p_point = centerline[p_idx]
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0, 150)))
            painter.drawEllipse(QPointF(p_point[1], p_point[0]), 4, 4)
            
            # Label
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(p_point[1] + 8, p_point[0] - 5, "P")
        
        # D point (distal boundary) - blue marker
        d_idx = boundaries.get('d_point', 0)
        if 0 <= d_idx < len(centerline):
            d_point = centerline[d_idx]
            painter.setPen(QPen(QColor(0, 100, 255), 2))
            painter.setBrush(QBrush(QColor(0, 100, 255, 150)))
            painter.drawEllipse(QPointF(d_point[1], d_point[0]), 4, 4)
            
            # Label
            painter.setPen(QPen(QColor(0, 100, 255), 1))
            painter.drawText(d_point[1] + 8, d_point[0] - 5, "D")
        
        # MLD/Throat point - red marker
        mld_idx = boundaries.get('mld_point', 0)
        if 0 <= mld_idx < len(centerline):
            mld_point = centerline[mld_idx]
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0, 150)))
            painter.drawEllipse(QPointF(mld_point[1], mld_point[0]), 4, 4)
            
            # Label
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(mld_point[1] + 8, mld_point[0] - 5, "MLD")
        
        # Draw reference diameter info
        ref_diam = boundaries.get('reference_diameter', 0)
        threshold = boundaries.get('threshold', 0)
        if ref_diam > 0:
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawText(10, 20, f"Ref: {ref_diam:.1f}mm (75%: {threshold:.1f}mm)")

    def _paint_mld_location(self, painter: QPainter, results: dict):
        """Paint MLD location marker and stenosis percentage"""
        mld_loc = results['mld_location']

        # Stenosis marker - small red square
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawRect(mld_loc[1] - 2, mld_loc[0] - 2, 4, 4)

        # Draw stenosis percentage
        if 'percent_stenosis' in results:
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)

            text = f"{results['percent_stenosis']:.1f}%"
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(QPointF(mld_loc[1] + 5, mld_loc[0] - 5), text)

    def _paint_diameter_measurements(self, painter: QPainter, results: dict):
        """Paint diameter measurement lines along the vessel"""
        if not hasattr(self, 'segmentation_centerline') or self.segmentation_centerline is None:
            logger.warning("No centerline available for diameter visualization")
            return
            
        centerline = self.segmentation_centerline
        # Try different keys for diameter data
        diameters = results.get('diameters') or results.get('diameters_px') or results.get('diameters_pixels')
        
        if diameters is None:
            logger.warning("No diameter data found in QCA results")
            return

        # Sample every Nth point to avoid clutter
        num_points = min(len(centerline), len(diameters))
        line_sample_interval = max(1, num_points // 30)  # Show approximately 30 lines
        
        logger.info(f"Diameter visualization: centerline={len(centerline)} points, diameters={len(diameters)} values, interval={line_sample_interval}")
        
        drawn_count = 0
        for i in range(0, num_points, line_sample_interval):
            if self._draw_diameter_line(painter, i, centerline, results):
                drawn_count += 1
        
        logger.info(f"Drew {drawn_count} diameter lines")

    def _draw_diameter_line(self, painter: QPainter, index: int, centerline: list, results: dict) -> bool:
        """Draw a single diameter line at given index"""
        if index <= 0 or index >= len(centerline) - 1:
            return False
            
        point = centerline[index]
        
        # Calculate perpendicular direction
        window = min(5, min(index, len(centerline) - 1 - index))
        
        # Centerline is in [y, x] format
        prev_point = centerline[max(0, index-window)]
        next_point = centerline[min(len(centerline)-1, index+window)]
        
        # Calculate tangent
        tangent_y = next_point[0] - prev_point[0]
        tangent_x = next_point[1] - prev_point[1]
        
        # Normalize tangent
        norm = np.sqrt(tangent_y**2 + tangent_x**2)
        if norm == 0:
            return False
            
        tangent_y /= norm
        tangent_x /= norm
        
        # Get edge information if available
        if ('left_edges' in results and 'right_edges' in results and 
            'perpendiculars' in results and index < len(results['left_edges'])):
            # Use exact edge positions from QCA
            left_edge = results['left_edges'][index]
            right_edge = results['right_edges'][index]
            perp = results['perpendiculars'][index]
            perp_y = perp[0]
            perp_x = perp[1]
        else:
            # Use simple perpendicular
            perp_y = -tangent_x
            perp_x = tangent_y
            
            # Get diameter
            diameters = results.get('diameters') or results.get('diameters_px') or results.get('diameters_pixels')
            if diameters and index < len(diameters):
                diameter_pixels = diameters[index]
            else:
                return False
                
            # Use half diameter as edge distances
            left_edge = diameter_pixels / 2
            right_edge = diameter_pixels / 2

        # Draw diameter line
        pen = QPen(QColor(255, 255, 0, 255), 2)  # Bright yellow, opaque, thicker
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        # point is in [y, x] format
        center_y = point[0]
        center_x = point[1]
        
        # Calculate endpoints
        p1 = QPointF(
            center_x - perp_x * left_edge,
            center_y - perp_y * left_edge
        )
        p2 = QPointF(
            center_x + perp_x * right_edge,
            center_y + perp_y * right_edge
        )
        painter.drawLine(p1, p2)
        return True

    def paint_user_points(self, painter: QPainter):
        """Paint user click points with tracking region visualization"""
        # Get points for current frame
        points = self.frame_points.get(self.current_frame_index, [])

        # Also paint user_points if available (for multi-frame mode)
        if self.user_points:
            points = self.user_points

        # Paint reference points (first and last) differently
        if len(points) >= 1:
            # Define which points to label
            labeled_points = [(0, points[0])]  # Always include first point
            if len(points) >= 2:
                labeled_points.append((len(points)-1, points[-1]))  # Add last point if different
            
            for idx, (point_idx, (x, y)) in enumerate(labeled_points):
                # Convert to int to avoid numpy.float32 type error
                x, y = int(x), int(y)
                # Draw 4x4 square like AngioPy encoding (green channel)
                painter.setPen(QPen(QColor(0, 255, 0), 1))  # Green, 1 pixel
                painter.setBrush(QBrush(QColor(0, 255, 0, 180)))  # Semi-transparent green fill
                painter.drawRect(x - 2, y - 2, 4, 4)  # 4x4 square centered at point

        # Paint additional guide points (all points except first and last)
        if len(points) > 2:
            # Draw 4x4 squares like AngioPy encoding (blue channel)
            painter.setPen(QPen(QColor(0, 0, 255), 1))  # Blue, 1 pixel
            painter.setBrush(QBrush(QColor(0, 0, 255, 180)))  # Semi-transparent blue fill
            for i in range(1, len(points) - 1):  # Skip first and last
                x, y = points[i]
                x, y = int(x), int(y)  # Convert to int
                painter.drawRect(x - 2, y - 2, 4, 4)  # 4x4 square centered at point

    def paint_calibration_points(self, painter: QPainter):
        """Paint calibration click points - same style as segmentation"""
        for i, (x, y) in enumerate(self.calibration_points[:2]):
            x, y = int(x), int(y)  # Convert to int
            # Draw 4x4 square for calibration point
            painter.setPen(QPen(QColor(255, 0, 0), 1))  # Red, 1 pixel
            painter.setBrush(QBrush(QColor(255, 0, 0, 180)))  # Semi-transparent red fill
            painter.drawRect(x - 2, y - 2, 4, 4)  # 4x4 square centered at point

            # Add label with background for better readability
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            label = "C1" if i == 0 else "C2"  # C for Calibration

            # Draw text background
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 180)))  # Semi-transparent black
            painter.drawRoundedRect(x + 8, y - 18, 25, 16, 3, 3)

            # Draw text
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(x + 10, y - 5, label)

    def paint_calibration_mask(self, painter: QPainter):
        """Paint calibration segmentation mask"""
        if self.calibration_mask is None:
            return

        # Create semi-transparent overlay
        h, w = self.calibration_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[self.calibration_mask > 0] = [0, 255, 0, 100]  # Semi-transparent green

        # Convert to QImage and draw
        qimage = QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, qimage)

        # Also draw contour for clarity
        contours, _ = cv2.findContours(self.calibration_mask.astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        painter.setPen(QPen(QColor(0, 255, 0), 3))  # Green outline, thicker
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for contour in contours:
            points = contour.squeeze()
            if len(points) > 2:
                path = QPainterPath()
                path.moveTo(points[0][0], points[0][1])
                for point in points[1:]:
                    path.lineTo(point[0], point[1])
                path.closeSubpath()
                painter.drawPath(path)
    
    def paint_calibration_centerline(self, painter: QPainter):
        """Paint automatic calibration centerline"""
        if self.calibration_centerline is None or len(self.calibration_centerline) < 2:
            return
        
        # Draw centerline with a distinct color
        painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow, 2 pixel width
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Create path from centerline points
        path = QPainterPath()
        # Centerline is in (y, x) format, convert to (x, y) for drawing
        first_point = self.calibration_centerline[0]
        path.moveTo(first_point[1], first_point[0])  # x, y
        
        for point in self.calibration_centerline[1:]:
            path.lineTo(point[1], point[0])  # x, y
        
        painter.drawPath(path)
        
        # Draw small dots at centerline points for visualization
        painter.setPen(QPen(QColor(255, 200, 0), 1))
        painter.setBrush(QBrush(QColor(255, 200, 0)))
        for point in self.calibration_centerline[::5]:  # Every 5th point
            painter.drawEllipse(QPointF(point[1], point[0]), 1, 1)  # x, y
    
    def paint_calibration_diameters(self, painter: QPainter):
        """Paint automatic calibration diameter measurements"""
        if (self.calibration_centerline is None or 
            self.calibration_diameters is None or
            self.calibration_left_edges is None or
            self.calibration_right_edges is None or
            self.calibration_perpendiculars is None):
            return
        
        # Draw diameter measurement lines at regular intervals
        painter.setPen(QPen(QColor(0, 255, 255), 1))  # Cyan, 1 pixel width
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Sample every N points to avoid clutter
        sample_interval = max(1, len(self.calibration_centerline) // 20)  # Show ~20 lines max
        
        for i in range(0, len(self.calibration_centerline), sample_interval):
            if i >= len(self.calibration_diameters):
                break
                
            center = self.calibration_centerline[i]
            left_dist = self.calibration_left_edges[i]
            right_dist = self.calibration_right_edges[i]
            perpendicular = self.calibration_perpendiculars[i]
            
            # Skip invalid measurements
            if left_dist <= 0 or right_dist <= 0:
                continue
            
            # Calculate edge points
            # Center is in (y, x) format
            cy, cx = center[0], center[1]
            
            # Left edge point
            left_x = cx - perpendicular[1] * left_dist
            left_y = cy - perpendicular[0] * left_dist
            
            # Right edge point  
            right_x = cx + perpendicular[1] * right_dist
            right_y = cy + perpendicular[0] * right_dist
            
            # Draw diameter line
            painter.drawLine(QPointF(left_x, left_y), QPointF(right_x, right_y))
            
            # Draw small circles at edge points
            painter.setPen(QPen(QColor(255, 0, 255), 1))  # Magenta
            painter.setBrush(QBrush(QColor(255, 0, 255, 150)))
            painter.drawEllipse(QPointF(left_x, left_y), 2, 2)
            painter.drawEllipse(QPointF(right_x, right_y), 2, 2)
            
            # Reset pen for next line
            painter.setPen(QPen(QColor(0, 255, 255), 1))