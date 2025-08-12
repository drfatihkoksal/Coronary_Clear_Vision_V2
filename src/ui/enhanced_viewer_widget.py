"""
Enhanced DICOM Viewer Widget with Overlay Support
MVP Phase 3-5: Supports segmentation and QCA overlays
"""

from typing import Dict, Optional, Tuple
import logging

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QCursor, QFont, QImage, QMouseEvent, QPainter,
                         QPainterPath, QPen, QPixmap, QWheelEvent)
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsLineItem, QGraphicsPathItem,
                             QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsTextItem, QGraphicsView,
                             QMessageBox, QProgressDialog, QGraphicsProxyWidget)

from ..core.dicom_parser import DicomParser
from ..core.threading import TrackingWorker
from ..core.simple_tracker import get_tracker
from ..core.domain_models import Point
from .viewer_modes import ViewerMode, ModeManager
from .heartbeat_overlay_widget import HeartbeatOverlayWidget

logger = logging.getLogger(__name__)


class OverlayItem(QGraphicsItem):
    """Custom graphics item for rendering overlays"""

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
        # Removed stenosis_points - no longer using reference points
        # Removed calibration_line - we don't draw lines between points
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
        logger.debug(f"OverlayItem.paint called - show_qca: {self.show_qca}, has_qca_results: {self.qca_results is not None}")
        
        # Draw calibration overlays first (background)
        if hasattr(self, 'calibration_mask') and self.calibration_mask is not None:
            self.paint_calibration_mask(painter)
        
        # Draw automatic calibration centerline if available and visible
        if (hasattr(self, 'calibration_centerline') and self.calibration_centerline is not None 
            and hasattr(self, 'show_calibration_centerline') and self.show_calibration_centerline):
            self.paint_calibration_centerline(painter)
            
        # Draw automatic calibration diameter measurements if available and visible
        if (hasattr(self, 'calibration_diameters') and self.calibration_diameters is not None 
            and hasattr(self, 'show_calibration_diameters') and self.show_calibration_diameters):
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

        # No calibration line drawing - just show points

        # Draw segmentation overlay
        if self.segmentation_mask is not None and self.show_segmentation:
            self.paint_segmentation(painter)

        # Draw QCA overlay
        if self.show_qca and self.qca_results:
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
            'PistachioGreen': QColor(147, 197, 114)  # Pistachio green for reference points
        }
        color = color_map.get(color_name, QColor(255, 0, 0))
        # Removed reference color - no longer using reference points

        if contour_only:
            # Draw only contours
            # For vessel contours
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
            
            # For reference point contours
            reference_mask = self.segmentation_mask == 128
            if np.any(reference_mask):
                contours, _ = cv2.findContours(reference_mask.astype(np.uint8),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                pen = QPen(reference_color, 2)
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
            # Create QImage from mask
            h, w = self.segmentation_mask.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Check if mask has multiple values (reference points marked differently)
            unique_values = np.unique(self.segmentation_mask)
            if len(unique_values) > 2:  # More than just 0 and 255/1
                # Handle multi-level mask
                # Value 1 or 255: vessel segmentation (use selected color)
                vessel_mask = (self.segmentation_mask == 1) | (self.segmentation_mask == 255)
                overlay[vessel_mask] = [color.red(), color.green(), color.blue(), int(255 * opacity)]
                
                # No longer handling reference points with value 128
            else:
                # Single color mask (backward compatibility)
                overlay[self.segmentation_mask > 0] = [color.red(), color.green(), color.blue(), int(255 * opacity)]

            qimage = QImage(overlay.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
            painter.drawImage(0, 0, qimage)

        # AngioPy no longer provides centerline - it will be drawn by QCA
        
        # Restore painter state
        painter.restore()


    def paint_qca(self, painter: QPainter):
        """Paint QCA overlay"""
        settings = self.qca_settings
        results = self.qca_results
        
        logger.info(f"paint_qca called - results: {results is not None}, success: {results.get('success') if results else False}")

        if not results or not results.get('success'):
            logger.warning("QCA paint skipped - no results or not successful")
            return

        # Draw centerline from QCA results if available
        if settings.get('show_centerline', True) and 'centerline' in results and results['centerline'] is not None:
            centerline = results['centerline']
            logger.info(f"Drawing centerline with {len(centerline)} points")
            if len(centerline) > 1:
                # Use thicker, more visible line
                pen = QPen(QColor(0, 255, 255), 3)  # Cyan centerline, thicker
                pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                
                # Draw centerline path
                path = QPainterPath()
                first_point = centerline[0]
                path.moveTo(first_point[1], first_point[0])  # centerline is in (y,x) format
                
                for i, point in enumerate(centerline[1:]):
                    path.lineTo(point[1], point[0])
                
                painter.drawPath(path)
                logger.info(f"Drew QCA centerline with {len(centerline)} points")
                
                # Draw centerline points for debugging
                painter.setPen(QPen(QColor(255, 255, 0), 1))  # Yellow dots
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                for i in range(0, len(centerline), max(1, len(centerline)//20)):  # Draw every Nth point
                    point = centerline[i]
                    painter.drawEllipse(QPointF(point[1], point[0]), 2, 2)
        else:
            logger.warning(f"Centerline not drawn - show_centerline: {settings.get('show_centerline', True)}, has_centerline: {'centerline' in results}")

        # Debug log for diameter data
        if 'diameters_pixels' in results:
            logger.info(f"QCA Results has diameters_pixels: {len(results['diameters_pixels'])} values")
        else:
            logger.warning(f"QCA Results keys: {list(results.keys())}")

        # Draw stenosis markers
        if settings.get('show_stenosis', True):
            # Draw P and D points (stenosis boundaries)
            if 'stenosis_boundaries' in results:
                boundaries = results['stenosis_boundaries']
                centerline = results.get('centerline', [])
                
                if boundaries and len(centerline) > 0:
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
                    
                    # Draw reference diameter line
                    ref_diam = boundaries.get('reference_diameter', 0)
                    threshold = boundaries.get('threshold', 0)
                    if ref_diam > 0:
                        painter.setPen(QPen(QColor(150, 150, 150), 1))
                        painter.drawText(10, 20, f"Ref: {ref_diam:.1f}mm (75%: {threshold:.1f}mm)")
            
            # Draw MLD location
            if 'mld_location' in results and results['mld_location'] is not None:
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


        # Draw diameter measurements along vessel if enabled
        # Use QCA centerline from results
        if settings.get('show_diameter', False) and 'centerline' in results and results['centerline'] is not None:
            centerline = results['centerline']
            # Try different keys for diameter data
            diameters = results.get('diameters') or results.get('diameters_px') or results.get('diameters_pixels')
            
            logger.info(f"Diameter visualization - show_diameter={settings.get('show_diameter')}, centerline_length={len(centerline)}, has_diameters={diameters is not None}")
            
            if diameters is None:
                logger.warning("No diameter data found in QCA results")
                logger.warning(f"Available QCA result keys: {list(results.keys())}")
                return  # No diameter data available

            # Sample every Nth point to avoid clutter
            num_points = min(len(centerline), len(diameters))
            line_sample_interval = max(1, num_points // 30)  # Show approximately 30 lines
            
            logger.info(f"Diameter visualization: centerline={len(centerline)} points, diameters={len(diameters)} values, interval={line_sample_interval}")
            
            drawn_count = 0
            for i in range(0, num_points, line_sample_interval):
                point = centerline[i]
                diameter = diameters[i]

                # Draw small perpendicular line showing diameter
                # Calculate perpendicular direction
                if i > 0 and i < len(centerline) - 1:
                    # Use larger window for smoother tangent calculation
                    window = min(5, min(i, len(centerline) - 1 - i))
                    
                    # Centerline is in [y, x] format
                    prev_point = centerline[max(0, i-window)]
                    next_point = centerline[min(len(centerline)-1, i+window)]
                    
                    # Calculate tangent in [y, x] format
                    tangent_y = next_point[0] - prev_point[0]
                    tangent_x = next_point[1] - prev_point[1]
                    
                    # Normalize tangent
                    norm = np.sqrt(tangent_y**2 + tangent_x**2)
                    if norm > 0:
                        tangent_y /= norm
                        tangent_x /= norm
                        
                        # Get perpendicular and edge information from QCA results
                        if ('left_edges' in results and 'right_edges' in results and 
                            'perpendiculars' in results and 
                            i < len(results['left_edges'])):
                            # Use exact edge positions from QCA
                            left_edge = results['left_edges'][i]
                            right_edge = results['right_edges'][i]
                            perp = results['perpendiculars'][i]
                            perp_y = perp[0]
                            perp_x = perp[1]
                        else:
                            # Use simple diameter drawing if edge info not available
                            perp_y = -tangent_x
                            perp_x = tangent_y
                            
                            # Get diameter in pixels for drawing
                            if 'diameters' in results and i < len(results['diameters']):
                                diameter_pixels = results['diameters'][i]
                            else:
                                diameter_pixels = diameter
                            
                            # Use half diameter as edge distances
                            left_edge = diameter_pixels / 2
                            right_edge = diameter_pixels / 2

                        # Draw diameter line across vessel
                        pen = QPen(QColor(255, 255, 0, 255), 2)  # Bright yellow, opaque, thicker
                        pen.setStyle(Qt.PenStyle.SolidLine)
                        painter.setPen(pen)
                        
                        # point is in [y, x] format
                        center_y = point[0]
                        center_x = point[1]
                        
                        # Calculate endpoints using exact edge distances
                        p1 = QPointF(
                            center_x - perp_x * left_edge,
                            center_y - perp_y * left_edge
                        )
                        p2 = QPointF(
                            center_x + perp_x * right_edge,
                            center_y + perp_y * right_edge
                        )
                        painter.drawLine(p1, p2)
                        drawn_count += 1
            
            logger.info(f"Drew {drawn_count} diameter lines out of {num_points // line_sample_interval} expected")


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

                # Skip drawing labels for P, D, M points to avoid confusion

        # Paint additional guide points (all points except first and last)
        if len(points) > 2:
            # Draw 4x4 squares like AngioPy encoding (blue channel)
            painter.setPen(QPen(QColor(0, 0, 255), 1))  # Blue, 1 pixel
            painter.setBrush(QBrush(QColor(0, 0, 255, 180)))  # Semi-transparent blue fill
            for i in range(1, len(points) - 1):  # Skip first and last
                x, y = points[i]
                x, y = int(x), int(y)  # Convert to int
                painter.drawRect(x - 2, y - 2, 4, 4)  # 4x4 square centered at point

    # Removed paint_calibration_line method - no line drawing needed

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
        
        logger.debug(f"🎨 Painting calibration mask - shape: {self.calibration_mask.shape}")

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
        if not hasattr(self, 'calibration_centerline') or self.calibration_centerline is None:
            return
            
        centerline = self.calibration_centerline
        if len(centerline) < 2:
            return
        
        logger.debug(f"🎨 Painting calibration centerline - {len(centerline)} points")
            
        # Draw centerline in cyan color
        painter.setPen(QPen(QColor(0, 255, 255), 3))  # Cyan, thick line
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        path = QPainterPath()
        # Centerline is in (y, x) format, convert to (x, y) for drawing
        first_point = centerline[0]
        path.moveTo(first_point[1], first_point[0])  # x, y
        
        for point in centerline[1:]:
            path.lineTo(point[1], point[0])  # x, y
            
        painter.drawPath(path)
        
        # Draw start and end points
        painter.setBrush(QBrush(QColor(0, 255, 255)))  # Cyan fill
        start_point = centerline[0]
        end_point = centerline[-1]
        
        # Draw circles at start and end
        painter.drawEllipse(QPointF(start_point[1], start_point[0]), 6, 6)  # x, y
        painter.drawEllipse(QPointF(end_point[1], end_point[0]), 6, 6)  # x, y
    
    def paint_calibration_diameters(self, painter: QPainter):
        """Paint automatic calibration diameter measurements"""
        # Check if all required data is available
        if (not hasattr(self, 'calibration_centerline') or self.calibration_centerline is None or
            not hasattr(self, 'calibration_diameters') or self.calibration_diameters is None or
            not hasattr(self, 'calibration_left_edges') or self.calibration_left_edges is None or
            not hasattr(self, 'calibration_right_edges') or self.calibration_right_edges is None or
            not hasattr(self, 'calibration_perpendiculars') or self.calibration_perpendiculars is None):
            return
            
        centerline = self.calibration_centerline
        diameters = self.calibration_diameters
        left_edges = self.calibration_left_edges
        right_edges = self.calibration_right_edges
        perpendiculars = self.calibration_perpendiculars
        
        # Ensure all data is numpy arrays and valid
        if centerline is not None and not isinstance(centerline, np.ndarray):
            centerline = np.array(centerline)
        if diameters is not None and not isinstance(diameters, np.ndarray):
            diameters = np.array(diameters)
        if left_edges is not None and not isinstance(left_edges, np.ndarray):
            left_edges = np.array(left_edges)
        if right_edges is not None and not isinstance(right_edges, np.ndarray):
            right_edges = np.array(right_edges)
        if perpendiculars is not None and not isinstance(perpendiculars, np.ndarray):
            perpendiculars = np.array(perpendiculars)
        
        # Check if data is valid
        if centerline is None or diameters is None:
            return
            
        # Convert to numpy arrays if needed
        if not isinstance(centerline, np.ndarray):
            centerline = np.array(centerline)
        if not isinstance(diameters, np.ndarray):
            diameters = np.array(diameters)
            
        # Check if arrays are empty
        if centerline.size == 0 or diameters.size == 0:
            return
        
        logger.info(f"🎨 PAINTING CALIBRATION DIAMETERS:")
        logger.info(f"  - Centerline points: {len(centerline)}")
        logger.info(f"  - Diameters: {len(diameters)}")
        logger.info(f"  - First 5 diameters: {diameters[:5] if len(diameters) >= 5 else diameters}")
        logger.info(f"  - Valid diameters (>0): {len([d for d in diameters if d > 0])}")
        
        # Draw diameter lines perpendicular to centerline
        painter.setPen(QPen(QColor(0, 255, 255), 1))  # Cyan, 1 pixel width
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Sample every N points to avoid clutter
        sample_interval = max(1, len(centerline) // 20)  # Show ~20 lines max
        
        lines_drawn = 0
        for i in range(0, len(centerline), sample_interval):
            if i >= len(diameters):
                break
                
            center = centerline[i]
            left_dist = left_edges[i]
            right_dist = right_edges[i]
            perpendicular = perpendiculars[i]
            
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
            lines_drawn += 1
            
            # Debug first few lines
            if i < 3:
                diameter = left_dist + right_dist
                logger.info(f"  - Line {i}: center=({cx:.1f},{cy:.1f}), diameter={diameter:.1f}, "
                           f"from=({left_x:.1f},{left_y:.1f}) to=({right_x:.1f},{right_y:.1f})")
        
        logger.info(f"🎨 DIAMETER PAINTING COMPLETE: {lines_drawn} lines drawn")
    

class EnhancedDicomViewer(QGraphicsView):
    """Enhanced DICOM viewer with overlay support"""

    # Signals
    zoom_changed = pyqtSignal(float)
    pixel_info_changed = pyqtSignal(str)
    user_clicked = pyqtSignal(int, int)  # x, y coordinates
    calibration_points_selected = pyqtSignal(tuple, tuple)  # two points
    calibration_completed = pyqtSignal(float, dict)  # calibration_factor, details
    calibration_cancelled = pyqtSignal()  # calibration cancelled
    segmentation_point_clicked = pyqtSignal(int, int)  # x, y for segmentation
    calibration_point_clicked = pyqtSignal(int, int)  # x, y for calibration
    points_changed = pyqtSignal()  # Emitted when points are added/removed
    point_selected = pyqtSignal(float, float)  # x, y for multi-frame mode
    
    # Mode change signals
    segmentation_toggled = pyqtSignal(bool)  # Segmentation mode on/off
    calibration_toggled = pyqtSignal(bool)  # Calibration mode on/off
    multi_frame_toggled = pyqtSignal(bool)  # Multi-frame mode on/off
    
    # Frame range selection signal
    frame_range_selected = pyqtSignal(int, int)  # start_frame, end_frame

    def __init__(self):
        super().__init__()

        # Setup scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


        # Graphics items
        self.pixmap_item = QGraphicsPixmapItem()
        self.overlay_item = OverlayItem(self)

        self.scene.addItem(self.pixmap_item)
        
        # Make overlay a child of pixmap so it follows transformations
        self.overlay_item.setParentItem(self.pixmap_item)
        
        # Ensure overlay is on top (relative to parent)
        self.overlay_item.setZValue(1)

        # Viewer settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)

        # State
        self.dicom_parser: Optional[DicomParser] = None
        self.main_window = None  # Reference to main window
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_index: int = 0
        self.window_center: float = 128
        self.window_width: float = 256
        self.zoom_factor: float = 1.0
        self.previous_frame: Optional[np.ndarray] = None  # For optical flow

        # Mode management
        self.mode_manager = ModeManager(self)
        self.mode_manager.mode_changed.connect(self._on_mode_changed)
        
        # Interaction state
        self.calibration_points = []
        self.segmentation_points = []  # Store segmentation points
        # Removed stenosis_points - no longer using reference points
        self.current_segmentation_result = None
        self.segmented_frame_index = None
        self.temp_calibration_end_point = None  # For mouse tracking during calibration

        # Point editing state
        self.hovered_point_index = -1  # Index of point being hovered
        self.dragging_point = False  # Is a point being dragged
        self.dragged_point_index = -1  # Index of point being dragged
        self.drag_start_pos = None  # Start position of drag
        
        # Additional attributes that might be set later
        self.multi_frame_points = []  # For multi-frame mode
        
        # Setup mode callbacks
        self._setup_mode_callbacks()
        self.tracking_method = 'template'  # Default tracking method
        self.catheter_size = None  # Set during calibration
        self.catheter_diameter_mm = None  # Set during calibration
        self.calibration_panning = False  # Pan mode during calibration

        # Window/Level adjustment
        self.adjusting_window_level = False
        self.wl_start_pos: Optional[QPointF] = None
        self.wl_start_center: float = 128
        self.wl_start_width: float = 256

        # Vessel tracking with TAPIR
        self.tracking_enabled = True
        self._prev_frame_index = None  # Track previous frame for auto-tracking
        
        
        # Initialize tracking adapter
        self.point_tracker = get_tracker()
        self._tracked_point_ids = {}  # Maps frame point indices to tracker IDs
        
        # Update tracking config when DICOM/calibration changes
        self._update_tracking_config()

        # Performance optimization
        self._frame_cache = {}  # Cache processed frames
        self._cache_size = 20  # Maximum cached frames
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._batch_update)
        self._update_timer.setInterval(16)  # 60 FPS max update rate
        self._pending_updates = False

        # Background workers
        self.tracking_worker = None
        self.frame_processor = None

        # QCA visualization state
        self.qca_visualization_items = []  # Graphics items for QCA results
        self.qca_centerline = None
        self.qca_diameters = None
        self.show_qca_overlay = False
        self.qca_stenosis_info = None
        
        # Frame range selection
        self.frame_range_start = None
        self.frame_range_end = None
        self.selected_beat_frames = None  # Specific frames for selected beat
        self.frame_timestamps = None  # Will be set by main window
        self.cardiac_phases = None  # Will be set by main window
        
        # Create heartbeat overlay widget
        self.heartbeat_overlay = HeartbeatOverlayWidget()  # Don't pass parent
        self.heartbeat_proxy = QGraphicsProxyWidget()
        self.heartbeat_proxy.setWidget(self.heartbeat_overlay)
        self.scene.addItem(self.heartbeat_proxy)
        self.heartbeat_proxy.setZValue(100)  # Always on top
        self.heartbeat_proxy.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)  # Don't scale with scene
        self.heartbeat_proxy.setAcceptedMouseButtons(Qt.MouseButton.NoButton)  # Make transparent to mouse events
        
        # Position overlay in top-right corner
        QTimer.singleShot(100, self._position_heartbeat_overlay)  # Delay initial positioning

    def _setup_mode_callbacks(self):
        """Setup mode manager callbacks"""
        # Register mode-specific callbacks
        self.mode_manager.register_enter_callback(ViewerMode.CALIBRATE, self._enter_calibrate_mode)
        self.mode_manager.register_exit_callback(ViewerMode.CALIBRATE, self._exit_calibrate_mode)
        
        self.mode_manager.register_enter_callback(ViewerMode.SEGMENT, self._enter_segment_mode)
        self.mode_manager.register_exit_callback(ViewerMode.SEGMENT, self._exit_segment_mode)
    
    def _enter_calibrate_mode(self):
        """Called when entering calibration mode"""
        # Clear any existing calibration points
        self.overlay_item.calibration_points = []
        self.calibration_points = []
        self.overlay_item.update()
        self._request_update()
        self.clear_calibration_points()
    
    def _exit_calibrate_mode(self):
        """Called when exiting calibration mode"""
        # Keep calibration points for display
        pass
    
    def _enter_segment_mode(self):
        """Called when entering segmentation mode"""
        # Check for existing tracking points
        if hasattr(self, '_check_and_import_tracking_points'):
            self._check_and_import_tracking_points()
    
    def _exit_segment_mode(self):
        """Called when exiting segmentation mode"""
        # Keep segmentation points for later use
        pass
    
    @property
    def interaction_mode(self) -> str:
        """Legacy property for interaction mode"""
        mode_map = {
            ViewerMode.VIEW: 'view',
            ViewerMode.SEGMENT: 'segment',
            ViewerMode.CALIBRATE: 'calibrate',
            ViewerMode.MULTI_FRAME: 'multi_frame',
            ViewerMode.QCA: 'qca'
        }
        return mode_map.get(self.mode_manager.current_mode, 'view')
    
    @property
    def segmentation_mode(self) -> bool:
        """Legacy property for segmentation mode"""
        return self.mode_manager.is_in_mode(ViewerMode.SEGMENT)
    
    @property
    def calibration_mode(self) -> bool:
        """Legacy property for calibration mode"""
        return self.mode_manager.is_in_mode(ViewerMode.CALIBRATE)
    
    @property
    def multi_frame_mode(self) -> bool:
        """Legacy property for multi-frame mode"""
        return self.mode_manager.is_in_mode(ViewerMode.MULTI_FRAME)
    
    def _on_mode_changed(self, old_mode: ViewerMode, new_mode: ViewerMode):
        """Handle mode changes"""
        logger.debug(f"Mode changed from {old_mode.name} to {new_mode.name}")
        
        # Update tracking if needed
        if new_mode == ViewerMode.VIEW and self.tracking_enabled:
            # Create custom crosshair cursor for tracking mode
            self._set_tracking_crosshair_cursor()
        
        # Emit legacy signals for compatibility
        if old_mode == ViewerMode.SEGMENT:
            if hasattr(self, 'segmentation_toggled'):
                self.segmentation_toggled.emit(False)
        if new_mode == ViewerMode.SEGMENT:
            if hasattr(self, 'segmentation_toggled'):
                self.segmentation_toggled.emit(True)
            
        if old_mode == ViewerMode.CALIBRATE:
            if hasattr(self, 'calibration_toggled'):
                self.calibration_toggled.emit(False)
        if new_mode == ViewerMode.CALIBRATE:
            if hasattr(self, 'calibration_toggled'):
                self.calibration_toggled.emit(True)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        # Check if we have a valid frame
        if self.current_frame is None:
            super().mousePressEvent(event)
            return
            
        scene_pos = self.mapToScene(event.pos())
        # Convert to DICOM coordinates
        dicom_x, dicom_y = self.scene_to_dicom_coords(scene_pos)
        
        # Check if current mode accepts points or tracking is enabled
        accepts_points = self.mode_manager.accepts_points()
        is_tracking_in_view = (self.tracking_enabled and 
                              self.mode_manager.is_in_mode(ViewerMode.VIEW))
        
        if (accepts_points or is_tracking_in_view) and event.button() == Qt.MouseButton.LeftButton:
            self._handle_point_placement(dicom_x, dicom_y, event)
            return
            
        # Call parent implementation for other cases
        super().mousePressEvent(event)
    
    def _handle_point_placement(self, dicom_x: int, dicom_y: int, event: QMouseEvent):
        """Unified point placement handler for all modes"""
        current_mode = self.mode_manager.current_mode
        
        # Initialize frame points if needed
        if self.current_frame_index not in self.overlay_item.frame_points:
            self.overlay_item.frame_points[self.current_frame_index] = []
        
        current_points = self.overlay_item.frame_points[self.current_frame_index]
        max_points = self.mode_manager.get_max_points()
        
        # Check if clicking on existing point to drag
        point_index = self.get_point_at_position(dicom_x, dicom_y)
        if point_index >= 0:
            # Start dragging existing point
            self.dragging_point = True
            self.dragged_point_index = point_index
            self.drag_start_pos = (dicom_x, dicom_y)
            # Keep the mode's cursor (cross cursor) instead of changing to hand
            self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Disable pan during drag
            event.accept()
            return
        
        # Check max points limit
        if max_points and len(current_points) >= max_points:
            # For modes with limits, clear and start over
            if current_mode in [ViewerMode.CALIBRATE, ViewerMode.QCA]:
                current_points.clear()
            else:
                event.accept()
                return
        
        # For tracking in VIEW mode, allow unlimited points (removed 3-point limit)
        # Users can now select 1, 2, 3, 4 or more points as needed
        
        # Add new point
        current_points.append((dicom_x, dicom_y))
        self.overlay_item.user_points = current_points.copy()
        self.overlay_item.update()
        self._request_update()
        
        
        # Emit appropriate signals based on mode
        if current_mode == ViewerMode.CALIBRATE:
            self.calibration_point_clicked.emit(dicom_x, dicom_y)
            if len(current_points) == 2:
                self.calibration_points_selected.emit(current_points[0], current_points[1])
        elif current_mode == ViewerMode.SEGMENT:
            self.segmentation_point_clicked.emit(dicom_x, dicom_y)
        elif current_mode == ViewerMode.MULTI_FRAME:
            self.point_selected.emit(float(dicom_x), float(dicom_y))
        
        # Always emit generic signals
        self.user_clicked.emit(dicom_x, dicom_y)
        self.points_changed.emit()
        
        event.accept()
    
    def set_dicom_parser(self, parser: DicomParser):
        """Set the DICOM parser"""
        self.dicom_parser = parser
        # Update tracking configuration with new DICOM metadata
        if parser:
            self._update_tracking_config()

    def display_frame(self, frame_index: int):
        """Display a specific frame with caching"""
        if not self.dicom_parser:
            return

        # Check cache first
        cache_key = (frame_index, self.window_center, self.window_width)

        if cache_key in self._frame_cache:
            windowed = self._frame_cache[cache_key]
            frame = self.dicom_parser.get_frame(frame_index)
        else:
            frame = self.dicom_parser.get_frame(frame_index)
            if frame is None:
                return

            # Apply window/level
            windowed = self.dicom_parser.apply_window_level(
                frame, self.window_center, self.window_width
            )

            # Update cache
            self._frame_cache[cache_key] = windowed

            # Limit cache size
            if len(self._frame_cache) > self._cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._frame_cache))
                del self._frame_cache[oldest_key]

        # Store previous frame for optical flow (use raw frame, not windowed)
        self.previous_frame = self.current_frame
        self.current_frame = frame.copy()  # Store raw frame copy
        self.current_frame_index = frame_index

        # Update overlay's current frame
        self.overlay_item.current_frame_index = frame_index

        # Always show the original windowed image
        # Overlays are rendered separately by the OverlayItem
        height, width = windowed.shape
        qimage = QImage(windowed.data, width, height, width, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)

        # Set scene rect to match DICOM dimensions
        self.scene.setSceneRect(0, 0, width, height)

        # Schedule batch update instead of immediate update
        self._request_update()
        
        # Reposition heartbeat overlay
        self._position_heartbeat_overlay()

        # Update user_points for current frame
        if frame_index in self.overlay_item.frame_points:
            self.overlay_item.user_points = self.overlay_item.frame_points[frame_index].copy()
        else:
            self.overlay_item.user_points = []

        # Auto-track points if tracking is enabled
        if self.tracking_enabled and hasattr(self, '_prev_frame_index') and self._prev_frame_index is not None:
            if self._prev_frame_index != frame_index:
                # Track from previous frame to current frame
                if self._prev_frame_index in self.overlay_item.frame_points:
                    prev_points = self.overlay_item.frame_points[self._prev_frame_index]
                    if prev_points and frame_index not in self.overlay_item.frame_points:
                        # Save current frame (already updated)
                        temp_frame = self.current_frame
                        temp_index = self.current_frame_index

                        # Get previous frame for tracking
                        prev_frame = self.dicom_parser.get_frame(self._prev_frame_index)
                        if prev_frame is not None:
                            # Set previous frame as current for tracking
                            self.current_frame = prev_frame
                            self.current_frame_index = self._prev_frame_index

                            # Track to new frame
                            result = self.track_points_to_frame(frame_index)

                            # Restore current frame
                            self.current_frame = temp_frame
                            self.current_frame_index = temp_index

                            # Update user_points with tracked points
                            if result and frame_index in self.overlay_item.frame_points:
                                self.overlay_item.user_points = self.overlay_item.frame_points[frame_index].copy()

        # Store current frame index for next time
        self._prev_frame_index = frame_index

    def set_interaction_mode(self, mode: str):
        """Set interaction mode (legacy method)"""
        mode_map = {
            'view': ViewerMode.VIEW,
            'segment': ViewerMode.SEGMENT,
            'calibrate': ViewerMode.CALIBRATE,
            'multi_frame': ViewerMode.MULTI_FRAME,
            'qca': ViewerMode.QCA
        }
        if mode in mode_map:
            self.mode_manager.set_mode(mode_map[mode])
        else:
            logger.warning(f"Unknown interaction mode: {mode}")

    def scene_to_dicom_coords(self, scene_pos: QPointF) -> Tuple[int, int]:
        """Convert scene coordinates to DICOM pixel coordinates"""
        # The pixmap item is at (0, 0) in scene coordinates
        # and its size matches the DICOM dimensions
        x = int(round(scene_pos.x()))
        y = int(round(scene_pos.y()))

        # Clamp to DICOM bounds
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

        return x, y

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        # Check if we have a valid frame
        if self.current_frame is None:
            return

        scene_pos = self.mapToScene(event.pos())

        # Convert to DICOM coordinates
        dicom_x, dicom_y = self.scene_to_dicom_coords(scene_pos)
        
        # Check if current mode accepts points or tracking is enabled
        accepts_points = self.mode_manager.accepts_points()
        is_tracking_in_view = (self.tracking_enabled and 
                              self.mode_manager.is_in_mode(ViewerMode.VIEW))
        
        if (accepts_points or is_tracking_in_view) and event.button() == Qt.MouseButton.LeftButton:
            self._handle_point_placement(dicom_x, dicom_y, event)
            return

        # Legacy multi-frame mode handling (will be replaced)
        if self.multi_frame_mode and False:
            if event.button() == Qt.MouseButton.LeftButton:
                print(f"Multi-frame click at: ({dicom_x}, {dicom_y})")
                # Add point to overlay for visual feedback
                if not hasattr(self, 'multi_frame_points'):
                    self.multi_frame_points = []

                # Clear if we have 3 points already
                if len(self.multi_frame_points) >= 3:
                    self.multi_frame_points.clear()

                # Add new point
                self.multi_frame_points.append((dicom_x, dicom_y))
                print(f"Current points: {self.multi_frame_points}")

                # Update overlay
                self.overlay_item.user_points = self.multi_frame_points.copy()
                self._request_update()

                # Force immediate update
                self.scene().update()

                # Emit signal
                self.point_selected.emit(dicom_x, dicom_y)
                event.accept()
                return
        elif self.calibration_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                # Add calibration point like segmentation
                self.add_calibration_point(dicom_x, dicom_y)
                self.calibration_point_clicked.emit(dicom_x, dicom_y)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # Enable pan mode with right click in calibration mode
                self.calibration_panning = True
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                # Store the press position to handle drag in parent class
                super().mousePressEvent(event)
                return
        elif self.segmentation_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check if clicking on existing point
                point_index = self.get_point_at_position(dicom_x, dicom_y)
                if point_index >= 0:
                    # Start dragging
                    self.dragging_point = True
                    self.dragged_point_index = point_index
                    self.drag_start_pos = (dicom_x, dicom_y)
                    # Keep cross cursor for consistency
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Disable pan during drag
                else:
                    # Add new point
                    self.add_segmentation_point(dicom_x, dicom_y)
                    self.segmentation_point_clicked.emit(dicom_x, dicom_y)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # Show context menu
                self.show_point_context_menu(event.globalPosition().toPoint(), dicom_x, dicom_y)
                event.accept()
                return
        elif self.interaction_mode == 'segment' and event.button() == Qt.MouseButton.LeftButton:
            # Legacy mode for compatibility - also use frame-specific storage
            if self.current_frame_index not in self.overlay_item.frame_points:
                self.overlay_item.frame_points[self.current_frame_index] = []

            # Check if we already have 3 points in current frame
            if len(self.overlay_item.frame_points[self.current_frame_index]) >= 3:
                event.accept()
                return

            self.overlay_item.frame_points[self.current_frame_index].append((dicom_x, dicom_y))
            self.overlay_item.user_points = self.overlay_item.frame_points[self.current_frame_index].copy()
            self.overlay_item.update()  # Force overlay update
            self._request_update()
            self.user_clicked.emit(dicom_x, dicom_y)
            # Emit signal to update tracking buttons
            self.points_changed.emit()

        elif self.tracking_enabled and event.button() == Qt.MouseButton.LeftButton and self.interaction_mode == 'view':
            # Check if we're clicking on an existing point to drag it
            point_index = self.get_point_at_position(dicom_x, dicom_y)

            if point_index >= 0:
                # Start dragging existing point
                self.dragging_point = True
                self.dragged_point_index = point_index
                # Keep current mode's cursor
                # self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

            # Add new tracking points in view mode when tracking is enabled
            if self.current_frame_index not in self.overlay_item.frame_points:
                self.overlay_item.frame_points[self.current_frame_index] = []

            # Check if we already have 3 points in current frame
            if len(self.overlay_item.frame_points[self.current_frame_index]) >= 3:
                event.accept()
                return

            self.overlay_item.frame_points[self.current_frame_index].append((dicom_x, dicom_y))
            self.overlay_item.user_points = self.overlay_item.frame_points[self.current_frame_index].copy()
            self.overlay_item.update()  # Force overlay update
            self._request_update()
            self.user_clicked.emit(dicom_x, dicom_y)
            # Emit signal to update tracking buttons
            self.points_changed.emit()

        elif self.interaction_mode == 'calibrate' and event.button() == Qt.MouseButton.LeftButton:
            # Add calibration point using DICOM coordinates
            self.calibration_points.append((dicom_x, dicom_y))

            if len(self.calibration_points) == 2:
                # Calibration complete
                # Don't draw line, just emit the points
                self._request_update()
                self.calibration_points_selected.emit(
                    self.calibration_points[0],
                    self.calibration_points[1]
                )

                # Calculate calibration factor if we have catheter info
                if hasattr(self, 'catheter_diameter_mm') and self.current_frame is not None:
                    p1, p2 = self.calibration_points

                    # Segment catheter and measure width
                    width_pixels = self.measure_catheter_width(p1, p2)

                    if width_pixels > 0:
                        calibration_factor = self.catheter_diameter_mm / width_pixels
                        
                        logger.info(f"=== CALIBRATION DEBUG ===")
                        logger.info(f"Catheter size: {getattr(self, 'catheter_size', 'Unknown')} = {self.catheter_diameter_mm}mm")
                        logger.info(f"Measured width: {width_pixels:.2f} pixels")
                        logger.info(f"Calibration factor: {calibration_factor:.5f} mm/pixel")
                        logger.info(f"Expected 3mm vessel = {3.0/calibration_factor:.1f} pixels")

                        details = {
                            'catheter_size': getattr(self, 'catheter_size', 'Unknown'),
                            'catheter_diameter_mm': self.catheter_diameter_mm,
                            'width_pixels': width_pixels,
                            'points': self.calibration_points
                        }

                        self.calibration_completed.emit(calibration_factor, details)
                    else:
                        pass  # Failed to measure catheter width

                # Don't exit calibration mode yet - wait for user to confirm/cancel
            elif len(self.calibration_points) == 1:
                # Just show the first point
                self._request_update()
                # Force update
                self._request_update()
                self.scene.update()

        elif event.button() == Qt.MouseButton.RightButton:
            # Check if Ctrl is pressed for window/level adjustment
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Start window/level adjustment with Ctrl+Right click
                self.adjusting_window_level = True
                self.wl_start_pos = event.pos()
                self.wl_start_center = self.window_center
                self.wl_start_width = self.window_width
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
            else:
                # Show context menu on normal right click
                dicom_x, dicom_y = self.scene_to_dicom_coords(scene_pos)
                self.show_point_context_menu(event.globalPosition().toPoint(), dicom_x, dicom_y)
                event.accept()
            return
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Middle button for window/level adjustment
            self.adjusting_window_level = True
            self.wl_start_pos = event.pos()
            self.wl_start_center = self.window_center
            self.wl_start_width = self.window_width
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # Call super for all modes except when we handled the event
        if not (self.segmentation_mode and event.button() == Qt.MouseButton.LeftButton):
            if not (self.interaction_mode in ['segment', 'calibrate'] and event.button() == Qt.MouseButton.LeftButton):
                if not (event.button() == Qt.MouseButton.RightButton and self.adjusting_window_level):
                    super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        scene_pos = self.mapToScene(event.pos())

        # Update pixel info using proper coordinate conversion
        if self.current_frame is not None:
            x, y = self.scene_to_dicom_coords(scene_pos)
            h, w = self.current_frame.shape

            if 0 <= x < w and 0 <= y < h:
                pixel_value = self.current_frame[y, x]
                info = f"DICOM ({x}, {y}): {pixel_value}"
                self.pixel_info_changed.emit(info)

            # Check for point dragging in segmentation mode or tracking mode
            if (self.segmentation_mode or (self.tracking_enabled and self.interaction_mode == 'view')) and self.dragging_point and self.dragged_point_index >= 0:
                # Drag the point
                if self.current_frame_index in self.overlay_item.frame_points:
                    points = self.overlay_item.frame_points[self.current_frame_index]
                    if 0 <= self.dragged_point_index < len(points):
                        points[self.dragged_point_index] = (x, y)
                        # Update user_points as well
                        self.overlay_item.user_points = points.copy()
                        self._request_update()
                        self.points_changed.emit()
                event.accept()
                return

            # Check for point hovering in segmentation mode or tracking mode
            if (self.segmentation_mode or (self.tracking_enabled and self.interaction_mode == 'view')) and not self.dragging_point:
                # Check if mouse is near any point
                self.hovered_point_index = self.get_point_at_position(x, y)
                if self.hovered_point_index >= 0:
                    # Let mode manager handle cursor
                    pass
                else:
                    # Keep the mode's cursor
                    pass

            # No line preview for calibration mode

        # Handle window/level adjustment
        if self.adjusting_window_level and self.wl_start_pos:
            delta = event.pos() - self.wl_start_pos

            # Adjust window center with vertical movement
            self.window_center = self.wl_start_center + delta.y()

            # Adjust window width with horizontal movement
            self.window_width = max(1, self.wl_start_width + delta.x())

            # Clear cache when window/level changes
            self.clear_cache()

            # Update display
            self.display_frame(self.current_frame_index)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if self.dragging_point and (self.segmentation_mode or self.tracking_enabled):
            # Check if this is a batch tracked frame and if we should re-track
            should_retrack = False
            if self.dragged_point_index >= 0 and hasattr(self, '_batch_tracked_frames'):
                # Check if current frame is part of a batch tracking session
                if self.current_frame_index in self._batch_tracked_frames:
                    should_retrack = True
            
            # End dragging for both segmentation and tracking modes
            self.dragging_point = False
            edited_frame_index = self.current_frame_index
            edited_point_index = self.dragged_point_index
            self.dragged_point_index = -1
            self.drag_start_pos = None
            # Let mode manager handle cursor
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Re-enable pan
            event.accept()
            
            # If this was a batch tracked point, ask user if they want to re-track
            if should_retrack:
                self._handle_batch_tracking_edit(edited_frame_index, edited_point_index)
        elif self.calibration_mode and event.button() == Qt.MouseButton.RightButton:
            # End pan mode in calibration, restore cross cursor
            self.calibration_panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
        elif (event.button() == Qt.MouseButton.RightButton or event.button() == Qt.MouseButton.MiddleButton) and self.adjusting_window_level:
            self.adjusting_window_level = False
            # Restore cursor
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events for zooming"""
        # Get the position before zoom
        old_pos = self.mapToScene(event.position().toPoint())

        # Calculate zoom
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
        self.zoom_factor *= zoom_factor

        # Get the new position after zoom
        new_pos = self.mapToScene(event.position().toPoint())

        # Move scene to old position
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        self.zoom_changed.emit(self.zoom_factor)

    def fit_to_window(self):
        """Fit the image to the window"""
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_factor = self.transform().m11()
        self.zoom_changed.emit(self.zoom_factor)

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.resetTransform()
        self.zoom_factor = 1.0
        self.zoom_changed.emit(self.zoom_factor)

    def track_points_to_next_frame(self):
        """Manually track points from current frame to next frame"""
        if self.current_frame_index >= self.dicom_parser.num_frames - 1:
            return False

        next_frame_index = self.current_frame_index + 1
        return self.track_points_to_frame(next_frame_index)

    def track_points_to_previous_frame(self):
        """Manually track points from current frame to previous frame"""
        if self.current_frame_index <= 0:
            return False

        prev_frame_index = self.current_frame_index - 1
        return self.track_points_to_frame(prev_frame_index)

    def track_points_to_frame(self, new_frame_index: int):
        """Track points from current frame to new frame using MHT-enabled PointTracker"""
        logger.debug(f"track_points_to_frame called: {self.current_frame_index} -> {new_frame_index}")
        
        if self.current_frame is None or not self.tracking_enabled:
            logger.debug("Tracking disabled or no current frame")
            return False

        # Don't track if we're on the same frame
        if self.current_frame_index == new_frame_index:
            logger.debug("Same frame, no tracking needed")
            return True

        # Get points from current frame
        current_points = self.overlay_item.frame_points.get(self.current_frame_index, [])
        if not current_points:
            logger.debug("No points to track")
            return False

        try:
            # Get new frame
            new_frame = self.dicom_parser.get_frame(new_frame_index)
            if new_frame is None:
                logger.error(f"Failed to get frame {new_frame_index}")
                return False

            # Ensure all points are being tracked
            self.point_tracker.clear()
            self.point_tracker.set_frame(self.current_frame)
            
            # Add current points to tracker
            for i, point in enumerate(current_points):
                point_id = f"point_{i}"
                self.point_tracker.add_point(point_id, Point(float(point[0]), float(point[1])))
                self._tracked_point_ids[(self.current_frame_index, i)] = point_id
            
            # Log active points
            active_points = self.point_tracker.get_tracked_points()
            logger.debug(f"Active points before tracking: {len(active_points)}")

            # Validate frames
            logger.debug(f"Current frame shape: {self.current_frame.shape if self.current_frame is not None else 'None'}")
            logger.debug(f"New frame shape: {new_frame.shape if new_frame is not None else 'None'}")
            logger.debug(f"Frame indices: {self.current_frame_index} -> {new_frame_index}")
            
            # Check if current frame is available
            if self.current_frame is None:
                logger.warning("Current frame is None, getting it from parser")
                self.current_frame = self.dicom_parser.get_frame(self.current_frame_index)
                if self.current_frame is None:
                    logger.error(f"Failed to get current frame {self.current_frame_index}")
                    return False
            
            # Track all points
            tracking_results = self.point_tracker.track_in_frame(new_frame)
            
            logger.debug(f"Tracking results: {len(tracking_results)} points tracked")
            
            # Convert tracking results to frame points
            if tracking_results:
                # Maintain order of points by using tracked IDs
                tracked_points = []
                for i, point in enumerate(current_points):
                    point_id = self._tracked_point_ids.get((self.current_frame_index, i))
                    if point_id and point_id in tracking_results:
                        new_pos = tracking_results[point_id]
                        tracked_points.append((float(new_pos.x), float(new_pos.y)))
                        # Update ID mapping for new frame
                        self._tracked_point_ids[(new_frame_index, i)] = point_id
                    else:
                        # Keep original position if tracking failed
                        tracked_points.append((float(point[0]), float(point[1])))
                        logger.warning(f"Failed to track point {i} (ID: {point_id}) at frame {new_frame_index}")
                
                # Store tracked points
                self.overlay_item.frame_points[new_frame_index] = tracked_points
                # Update overlay
                self.overlay_item.update()
                self._request_update()
                return True
            else:
                # Use simple copy if no tracking results
                logger.warning(f"No tracking results from frame {self.current_frame_index} to frame {new_frame_index}, copying points")
                self.overlay_item.frame_points[new_frame_index] = current_points.copy()
                self.overlay_item.update()
                self._request_update()
                return False

        except Exception as e:
            logger.error(f"Tracking error: {str(e)}")
            # Copy points from current frame
            self.overlay_item.frame_points[new_frame_index] = current_points.copy()
            return False

        # Return False if no points were tracked
        return False

    def _set_tracking_crosshair_cursor(self):
        """Create and set a custom crosshair cursor for tracking mode"""
        # Create a transparent pixmap for custom cursor
        cursor_size = 31  # Should be odd for center alignment
        pixmap = QPixmap(cursor_size, cursor_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw crosshair
        center = cursor_size // 2
        
        # White outline for visibility on dark backgrounds
        painter.setPen(QPen(Qt.GlobalColor.white, 3))
        painter.drawLine(center, 0, center, cursor_size)
        painter.drawLine(0, center, cursor_size, center)
        
        # Thin black crosshair
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.drawLine(center, 0, center, cursor_size)
        painter.drawLine(0, center, cursor_size, center)
        
        # Small gap in center
        gap = 3
        painter.setPen(QPen(Qt.GlobalColor.transparent, 3))
        painter.drawLine(center - gap, center, center + gap, center)
        painter.drawLine(center, center - gap, center, center + gap)
        
        painter.end()
        
        # Create cursor with center as hotspot
        cursor = QCursor(pixmap, center, center)
        self.setCursor(cursor)

    def set_tracking_enabled(self, enabled: bool):
        """Enable or disable tracking"""
        self.tracking_enabled = enabled
        if not enabled:
            self.point_tracker.clear()
            self._tracked_point_ids.clear()
            # Restore default cursor
            if self.mode_manager.current_mode == ViewerMode.VIEW:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Set custom crosshair cursor for tracking
            # Use a small delay to ensure mode manager doesn't override
            QTimer.singleShot(100, self._apply_tracking_cursor)
            logger.info(f"Tracking enabled, cursor should be crosshair")
    
    def _apply_tracking_cursor(self):
        """Apply tracking cursor after delay"""
        if self.tracking_enabled and self.mode_manager.current_mode == ViewerMode.VIEW:
            self._set_tracking_crosshair_cursor()
    
    def _update_tracked_points(self, frame_index: int):
        """Ensure all points in the current frame are tracked"""
        frame_points = self.overlay_item.frame_points.get(frame_index, [])
        
        # Clear and re-add all points for simplicity
        self.point_tracker.clear()
        
        # Get the frame for this index
        frame = self.dicom_parser.get_frame(frame_index) if frame_index != self.current_frame_index else self.current_frame
        if frame is None:
            logger.error(f"Cannot get frame {frame_index} for point tracking")
            return
            
        # Set current frame in tracker
        self.point_tracker.set_frame(frame)
        
        for i, point in enumerate(frame_points):
            point_id = f"point_{i}"
            success = self.point_tracker.add_point(
                point_id,
                Point(float(point[0]), float(point[1]))
            )
            if success:
                self._tracked_point_ids[(frame_index, i)] = point_id
                logger.info(f"Added point {i} to tracker with ID {point_id}")
            else:
                logger.error(f"Failed to add point {i} to tracker")

    def track_all_frames(self, progress_callback=None, start_frame=None, end_frame=None):
        """Track points through all frames or a specified range"""
        if not self.dicom_parser or self.current_frame is None:
            return False

        # Get points from current frame
        current_points = self.overlay_item.frame_points.get(self.current_frame_index, [])
        if not current_points:
            return False

        # Set range limits
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.dicom_parser.num_frames - 1
            
        # Ensure current frame is within range
        if self.current_frame_index < start_frame or self.current_frame_index > end_frame:
            logger.warning(f"Current frame {self.current_frame_index} is outside range {start_frame}-{end_frame}")
            return False
        
        # Initialize batch tracking set if not exists
        if not hasattr(self, '_batch_tracked_frames'):
            self._batch_tracked_frames = set()
            
        # Store batch tracking info
        self._batch_tracking_start_frame = self.current_frame_index
        self._batch_tracking_range = (start_frame, end_frame)

        # Store original frame index
        original_frame = self.current_frame_index

        # Track forward from current frame (but only up to end_frame)
        for frame_idx in range(self.current_frame_index + 1, min(end_frame + 1, self.dicom_parser.num_frames)):
            if progress_callback:
                total_forward = end_frame - self.current_frame_index
                if total_forward > 0:
                    progress = int(((frame_idx - self.current_frame_index) / total_forward) * 50)
                else:
                    progress = 50
                if not progress_callback(progress, 100):
                    break

            # Update current frame index for template matching
            self.display_frame(frame_idx - 1)
            if not self.track_points_to_frame(frame_idx):
                break
            # Mark this frame as part of batch tracking
            self._batch_tracked_frames.add(frame_idx)

        # Reset to original frame and track backward
        self.display_frame(original_frame)

        # Track backward from current frame (but only down to start_frame)
        for frame_idx in range(self.current_frame_index - 1, max(start_frame - 1, -1), -1):
            if progress_callback:
                total_backward = self.current_frame_index - start_frame
                if total_backward > 0:
                    progress = 50 + int(((self.current_frame_index - frame_idx) / total_backward) * 50)
                else:
                    progress = 100
                if not progress_callback(progress, 100):
                    break

            # Update current frame index for template matching
            self.display_frame(frame_idx + 1)
            if not self.track_points_to_frame(frame_idx):
                break
            # Mark this frame as part of batch tracking
            self._batch_tracked_frames.add(frame_idx)

        # Restore original frame
        self.display_frame(original_frame)
        self._request_update()
        return True
    
    def _handle_batch_tracking_edit(self, edited_frame_index: int, edited_point_index: int):
        """Handle editing of a point that was part of batch tracking"""
        
        # Create message box with custom buttons
        msg = QMessageBox(self)
        msg.setWindowTitle("Batch Tracking Point Edited")
        msg.setText(f"You've edited a point that was tracked through multiple frames.\n\n"
                   f"Would you like to re-track this point from frame {edited_frame_index} onwards?")
        msg.setIcon(QMessageBox.Icon.Question)
        
        # Add custom buttons
        retrack_btn = msg.addButton("Re-track from here", QMessageBox.ButtonRole.YesRole)
        edit_only_btn = msg.addButton("Edit this frame only", QMessageBox.ButtonRole.NoRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        
        msg.exec()
        clicked_button = msg.clickedButton()
        
        if clicked_button == retrack_btn:
            # Re-track from the edited frame onwards
            self._retrack_from_frame(edited_frame_index, edited_point_index)
        elif clicked_button == edit_only_btn:
            # Just keep the edit, remove this frame from batch tracked set
            self._batch_tracked_frames.discard(edited_frame_index)
        else:
            pass
    
    def _retrack_from_frame(self, start_frame: int, point_index: int):
        """Re-track points from a specific frame after editing"""
        
        # Ensure we're on the right frame
        self.display_frame(start_frame)
        
        # Get the tracking range
        if hasattr(self, '_batch_tracking_range'):
            _, end_frame = self._batch_tracking_range
        else:
            end_frame = self.dicom_parser.num_frames - 1
        
        # Clear tracked points in frames after the edited one
        for frame_idx in range(start_frame + 1, end_frame + 1):
            if frame_idx in self._batch_tracked_frames:
                # Keep only non-edited points
                if frame_idx in self.overlay_item.frame_points:
                    # For simplicity, we'll clear and re-track all points
                    # In a more sophisticated implementation, we could track only the edited point
                    pass
        
        # Create progress dialog
        progress = QProgressDialog("Re-tracking points from edited frame...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        
        def update_progress(current, total):
            progress.setValue(current)
            return not progress.wasCanceled()
        
        # Perform tracking from current frame to end
        success = self.track_all_frames(progress_callback=update_progress, 
                                      start_frame=start_frame, 
                                      end_frame=end_frame)
        
        progress.close()
        
        if success:
            # Show completion dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Re-tracking Complete")
            msg.setText("Points have been successfully re-tracked from the edited frame.")
            msg.setIcon(QMessageBox.Icon.Information)
            
            # Add options for next action
            analyze_btn = msg.addButton("Analyze Results", QMessageBox.ButtonRole.ActionRole)
            edit_btn = msg.addButton("Continue Editing", QMessageBox.ButtonRole.ActionRole)
            done_btn = msg.addButton("Done", QMessageBox.ButtonRole.AcceptRole)
            
            msg.exec()
            clicked = msg.clickedButton()
            
            if clicked == analyze_btn:
                # Trigger analysis (e.g., QCA)
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.start_qca_analysis()
            elif clicked == edit_btn:
                # Stay in tracking mode
                pass
        else:
            QMessageBox.warning(self, "Re-tracking Failed", 
                              "Failed to re-track points from the edited frame.")

    def get_point_at_position(self, x: int, y: int, threshold: int = 10) -> int:
        """Get index of point at given position, -1 if none found"""
        if self.current_frame_index not in self.overlay_item.frame_points:
            return -1

        points = self.overlay_item.frame_points[self.current_frame_index]
        for i, (px, py) in enumerate(points):
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance <= threshold:
                return i
        return -1

    def show_point_context_menu(self, global_pos, x: int, y: int):
        """Show context menu for point operations"""
        from PyQt6.QtGui import QAction
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)

        # Check if clicking on existing point
        point_index = self.get_point_at_position(x, y)

        if point_index >= 0:
            # Point-specific actions
            delete_action = QAction("🗑 Delete Point", self)
            delete_action.triggered.connect(lambda: self.delete_point(point_index))
            menu.addAction(delete_action)

            menu.addSeparator()

            # Clear all points action
            clear_all_action = QAction("🗑 Clear All Points", self)
            clear_all_action.triggered.connect(self.clear_all_points_in_frame)
            menu.addAction(clear_all_action)
        else:
            # Add point action
            add_action = QAction("➕ Add Point Here", self)
            add_action.triggered.connect(lambda: self.add_point_at_position(x, y))
            menu.addAction(add_action)

            if self.current_frame_index in self.overlay_item.frame_points and \
               len(self.overlay_item.frame_points[self.current_frame_index]) > 0:
                menu.addSeparator()

                # Clear all points action
                clear_all_action = QAction("🗑 Clear All Points", self)
                clear_all_action.triggered.connect(self.clear_all_points_in_frame)
                menu.addAction(clear_all_action)
        
        # Frame range selection actions
        menu.addSeparator()
        
        # Set as start frame
        start_frame_action = QAction("🎬 Set as Start Frame", self)
        start_frame_action.triggered.connect(lambda: self.set_frame_range_start(self.current_frame_index))
        menu.addAction(start_frame_action)
        
        # Set as end frame
        end_frame_action = QAction("🏁 Set as End Frame", self)
        end_frame_action.triggered.connect(lambda: self.set_frame_range_end(self.current_frame_index))
        menu.addAction(end_frame_action)
        
        # Beat selection submenu
        if self.cardiac_phases is not None and 'frame_phases' in self.cardiac_phases:
            menu.addSeparator()
            
            # Create beat submenu
            beat_menu = menu.addMenu("💓 Select Beat")
            logger.info(f"Context menu: cardiac phases available, showing beat selection")
            
            # Get unique beat numbers
            beat_numbers = set()
            for phase_info in self.cardiac_phases['frame_phases']:
                if 'beat_number' in phase_info:
                    beat_numbers.add(phase_info['beat_number'])
            
            # Add beat options
            for beat_num in sorted(beat_numbers):
                beat_action = QAction(f"Beat {beat_num}", self)
                beat_action.triggered.connect(lambda checked, b=beat_num: self.select_beat_number(b))
                beat_menu.addAction(beat_action)
            
            # Also keep the original "Select Beat Around Frame" option
            beat_menu.addSeparator()
            beat_around_action = QAction("Select Beat Around Current Frame", self)
            beat_around_action.triggered.connect(lambda: self.select_beat_around_frame(self.current_frame_index))
            beat_menu.addAction(beat_around_action)
        
        # Show All Frames / Clear Beat Selection
        if (self.frame_range_start is not None or self.frame_range_end is not None or 
            (self.main_window and hasattr(self.main_window, 'navigation_range'))):
            menu.addSeparator()
            show_all_action = QAction("🎬 Show All Frames", self)
            show_all_action.triggered.connect(self.show_all_frames)
            menu.addAction(show_all_action)

        menu.exec(global_pos)

    def delete_point(self, index: int):
        """Delete point at given index"""
        if self.current_frame_index in self.overlay_item.frame_points:
            points = self.overlay_item.frame_points[self.current_frame_index]
            if 0 <= index < len(points):
                points.pop(index)
                self._request_update()
                self.points_changed.emit()

    def add_point_at_position(self, x: int, y: int):
        """Add a new point at the given position"""
        if self.current_frame_index not in self.overlay_item.frame_points:
            self.overlay_item.frame_points[self.current_frame_index] = []

        # Check if we already have 3 points in current frame
        if len(self.overlay_item.frame_points[self.current_frame_index]) >= 3:
            return

        self.overlay_item.frame_points[self.current_frame_index].append((x, y))
        self._request_update()
        self.points_changed.emit()
        self.segmentation_point_clicked.emit(x, y)

    def clear_all_points_in_frame(self):
        """Clear all points in current frame"""
        if self.current_frame_index in self.overlay_item.frame_points:
            self.overlay_item.frame_points[self.current_frame_index] = []
            self._request_update()
            self.points_changed.emit()
    
    def set_frame_range_start(self, frame_index: int):
        """Set the start frame for range selection"""
        self.frame_range_start = frame_index
        logger.info(f"Frame range start set to: {frame_index}")
        
        # If end frame is already set and valid, emit the range
        if self.frame_range_end is not None and self.frame_range_end > self.frame_range_start:
            self.frame_range_selected.emit(self.frame_range_start, self.frame_range_end)
    
    def set_frame_range_end(self, frame_index: int):
        """Set the end frame for range selection"""
        self.frame_range_end = frame_index
        logger.info(f"Frame range end set to: {frame_index}")
        
        # If start frame is already set and valid, emit the range
        if self.frame_range_start is not None and self.frame_range_start < self.frame_range_end:
            self.frame_range_selected.emit(self.frame_range_start, self.frame_range_end)
    
    def select_beat_around_frame(self, frame_index: int):
        """Select beat boundaries around the given frame using D2 points"""
        if self.cardiac_phases is None:
            logger.warning("No cardiac phases data available for beat selection")
            return
        
        # Get D2 points from cardiac phases
        d2_points = self.cardiac_phases.get('D2', [])
        if not d2_points:
            logger.warning("No D2 points found in cardiac phases")
            return
        
        # Get frame timestamps
        if self.frame_timestamps is None or frame_index >= len(self.frame_timestamps):
            logger.warning("No frame timestamps available")
            return
        
        current_timestamp = self.frame_timestamps[frame_index]
        
        # Find the D2 points before and after current timestamp
        d2_before = None
        d2_after = None
        
        for d2_time in d2_points:
            if d2_time <= current_timestamp:
                d2_before = d2_time
            elif d2_time > current_timestamp and d2_after is None:
                d2_after = d2_time
                break
        
        if d2_before is None or d2_after is None:
            logger.warning("Could not find beat boundaries around frame")
            return
        
        # Convert D2 times to frame indices
        start_frame = None
        end_frame = None
        
        for i, timestamp in enumerate(self.frame_timestamps):
            if start_frame is None and timestamp >= d2_before:
                start_frame = i
            if end_frame is None and timestamp >= d2_after:
                end_frame = i
                break
        
        if start_frame is not None and end_frame is not None:
            self.frame_range_start = start_frame
            self.frame_range_end = end_frame
            logger.info(f"Beat selected: frames {start_frame} to {end_frame}")
            self.frame_range_selected.emit(start_frame, end_frame)
    
    def select_beat_number(self, beat_number: int):
        """Select a specific beat by its number"""
        if self.cardiac_phases is None or 'frame_phases' not in self.cardiac_phases:
            logger.warning("No cardiac phase frame mapping available")
            return
        
        # Find all frames belonging to this beat
        beat_frames = []
        for phase_info in self.cardiac_phases['frame_phases']:
            if phase_info.get('beat_number') == beat_number:
                # Add all frames in this phase range
                for frame in range(phase_info['frame_start'], phase_info['frame_end'] + 1):
                    if frame not in beat_frames:
                        beat_frames.append(frame)
        
        if not beat_frames:
            logger.warning(f"No frames found for beat {beat_number}")
            return
        
        # Get the range of frames for this beat
        start_frame = min(beat_frames)
        end_frame = max(beat_frames)
        
        # Set frame range
        self.frame_range_start = start_frame
        self.frame_range_end = end_frame
        
        # Store beat-specific frames for precise filtering
        self.selected_beat_frames = sorted(beat_frames)
        
        logger.info(f"Beat {beat_number} selected: frames {start_frame} to {end_frame} (total {len(beat_frames)} frames) [UI: {start_frame + 1} to {end_frame + 1}]")
        
        # Navigate to the first frame of the beat
        # For beat 1, use the first frame; for other beats, use D2 frame
        if beat_number == 1:
            navigate_frame = start_frame
        else:
            # Find the D2 frame for this beat
            navigate_frame = start_frame
            for phase_info in self.cardiac_phases['frame_phases']:
                if (phase_info.get('beat_number') == beat_number and 
                    phase_info.get('phase') == 'd2'):
                    # Navigate to D2 frame (not after D2 ends)
                    navigate_frame = phase_info['frame_start']
                    break
        
        # Emit the range selection first
        self.frame_range_selected.emit(start_frame, end_frame)
        
        # Notify main window to update navigation and navigate to frame
        if self.main_window:
            self.main_window.set_navigation_range(start_frame, end_frame)
            # Navigate to the appropriate frame through main window
            if hasattr(self.main_window, 'navigate_to_frame'):
                self.main_window.navigate_to_frame(navigate_frame)
            elif hasattr(self.main_window, 'frame_slider'):
                self.main_window.frame_slider.setValue(navigate_frame)
    
    def clear_frame_range(self):
        """Clear the frame range selection"""
        self.frame_range_start = None
        self.frame_range_end = None
        # Also clear beat-specific frames
        if hasattr(self, 'selected_beat_frames'):
            self.selected_beat_frames = None
        logger.info("Frame range and beat selection cleared")
    
    def show_all_frames(self):
        """Restore full frame navigation by clearing both frame range and navigation limits"""
        # Clear frame range selection
        self.clear_frame_range()
        
        # Clear navigation limits in main window
        if self.main_window and hasattr(self.main_window, 'clear_navigation_range'):
            self.main_window.clear_navigation_range()
            
        logger.info("All frames restored - navigation limits cleared")

    def clear_overlays(self):
        """Clear all overlays"""
        self.overlay_item.segmentation_mask = None
        self.overlay_item.qca_results = None
        self.overlay_item.user_points = []
        self.overlay_item.frame_points = {}  # Clear all frame points
        # No calibration line to clear
        self._request_update()

    def set_segmentation_mask_overlay(self, mask: np.ndarray, settings: Dict):
        """Set segmentation mask for overlay painting"""
        self.overlay_item.segmentation_mask = mask
        self.overlay_item.segmentation_settings = settings
        self._request_update()

    def set_qca_overlay(self, results: Dict, settings: Dict):
        """Set QCA overlay"""
        self.overlay_item.qca_results = results
        self.overlay_item.qca_settings = settings
        logger.info(f"QCA overlay set with {len(results.get('centerline', []))} centerline points")
        logger.info(f"QCA settings: {settings}")
        self._request_update()


    def clear_user_points(self):
        """Clear user click points from all frames"""
        self.overlay_item.user_points = []
        self.overlay_item.frame_points = {}  # Clear all frame points
        self.overlay_item.calibration_points = []  # Clear calibration points too
        self._request_update()

    def get_current_pixmap(self) -> Optional[QPixmap]:
        """Get the current displayed pixmap"""
        return self.pixmap_item.pixmap()

    def toggle_segmentation_overlay(self, enabled: bool):
        """Toggle segmentation overlay visibility"""
        self.overlay_item.segmentation_settings['enabled'] = enabled
        self._request_update()

    def toggle_qca_overlay(self, enabled: bool):
        """Toggle QCA overlay visibility"""
        self.overlay_item.qca_settings['enabled'] = enabled
        self._request_update()


    def update_qca_overlay_settings(self, settings: Dict):
        """Update QCA overlay settings"""
        self.overlay_item.qca_settings.update(settings)
        self._request_update()


    def update_segmentation_overlay_settings(self, settings: Dict):
        """Update segmentation overlay settings"""
        self.overlay_item.segmentation_settings.update(settings)
        self._request_update()

    def enable_segmentation_mode(self, enabled: bool):
        """Enable/disable segmentation mode"""
        if enabled:
            self.mode_manager.set_mode(ViewerMode.SEGMENT)
        else:
            self.mode_manager.return_to_view()
    
    def enable_calibration_mode(self, enabled: bool):
        """Enable/disable calibration mode"""
        if enabled:
            self.mode_manager.set_mode(ViewerMode.CALIBRATE)
        else:
            self.mode_manager.return_to_view()
    
    def enable_multi_frame_mode(self, enabled: bool):
        """Enable/disable multi-frame segmentation mode"""
        if enabled:
            self.multi_frame_points = []  # Reset points
            self.mode_manager.set_mode(ViewerMode.MULTI_FRAME)
        else:
            # Clear multi-frame points
            self.multi_frame_points = []
            self.mode_manager.return_to_view()
    
    def set_segmentation_mode(self, enabled: bool):
        """Enable or disable segmentation mode"""
        # Use mode manager instead of setting property
        if enabled:
            self.mode_manager.set_mode(ViewerMode.SEGMENT)
        else:
            self.mode_manager.return_to_view()

        # Mode manager handles cursor and drag mode
        if not enabled:
            self.clear_segmentation_graphics()

    def set_calibration_mode(self, enabled: bool):
        """Enable or disable calibration mode"""
        # Use mode manager instead of setting property
        if enabled:
            self.mode_manager.set_mode(ViewerMode.CALIBRATE)
        else:
            self.mode_manager.return_to_view()

        # Mode manager handles cursor and drag mode
        if enabled:
            # Clear any existing calibration points
            self.clear_calibration_points()

    def set_multi_frame_mode(self, enabled: bool):
        """Enable or disable multi-frame selection mode"""
        # Use mode manager instead of setting property
        if enabled:
            self.mode_manager.set_mode(ViewerMode.MULTI_FRAME)
        else:
            self.mode_manager.return_to_view()

        # Mode manager handles cursor and drag mode
        if enabled:
            print(f"Multi-frame mode enabled in viewer")
            # Initialize multi-frame points
            self.multi_frame_points = []
        else:
            print(f"Multi-frame mode disabled in viewer")
            # Clear multi-frame points
            if hasattr(self, 'multi_frame_points'):
                self.multi_frame_points.clear()
            self.overlay_item.user_points = []
            self._request_update()

    def add_calibration_point(self, x: int, y: int):
        """Add a calibration point and visualize it"""
        # Use overlay item's calibration points
        if not hasattr(self.overlay_item, 'calibration_points'):
            self.overlay_item.calibration_points = []

        # Store in frame_points like segmentation does
        if self.current_frame_index not in self.overlay_item.frame_points:
            self.overlay_item.frame_points[self.current_frame_index] = []

        # Add point (max 2 for calibration)
        if len(self.overlay_item.calibration_points) < 2:
            point = (x, y)
            self.overlay_item.calibration_points.append(point)
            # Also add to frame points for consistent visualization
            self.overlay_item.frame_points[self.current_frame_index].append(point)

        # No line drawing between calibration points

        self._request_update()

    def clear_calibration_points(self):
        """Clear all calibration points and graphics"""
        if hasattr(self.overlay_item, 'calibration_points'):
            self.overlay_item.calibration_points = []
        if hasattr(self.overlay_item, 'calibration_line'):
            # No calibration line to clear
            pass
        if hasattr(self.overlay_item, 'calibration_mask'):
            self.overlay_item.calibration_mask = None

        # Also clear from frame_points
        if self.current_frame_index in self.overlay_item.frame_points:
            self.overlay_item.frame_points[self.current_frame_index] = []

        self._request_update()

    def add_segmentation_point(self, x: int, y: int):
        """Add a segmentation point and visualize it"""
        # Check if we already have 3 points in current frame
        if self.current_frame_index in self.overlay_item.frame_points:
            if len(self.overlay_item.frame_points[self.current_frame_index]) >= 3:
                return

        point = (x, y)
        self.segmentation_points.append(point)

        # No longer tracking stenosis points separately

        # Add to overlay for visualization - store in frame-specific dict
        if self.current_frame_index not in self.overlay_item.frame_points:
            self.overlay_item.frame_points[self.current_frame_index] = []
        self.overlay_item.frame_points[self.current_frame_index].append(point)
        self._request_update()
        # Emit signal to update tracking buttons
        self.points_changed.emit()

        # Force scene update
        self.scene.update()


    def clear_segmentation_graphics(self):
        """Clear segmentation graphics only, keep points"""
        self.overlay_item.segmentation_mask = None
        self.overlay_item.segmentation_centerline = None
        self.overlay_item.segmentation_boundaries = None
        self._request_update()

        # Clear segmentation results
        self.current_segmentation_result = None
        self.segmented_frame_index = None

        # Restore original image
        if self.current_frame_index is not None:
            self.display_frame(self.current_frame_index)


    def create_segmented_image(self, original_image: np.ndarray, result: dict) -> np.ndarray:
        """Create visualization combining original image with segmentation results"""

        # Make a copy to avoid modifying original
        if len(original_image.shape) == 2:
            # Convert grayscale to RGB for colored visualization
            image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            image = original_image.copy()

        mask = result.get('mask')
        result.get('centerline')
        result.get('boundaries')

        if mask is not None:

            # Ensure mask and image have same dimensions
            if mask.shape != image.shape[:2]:
                logger.warning(f"Mask shape {mask.shape} != Image shape {image.shape[:2]}, resizing mask")
                # Resize mask to match image with proper aspect ratio preservation
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Debug: Check if mask has any positive values
            np.sum(mask > 0)

            # Create colored mask overlay
            mask_colored = np.zeros_like(image)
            # Use bright red for better visibility
            mask_bool = mask > 0
            mask_colored[mask_bool, 0] = 255  # Red channel (RGB format)
            mask_colored[mask_bool, 1] = 0    # Green channel
            mask_colored[mask_bool, 2] = 0    # Blue channel

            # Debug: Check colored mask
            colored_sum = np.sum(mask_colored[:,:,0] > 0)

            # Get settings for opacity from result if available
            result.get('settings', {})
            # Use higher opacity for better visibility
            alpha = 0.5  # 50% opacity for better visibility

            # Blend with original image
            image = cv2.addWeighted(image, 1-alpha, mask_colored, alpha, 0)

        # Draw centerline if available - DISABLED
        # if centerline:
        #     for i in range(len(centerline) - 1):
        #         pt1 = (int(centerline[i][0]), int(centerline[i][1]))
        #         pt2 = (int(centerline[i+1][0]), int(centerline[i+1][1]))
        #         cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Green centerline

        # Draw boundaries if available - DISABLED
        # if boundaries:
        #     for boundary in boundaries:
        #         if len(boundary) > 2:
        #             pts = np.array(boundary, np.int32)
        #             pts = pts.reshape((-1, 1, 2))
        #             cv2.polylines(image, [pts], True, (0, 255, 255), 2)  # Yellow boundaries

        # Don't draw user click points on the segmented image - they're already visible on the overlay
        # Comment out to remove the large yellow/cyan dots
        # for i, point in enumerate(self.segmentation_points):
        #     color = (255, 255, 0) if i < 2 else (0, 255, 255)  # Yellow for stenosis, cyan for guide
        #     # Draw single pixel
        #     if 0 <= point[1] < image.shape[0] and 0 <= point[0] < image.shape[1]:
        #         image[point[1], point[0]] = color

        return image

    def set_segmentation_overlay(self, result: dict, settings: dict):
        """Set segmentation mask for overlay rendering"""

        if result.get('success') and result.get('mask') is not None:
            mask = result['mask']

            # Ensure mask is binary (0 or 1)
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)

            # Debug: Log mask statistics
            logger.info(f"Segmentation mask shape: {mask.shape}")
            logger.info(f"Non-zero pixels in mask: {np.sum(mask > 0)}")
            if np.sum(mask > 0) > 0:
                y_indices, x_indices = np.where(mask > 0)
                logger.info(f"Mask bounds - Y: [{y_indices.min()}, {y_indices.max()}], X: [{x_indices.min()}, {x_indices.max()}]")
                logger.info(f"Mask area: {(y_indices.max() - y_indices.min()) * (x_indices.max() - x_indices.min())} pixels")

            # Check if mask needs resizing to match current frame
            if self.current_frame is not None:
                frame_shape = self.current_frame.shape[:2]
                if mask.shape != frame_shape:
                    logger.warning(f"Resizing mask from {mask.shape} to match frame {frame_shape}")
                    # Use INTER_NEAREST to preserve binary mask values
                    mask = cv2.resize(mask.astype(np.uint8), (frame_shape[1], frame_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Set mask and settings in overlay item
            self.overlay_item.segmentation_mask = mask
            self.overlay_item.segmentation_settings = settings
            self.overlay_item.segmentation_settings['enabled'] = True

            # Store centerline and boundaries if available
            if 'centerline' in result:
                centerline = result['centerline']
                logger.info(f"Received centerline with {len(centerline) if centerline is not None else 0} points")
                if centerline is not None and len(centerline) > 0:
                    logger.info(f"Centerline shape: {np.array(centerline).shape}")
                    if len(centerline) <= 10:
                        logger.info(f"All centerline points: {centerline}")
                    else:
                        logger.info(f"Sample centerline - First: {centerline[0]}, Last: {centerline[-1]}")
                self.overlay_item.segmentation_centerline = centerline
            if 'boundaries' in result:
                self.overlay_item.segmentation_boundaries = result['boundaries']

            # Store result for potential save/export
            self.current_segmentation_result = result
            self.segmented_frame_index = self.current_frame_index

            self._request_update()

    def set_segmentation_overlay_settings(self, settings: dict):
        """Update segmentation overlay settings"""
        self.overlay_item.segmentation_settings.update(settings)
        self._request_update()

    def zoom_in(self):
        """Zoom in"""
        self.scale(1.2, 1.2)
        self.zoom_factor *= 1.2
        self.zoom_changed.emit(self.zoom_factor)

    def zoom_out(self):
        """Zoom out"""
        self.scale(0.8, 0.8)
        self.zoom_factor *= 0.8
        self.zoom_changed.emit(self.zoom_factor)

    def set_window_level(self, center: float, width: float):
        """Set window/level values"""
        self.window_center = center
        self.window_width = width
        self.clear_cache()
        if self.current_frame_index is not None:
            self.display_frame(self.current_frame_index)

    def _update_tracking_config(self):
        """Update tracking configuration based on current pixel spacing and frame rate"""
        # TAPIR tracker automatically handles configuration
        # No need to update config as TAPIR uses optimized settings
        pass
    
    def set_calibration_factor(self, factor: float):
        """Set calibration factor and update tracking config"""
        self.calibration_factor = factor
        self._update_tracking_config()
    
    def set_tracking_method(self, method: str):
        """Set point tracking method - now using TAPIR for all methods"""
        # TAPIR handles all tracking methods internally
        logger.info(f"Tracking method '{method}' requested - using TAPIR optimized tracking")

    def start_calibration(self, catheter_size: str, catheter_diameter_mm: float):
        """Start calibration mode"""
        self.set_interaction_mode('calibrate')
        self.catheter_size = catheter_size
        self.catheter_diameter_mm = catheter_diameter_mm

        # Ensure widget has focus
        self.setFocus()

    def stop_calibration(self):
        """Stop calibration mode"""
        self.set_interaction_mode('view')
        # Clear calibration graphics when stopping
        self.clear_calibration_graphics()

    def clear_calibration_graphics(self):
        """Clear calibration graphics"""
        # No calibration line to clear
        self.calibration_points = []
        self._request_update()

    def measure_catheter_width(self, p1: tuple, p2: tuple) -> float:
        """
        Measure catheter width between two points using improved algorithm

        Args:
            p1: First point (x, y)
            p2: Second point (x, y)

        Returns:
            Width in pixels, or 0 if measurement failed
        """
        if self.current_frame is None:
            return 0

        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate line direction
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.sqrt(dx*dx + dy*dy)
        
        if line_length == 0:
            return 0
        
        # Normalized direction
        line_dx = dx / line_length
        line_dy = dy / line_length
        
        # Perpendicular direction
        perp_dx = -line_dy
        perp_dy = line_dx
        
        # Sample multiple points along the line
        num_samples = 50  # Increased for better sampling
        all_widths = []
        
        logger.debug(f"Measuring catheter between ({x1},{y1}) and ({x2},{y2}), line length: {line_length:.1f}px")
        
        for i in range(num_samples):
            # Interpolate point along the line
            t = i / (num_samples - 1)
            center_x = x1 + t * dx
            center_y = y1 + t * dy
            
            # Get intensity profile perpendicular to line
            profile_length = 80  # pixels to scan on each side
            profile_values = []
            positions = []
            
            for dist in range(-profile_length, profile_length + 1):
                sample_x = int(center_x + dist * perp_dx)
                sample_y = int(center_y + dist * perp_dy)
                
                if 0 <= sample_x < self.current_frame.shape[1] and 0 <= sample_y < self.current_frame.shape[0]:
                    profile_values.append(self.current_frame[sample_y, sample_x])
                    positions.append(dist)
            
            if len(profile_values) < 10:
                continue
                
            profile_values = np.array(profile_values)
            positions = np.array(positions)
            
            # Find catheter width using gradient-based method
            width = self._find_catheter_width_from_profile(profile_values, positions)
            if width > 0:
                all_widths.append(width)
        
        if all_widths:
            # Use 75th percentile to avoid underestimation
            # Catheters often have varying apparent width due to contrast/angle
            width = np.percentile(all_widths, 75)
            logger.info(f"Catheter width measurement: {width:.2f} pixels (from {len(all_widths)} samples)")
            return width
        
        logger.warning("Failed to measure catheter width")
        return 0
    
    def _find_catheter_width_from_profile(self, profile: np.ndarray, positions: np.ndarray) -> float:
        """
        Find catheter width from intensity profile using gradient analysis.
        
        Args:
            profile: Intensity values
            positions: Position values corresponding to profile
            
        Returns:
            Width in pixels
        """
        if len(profile) < 5:
            return 0
        
        # Smooth the profile first
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(profile.astype(float), sigma=2)
        
        # Calculate gradient
        gradient = np.gradient(smoothed)
        
        # Find significant edges
        center_idx = len(profile) // 2
        threshold = np.std(gradient) * 0.3  # Reduced threshold for better edge detection
        
        # Debug logging
        logger.debug(f"Profile analysis - length: {len(profile)}, center: {center_idx}")
        logger.debug(f"Gradient std: {np.std(gradient):.3f}, threshold: {threshold:.3f}")
        
        # Find left edge (positive gradient - dark to bright)
        left_edge = None
        for i in range(center_idx, 0, -1):
            if gradient[i] > threshold:
                # Sub-pixel refinement
                if i > 0 and i < len(gradient) - 1:
                    y1, y2, y3 = gradient[i-1], gradient[i], gradient[i+1]
                    if abs(y1 - 2*y2 + y3) > 1e-8:
                        offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                        left_edge = positions[i] + offset
                    else:
                        left_edge = positions[i]
                else:
                    left_edge = positions[i]
                break
        
        # Find right edge (negative gradient - bright to dark)
        right_edge = None
        for i in range(center_idx, len(gradient)):
            if gradient[i] < -threshold:
                # Sub-pixel refinement
                if i > 0 and i < len(gradient) - 1:
                    y1, y2, y3 = gradient[i-1], gradient[i], gradient[i+1]
                    if abs(y1 - 2*y2 + y3) > 1e-8:
                        offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
                        right_edge = positions[i] + offset
                    else:
                        right_edge = positions[i]
                else:
                    right_edge = positions[i]
                break
        
        if left_edge is not None and right_edge is not None:
            width = right_edge - left_edge
            logger.debug(f"Edge detection: left={left_edge:.1f}, right={right_edge:.1f}, width={width:.1f}")
            # Sanity check - catheter should be between 10-50 pixels typically
            if 5 < width < 100:
                return width
            else:
                logger.debug(f"Width {width:.1f} outside reasonable range (5-100)")
        
        logger.debug("Failed to measure width using edge detection")
        return 0

    def measure_width_at_point(self, edges: np.ndarray, point: tuple, direction: tuple) -> float:
        """
        Measure width at a specific point along a direction

        Args:
            edges: Edge image
            point: Center point (x, y)
            direction: Normalized direction vector (dx, dy)

        Returns:
            Width in pixels
        """
        x, y = point
        dx, dy = direction
        max_dist = 30  # Maximum search distance

        # Find edges on both sides
        edge_distances = []

        for sign in [-1, 1]:
            for dist in range(1, max_dist):
                sample_x = int(x + sign * dist * dx)
                sample_y = int(y + sign * dist * dy)

                # Check bounds
                if (0 <= sample_x < edges.shape[1] and
                    0 <= sample_y < edges.shape[0]):
                    if edges[sample_y, sample_x] > 0:
                        edge_distances.append(dist)
                        break

        if len(edge_distances) == 2:
            return sum(edge_distances)

        return 0

    def get_current_frame(self):
        """Get the current frame being displayed"""
        return self.current_frame

    def keyPressEvent(self, event):
        """Handle key press events"""
        # First check for mode-specific actions that we want to handle
        if event.key() == Qt.Key.Key_Escape:
            # Cancel current operation based on mode
            if self.interaction_mode == 'calibrate':
                # Clear calibration and emit cancelled signal
                self.clear_calibration_graphics()
                self.set_interaction_mode('view')
                # Emit a signal that calibration was cancelled
                self.calibration_cancelled.emit()
                event.accept()
                return
            elif self.interaction_mode == 'segment':
                # Clear segmentation mode
                self.set_interaction_mode('view')
                event.accept()
                return
        elif event.key() == Qt.Key.Key_F:
            self.fit_to_window()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.zoom_in()
            event.accept()
            return
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
            return

        # For navigation keys (arrows, space, home, end), ignore the event
        # so it bubbles up to the main window for playback control
        if event.key() in [Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Space,
                          Qt.Key.Key_Home, Qt.Key.Key_End]:
            event.ignore()
        else:
            # For other keys, let parent handle them
            super().keyPressEvent(event)

    def _request_update(self):
        """Request a batched update"""
        self._pending_updates = True
        if not self._update_timer.isActive():
            self._update_timer.start()

    def _batch_update(self):
        """Perform batched updates"""
        if self._pending_updates:
            self.overlay_item.prepareGeometryChange()
            self.overlay_item.setPos(0, 0)
            self.overlay_item.update()
            self.scene.update()
            self._pending_updates = False
        self._update_timer.stop()

    def _track_points_between_frames(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                                   points: list) -> list:
        """Track points between two frames (for use in worker thread)"""
        # Create a temporary tracker for thread-safe operation
        from ..core.simple_tracker import SimpleTracker
        temp_tracker = SimpleTracker()
        
        # Set initial frame
        temp_tracker.set_frame(prev_frame)
        
        # Add all points to the temporary tracker
        for i, point in enumerate(points):
            temp_tracker.add_point(
                str(i),
                Point(point[0], point[1])
            )
        
        # Track to next frame
        tracking_results = temp_tracker.track_in_frame(curr_frame)
        
        # Convert results back to list format
        tracked_points = []
        for i, point in enumerate(points):
            point_id = str(i)
            if point_id in tracking_results:
                new_pos = tracking_results[point_id]
                tracked_points.append((new_pos.x, new_pos.y))
            else:
                # Keep original position if tracking failed
                tracked_points.append(point)
        
        return tracked_points

    def start_background_tracking(self, target_frame: int):
        """Start tracking in background thread"""
        if self.tracking_worker and self.tracking_worker.isRunning():
            return

        current_points = self.overlay_item.frame_points.get(self.current_frame_index, [])
        if not current_points:
            return

        # Create worker
        self.tracking_worker = TrackingWorker(
            self, self.current_frame_index, target_frame, current_points
        )

        # Connect signals
        self.tracking_worker.signals.result.connect(self._on_tracking_complete)
        self.tracking_worker.signals.progress.connect(self._on_tracking_progress)

        # Start worker
        self.tracking_worker.start()

    def _on_tracking_complete(self, result: dict):
        """Handle tracking completion"""
        if result['success']:
            tracked_frames = result['data']
            # Update frame points
            for frame_idx, points in tracked_frames.items():
                self.overlay_item.frame_points[frame_idx] = points
            self._request_update()
            self.points_changed.emit()

    def _on_tracking_progress(self, current: int, total: int):
        """Handle tracking progress"""
        # Could emit to progress bar

    def clear_cache(self):
        """Clear frame cache when window/level changes"""
        self._frame_cache.clear()

    def display_qca_results(self, qca_results: Dict):
        """Display QCA analysis results on the viewer"""
        # Use the overlay item for QCA results
        if qca_results.get('success'):
            self.overlay_item.qca_results = qca_results
            self.overlay_item.qca_settings['enabled'] = True
            self._request_update()

    def _draw_qca_centerline(self):
        """Draw the vessel centerline"""
        # Create path for centerline
        path = QPainterPath()
        first_point = True

        if self.qca_centerline is not None and len(self.qca_centerline) > 0:
            for point in self.qca_centerline:
                x, y = point[1], point[0]  # Note: centerline is in (y,x) format
                if first_point:
                    path.moveTo(x, y)
                    first_point = False
                else:
                    path.lineTo(x, y)

        # Create path item
        centerline_item = QGraphicsPathItem(path)
        centerline_item.setPen(QPen(QColor(0, 255, 255), 2))  # Cyan color, 2px width
        centerline_item.setZValue(10)  # Ensure it's on top

        self.scene.addItem(centerline_item)
        self.qca_visualization_items.append(centerline_item)

    def _draw_diameter_measurements(self):
        """Draw diameter measurements along the centerline"""
        if self.qca_centerline is None or self.qca_diameters is None:
            return
            
        if len(self.qca_centerline) == 0 or len(self.qca_diameters) == 0:
            return
            
        # Sample every N points to avoid overcrowding
        sample_interval = max(1, len(self.qca_centerline) // 50)

        for i in range(0, len(self.qca_centerline), sample_interval):
            if i >= len(self.qca_diameters):
                break

            point = self.qca_centerline[i]
            diameter = self.qca_diameters[i]

            if diameter > 0:
                # Calculate perpendicular direction
                if i > 0 and i < len(self.qca_centerline) - 1:
                    prev = self.qca_centerline[i-1]
                    next_point = self.qca_centerline[i+1]
                    # Centerline is in (y, x) format
                    direction = next_point - prev

                    # Normalize and get perpendicular
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        # Get perpendicular in (y, x) format
                        perp = np.array([-direction[1], direction[0]])

                        # Draw diameter line
                        # point is in (y, x) format
                        center_y, center_x = point[0], point[1]
                        half_diameter = diameter / 2

                        # Apply perpendicular in correct coordinate system
                        y1 = center_y + perp[0] * half_diameter
                        x1 = center_x + perp[1] * half_diameter
                        y2 = center_y - perp[0] * half_diameter
                        x2 = center_x - perp[1] * half_diameter

                        line = QGraphicsLineItem(x1, y1, x2, y2)
                        line.setPen(QPen(QColor(255, 255, 0, 100), 1))  # Yellow, semi-transparent
                        line.setZValue(5)

                        self.scene.addItem(line)
                        self.qca_visualization_items.append(line)

    def _draw_stenosis_markers(self, qca_results: Dict):
        """Draw markers for stenosis locations"""
        # Draw MLD (Minimal Lumen Diameter) location
        mld_location = qca_results.get('mld_location')
        if mld_location is not None:
            y, x = mld_location[0], mld_location[1]

            # Create small 2x2 pixel red square for MLD
            mld_marker = QGraphicsRectItem(x-1, y-1, 2, 2)
            mld_marker.setPen(QPen(QColor(255, 0, 0), 1))
            mld_marker.setBrush(QBrush(QColor(255, 0, 0)))
            mld_marker.setZValue(16)

            self.scene.addItem(mld_marker)
            self.qca_visualization_items.append(mld_marker)

            # Add text label
            mld_text = QGraphicsTextItem(f"MLD: {qca_results.get('mld', 0):.3f} mm")
            mld_text.setPos(x + 5, y - 10)
            mld_text.setDefaultTextColor(QColor(255, 0, 0))
            font = QFont("Arial", 10)
            font.setBold(True)
            mld_text.setFont(font)
            mld_text.setZValue(20)

            self.scene.addItem(mld_text)
            self.qca_visualization_items.append(mld_text)


    def clear_qca_visualization(self):
        """Clear all QCA visualization items"""
        for item in self.qca_visualization_items:
            self.scene.removeItem(item)
        self.qca_visualization_items = []
        self.show_qca_overlay = False
    
    def _position_heartbeat_overlay(self):
        """Position heartbeat overlay in top-right corner"""
        if hasattr(self, 'heartbeat_proxy') and hasattr(self, 'heartbeat_overlay'):
            # Get viewport rect in widget coordinates
            viewport_rect = self.viewport().rect()
            
            # Convert to scene coordinates
            top_right = self.mapToScene(viewport_rect.topRight())
            
            # Position with margin from right edge
            x = top_right.x() - self.heartbeat_overlay.width() - 20
            y = top_right.y() + 20
            
            self.heartbeat_proxy.setPos(x, y)
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # Reposition overlay when window resizes
        if hasattr(self, '_position_heartbeat_overlay'):
            self._position_heartbeat_overlay()
    
    def update_heartbeat_sync_status(self, is_synced: bool):
        """Update heartbeat overlay sync status"""
        if hasattr(self, 'heartbeat_overlay'):
            self.heartbeat_overlay.set_sync_status(is_synced)
    
    def trigger_heartbeat(self):
        """Trigger heartbeat animation"""
        if hasattr(self, 'heartbeat_overlay'):
            self.heartbeat_overlay.heartbeat()
    
    def update_heart_rate(self, rate: float):
        """Update heart rate display"""
        if hasattr(self, 'heartbeat_overlay'):
            self.heartbeat_overlay.update_heart_rate(rate)
    
    def reset_heartbeat_overlay(self):
        """Reset heartbeat overlay"""
        if hasattr(self, 'heartbeat_overlay'):
            self.heartbeat_overlay.reset()
    
    def set_current_beat(self, current_beat: int, total_beats: int):
        """Set current beat and total beats in heartbeat overlay"""
        if hasattr(self, 'heartbeat_overlay'):
            self.heartbeat_overlay.set_current_beat(current_beat, total_beats)
    
    
    
    def perform_single_point_tracking(self, point_x: float, point_y: float):
        """
        Perform single-point tracking:
        1. Segment vessel around the point
        2. Extract centerline
        3. Align centerline midpoint to selected point
        4. Create proximal/distal segments
        
        Args:
            point_x: X coordinate of selected point
            point_y: Y coordinate of selected point
        """
        logger.info(f"Performing single-point tracking at ({point_x}, {point_y})")
        
        if self.current_frame is None:
            logger.error("No current frame available for single-point tracking")
            return False
        
        try:
            # Step 1: Segment vessel around the point
            segmentation_result = self._segment_around_point(point_x, point_y)
            if segmentation_result is None:
                logger.error("Segmentation failed")
                return False
            
            self.single_point_segmentation_result = segmentation_result
            
            # Step 2: Extract centerline from segmentation
            centerline = self._extract_centerline_from_segmentation(segmentation_result)
            if centerline is None or len(centerline) < 10:
                logger.error("Centerline extraction failed or too short")
                return False
            
            self.single_point_centerline = centerline
            
            # Step 3: Align centerline midpoint to selected point
            aligned_centerline = self._align_centerline_to_point(centerline, point_x, point_y)
            self.single_point_aligned_centerline = aligned_centerline
            
            # Step 4: Create proximal/distal segments
            proximal_segment, distal_segment = self._create_proximal_distal_segments(aligned_centerline)
            self.single_point_proximal_segment = proximal_segment
            self.single_point_distal_segment = distal_segment
            
            # Update visualization
            self.overlay_item.single_point_results = {
                'segmentation': segmentation_result,
                'centerline': aligned_centerline,
                'proximal_segment': proximal_segment,
                'distal_segment': distal_segment,
                'selected_point': (point_x, point_y)
            }
            self.overlay_item.update()
            self._request_update()
            
            logger.info("Single-point tracking completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Single-point tracking failed: {str(e)}")
            return False
    
    def _segment_around_point(self, point_x: float, point_y: float):
        """Segment vessel around the selected point"""
        try:
            # Import segmentation module
            from ..analysis.angiopy_segmentation import AngioPySegmentation
            
            # Create segmentation instance
            segmenter = AngioPySegmentation()
            
            # Perform segmentation on current frame
            result = segmenter.segment_frame(self.current_frame)
            
            if result is None:
                logger.error("Segmentation returned None")
                return None
            
            # Extract binary mask
            if 'binary_mask' in result:
                return result['binary_mask']
            elif 'segmentation_mask' in result:
                return result['segmentation_mask']
            else:
                logger.error("No segmentation mask found in result")
                return None
                
        except Exception as e:
            logger.error(f"Segmentation error: {str(e)}")
            return None
    
    def _extract_centerline_from_segmentation(self, segmentation_mask):
        """Extract centerline from segmentation mask"""
        try:
            from skimage.morphology import skeletonize
            from scipy import ndimage
            
            # Ensure binary mask
            if segmentation_mask.dtype != bool:
                binary_mask = segmentation_mask > 0.5 if segmentation_mask.max() <= 1.0 else segmentation_mask > 127
            else:
                binary_mask = segmentation_mask
            
            # Clean up mask
            binary_mask = ndimage.binary_opening(binary_mask, iterations=2)
            binary_mask = ndimage.binary_closing(binary_mask, iterations=2)
            
            # Extract skeleton
            skeleton = skeletonize(binary_mask)
            
            # Convert skeleton to points
            y_coords, x_coords = np.where(skeleton)
            if len(y_coords) == 0:
                return None
            
            # Sort points to form a continuous line
            centerline_points = self._order_skeleton_points(y_coords, x_coords)
            
            return centerline_points
            
        except Exception as e:
            logger.error(f"Centerline extraction error: {str(e)}")
            return None
    
    def _order_skeleton_points(self, y_coords, x_coords):
        """Order skeleton points to form a continuous centerline"""
        if len(y_coords) == 0:
            return np.array([])
        
        points = np.column_stack((y_coords, x_coords))
        
        # Simple ordering: start from one end and follow nearest neighbors
        ordered_points = [points[0]]
        remaining_points = list(range(1, len(points)))
        
        while remaining_points:
            current_point = ordered_points[-1]
            
            # Find nearest remaining point
            distances = [np.linalg.norm(points[i] - current_point) for i in remaining_points]
            nearest_idx = remaining_points[np.argmin(distances)]
            
            ordered_points.append(points[nearest_idx])
            remaining_points.remove(nearest_idx)
        
        return np.array(ordered_points)
    
    def _align_centerline_to_point(self, centerline, point_x, point_y):
        """Align centerline midpoint to selected point"""
        if len(centerline) == 0:
            return centerline
        
        # Find midpoint of centerline
        mid_index = len(centerline) // 2
        centerline_midpoint = centerline[mid_index]
        
        # Calculate translation
        target_point = np.array([point_y, point_x])  # Note: centerline is in (y, x) format
        translation = target_point - centerline_midpoint
        
        # Apply translation to all centerline points
        aligned_centerline = centerline + translation
        
        return aligned_centerline
    
    def _create_proximal_distal_segments(self, centerline):
        """Create proximal and distal segments from centerline"""
        if len(centerline) == 0:
            return None, None
        
        # Find midpoint
        mid_index = len(centerline) // 2
        
        # Split into proximal and distal segments
        proximal_segment = centerline[:mid_index + 1]  # Include midpoint
        distal_segment = centerline[mid_index:]        # Include midpoint
        
        return proximal_segment, distal_segment

    # Calibration overlay methods
    def clear_calibration_overlays(self):
        """Clear all calibration overlays"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            # Clear calibration-specific overlays
            self.clear_calibration_mask_overlay()
            self.clear_calibration_centerline_overlay() 
            self.clear_calibration_diameter_overlay()
            self._request_update()
        logger.info("Cleared all calibration overlays")

    def set_calibration_mask_overlay(self, mask: np.ndarray):
        """Set AngioPy segmentation mask overlay for calibration"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            self.overlay_item.calibration_mask = mask
            self._request_update()
            logger.info(f"✅ Set calibration mask overlay - shape: {mask.shape}, nonzero: {np.count_nonzero(mask)}")
        else:
            logger.error("❌ Cannot set calibration mask overlay - no overlay_item")

    def clear_calibration_mask_overlay(self):
        """Clear AngioPy mask overlay"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            self.overlay_item.calibration_mask = None
            self._request_update()
            logger.info("✅ Cleared calibration mask overlay")
        else:
            logger.warning("❌ Cannot clear calibration mask overlay - no overlay_item")

    def set_calibration_centerline_overlay(self, centerline: np.ndarray):
        """Set automatic centerline overlay for calibration"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            if not hasattr(self.overlay_item, 'calibration_centerline'):
                self.overlay_item.calibration_centerline = None
            self.overlay_item.calibration_centerline = centerline
            self._request_update()
            logger.info(f"✅ Set calibration centerline overlay: {len(centerline)} points")
        else:
            logger.error("❌ Cannot set calibration centerline overlay - no overlay_item")

    def clear_calibration_centerline_overlay(self):
        """Clear automatic centerline overlay"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            if hasattr(self.overlay_item, 'calibration_centerline'):
                self.overlay_item.calibration_centerline = None
            self._request_update()

    def set_calibration_diameter_overlay(self, centerline: np.ndarray, diameters: list):
        """Set automatic diameter overlay for calibration"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            if not hasattr(self.overlay_item, 'calibration_diameters'):
                self.overlay_item.calibration_diameters = None
            self.overlay_item.calibration_diameters = {
                'centerline': centerline,
                'diameters': diameters
            }
            self._request_update()
            logger.info(f"✅ Set calibration diameter overlay: {len(centerline)} points, {len(diameters)} diameters")
        else:
            logger.error("❌ Cannot set calibration diameter overlay - no overlay_item")

    def clear_calibration_diameter_overlay(self):
        """Clear automatic diameter overlay"""
        if hasattr(self, 'overlay_item') and self.overlay_item:
            if hasattr(self.overlay_item, 'calibration_diameters'):
                self.overlay_item.calibration_diameters = None
            self._request_update()

