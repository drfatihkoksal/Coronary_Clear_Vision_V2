"""
Heartbeat overlay widget with animated heart and sync status
"""

from PyQt6.QtWidgets import QWidget, QLabel, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QSequentialAnimationGroup, pyqtProperty, QEasingCurve, QRect
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath
import math
import logging

logger = logging.getLogger(__name__)


class HeartbeatOverlayWidget(QWidget):
    """Overlay widget showing animated heartbeat with sync status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Don't set these attributes when embedding in QGraphicsProxyWidget
        # self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        
        # State
        self._beat_count = 0
        self._current_beat = 0
        self._total_beats = 0
        self._heart_rate = 0
        self._is_synced = True
        self._heart_scale = 1.0
        self._heart_opacity = 0.8
        self._pulse_active = False
        
        # Animation setup
        self.setup_animations()
        
        # Size and position
        self.setFixedSize(150, 100)
        
    def setup_animations(self):
        """Setup heart beat animations"""
        # Heart scale animation (beat effect)
        self.scale_animation = QPropertyAnimation(self, b"heart_scale")
        self.scale_animation.setDuration(150)
        self.scale_animation.setStartValue(1.0)
        self.scale_animation.setEndValue(1.3)
        self.scale_animation.setEasingCurve(QEasingCurve.Type.OutElastic)
        
        # Return to normal size
        self.scale_back_animation = QPropertyAnimation(self, b"heart_scale")
        self.scale_back_animation.setDuration(350)
        self.scale_back_animation.setStartValue(1.3)
        self.scale_back_animation.setEndValue(1.0)
        self.scale_back_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Combine animations
        self.beat_animation = QSequentialAnimationGroup()
        self.beat_animation.addAnimation(self.scale_animation)
        self.beat_animation.addAnimation(self.scale_back_animation)
        
        # Opacity pulse for warning
        self.opacity_animation = QPropertyAnimation(self, b"heart_opacity")
        self.opacity_animation.setDuration(1000)
        self.opacity_animation.setStartValue(0.3)
        self.opacity_animation.setEndValue(0.9)
        self.opacity_animation.setLoopCount(-1)  # Infinite loop
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        
    @pyqtProperty(float)
    def heart_scale(self):
        return self._heart_scale
    
    @heart_scale.setter
    def heart_scale(self, value):
        self._heart_scale = value
        self.update()
        
    @pyqtProperty(float)
    def heart_opacity(self):
        return self._heart_opacity
    
    @heart_opacity.setter  
    def heart_opacity(self, value):
        self._heart_opacity = value
        self.update()
        
    def set_sync_status(self, is_synced: bool):
        """Update synchronization status"""
        self._is_synced = is_synced
        
        if not is_synced:
            # Start pulsing for out of sync
            if self.opacity_animation.state() != QPropertyAnimation.State.Running:
                self.opacity_animation.start()
        else:
            # Stop pulsing when in sync
            self.opacity_animation.stop()
            self._heart_opacity = 0.8
            
        self.update()
        
    def heartbeat(self):
        """Trigger heartbeat animation"""
        self._beat_count += 1
        self.beat_animation.start()
        self.update()
        
    def update_heart_rate(self, rate: float):
        """Update displayed heart rate"""
        self._heart_rate = rate
        self.update()
        
    def set_current_beat(self, current_beat: int, total_beats: int):
        """Set current beat and total beats"""
        self._current_beat = current_beat
        self._total_beats = total_beats
        self.update()
        
    def reset(self):
        """Reset counter and animations"""
        self._beat_count = 0
        self._current_beat = 0
        self._total_beats = 0
        self._heart_rate = 0
        self._is_synced = True
        self.beat_animation.stop()
        self.opacity_animation.stop()
        self._heart_scale = 1.0
        self._heart_opacity = 0.8
        self.update()
        
    def paintEvent(self, event):
        """Custom paint for overlay"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Transparent background - no fill
        # bg_color = QColor(20, 20, 20, 180)
        # painter.fillRect(self.rect(), bg_color)
        
        # Draw border with light color
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        # Calculate center for heart
        center_x = self.width() // 2
        center_y = self.height() // 2 - 10
        
        # Draw heart shape
        self.draw_heart(painter, center_x, center_y)
        
        # Draw beat info
        font = QFont("Arial", 20, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(Qt.GlobalColor.white))
        
        # Show current beat / total beats
        if self._total_beats > 0:
            beat_text = f"{self._current_beat}/{self._total_beats}"
        else:
            beat_text = str(self._beat_count)
        
        text_rect = painter.fontMetrics().boundingRect(beat_text)
        text_x = center_x - text_rect.width() // 2
        text_y = center_y + 5
        painter.drawText(text_x, text_y, beat_text)
        
        # Draw heart rate
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(QPen(QColor(50, 50, 50)))
        
        if self._heart_rate > 0:
            hr_text = f"{self._heart_rate:.0f} BPM"
        else:
            hr_text = "-- BPM"
            
        hr_rect = painter.fontMetrics().boundingRect(hr_text)
        hr_x = center_x - hr_rect.width() // 2
        hr_y = self.height() - 10
        painter.drawText(hr_x, hr_y, hr_text)
        
        # Draw sync status text
        status_text = "SYNC" if self._is_synced else "ASYNC"
        status_color = QColor(34, 139, 34) if self._is_synced else QColor(220, 20, 60)
        painter.setPen(QPen(status_color))
        
        status_rect = painter.fontMetrics().boundingRect(status_text)
        status_x = center_x - status_rect.width() // 2
        status_y = 15
        painter.drawText(status_x, status_y, status_text)
        
    def draw_heart(self, painter: QPainter, cx: int, cy: int):
        """Draw animated heart shape"""
        # Heart color based on sync status
        if self._is_synced:
            heart_color = QColor(34, 139, 34, int(255 * self._heart_opacity))  # Forest Green
        else:
            heart_color = QColor(220, 20, 60, int(255 * self._heart_opacity))  # Crimson
            
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(QBrush(heart_color))
        
        # Create heart path
        path = QPainterPath()
        
        # Scale factor
        scale = 20 * self._heart_scale
        
        # Heart shape using bezier curves
        # Start at bottom point
        path.moveTo(cx, cy + scale * 0.8)
        
        # Left curve
        path.cubicTo(
            cx - scale * 0.5, cy + scale * 0.5,
            cx - scale * 1.0, cy + scale * 0.1,
            cx - scale * 1.0, cy - scale * 0.2
        )
        
        # Left top arc
        path.cubicTo(
            cx - scale * 1.0, cy - scale * 0.6,
            cx - scale * 0.5, cy - scale * 0.8,
            cx, cy - scale * 0.4
        )
        
        # Right top arc
        path.cubicTo(
            cx + scale * 0.5, cy - scale * 0.8,
            cx + scale * 1.0, cy - scale * 0.6,
            cx + scale * 1.0, cy - scale * 0.2
        )
        
        # Right curve
        path.cubicTo(
            cx + scale * 1.0, cy + scale * 0.1,
            cx + scale * 0.5, cy + scale * 0.5,
            cx, cy + scale * 0.8
        )
        
        painter.drawPath(path)
        
        # Add highlight for 3D effect
        if self._heart_scale > 1.0:
            highlight_color = QColor(255, 255, 255, int(50 * self._heart_opacity))
            painter.setBrush(QBrush(highlight_color))
            
            # Smaller highlight heart
            highlight_path = QPainterPath()
            h_scale = scale * 0.6
            
            highlight_path.moveTo(cx - h_scale * 0.3, cy - h_scale * 0.2)
            highlight_path.cubicTo(
                cx - h_scale * 0.5, cy - h_scale * 0.5,
                cx - h_scale * 0.2, cy - h_scale * 0.6,
                cx, cy - h_scale * 0.3
            )
            highlight_path.cubicTo(
                cx + h_scale * 0.2, cy - h_scale * 0.6,
                cx + h_scale * 0.5, cy - h_scale * 0.5,
                cx + h_scale * 0.3, cy - h_scale * 0.2
            )
            
            painter.drawPath(highlight_path)