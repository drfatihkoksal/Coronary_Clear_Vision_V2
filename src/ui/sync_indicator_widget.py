"""
Synchronization status indicator and heartbeat counter widget
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont
import logging

logger = logging.getLogger(__name__)


class SyncStatusLight(QWidget):
    """Green/yellow/red status light for synchronization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self._color = QColor(128, 128, 128)  # Gray by default
        self._status = "unknown"
        
    def set_status(self, status: str):
        """Set sync status: 'good', 'warning', 'error', 'unknown'"""
        self._status = status
        if status == "good":
            self._color = QColor(0, 255, 0)  # Green
        elif status == "warning":
            self._color = QColor(255, 255, 0)  # Yellow
        elif status == "error":
            self._color = QColor(255, 0, 0)  # Red
        else:
            self._color = QColor(128, 128, 128)  # Gray
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw outer circle
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(2, 2, 26, 26)
        
        # Add highlight for 3D effect
        highlight = QColor(255, 255, 255, 100)
        painter.setBrush(QBrush(highlight))
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.drawEllipse(6, 6, 10, 10)


class HeartbeatCounter(QWidget):
    """Large counter display for heartbeat number"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(120, 80)
        self._beat_count = 0
        self._flash_opacity = 0
        
        # Animation for flash effect
        self._flash_animation = QPropertyAnimation(self, b"flash_opacity")
        self._flash_animation.setDuration(300)
        self._flash_animation.setStartValue(255)
        self._flash_animation.setEndValue(0)
        
    @pyqtProperty(int)
    def flash_opacity(self):
        return self._flash_opacity
    
    @flash_opacity.setter
    def flash_opacity(self, value):
        self._flash_opacity = value
        self.update()
        
    def increment_beat(self):
        """Increment beat counter and flash"""
        self._beat_count += 1
        self._flash_animation.start()
        self.update()
        
    def reset_count(self):
        """Reset beat counter"""
        self._beat_count = 0
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        # Border
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        # Beat number
        font = QFont("Arial", 36, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Flash effect
        if self._flash_opacity > 0:
            flash_color = QColor(255, 0, 0, self._flash_opacity)
            painter.setPen(QPen(flash_color))
        else:
            painter.setPen(QPen(QColor(255, 255, 255)))
            
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, str(self._beat_count))
        
        # Label
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.setPen(QPen(QColor(200, 200, 200)))
        label_rect = self.rect().adjusted(0, -20, 0, 0)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, "Beat #")


class SyncIndicatorWidget(QWidget):
    """Combined widget for sync status and heartbeat counter"""
    
    # Signals
    sync_status_changed = pyqtSignal(str)
    beat_detected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._sync_status = "unknown"
        self._total_beats = 0
        
    def setup_ui(self):
        """Initialize UI components"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Sync status section
        sync_frame = QFrame()
        sync_frame.setFrameStyle(QFrame.Shape.Box)
        sync_layout = QVBoxLayout(sync_frame)
        
        sync_label = QLabel("ECG Sync")
        sync_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sync_layout.addWidget(sync_label)
        
        self.sync_light = SyncStatusLight()
        sync_layout.addWidget(self.sync_light, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.sync_status_label = QLabel("Unknown")
        self.sync_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sync_status_label.setStyleSheet("QLabel { font-size: 10px; }")
        sync_layout.addWidget(self.sync_status_label)
        
        layout.addWidget(sync_frame)
        
        # Heartbeat counter section
        counter_frame = QFrame()
        counter_frame.setFrameStyle(QFrame.Shape.Box)
        counter_layout = QVBoxLayout(counter_frame)
        
        self.beat_counter = HeartbeatCounter()
        counter_layout.addWidget(self.beat_counter)
        
        self.heart_rate_label = QLabel("HR: -- bpm")
        self.heart_rate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heart_rate_label.setStyleSheet("QLabel { font-size: 12px; font-weight: bold; }")
        counter_layout.addWidget(self.heart_rate_label)
        
        layout.addWidget(counter_frame)
        
        # Style
        self.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: #333333;
            }
        """)
        
    def set_sync_status(self, status: str, message: str = ""):
        """Update synchronization status"""
        self._sync_status = status
        self.sync_light.set_status(status)
        
        if status == "good":
            self.sync_status_label.setText("Synced")
            self.sync_status_label.setStyleSheet("QLabel { font-size: 10px; color: #00ff00; }")
        elif status == "warning":
            self.sync_status_label.setText("Warning")
            self.sync_status_label.setStyleSheet("QLabel { font-size: 10px; color: #ffff00; }")
        elif status == "error":
            self.sync_status_label.setText("Error")
            self.sync_status_label.setStyleSheet("QLabel { font-size: 10px; color: #ff0000; }")
        else:
            self.sync_status_label.setText("Unknown")
            self.sync_status_label.setStyleSheet("QLabel { font-size: 10px; color: #808080; }")
            
        if message:
            self.sync_status_label.setToolTip(message)
            
        self.sync_status_changed.emit(status)
        
    def on_beat_detected(self):
        """Handle heartbeat detection"""
        self._total_beats += 1
        self.beat_counter.increment_beat()
        self.beat_detected.emit(self._total_beats)
        
    def update_heart_rate(self, heart_rate: float):
        """Update heart rate display"""
        if heart_rate > 0:
            self.heart_rate_label.setText(f"HR: {heart_rate:.0f} bpm")
        else:
            self.heart_rate_label.setText("HR: -- bpm")
            
    def reset(self):
        """Reset all counters and status"""
        self._total_beats = 0
        self.beat_counter.reset_count()
        self.set_sync_status("unknown")
        self.heart_rate_label.setText("HR: -- bpm")
        
    def check_sync_status(self, video_duration: float, ecg_duration: float, tolerance_ms: float = 1.0):
        """Check and update sync status based on durations"""
        if video_duration <= 0 or ecg_duration <= 0:
            self.set_sync_status("unknown", "No duration data available")
            return
            
        diff_ms = abs(video_duration - ecg_duration) * 1000
        
        if diff_ms < tolerance_ms:
            self.set_sync_status("good", f"Duration difference: {diff_ms:.2f}ms")
        elif diff_ms < tolerance_ms * 5:  # 5ms warning threshold
            self.set_sync_status("warning", f"Duration difference: {diff_ms:.2f}ms")
        else:
            self.set_sync_status("error", f"Duration difference: {diff_ms:.2f}ms")