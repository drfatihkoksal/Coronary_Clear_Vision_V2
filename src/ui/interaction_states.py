"""Interaction state pattern for viewer widgets."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QObject, QRectF
from PyQt6.QtGui import QMouseEvent, QKeyEvent, QPainter, QPen, QColor
from ..core.domain_models import Point
from ..core.simple_tracker import get_tracker


class InteractionState(ABC):
    """Base class for interaction states."""

    def __init__(self, viewer: 'ViewerWidget'):
        """Initialize with reference to viewer widget."""
        self.viewer = viewer

    @abstractmethod
    def enter(self):
        """Called when entering this state."""

    @abstractmethod
    def exit(self):
        """Called when exiting this state."""

    @abstractmethod
    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Handle mouse press event. Return True if handled."""

    @abstractmethod
    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Handle mouse move event. Return True if handled."""

    @abstractmethod
    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Handle mouse release event. Return True if handled."""

    @abstractmethod
    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle key press event. Return True if handled."""

    @abstractmethod
    def draw_overlay(self, painter: QPainter):
        """Draw any state-specific overlay."""

    @abstractmethod
    def get_state_name(self) -> str:
        """Get the name of this state."""


class ViewState(InteractionState):
    """Default viewing state with pan and zoom."""

    def __init__(self, viewer: 'ViewerWidget'):
        super().__init__(viewer)
        self.is_panning = False
        self.last_pan_point = None

    def enter(self):
        """Enter view state."""
        self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def exit(self):
        """Exit view state."""
        self.is_panning = False
        self.last_pan_point = None

    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Start panning on middle button or with modifier."""
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.is_panning = True
            self.last_pan_point = event.position()
            self.viewer.setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        return False

    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Handle panning."""
        if self.is_panning and self.last_pan_point:
            delta = event.position() - self.last_pan_point
            self.viewer.pan(delta.x(), delta.y())
            self.last_pan_point = event.position()
            return True
        return False

    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Stop panning."""
        if self.is_panning:
            self.is_panning = False
            self.last_pan_point = None
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)
            return True
        return False

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle keyboard shortcuts."""
        # Reset view on 'R'
        if event.key() == Qt.Key.Key_R:
            self.viewer.reset_view()
            return True
        return False

    def draw_overlay(self, painter: QPainter):
        """No overlay in view state."""

    def get_state_name(self) -> str:
        return "view"


class CalibrationState(InteractionState):
    """State for catheter calibration."""

    def __init__(self, viewer: 'ViewerWidget'):
        super().__init__(viewer)
        self.start_point: Optional[Point] = None
        self.end_point: Optional[Point] = None
        self.is_drawing = False
        self.temp_end_point: Optional[Point] = None

    def enter(self):
        """Enter calibration state."""
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.start_point = None
        self.end_point = None
        self.is_drawing = False

    def exit(self):
        """Exit calibration state."""
        self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Start calibration line."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.viewer.map_to_image(event.position())
            self.start_point = Point(pos.x(), pos.y())
            self.end_point = None
            self.is_drawing = True
            return True
        elif event.button() == Qt.MouseButton.RightButton:
            # Cancel current calibration
            self.start_point = None
            self.end_point = None
            self.is_drawing = False
            self.viewer.update()
            return True
        return False

    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Update calibration line end point."""
        if self.is_drawing:
            pos = self.viewer.map_to_image(event.position())
            self.temp_end_point = Point(pos.x(), pos.y())
            self.viewer.update()
            return True
        return False

    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Complete calibration line."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            pos = self.viewer.map_to_image(event.position())
            self.end_point = Point(pos.x(), pos.y())
            self.is_drawing = False
            self.temp_end_point = None

            # Emit calibration complete signal
            if self.start_point and self.end_point:
                distance = self.start_point.distance_to(self.end_point)
                if distance > 10:  # Minimum 10 pixels
                    self.viewer.calibration_completed.emit(
                        self.start_point, self.end_point
                    )

            self.viewer.update()
            return True
        return False

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle escape to cancel."""
        if event.key() == Qt.Key.Key_Escape:
            self.start_point = None
            self.end_point = None
            self.is_drawing = False
            self.viewer.update()
            return True
        return False

    def draw_overlay(self, painter: QPainter):
        """Draw calibration line."""
        if self.start_point:
            pen = QPen(QColor(255, 255, 0), 2)
            painter.setPen(pen)

            start = self.viewer.map_from_image(
                QPointF(self.start_point.x, self.start_point.y)
            )

            if self.end_point:
                end = self.viewer.map_from_image(
                    QPointF(self.end_point.x, self.end_point.y)
                )
            elif self.temp_end_point:
                end = self.viewer.map_from_image(
                    QPointF(self.temp_end_point.x, self.temp_end_point.y)
                )
            else:
                return

            painter.drawLine(start, end)

            # Draw endpoints
            painter.setBrush(QColor(255, 255, 0))
            painter.drawEllipse(start, 5, 5)
            painter.drawEllipse(end, 5, 5)

            # Draw distance
            distance = self.start_point.distance_to(
                self.end_point if self.end_point else self.temp_end_point
            )
            mid_point = QPointF((start.x() + end.x()) / 2,
                               (start.y() + end.y()) / 2)
            painter.drawText(mid_point, f"{distance:.1f} px")

    def get_state_name(self) -> str:
        return "calibration"


class SegmentationState(InteractionState):
    """State for vessel segmentation selection."""

    def __init__(self, viewer: 'ViewerWidget'):
        super().__init__(viewer)
        self.seed_points: List[Point] = []
        self.is_drawing_box = False
        self.box_start: Optional[Point] = None
        self.box_end: Optional[Point] = None
        self.temp_box_end: Optional[Point] = None

    def enter(self):
        """Enter segmentation state."""
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)
        self.seed_points.clear()
        self.is_drawing_box = False

    def exit(self):
        """Exit segmentation state."""
        self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Handle segmentation interaction."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.viewer.map_to_image(event.position())

            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Start box selection
                self.box_start = Point(pos.x(), pos.y())
                self.is_drawing_box = True
            else:
                # Add seed point
                self.seed_points.append(Point(pos.x(), pos.y()))
                self.viewer.update()
            return True

        elif event.button() == Qt.MouseButton.RightButton:
            if self.seed_points:
                # Remove last seed point
                self.seed_points.pop()
                self.viewer.update()
            return True

        return False

    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Update box selection."""
        if self.is_drawing_box:
            pos = self.viewer.map_to_image(event.position())
            self.temp_box_end = Point(pos.x(), pos.y())
            self.viewer.update()
            return True
        return False

    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Complete box selection."""
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing_box:
            pos = self.viewer.map_to_image(event.position())
            self.box_end = Point(pos.x(), pos.y())
            self.is_drawing_box = False
            self.temp_box_end = None

            # Emit segmentation region selected
            if self.box_start and self.box_end:
                self.viewer.segmentation_region_selected.emit(
                    self.box_start, self.box_end
                )

            self.viewer.update()
            return True
        return False

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle segmentation shortcuts."""
        if event.key() == Qt.Key.Key_Return and self.seed_points:
            # Trigger segmentation with seed points
            self.viewer.segmentation_seeds_selected.emit(self.seed_points)
            return True
        elif event.key() == Qt.Key.Key_Escape:
            # Clear selection
            self.seed_points.clear()
            self.box_start = None
            self.box_end = None
            self.is_drawing_box = False
            self.viewer.update()
            return True
        return False

    def draw_overlay(self, painter: QPainter):
        """Draw segmentation selection."""
        # Draw seed points
        if self.seed_points:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 100))

            for point in self.seed_points:
                pos = self.viewer.map_from_image(QPointF(point.x, point.y))
                painter.drawEllipse(pos, 5, 5)

        # Draw selection box
        if self.box_start:
            pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            start = self.viewer.map_from_image(
                QPointF(self.box_start.x, self.box_start.y)
            )

            if self.box_end:
                end = self.viewer.map_from_image(
                    QPointF(self.box_end.x, self.box_end.y)
                )
            elif self.temp_box_end:
                end = self.viewer.map_from_image(
                    QPointF(self.temp_box_end.x, self.temp_box_end.y)
                )
            else:
                return

            rect = QRectF(start, end).normalized()
            painter.drawRect(rect)

    def get_state_name(self) -> str:
        return "segmentation"


class TrackingState(InteractionState):
    """State for point tracking across frames."""

    def __init__(self, viewer: 'ViewerWidget'):
        super().__init__(viewer)
        self.point_tracker = get_tracker()
        self.selected_point_id: Optional[str] = None

    def enter(self):
        """Enter tracking state."""
        self.viewer.setCursor(Qt.CursorShape.CrossCursor)

    def exit(self):
        """Exit tracking state."""
        self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Handle point selection for tracking."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.viewer.map_to_image(event.position())
            point = Point(pos.x(), pos.y())

            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Add new tracking point
                point_id = self.point_tracker.add_point(
                    point, self.viewer.current_frame
                )
                self.viewer.tracking_point_added.emit(point_id, point)
            else:
                # Select existing point
                self.selected_point_id = self._find_nearest_point(point)
                if self.selected_point_id:
                    self.viewer.tracking_point_selected.emit(self.selected_point_id)

            self.viewer.update()
            return True

        elif event.button() == Qt.MouseButton.RightButton:
            # Remove tracking point
            pos = self.viewer.map_to_image(event.position())
            point = Point(pos.x(), pos.y())
            point_id = self._find_nearest_point(point)

            if point_id:
                self.point_tracker.remove_point(point_id)
                self.viewer.tracking_point_removed.emit(point_id)
                self.viewer.update()

            return True

        return False

    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Handle point dragging."""
        if (event.buttons() & Qt.MouseButton.LeftButton and
            self.selected_point_id):
            pos = self.viewer.map_to_image(event.position())
            point = Point(pos.x(), pos.y())

            tracked_point = self.point_tracker.tracked_points.get(self.selected_point_id)
            if tracked_point:
                tracked_point.add_position(self.viewer.current_frame, point)
                self.viewer.update()

            return True

        return False

    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Complete point dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_point_id = None
            return True
        return False

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle tracking shortcuts."""
        if event.key() == Qt.Key.Key_T:
            # Track all points to current frame
            self.track_to_current_frame()
            return True
        elif event.key() == Qt.Key.Key_C:
            # Clear inactive points
            self.point_tracker.clear_inactive_points()
            self.viewer.update()
            return True
        return False

    def draw_overlay(self, painter: QPainter):
        """Draw tracking points and trajectories."""
        # Draw tracking points
        for point_id, tracked_point in self.point_tracker.tracked_points.items():
            if tracked_point.is_active:
                pos = tracked_point.get_position_at_frame(self.viewer.current_frame)
                if pos:
                    screen_pos = self.viewer.map_from_image(QPointF(pos.x, pos.y))

                    # Draw point
                    color = QColor(255, 0, 0) if point_id == self.selected_point_id else QColor(0, 255, 255)
                    pen = QPen(color, 2)
                    painter.setPen(pen)
                    painter.setBrush(color)
                    painter.drawEllipse(screen_pos, 5, 5)

                    # Draw ID
                    painter.drawText(screen_pos + QPointF(10, -5), point_id[:4])

    def _find_nearest_point(self, point: Point, max_distance: float = 20) -> Optional[str]:
        """Find nearest tracked point within max distance."""
        min_distance = max_distance
        nearest_id = None

        for point_id, tracked_point in self.point_tracker.tracked_points.items():
            pos = tracked_point.get_position_at_frame(self.viewer.current_frame)
            if pos:
                distance = point.distance_to(pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = point_id

        return nearest_id

    def track_to_current_frame(self):
        """Track all points to current frame."""
        if not hasattr(self.viewer, 'get_frame'):
            return

        # Get frames
        source_frame_num = next(iter(self.point_tracker.tracked_points.values())).frame_history.keys()[0] if self.point_tracker.tracked_points else 0
        target_frame_num = self.viewer.current_frame

        if source_frame_num == target_frame_num:
            return

        source_frame = self.viewer.get_frame(source_frame_num)
        target_frame = self.viewer.get_frame(target_frame_num)

        if source_frame is not None and target_frame is not None:
            self.point_tracker.track_to_frame(
                source_frame, target_frame,
                source_frame_num, target_frame_num
            )
            self.viewer.update()

    def get_state_name(self) -> str:
        return "tracking"


class InteractionController(QObject):
    """Controls interaction state transitions."""

    # Signals
    state_changed = pyqtSignal(str)

    def __init__(self, viewer: 'ViewerWidget'):
        super().__init__()
        self.viewer = viewer

        # Create states
        self.states: Dict[str, InteractionState] = {
            'view': ViewState(viewer),
            'calibration': CalibrationState(viewer),
            'segmentation': SegmentationState(viewer),
            'tracking': TrackingState(viewer)
        }

        # Set initial state
        self.current_state: InteractionState = self.states['view']
        self.current_state.enter()

    def set_state(self, state_name: str):
        """Change to a new interaction state."""
        if state_name not in self.states:
            raise ValueError(f"Unknown state: {state_name}")

        if self.states[state_name] == self.current_state:
            return

        # Exit current state
        self.current_state.exit()

        # Enter new state
        self.current_state = self.states[state_name]
        self.current_state.enter()

        # Emit signal
        self.state_changed.emit(state_name)

    def get_current_state_name(self) -> str:
        """Get the name of the current state."""
        return self.current_state.get_state_name()

    def handle_mouse_press(self, event: QMouseEvent) -> bool:
        """Forward to current state."""
        return self.current_state.handle_mouse_press(event)

    def handle_mouse_move(self, event: QMouseEvent) -> bool:
        """Forward to current state."""
        return self.current_state.handle_mouse_move(event)

    def handle_mouse_release(self, event: QMouseEvent) -> bool:
        """Forward to current state."""
        return self.current_state.handle_mouse_release(event)

    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Forward to current state."""
        return self.current_state.handle_key_press(event)

    def draw_overlay(self, painter: QPainter):
        """Forward to current state."""
        self.current_state.draw_overlay(painter)