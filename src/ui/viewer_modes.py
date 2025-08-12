"""
Centralized mode management for the viewer widget.
Ensures consistent state transitions and prevents mode conflicts.
"""

from enum import Enum, auto
from typing import Optional, Callable, Dict, Any
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt
import logging

logger = logging.getLogger(__name__)


class ViewerMode(Enum):
    """Available viewer modes"""
    VIEW = auto()          # Default viewing mode
    CALIBRATE = auto()     # Calibration point collection
    SEGMENT = auto()       # Segmentation point collection
    MULTI_FRAME = auto()   # Multi-frame segmentation
    QCA = auto()          # QCA analysis mode


class ModeManager(QObject):
    """Manages viewer interaction modes"""
    
    # Signals
    mode_changed = pyqtSignal(ViewerMode, ViewerMode)  # old_mode, new_mode
    mode_enter = pyqtSignal(ViewerMode)
    mode_exit = pyqtSignal(ViewerMode)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self._current_mode = ViewerMode.VIEW
        self._previous_mode = ViewerMode.VIEW
        self._mode_stack = []  # For nested mode support if needed
        
        # Mode-specific settings storage
        self._mode_settings: Dict[ViewerMode, Dict[str, Any]] = {
            ViewerMode.VIEW: {
                'drag_mode': QGraphicsView.DragMode.ScrollHandDrag,
                'cursor': Qt.CursorShape.ArrowCursor,
                'accepts_points': False,  # Will be overridden by tracking
                'max_points': 0,
                'clear_points_on_exit': False
            },
            ViewerMode.CALIBRATE: {
                'drag_mode': QGraphicsView.DragMode.NoDrag,
                'cursor': Qt.CursorShape.CrossCursor,
                'accepts_points': True,
                'max_points': 2,
                'clear_points_on_exit': True
            },
            ViewerMode.SEGMENT: {
                'drag_mode': QGraphicsView.DragMode.NoDrag,
                'cursor': Qt.CursorShape.CrossCursor,
                'accepts_points': True,
                'max_points': None,  # Unlimited
                'clear_points_on_exit': False
            },
            ViewerMode.MULTI_FRAME: {
                'drag_mode': QGraphicsView.DragMode.NoDrag,
                'cursor': Qt.CursorShape.CrossCursor,
                'accepts_points': True,
                'max_points': None,
                'clear_points_on_exit': False
            },
            ViewerMode.QCA: {
                'drag_mode': QGraphicsView.DragMode.NoDrag,
                'cursor': Qt.CursorShape.CrossCursor,
                'accepts_points': True,
                'max_points': 2,  # Proximal and distal points
                'clear_points_on_exit': False
            }
        }
        
        # Mode transition callbacks
        self._enter_callbacks: Dict[ViewerMode, Callable] = {}
        self._exit_callbacks: Dict[ViewerMode, Callable] = {}
        
        # Blocked transitions
        self._blocked_transitions: Dict[ViewerMode, set] = {
            # Can't go directly from calibration to segmentation without clearing
            ViewerMode.CALIBRATE: set(),
            ViewerMode.SEGMENT: set(),
            ViewerMode.MULTI_FRAME: {ViewerMode.CALIBRATE},  # Can't calibrate during multi-frame
        }
    
    @property
    def current_mode(self) -> ViewerMode:
        """Get current mode"""
        return self._current_mode
    
    @property
    def previous_mode(self) -> ViewerMode:
        """Get previous mode"""
        return self._previous_mode
    
    def get_mode_settings(self, mode: Optional[ViewerMode] = None) -> Dict[str, Any]:
        """Get settings for a mode"""
        mode = mode or self._current_mode
        return self._mode_settings.get(mode, {}).copy()
    
    def can_transition_to(self, new_mode: ViewerMode) -> bool:
        """Check if transition to new mode is allowed"""
        if new_mode == self._current_mode:
            return True
            
        blocked = self._blocked_transitions.get(self._current_mode, set())
        return new_mode not in blocked
    
    def set_mode(self, new_mode: ViewerMode, force: bool = False) -> bool:
        """
        Change viewer mode
        
        Args:
            new_mode: Target mode
            force: Force transition even if blocked
            
        Returns:
            True if mode was changed
        """
        if new_mode == self._current_mode:
            # logger.debug(f"Already in mode {new_mode.name}")
            return True
        
        # Check if transition is allowed
        if not force and not self.can_transition_to(new_mode):
            logger.warning(f"Transition from {self._current_mode.name} to {new_mode.name} is blocked")
            return False
        
        old_mode = self._current_mode
        
        # Exit current mode
        self._exit_mode(old_mode)
        
        # Update mode
        self._previous_mode = old_mode
        self._current_mode = new_mode
        
        # Enter new mode
        self._enter_mode(new_mode)
        
        # Emit signals
        self.mode_changed.emit(old_mode, new_mode)
        
        logger.info(f"Mode changed: {old_mode.name} -> {new_mode.name}")
        return True
    
    def _exit_mode(self, mode: ViewerMode):
        """Handle mode exit"""
        settings = self.get_mode_settings(mode)
        
        # Clear points if needed
        if settings.get('clear_points_on_exit', False):
            self.viewer.clear_user_points()
        
        # Call exit callback
        if mode in self._exit_callbacks:
            self._exit_callbacks[mode]()
        
        self.mode_exit.emit(mode)
    
    def _enter_mode(self, mode: ViewerMode):
        """Handle mode entry"""
        settings = self.get_mode_settings(mode)
        
        # Apply mode settings
        if 'drag_mode' in settings:
            self.viewer.setDragMode(settings['drag_mode'])
        
        # Special handling for cursor in VIEW mode with tracking
        if mode == ViewerMode.VIEW and hasattr(self.viewer, 'tracking_enabled') and self.viewer.tracking_enabled:
            # Use custom crosshair cursor for tracking
            self.viewer._set_tracking_crosshair_cursor()
        elif 'cursor' in settings:
            self.viewer.setCursor(settings['cursor'])
        
        # Call enter callback
        if mode in self._enter_callbacks:
            self._enter_callbacks[mode]()
        
        self.mode_enter.emit(mode)
    
    def register_enter_callback(self, mode: ViewerMode, callback: Callable):
        """Register callback for mode entry"""
        self._enter_callbacks[mode] = callback
    
    def register_exit_callback(self, mode: ViewerMode, callback: Callable):
        """Register callback for mode exit"""
        self._exit_callbacks[mode] = callback
    
    def toggle_mode(self, mode: ViewerMode) -> bool:
        """Toggle between a mode and VIEW mode"""
        if self._current_mode == mode:
            return self.set_mode(ViewerMode.VIEW)
        else:
            return self.set_mode(mode)
    
    def return_to_view(self) -> bool:
        """Return to VIEW mode"""
        return self.set_mode(ViewerMode.VIEW)
    
    def return_to_previous(self) -> bool:
        """Return to previous mode"""
        return self.set_mode(self._previous_mode)
    
    def is_in_mode(self, mode: ViewerMode) -> bool:
        """Check if currently in specified mode"""
        return self._current_mode == mode
    
    def accepts_points(self) -> bool:
        """Check if current mode accepts point input"""
        return self.get_mode_settings().get('accepts_points', False)
    
    def get_max_points(self) -> Optional[int]:
        """Get maximum points for current mode"""
        return self.get_mode_settings().get('max_points')
    
    def block_transition(self, from_mode: ViewerMode, to_mode: ViewerMode):
        """Block a specific transition"""
        if from_mode not in self._blocked_transitions:
            self._blocked_transitions[from_mode] = set()
        self._blocked_transitions[from_mode].add(to_mode)
    
    def unblock_transition(self, from_mode: ViewerMode, to_mode: ViewerMode):
        """Unblock a specific transition"""
        if from_mode in self._blocked_transitions:
            self._blocked_transitions[from_mode].discard(to_mode)
    
    def clear_all_blocks(self):
        """Clear all transition blocks"""
        self._blocked_transitions.clear()