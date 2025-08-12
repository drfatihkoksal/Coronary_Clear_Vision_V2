"""
Event bus for decoupled communication between components.
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event class."""
    name: str
    data: Any
    timestamp: datetime = None
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus(QObject):
    """
    Centralized event bus for application-wide communication.
    Implements singleton pattern.
    """
    
    # Qt signal for thread-safe event emission
    _event_signal = pyqtSignal(Event)
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        super().__init__()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._history_limit = 1000
        
        # Connect internal signal
        self._event_signal.connect(self._handle_event)
        
        self._initialized = True
    
    def subscribe(self, event_name: str, callback: Callable) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event to subscribe to
            callback: Function to call when event is emitted
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        
        if callback not in self._subscribers[event_name]:
            self._subscribers[event_name].append(callback)
            logger.debug(f"Subscribed {callback.__name__} to event '{event_name}'")
    
    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Name of the event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_name in self._subscribers and callback in self._subscribers[event_name]:
            self._subscribers[event_name].remove(callback)
            logger.debug(f"Unsubscribed {callback.__name__} from event '{event_name}'")
    
    def emit(self, event_name: str, data: Any = None, source: Optional[str] = None) -> None:
        """
        Emit an event.
        
        Args:
            event_name: Name of the event
            data: Event data
            source: Optional source identifier
        """
        event = Event(name=event_name, data=data, source=source)
        
        # Use Qt signal for thread safety
        self._event_signal.emit(event)
    
    def _handle_event(self, event: Event) -> None:
        """Handle event emission (called in main thread)."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._history_limit:
            self._event_history.pop(0)
        
        # Notify subscribers
        if event.name in self._subscribers:
            for callback in self._subscribers[event.name]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler {callback.__name__} for event '{event.name}': {e}")
    
    def get_history(self, event_name: Optional[str] = None, limit: int = 100) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_name: Filter by event name (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        if event_name:
            filtered = [e for e in self._event_history if e.name == event_name]
            return filtered[-limit:]
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
    
    def reset(self) -> None:
        """Reset event bus (clear all subscribers and history)."""
        self._subscribers.clear()
        self._event_history.clear()
        logger.info("Event bus reset")


# Common event names
class Events:
    """Common event names used throughout the application."""
    
    # File events
    FILE_OPENED = "file.opened"
    FILE_CLOSED = "file.closed"
    FILE_SAVED = "file.saved"
    
    # DICOM events
    DICOM_LOADED = "dicom.loaded"
    DICOM_FRAME_CHANGED = "dicom.frame_changed"
    
    # Segmentation events
    SEGMENTATION_STARTED = "segmentation.started"
    SEGMENTATION_COMPLETED = "segmentation.completed"
    SEGMENTATION_FAILED = "segmentation.failed"
    SEGMENTATION_MODE_CHANGED = "segmentation.mode_changed"
    
    # QCA events
    QCA_STARTED = "qca.started"
    QCA_COMPLETED = "qca.completed"
    QCA_FAILED = "qca.failed"
    
    # Calibration events
    CALIBRATION_STARTED = "calibration.started"
    CALIBRATION_COMPLETED = "calibration.completed"
    CALIBRATION_CLEARED = "calibration.cleared"
    
    # UI events
    MODE_CHANGED = "ui.mode_changed"
    ZOOM_CHANGED = "ui.zoom_changed"
    OVERLAY_TOGGLED = "ui.overlay_toggled"
    
    # Processing events
    
    # Model events
    MODEL_LOADING = "model.loading"
    MODEL_LOADED = "model.loaded"
    MODEL_FAILED = "model.failed"


# Global event bus instance
event_bus = EventBus()