"""Main window module - refactored into smaller components"""

# Import from the parent directory
from ..main_window import MainWindow

# Import helper modules for internal use
from .menu_manager import MenuManager
from .playback_controller import PlaybackController
from .dicom_loader import DicomLoader
from .analysis_coordinator import AnalysisCoordinator

__all__ = ['MainWindow', 'MenuManager', 'PlaybackController', 'DicomLoader', 'AnalysisCoordinator']