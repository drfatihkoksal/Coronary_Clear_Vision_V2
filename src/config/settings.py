"""
Application settings management.
"""

from typing import Any, Optional
from PyQt6.QtCore import QSettings
from .constants import APP_NAME, ORGANIZATION_NAME


class SettingsManager:
    """Centralized settings management."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._settings = QSettings(ORGANIZATION_NAME, APP_NAME)
        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return self._settings.value(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a setting value."""
        self._settings.setValue(key, value)

    def remove(self, key: str) -> None:
        """Remove a setting."""
        self._settings.remove(key)

    def sync(self) -> None:
        """Force settings to be written to permanent storage."""
        self._settings.sync()

    # Convenience methods for common settings
    def get_last_directory(self) -> Optional[str]:
        """Get the last used directory."""
        return self.get("last_directory")

    def set_last_directory(self, directory: str) -> None:
        """Set the last used directory."""
        self.set("last_directory", directory)

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get("debug_mode", False)

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        self.set("debug_mode", enabled)

    def get_recent_files(self) -> list:
        """Get list of recent files."""
        return self.get("recent_files", [])

    def add_recent_file(self, file_path: str, max_files: int = 10) -> None:
        """Add a file to recent files list."""
        recent = self.get_recent_files()
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        self.set("recent_files", recent[:max_files])

    def get_window_state(self) -> Optional[bytes]:
        """Get saved window state."""
        return self.get("window_state")

    def set_window_state(self, state: bytes) -> None:
        """Save window state."""
        self.set("window_state", state)

    def get_window_geometry(self) -> Optional[bytes]:
        """Get saved window geometry."""
        return self.get("window_geometry")

    def set_window_geometry(self, geometry: bytes) -> None:
        """Save window geometry."""
        self.set("window_geometry", geometry)
