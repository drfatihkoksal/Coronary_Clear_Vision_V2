"""
Centralized Application Configuration using Pydantic

This module provides a type-safe, validated configuration system for the entire application.
Configuration can be loaded from environment variables, .env files, or settings.json.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CalibrationConfig(BaseSettings):
    """Calibration-related configuration"""
    
    min_calibration_factor: float = Field(
        default=0.01,
        description="Minimum calibration factor in px/mm",
        ge=0.001,
        le=1.0
    )
    
    max_calibration_factor: float = Field(
        default=100.0,
        description="Maximum calibration factor in px/mm",
        ge=10.0,
        le=1000.0
    )
    
    min_catheter_width_pixels: int = Field(
        default=15,
        description="Minimum catheter width in pixels",
        ge=5,
        le=100
    )
    
    min_manual_distance_pixels: float = Field(
        default=10.0,
        description="Minimum distance between manual calibration points",
        ge=5.0,
        le=50.0
    )
    
    max_reference_distance_mm: float = Field(
        default=50.0,
        description="Maximum reference distance in mm",
        ge=10.0,
        le=100.0
    )
    
    class Config:
        env_prefix = "CALIBRATION_"


class RWSConfig(BaseSettings):
    """RWS (Relative Wall Score) configuration"""
    
    min_mld_mm: float = Field(
        default=0.2,
        description="Minimum MLD (Minimal Lumen Diameter) in mm",
        ge=0.1,
        le=1.0
    )
    
    max_mld_mm: float = Field(
        default=6.0,
        description="Maximum MLD in mm",
        ge=3.0,
        le=10.0
    )
    
    min_frames_for_analysis: int = Field(
        default=3,
        description="Minimum frames required for RWS analysis",
        ge=2,
        le=10
    )
    
    max_rws_percentage: float = Field(
        default=200.0,
        description="Maximum reasonable RWS percentage",
        ge=100.0,
        le=500.0
    )
    
    iqr_multiplier: float = Field(
        default=6.0,
        description="IQR multiplier for outlier detection (higher = wider range kept, less conservative)",
        ge=1.0,
        le=8.0
    )
    
    class Config:
        env_prefix = "RWS_"


class QCAConfig(BaseSettings):
    """QCA (Quantitative Coronary Analysis) configuration"""
    
    min_centerline_points: int = Field(
        default=10,
        description="Minimum centerline points for analysis",
        ge=5,
        le=50
    )
    
    diameter_smoothing_window: int = Field(
        default=5,
        description="Window size for diameter smoothing",
        ge=3,
        le=15
    )
    
    max_search_distance_pixels: int = Field(
        default=30,
        description="Maximum search distance for diameter measurement",
        ge=10,
        le=100
    )
    
    stenosis_threshold_percent: float = Field(
        default=50.0,
        description="Threshold for significant stenosis",
        ge=20.0,
        le=90.0
    )
    
    global_ref_percentile: float = Field(
        default=75.0,
        description="Percentile for global reference diameter",
        ge=50.0,
        le=95.0
    )
    
    class Config:
        env_prefix = "QCA_"


class PerformanceConfig(BaseSettings):
    """Performance and optimization configuration"""
    
    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for frame analysis"
    )
    
    max_worker_threads: int = Field(
        default=4,
        description="Maximum worker threads for parallel processing",
        ge=1,
        le=16
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching for analysis results"
    )
    
    cache_size_mb: int = Field(
        default=500,
        description="Maximum cache size in MB",
        ge=100,
        le=2000
    )
    
    lazy_load_models: bool = Field(
        default=True,
        description="Load AI models only when needed"
    )
    
    progress_update_interval_ms: int = Field(
        default=100,
        description="Progress bar update interval in milliseconds",
        ge=50,
        le=1000
    )
    
    class Config:
        env_prefix = "PERFORMANCE_"


class UIConfig(BaseSettings):
    """User interface configuration"""
    
    default_font_size: int = Field(
        default=13,
        description="Default font size for UI",
        ge=8,
        le=24
    )
    
    show_tooltips: bool = Field(
        default=True,
        description="Show tooltips in UI"
    )
    
    auto_save_interval_seconds: int = Field(
        default=300,
        description="Auto-save interval in seconds (0 to disable)",
        ge=0,
        le=3600
    )
    
    max_recent_files: int = Field(
        default=10,
        description="Maximum number of recent files to remember",
        ge=1,
        le=50
    )
    
    dark_mode: bool = Field(
        default=False,
        description="Enable dark mode theme"
    )
    
    class Config:
        env_prefix = "UI_"


class ApplicationConfig(BaseSettings):
    """Main application configuration"""
    
    # General settings
    app_name: str = Field(
        default="Coronary Clear Vision",
        description="Application name"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode",
        env="DEBUG"
    )
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    
    log_to_file: bool = Field(
        default=True,
        description="Enable logging to file"
    )
    
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files"
    )
    
    # Sub-configurations
    calibration: CalibrationConfig = Field(
        default_factory=CalibrationConfig,
        description="Calibration settings"
    )
    
    rws: RWSConfig = Field(
        default_factory=RWSConfig,
        description="RWS analysis settings"
    )
    
    qca: QCAConfig = Field(
        default_factory=QCAConfig,
        description="QCA analysis settings"
    )
    
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance settings"
    )
    
    ui: UIConfig = Field(
        default_factory=UIConfig,
        description="UI settings"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("log_dir")
    def create_log_dir(cls, v: Path) -> Path:
        """Ensure log directory exists"""
        v.mkdir(exist_ok=True)
        return v
    
    @classmethod
    def load_from_json(cls, json_path: Path) -> "ApplicationConfig":
        """Load configuration from JSON file"""
        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            else:
                logger.warning(f"Config file not found: {json_path}")
                return cls()
        except Exception as e:
            logger.error(f"Error loading config from {json_path}: {e}")
            return cls()
    
    def save_to_json(self, json_path: Path) -> None:
        """Save configuration to JSON file"""
        try:
            with open(json_path, 'w') as f:
                json.dump(self.dict(), f, indent=2, default=str)
            logger.info(f"Config saved to {json_path}")
        except Exception as e:
            logger.error(f"Error saving config to {json_path}: {e}")
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), 'update_from_dict'):
                    getattr(self, key).update_from_dict(value)
                else:
                    setattr(self, key, value)
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation
        
        Example: config.get_nested('rws.min_mld_mm')
        """
        keys = key_path.split('.')
        value = self
        
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return default
        
        return value
    
    def set_nested(self, key_path: str, value: Any) -> bool:
        """Set nested configuration value using dot notation
        
        Example: config.set_nested('rws.min_mld_mm', 0.3)
        """
        keys = key_path.split('.')
        target = self
        
        # Navigate to parent object
        for key in keys[:-1]:
            if hasattr(target, key):
                target = getattr(target, key)
            else:
                return False
        
        # Set the value
        if hasattr(target, keys[-1]):
            setattr(target, keys[-1], value)
            return True
        
        return False


# Singleton instance
_config_instance: Optional[ApplicationConfig] = None


def get_config() -> ApplicationConfig:
    """Get the global configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        # Try to load from settings.json first
        settings_path = Path("settings.json")
        if settings_path.exists():
            _config_instance = ApplicationConfig.load_from_json(settings_path)
        else:
            # Fall back to environment variables and defaults
            _config_instance = ApplicationConfig()
        
        # Apply debug mode from environment if set
        if Path(".env").exists():
            from dotenv import dotenv_values
            env_values = dotenv_values(".env")
            if "DEBUG" in env_values:
                _config_instance.debug_mode = env_values["DEBUG"].lower() in ('true', '1', 'yes')
    
    return _config_instance


def reload_config() -> ApplicationConfig:
    """Reload configuration from files"""
    global _config_instance
    _config_instance = None
    return get_config()


# Convenience function for backward compatibility
def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return get_config().debug_mode