"""
Tracking Configuration for Different FPS Settings

Optimize edilmiş tracking parametreleri farklı frame rate'ler için.
15 FPS için özel optimizasyonlar içerir.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrackingConfig:
    """Tracking configuration parameters."""
    
    # Frame rate
    fps: float
    
    # Search parameters
    search_radius: int
    template_size: int
    
    # Pyramid tracking
    use_pyramid: bool
    pyramid_levels: int
    
    # Confidence thresholds
    confidence_threshold: float
    min_correlation: float
    
    # Motion constraints
    max_motion: float  # Maximum motion per frame (pixels)
    
    # Template update
    adaptive_template: bool
    update_rate: float
    
    # Motion prediction
    enable_prediction: bool
    prediction_lookahead: int  # Frames to look ahead
    
    # Hybrid tracking
    use_hybrid: bool
    hybrid_switch_threshold: float  # Confidence threshold to switch methods
    
    # Optical flow parameters
    optical_flow_window_size: tuple
    optical_flow_max_corners: int
    optical_flow_quality_level: float
    

# Predefined configurations for different FPS
FPS_CONFIGS = {
    "15fps": TrackingConfig(
        fps=15.0,
        # Geniş arama alanı - frame'ler arası büyük hareketler için
        search_radius=60,
        template_size=25,
        
        # Pyramid tracking aktif - hızlı hareket için
        use_pyramid=True,
        pyramid_levels=3,
        
        # Daha toleranslı eşikler
        confidence_threshold=0.4,
        min_correlation=0.35,
        
        # Frame başına max hareket - yüksek
        max_motion=80.0,
        
        # Template update - düşük (motion blur'dan kaçın)
        adaptive_template=True,
        update_rate=0.05,
        
        # Motion prediction aktif ve agresif
        enable_prediction=True,
        prediction_lookahead=2,
        
        # Hybrid tracking aktif
        use_hybrid=True,
        hybrid_switch_threshold=0.6,
        
        # Optical flow - geniş pencere
        optical_flow_window_size=(21, 21),
        optical_flow_max_corners=150,
        optical_flow_quality_level=0.2,
    ),
    
    "20fps": TrackingConfig(
        fps=20.0,
        search_radius=45,
        template_size=23,
        
        use_pyramid=True,
        pyramid_levels=2,
        
        confidence_threshold=0.45,
        min_correlation=0.4,
        
        max_motion=60.0,
        
        adaptive_template=True,
        update_rate=0.08,
        
        enable_prediction=True,
        prediction_lookahead=1,
        
        use_hybrid=True,
        hybrid_switch_threshold=0.65,
        
        optical_flow_window_size=(17, 17),
        optical_flow_max_corners=120,
        optical_flow_quality_level=0.25,
    ),
    
    "25fps": TrackingConfig(
        fps=25.0,
        search_radius=35,
        template_size=21,
        
        use_pyramid=False,
        pyramid_levels=2,
        
        confidence_threshold=0.5,
        min_correlation=0.45,
        
        max_motion=45.0,
        
        adaptive_template=True,
        update_rate=0.1,
        
        enable_prediction=True,
        prediction_lookahead=1,
        
        use_hybrid=False,
        hybrid_switch_threshold=0.7,
        
        optical_flow_window_size=(15, 15),
        optical_flow_max_corners=100,
        optical_flow_quality_level=0.3,
    ),
    
    "30fps": TrackingConfig(
        fps=30.0,
        # Standart parametreler - frame'ler arası küçük hareketler
        search_radius=30,
        template_size=21,
        
        # Pyramid tracking gerekli değil
        use_pyramid=False,
        pyramid_levels=2,
        
        # Normal eşikler
        confidence_threshold=0.5,
        min_correlation=0.5,
        
        # Frame başına max hareket - normal
        max_motion=40.0,
        
        # Template update - normal
        adaptive_template=True,
        update_rate=0.1,
        
        # Motion prediction - opsiyonel
        enable_prediction=True,
        prediction_lookahead=1,
        
        # Hybrid tracking kapalı
        use_hybrid=False,
        hybrid_switch_threshold=0.75,
        
        # Optical flow - standart
        optical_flow_window_size=(15, 15),
        optical_flow_max_corners=100,
        optical_flow_quality_level=0.3,
    ),
}


def get_tracking_config(fps: float) -> TrackingConfig:
    """
    Get optimal tracking configuration for given FPS.
    
    Args:
        fps: Video frame rate
        
    Returns:
        TrackingConfig: Optimal configuration
    """
    if fps <= 15:
        return FPS_CONFIGS["15fps"]
    elif fps <= 20:
        return FPS_CONFIGS["20fps"]
    elif fps <= 25:
        return FPS_CONFIGS["25fps"]
    else:
        return FPS_CONFIGS["30fps"]


def get_tracker_params(config: TrackingConfig) -> Dict[str, Any]:
    """
    Convert config to tracker parameters.
    
    Args:
        config: Tracking configuration
        
    Returns:
        Dict: Tracker parameters
    """
    return {
        "fps": config.fps,
        "search_radius": config.search_radius,
        "template_size": config.template_size,
        "use_pyramid": config.use_pyramid,
        "pyramid_levels": config.pyramid_levels,
        "matching_method": "cv2.TM_CCOEFF_NORMED",
    }


def get_motion_predictor_params(config: TrackingConfig) -> Dict[str, Any]:
    """
    Convert config to motion predictor parameters.
    
    Args:
        config: Tracking configuration
        
    Returns:
        Dict: Motion predictor parameters
    """
    return {
        "fps": config.fps,
        "history_size": 15 if config.fps < 20 else 10,
        "process_noise": 0.02 if config.fps < 20 else 0.01,
        "measurement_noise": 1.5 if config.fps < 20 else 1.0,
    }


# Special configuration for coronary angiography at 15 FPS
CORONARY_15FPS_CONFIG = TrackingConfig(
    fps=15.0,
    
    # Extra wide search for rapid vessel motion
    search_radius=70,
    template_size=27,
    
    # Multi-scale tracking essential
    use_pyramid=True,
    pyramid_levels=4,
    
    # Very tolerant thresholds
    confidence_threshold=0.35,
    min_correlation=0.3,
    
    # High motion tolerance (cardiac motion + breathing)
    max_motion=100.0,
    
    # Minimal template update (avoid motion blur)
    adaptive_template=True,
    update_rate=0.03,
    
    # Aggressive prediction
    enable_prediction=True,
    prediction_lookahead=3,
    
    # Always use hybrid
    use_hybrid=True,
    hybrid_switch_threshold=0.55,
    
    # Large optical flow window
    optical_flow_window_size=(25, 25),
    optical_flow_max_corners=200,
    optical_flow_quality_level=0.15,
)