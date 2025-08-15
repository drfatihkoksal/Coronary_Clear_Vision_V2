"""
CoTracker3 Configuration for Medical Imaging
Fine-tuned parameters for coronary angiography
"""


class CoTracker3Config:
    """Configuration for optimized CoTracker3 tracking."""

    # Model settings
    DEVICE = "cuda"  # or "cpu"
    USE_MIXED_PRECISION = True  # Faster inference on GPU

    # Tracking parameters
    WINDOW_SIZE = 8  # Number of frames to process together
    MULTI_SCALE_LEVELS = [0.75, 1.0, 1.25]  # Scale levels for multi-scale tracking

    # Confidence and thresholds
    CONFIDENCE_THRESHOLD = 0.15  # Lower = more sensitive (0.0-1.0)
    MOTION_THRESHOLD = 80.0  # Maximum pixel movement between frames
    VISIBILITY_THRESHOLD = 0.3  # Minimum visibility score

    # Preprocessing
    ENHANCE_CONTRAST = True  # CLAHE for medical images
    CLAHE_CLIP_LIMIT = 3.0  # Contrast enhancement strength
    CLAHE_GRID_SIZE = (8, 8)
    DENOISE_STRENGTH = 10  # Denoising filter strength

    # Smoothing and filtering
    USE_KALMAN_FILTER = True  # Smooth trajectories
    USE_MOTION_PREDICTION = True  # Predict next position
    HISTORY_LENGTH = 10  # Number of past positions to keep

    # Post-processing
    SMOOTHING_WEIGHTS = [0.5, 0.3, 0.2]  # Weights for temporal smoothing
    OUTLIER_REJECTION = True  # Remove outlier detections

    # Performance optimizations
    BATCH_PROCESSING = True  # Process multiple points together
    CACHE_PREPROCESSED = True  # Cache preprocessed frames

    # Medical imaging specific
    VESSEL_AWARE_TRACKING = True  # Use vessel structure hints
    CARDIAC_MOTION_COMP = True  # Compensate for cardiac motion

    # Debug and logging
    DEBUG_MODE = False
    SAVE_TRACKING_HISTORY = False

    @classmethod
    def get_angiography_preset(cls):
        """Get optimized settings for coronary angiography."""
        return {
            "window_size": 6,  # Smaller window for faster motion
            "confidence_threshold": 0.1,  # Very sensitive
            "motion_threshold": 100.0,  # Allow larger motion
            "enhance_contrast": True,
            "clahe_clip_limit": 4.0,  # Strong enhancement
            "use_motion_prediction": True,
            "smoothing_weights": [0.6, 0.3, 0.1],  # More weight on current
        }

    @classmethod
    def get_high_accuracy_preset(cls):
        """Get settings for maximum accuracy (slower)."""
        return {
            "window_size": 12,  # More context
            "multi_scale_levels": [0.5, 0.75, 1.0, 1.25, 1.5],  # More scales
            "confidence_threshold": 0.3,  # More selective
            "use_kalman_filter": True,
            "smoothing_weights": [0.4, 0.3, 0.2, 0.1],  # Use more history
        }

    @classmethod
    def get_fast_preset(cls):
        """Get settings for fast tracking (less accurate)."""
        return {
            "window_size": 4,  # Minimal context
            "multi_scale_levels": [1.0],  # Single scale only
            "enhance_contrast": False,  # Skip preprocessing
            "use_kalman_filter": False,
            "use_motion_prediction": False,
        }
