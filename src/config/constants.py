"""
Application-wide constants.
"""

# Application metadata
APP_NAME = "Coronary Clear Vision"
APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "CoronaryClearVision"

# File types
DICOM_EXTENSIONS = [".dcm", ".DCM", ".dicom", ".DICOM"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]

# Default values
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
DEFAULT_FPS = 30
DEFAULT_PLAYBACK_SPEED = 1.0

# Colors (RGB tuples)
COLOR_PRIMARY = (0, 120, 215)
COLOR_SECONDARY = (255, 255, 0)
COLOR_SUCCESS = (0, 255, 0)
COLOR_WARNING = (255, 165, 0)
COLOR_ERROR = (255, 0, 0)
COLOR_PISTACHIO_GREEN = (147, 197, 114)  # Reference points color

# Calibration
MIN_CALIBRATION_DISTANCE = 10  # pixels
DEFAULT_CATHETER_SIZE = 5.0  # French

# Segmentation
SEGMENTATION_MODEL_NAME = "angiopy_model.pth"
SEGMENTATION_CONFIDENCE_THRESHOLD = 0.5

# QCA Analysis
QCA_DEFAULT_SMOOTHING = 5
QCA_MIN_VESSEL_LENGTH = 50  # pixels
QCA_DIAMETER_PRECISION = 2  # decimal places

# UI Constants
TOOLTIP_SHOW_DELAY = 500  # milliseconds
DOUBLE_CLICK_INTERVAL = 400  # milliseconds
ANIMATION_DURATION = 200  # milliseconds

# Limits
MAX_UNDO_STACK_SIZE = 50
MAX_RECENT_FILES = 10
MAX_BATCH_SIZE = 100
