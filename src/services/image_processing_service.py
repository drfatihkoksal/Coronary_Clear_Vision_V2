"""
Image processing service for handling all image manipulation operations.
"""

from typing import Optional, Tuple, List
import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class ImageProcessingService:
    """Service for image processing operations."""
    
    @staticmethod
    def qimage_to_numpy(qimage: QImage) -> np.ndarray:
        """Convert QImage to numpy array."""
        width = qimage.width()
        height = qimage.height()
        
        # Convert to Format_RGB888 if needed
        if qimage.format() != QImage.Format.Format_RGB888:
            qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
        
        # Get the raw data
        ptr = qimage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def numpy_to_qimage(array: np.ndarray) -> QImage:
        """Convert numpy array to QImage."""
        if array.ndim == 2:
            # Grayscale
            height, width = array.shape
            bytes_per_line = width
            return QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif array.ndim == 3:
            # Color
            height, width, channels = array.shape
            bytes_per_line = width * channels
            
            if channels == 3:
                # Convert BGR to RGB
                rgb_array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                return QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            elif channels == 4:
                # BGRA to RGBA
                rgba_array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
                return QImage(rgba_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        
        raise ValueError(f"Unsupported array shape: {array.shape}")
    
    @staticmethod
    def numpy_to_pixmap(array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        qimage = ImageProcessingService.numpy_to_qimage(array)
        return QPixmap.fromImage(qimage)
    
    @staticmethod
    def pixmap_to_numpy(pixmap: QPixmap) -> np.ndarray:
        """Convert QPixmap to numpy array."""
        qimage = pixmap.toImage()
        return ImageProcessingService.qimage_to_numpy(qimage)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    @staticmethod
    def convert_color_space(image: np.ndarray, conversion: int) -> np.ndarray:
        """Convert image color space."""
        return cv2.cvtColor(image, conversion)
    
    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Apply mask overlay to image."""
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Create colored overlay
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green overlay
        
        # Blend with original
        return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    @staticmethod
    def extract_roi(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Extract region of interest from image."""
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def draw_circle(image: np.ndarray, center: Tuple[int, int], radius: int, 
                   color: Tuple[int, int, int], thickness: int = -1) -> np.ndarray:
        """Draw circle on image."""
        result = image.copy()
        cv2.circle(result, center, radius, color, thickness)
        return result
    
    @staticmethod
    def draw_line(image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                 color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
        """Draw line on image."""
        result = image.copy()
        cv2.line(result, pt1, pt2, color, thickness)
        return result
    
    @staticmethod
    def draw_polyline(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], 
                     thickness: int = 1, closed: bool = False) -> np.ndarray:
        """Draw polyline on image."""
        result = image.copy()
        if points.dtype != np.int32:
            points = points.astype(np.int32)
        cv2.polylines(result, [points], closed, color, thickness)
        return result
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Apply Gaussian blur to image."""
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def normalize_image(image: np.ndarray, min_val: int = 0, max_val: int = 255) -> np.ndarray:
        """Normalize image values to specified range."""
        normalized = cv2.normalize(image, None, min_val, max_val, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    @staticmethod
    def calculate_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """Calculate image histogram."""
        if len(image.shape) == 3:
            # Calculate histogram for each channel
            hist_b = cv2.calcHist([image], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [bins], [0, 256])
            return np.stack([hist_b, hist_g, hist_r], axis=1)
        else:
            # Grayscale
            return cv2.calcHist([image], [0], None, [bins], [0, 256])
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, 
                        tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)