"""
File service for handling all file I/O operations.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
import cv2
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QByteArray
import json
import csv
from ..config.constants import DICOM_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


class FileService:
    """Service for handling file operations."""
    
    @staticmethod
    def is_valid_file(file_path: str) -> bool:
        """Check if file exists and is readable."""
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension in lowercase."""
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def is_dicom_file(file_path: str) -> bool:
        """Check if file is a DICOM file."""
        return FileService.get_file_extension(file_path) in DICOM_EXTENSIONS
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is an image file."""
        return FileService.get_file_extension(file_path) in IMAGE_EXTENSIONS
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Check if file is a video file."""
        return FileService.get_file_extension(file_path) in VIDEO_EXTENSIONS
    
    @staticmethod
    def save_image(image: np.ndarray, file_path: str) -> bool:
        """Save numpy array as image file."""
        try:
            return cv2.imwrite(file_path, image)
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def load_image(file_path: str) -> Optional[np.ndarray]:
        """Load image file as numpy array."""
        try:
            return cv2.imread(file_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def save_numpy_data(data: np.ndarray, file_path: str, compressed: bool = True) -> bool:
        """Save numpy array to file."""
        try:
            if compressed:
                np.savez_compressed(file_path, data=data)
            else:
                np.save(file_path, data)
            return True
        except Exception as e:
            print(f"Error saving numpy data: {e}")
            return False
    
    @staticmethod
    def load_numpy_data(file_path: str) -> Optional[np.ndarray]:
        """Load numpy array from file."""
        try:
            if file_path.endswith('.npz'):
                loaded = np.load(file_path)
                return loaded['data'] if 'data' in loaded else loaded[loaded.files[0]]
            else:
                return np.load(file_path)
        except Exception as e:
            print(f"Error loading numpy data: {e}")
            return None
    
    @staticmethod
    def save_pixmap(pixmap: QPixmap, file_path: str, format: str = "PNG") -> bool:
        """Save QPixmap to file."""
        try:
            return pixmap.save(file_path, format)
        except Exception as e:
            print(f"Error saving pixmap: {e}")
            return False
    
    @staticmethod
    def load_pixmap(file_path: str) -> Optional[QPixmap]:
        """Load QPixmap from file."""
        try:
            pixmap = QPixmap(file_path)
            return pixmap if not pixmap.isNull() else None
        except Exception as e:
            print(f"Error loading pixmap: {e}")
            return None
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
        """Save data as JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Any]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None
    
    @staticmethod
    def save_csv(data: List[List[Any]], file_path: str, headers: Optional[List[str]] = None) -> bool:
        """Save data as CSV file."""
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    @staticmethod
    def load_csv(file_path: str, has_header: bool = True) -> Optional[Tuple[Optional[List[str]], List[List[str]]]]:
        """Load data from CSV file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                if has_header:
                    headers = next(reader, None)
                    data = list(reader)
                    return headers, data
                else:
                    data = list(reader)
                    return None, data
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    @staticmethod
    def create_directory(directory_path: str) -> bool:
        """Create directory if it doesn't exist."""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    
    @staticmethod
    def get_file_info(file_path: str) -> Optional[dict]:
        """Get file information."""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'created': stat.st_ctime,
                'is_file': os.path.isfile(file_path),
                'is_dir': os.path.isdir(file_path),
                'extension': FileService.get_file_extension(file_path)
            }
        except Exception as e:
            print(f"Error getting file info: {e}")
            return None