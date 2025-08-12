"""
DICOM service for handling DICOM file operations and data management.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import cv2
from ..core.dicom_parser import DicomParser


class DicomService:
    """Service for handling DICOM operations."""

    def __init__(self):
        self.dicom_parser: Optional[DicomParser] = None
        self.current_file_path: Optional[str] = None
        self.metadata_cache: Dict = {}

    def load_dicom_file(self, file_path: str) -> Dict:
        """
        Load a DICOM file.

        Args:
            file_path: Path to DICOM file

        Returns:
            Dictionary with load results
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Create parser
            self.dicom_parser = DicomParser()
            self.current_file_path = file_path

            # Clear metadata cache
            self.metadata_cache.clear()

            # Get basic info
            info = {
                'success': True,
                'file_path': file_path,
                'num_frames': self.get_frame_count(),
                'dimensions': self.get_dimensions(),
                'pixel_spacing': self.get_pixel_spacing(),
                'modality': self.get_modality(),
                'has_ekg': self.has_ekg_data()
            }

            return info

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get a specific frame."""
        if self.dicom_parser:
            return self.dicom_parser.get_frame(index)
        return None

    def get_frame_count(self) -> int:
        """Get total number of frames."""
        if self.dicom_parser:
            return self.dicom_parser.get_frame_count()
        return 0

    def get_dimensions(self) -> Tuple[int, int]:
        """Get frame dimensions."""
        if self.dicom_parser:
            frame = self.dicom_parser.get_frame(0)
            if frame is not None:
                return frame.shape[:2]
        return (0, 0)

    def get_pixel_spacing(self) -> Optional[Tuple[float, float]]:
        """Get pixel spacing if available."""
        if self.dicom_parser:
            return self.dicom_parser.get_pixel_spacing()
        return None

    def get_modality(self) -> str:
        """Get imaging modality."""
        if self.dicom_parser:
            return self.dicom_parser.get_modality()
        return "Unknown"

    def has_ekg_data(self) -> bool:
        """Check if DICOM has EKG data."""
        if self.dicom_parser:
            return self.dicom_parser.has_ekg_data()
        return False

    def get_ekg_data(self) -> Optional[Dict]:
        """Get EKG data if available."""
        if self.dicom_parser and self.has_ekg_data():
            return self.dicom_parser.get_ekg_data()
        return None

    def get_metadata(self, tag: str) -> Optional[str]:
        """
        Get specific metadata tag.

        Args:
            tag: DICOM tag name

        Returns:
            Tag value or None
        """
        # Check cache first
        if tag in self.metadata_cache:
            return self.metadata_cache[tag]

        if self.dicom_parser:
            value = self.dicom_parser.get_tag_value(tag)
            self.metadata_cache[tag] = value
            return value

        return None

    def get_patient_info(self) -> Dict:
        """Get patient information."""
        info = {}

        if self.dicom_parser:
            info['patient_name'] = self.get_metadata('PatientName')
            info['patient_id'] = self.get_metadata('PatientID')
            info['patient_birth_date'] = self.get_metadata('PatientBirthDate')
            info['patient_sex'] = self.get_metadata('PatientSex')

        return info

    def get_study_info(self) -> Dict:
        """Get study information."""
        info = {}

        if self.dicom_parser:
            info['study_date'] = self.get_metadata('StudyDate')
            info['study_time'] = self.get_metadata('StudyTime')
            info['study_description'] = self.get_metadata('StudyDescription')
            info['accession_number'] = self.get_metadata('AccessionNumber')

        return info

    def export_frame(self, frame_index: int, output_path: str, format: str = 'png') -> bool:
        """
        Export a frame to image file.

        Args:
            frame_index: Frame index to export
            output_path: Output file path
            format: Output format (png, jpg, etc.)

        Returns:
            True if successful
        """
        try:
            frame = self.get_frame(frame_index)
            if frame is None:
                return False

            # Normalize to 8-bit
            frame_normalized = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )

            # Save image
            cv2.imwrite(output_path, frame_normalized)
            return True

        except Exception:
            return False

    def export_video(self, output_path: str, fps: int = 30) -> bool:
        """
        Export all frames as video.

        Args:
            output_path: Output video path
            fps: Frames per second

        Returns:
            True if successful
        """
        try:
            if not self.dicom_parser:
                return False

            # Get dimensions
            height, width = self.get_dimensions()

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write frames
            for i in range(self.get_frame_count()):
                frame = self.get_frame(i)
                if frame is not None:
                    # Convert to 8-bit RGB
                    frame_8bit = cv2.normalize(
                        frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )
                    frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)
                    out.write(frame_rgb)

            out.release()
            return True

        except Exception:
            return False

    def close(self):
        """Close current DICOM file."""
        self.dicom_parser = None
        self.current_file_path = None
        self.metadata_cache.clear()