"""
DICOM Parser Module
Handles reading and parsing of DICOM files with multi-frame support
"""

import pydicom
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class DicomParser:
    """Parser for DICOM files with support for Siemens/GE formats"""

    def __init__(self):
        self.dicom_data: Optional[pydicom.Dataset] = None
        self.pixel_array: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.is_multi_frame: bool = False
        self.num_frames: int = 0
        self.pixel_spacing: Optional[float] = None  # mm per pixel from DICOM metadata

    def load_dicom(self, file_path: str) -> bool:
        """
        Load a DICOM file or DICOMDIR

        Args:
            file_path: Path to DICOM file or DICOMDIR

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"DICOM file not found: {file_path}")
                return False

            # Check if it's a DICOMDIR file
            if file_path.name == "DICOMDIR":
                return self._load_dicomdir(file_path)

            # Read DICOM file
            self.dicom_data = pydicom.dcmread(str(file_path), force=True)

            # Extract pixel data
            self.pixel_array = self.dicom_data.pixel_array

            # Extract pixel spacing (prefer PixelSpacing, fallback to ImagerPixelSpacing)
            if hasattr(self.dicom_data, "PixelSpacing"):
                pixel_spacing_row = float(self.dicom_data.PixelSpacing[0])
                pixel_spacing_col = (
                    float(self.dicom_data.PixelSpacing[1])
                    if len(self.dicom_data.PixelSpacing) > 1
                    else pixel_spacing_row
                )
                self.pixel_spacing = pixel_spacing_row  # Keep for backward compatibility
                self.pixel_spacing_row = pixel_spacing_row
                self.pixel_spacing_col = pixel_spacing_col
                logger.info(
                    f"DICOM PixelSpacing: row={pixel_spacing_row:.5f}, col={pixel_spacing_col:.5f} mm/pixel"
                )
                if abs(pixel_spacing_row - pixel_spacing_col) > 0.001:
                    logger.warning(
                        f"Non-square pixels detected! Aspect ratio: {pixel_spacing_row/pixel_spacing_col:.3f}"
                    )
            elif hasattr(self.dicom_data, "ImagerPixelSpacing"):
                pixel_spacing_row = float(self.dicom_data.ImagerPixelSpacing[0])
                pixel_spacing_col = (
                    float(self.dicom_data.ImagerPixelSpacing[1])
                    if len(self.dicom_data.ImagerPixelSpacing) > 1
                    else pixel_spacing_row
                )
                self.pixel_spacing = pixel_spacing_row  # Keep for backward compatibility
                self.pixel_spacing_row = pixel_spacing_row
                self.pixel_spacing_col = pixel_spacing_col
                logger.info(
                    f"DICOM ImagerPixelSpacing: row={pixel_spacing_row:.5f}, col={pixel_spacing_col:.5f} mm/pixel"
                )
                if abs(pixel_spacing_row - pixel_spacing_col) > 0.001:
                    logger.warning(
                        f"Non-square pixels detected! Aspect ratio: {pixel_spacing_row/pixel_spacing_col:.3f}"
                    )
            else:
                self.pixel_spacing = None
                self.pixel_spacing_row = None
                self.pixel_spacing_col = None
                logger.info("No pixel spacing found in DICOM")

            # Check if multi-frame
            if len(self.pixel_array.shape) == 3:
                self.is_multi_frame = True
                self.num_frames = self.pixel_array.shape[0]
            else:
                self.is_multi_frame = False
                self.num_frames = 1
                # Add frame dimension for consistency
                self.pixel_array = self.pixel_array[np.newaxis, ...]

            # Extract metadata
            self._extract_metadata()

            logger.info(f"Successfully loaded DICOM: {file_path.name}")
            logger.info(f"Multi-frame: {self.is_multi_frame}, Frames: {self.num_frames}")

            return True

        except (AttributeError, KeyError, ValueError, pydicom.errors.InvalidDicomError) as e:
            logger.error(f"Error loading DICOM file: {e}")
            return False

    def _extract_metadata(self):
        """Extract relevant metadata from DICOM"""
        if not self.dicom_data:
            return

        # Basic patient info (anonymized)
        self.metadata["patient_name"] = getattr(self.dicom_data, "PatientName", "Anonymous")
        self.metadata["patient_id"] = getattr(self.dicom_data, "PatientID", "Unknown")
        self.metadata["study_date"] = getattr(self.dicom_data, "StudyDate", "")
        self.metadata["modality"] = getattr(self.dicom_data, "Modality", "")

        # Image info
        self.metadata["rows"] = getattr(self.dicom_data, "Rows", 0)
        self.metadata["columns"] = getattr(self.dicom_data, "Columns", 0)
        self.metadata["bits_stored"] = getattr(self.dicom_data, "BitsStored", 0)

        # Window/Level defaults
        self.metadata["window_center"] = getattr(self.dicom_data, "WindowCenter", 128)
        self.metadata["window_width"] = getattr(self.dicom_data, "WindowWidth", 256)

        # Handle list values
        if isinstance(self.metadata["window_center"], (list, pydicom.multival.MultiValue)):
            self.metadata["window_center"] = float(self.metadata["window_center"][0])
        if isinstance(self.metadata["window_width"], (list, pydicom.multival.MultiValue)):
            self.metadata["window_width"] = float(self.metadata["window_width"][0])

        # Frame timing info
        if hasattr(self.dicom_data, "FrameTime"):
            self.metadata["frame_time"] = float(self.dicom_data.FrameTime)
        elif hasattr(self.dicom_data, "FrameTimeVector"):
            self.metadata["frame_time_vector"] = list(self.dicom_data.FrameTimeVector)

        # Projection/View angle information
        if (
            hasattr(self.dicom_data, "PositionerPrimaryAngle")
            and self.dicom_data.PositionerPrimaryAngle is not None
        ):
            try:
                self.metadata["positioner_primary_angle"] = float(
                    self.dicom_data.PositionerPrimaryAngle
                )
            except (ValueError, TypeError):
                pass

        if (
            hasattr(self.dicom_data, "PositionerSecondaryAngle")
            and self.dicom_data.PositionerSecondaryAngle is not None
        ):
            try:
                self.metadata["positioner_secondary_angle"] = float(
                    self.dicom_data.PositionerSecondaryAngle
                )
            except (ValueError, TypeError):
                pass

        if hasattr(self.dicom_data, "ViewPosition"):
            self.metadata["view_position"] = str(self.dicom_data.ViewPosition)

        if hasattr(self.dicom_data, "SeriesDescription"):
            self.metadata["series_description"] = str(self.dicom_data.SeriesDescription)

        # Frame count for multi-frame files
        self.metadata["num_frames"] = self.num_frames

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the DICOM

        Args:
            frame_index: Index of the frame to retrieve

        Returns:
            numpy array of the frame or None
        """
        if self.pixel_array is None:
            return None

        if frame_index < 0 or frame_index >= self.num_frames:
            logger.warning(f"Frame index {frame_index} out of range [0, {self.num_frames})")
            return None

        return self.pixel_array[frame_index]

    def apply_window_level(
        self, frame: np.ndarray, window_center: float = None, window_width: float = None
    ) -> np.ndarray:
        """
        Apply window/level transformation to frame

        Args:
            frame: Input frame
            window_center: Window center value (uses default if None)
            window_width: Window width value (uses default if None)

        Returns:
            Windowed frame as uint8
        """
        if window_center is None:
            window_center = self.metadata.get("window_center", 128)
        if window_width is None:
            window_width = self.metadata.get("window_width", 256)

        # Calculate window bounds
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2

        # Apply windowing
        windowed = np.clip(frame, lower, upper)

        # Scale to 0-255
        if upper > lower:
            windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        else:
            windowed = np.zeros_like(frame, dtype=np.uint8)

        return windowed

    def get_window_presets(self) -> Dict[str, tuple]:
        """Get common window/level presets for angiography"""
        return {
            "Default": (
                self.metadata.get("window_center", 128),
                self.metadata.get("window_width", 256),
            ),
            "Angio": (300, 600),
            "Bone": (400, 1500),
            "Soft Tissue": (40, 400),
            "Lung": (-600, 1500),
        }

    def _load_dicomdir(self, dicomdir_path: Path) -> bool:
        """
        Load first image from DICOMDIR or return list of all available projections

        Args:
            dicomdir_path: Path to DICOMDIR file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read DICOMDIR
            dicomdir = pydicom.dcmread(str(dicomdir_path))

            # Get the directory containing DICOMDIR
            base_dir = dicomdir_path.parent

            # Find first image record
            for record in dicomdir.DirectoryRecordSequence:
                if record.DirectoryRecordType == "IMAGE":
                    # Get referenced file
                    if hasattr(record, "ReferencedFileID"):
                        file_path = base_dir
                        for part in record.ReferencedFileID:
                            file_path = file_path / part

                        # Try to load the referenced file
                        if file_path.exists():
                            logger.info(f"Loading from DICOMDIR: {file_path}")
                            return self.load_dicom(str(file_path))

            logger.error("No valid images found in DICOMDIR")
            return False

        except (AttributeError, KeyError, ValueError, pydicom.errors.InvalidDicomError) as e:
            logger.error(f"Error loading DICOMDIR: {e}")
            return False

    @staticmethod
    def get_dicomdir_projections(dicomdir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Get all projections from a DICOMDIR

        Args:
            dicomdir_path: Path to DICOMDIR file

        Returns:
            List of dictionaries with file_path and metadata for each projection
        """
        projections = []

        try:
            dicomdir_path = Path(dicomdir_path)
            dicomdir = pydicom.dcmread(str(dicomdir_path))
            base_dir = dicomdir_path.parent

            # Group by series and then by projection angles
            series_projections = {}

            # First pass: collect all series UIDs from patient/study level
            series_records = []
            for patient_record in dicomdir.DirectoryRecordSequence:
                if patient_record.DirectoryRecordType == "PATIENT":
                    for study_record in patient_record.DirectoryRecordSequence:
                        if study_record.DirectoryRecordType == "STUDY":
                            for series_record in study_record.DirectoryRecordSequence:
                                if series_record.DirectoryRecordType == "SERIES":
                                    series_records.append(series_record)

            # Process each series
            for series_record in series_records:
                series_uid = getattr(series_record, "SeriesInstanceUID", None)
                if not series_uid:
                    continue

                # Find first image in this series to get metadata
                for image_record in series_record.DirectoryRecordSequence:
                    if image_record.DirectoryRecordType == "IMAGE":
                        if hasattr(image_record, "ReferencedFileID"):
                            file_path = base_dir
                            for part in image_record.ReferencedFileID:
                                file_path = file_path / part

                            if file_path.exists():
                                try:
                                    # Read just the metadata
                                    dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)

                                    # Extract projection metadata
                                    metadata = {
                                        "file_path": str(file_path),
                                        "series_uid": series_uid,
                                    }

                                    # Extract projection angles
                                    primary_angle = None
                                    secondary_angle = None

                                    if hasattr(dcm, "PositionerPrimaryAngle"):
                                        primary_angle = float(dcm.PositionerPrimaryAngle)
                                        metadata["positioner_primary_angle"] = primary_angle

                                    if hasattr(dcm, "PositionerSecondaryAngle"):
                                        secondary_angle = float(dcm.PositionerSecondaryAngle)
                                        metadata["positioner_secondary_angle"] = secondary_angle

                                    if hasattr(dcm, "ViewPosition"):
                                        metadata["view_position"] = str(dcm.ViewPosition)

                                    if hasattr(dcm, "SeriesDescription"):
                                        metadata["series_description"] = str(dcm.SeriesDescription)

                                    # Basic info
                                    metadata["patient_name"] = getattr(
                                        dcm, "PatientName", "Anonymous"
                                    )
                                    metadata["study_date"] = getattr(dcm, "StudyDate", "")
                                    metadata["rows"] = getattr(dcm, "Rows", 0)
                                    metadata["columns"] = getattr(dcm, "Columns", 0)

                                    # Check if multi-frame
                                    if hasattr(dcm, "NumberOfFrames"):
                                        metadata["num_frames"] = int(dcm.NumberOfFrames)
                                    else:
                                        metadata["num_frames"] = 1

                                    # Create unique key for this projection
                                    # Use angles if available, otherwise use series UID
                                    if primary_angle is not None and secondary_angle is not None:
                                        projection_key = (
                                            f"{primary_angle:.1f}_{secondary_angle:.1f}"
                                        )
                                    else:
                                        projection_key = series_uid

                                    # Only add if we haven't seen this projection
                                    if projection_key not in series_projections:
                                        projections.append(
                                            {"file_path": str(file_path), "metadata": metadata}
                                        )
                                        series_projections[projection_key] = True

                                    break  # Only need first image per series

                                except (AttributeError, KeyError, ValueError) as e:
                                    logger.warning(f"Could not read metadata from {file_path}: {e}")

            return projections

        except (AttributeError, KeyError, ValueError, pydicom.errors.InvalidDicomError) as e:
            logger.error(f"Error reading DICOMDIR projections: {e}")
            return []

    @staticmethod
    def get_folder_projections(folder_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Get all DICOM projections from a folder

        Args:
            folder_path: Path to folder containing DICOM files

        Returns:
            List of dictionaries with file_path and metadata for each projection
        """
        projections = []
        projection_map = {}  # To track unique projections

        try:
            folder_path = Path(folder_path)

            # Find all DICOM files
            dicom_files = []
            for ext in ["*.dcm", "*.DCM", "*.dicom", "*.DICOM"]:
                dicom_files.extend(folder_path.glob(ext))

            # Also check files without extension
            for file in folder_path.iterdir():
                if file.is_file() and not file.suffix:
                    try:
                        # Try to read as DICOM
                        pydicom.dcmread(str(file), stop_before_pixels=True)
                        dicom_files.append(file)
                    except (AttributeError, KeyError, ValueError) as e:

                        logger.warning(f"Ignored exception: {e}")

            # Process each DICOM file
            for file_path in dicom_files:
                try:
                    dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)

                    # Extract metadata
                    metadata = {"file_path": str(file_path)}

                    # Extract projection angles
                    primary_angle = None
                    secondary_angle = None

                    if (
                        hasattr(dcm, "PositionerPrimaryAngle")
                        and dcm.PositionerPrimaryAngle is not None
                    ):
                        try:
                            primary_angle = float(dcm.PositionerPrimaryAngle)
                            metadata["positioner_primary_angle"] = primary_angle
                        except (ValueError, TypeError):
                            pass

                    if (
                        hasattr(dcm, "PositionerSecondaryAngle")
                        and dcm.PositionerSecondaryAngle is not None
                    ):
                        try:
                            secondary_angle = float(dcm.PositionerSecondaryAngle)
                            metadata["positioner_secondary_angle"] = secondary_angle
                        except (ValueError, TypeError):
                            pass

                    if hasattr(dcm, "ViewPosition"):
                        metadata["view_position"] = str(dcm.ViewPosition)

                    if hasattr(dcm, "SeriesDescription"):
                        metadata["series_description"] = str(dcm.SeriesDescription)

                    # Basic info
                    metadata["patient_name"] = getattr(dcm, "PatientName", "Anonymous")
                    metadata["study_date"] = getattr(dcm, "StudyDate", "")
                    metadata["rows"] = getattr(dcm, "Rows", 0)
                    metadata["columns"] = getattr(dcm, "Columns", 0)

                    # Check if multi-frame
                    if hasattr(dcm, "NumberOfFrames"):
                        metadata["num_frames"] = int(dcm.NumberOfFrames)
                    else:
                        metadata["num_frames"] = 1

                    # Get series UID
                    series_uid = getattr(dcm, "SeriesInstanceUID", str(file_path))
                    metadata["series_uid"] = series_uid

                    # Create unique key for this projection
                    if primary_angle is not None and secondary_angle is not None:
                        projection_key = f"{series_uid}_{primary_angle:.1f}_{secondary_angle:.1f}"
                    else:
                        projection_key = series_uid

                    # Only add if we haven't seen this projection
                    if projection_key not in projection_map:
                        projections.append({"file_path": str(file_path), "metadata": metadata})
                        projection_map[projection_key] = True

                except (AttributeError, KeyError, ValueError) as e:
                    logger.warning(f"Could not read metadata from {file_path}: {e}")

            return projections

        except (AttributeError, KeyError, ValueError, pydicom.errors.InvalidDicomError) as e:
            logger.error(f"Error reading folder projections: {e}")
            return []

    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return self.num_frames

    def get_frame_rate(self) -> float:
        """Get frame rate in frames per second"""
        if "frame_time" in self.metadata:
            # Frame time is in milliseconds
            frame_time_ms = self.metadata["frame_time"]
            if frame_time_ms > 0:
                return 1000.0 / frame_time_ms

        # Default frame rate if not available
        return 30.0  # Assume 30 fps

    def has_data(self) -> bool:
        """Check if DICOM data is loaded"""
        return self.dicom_data is not None and self.pixel_array is not None
