"""
DICOM Folder Loader Module
Handles loading DICOM studies from folders with multiple series/projections
"""

import pydicom
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DicomSeries:
    """Represents a single DICOM series"""

    series_number: int
    series_description: str
    series_uid: str
    modality: str
    num_instances: int
    files: List[Path]
    primary_angle: Optional[float] = None
    secondary_angle: Optional[float] = None
    view_position: Optional[str] = None
    num_frames: Optional[int] = None

    @property
    def projection_name(self) -> str:
        """Get a descriptive name for the projection"""
        if self.primary_angle is not None and self.secondary_angle is not None:
            return f"Series {self.series_number}: LAO/RAO {self.primary_angle:.1f}° / CRAN/CAUD {self.secondary_angle:.1f}°"
        elif self.view_position:
            return f"Series {self.series_number}: {self.view_position}"
        else:
            return f"Series {self.series_number}: {self.series_description}"


class DicomFolderLoader:
    """Loads and manages DICOM studies from folders"""

    def __init__(self):
        self.study_path: Optional[Path] = None
        self.series_list: List[DicomSeries] = []
        self.dicomdir: Optional[pydicom.Dataset] = None

    def load_folder(self, folder_path: str) -> bool:
        """
        Load DICOM study from a folder

        Args:
            folder_path: Path to the study folder

        Returns:
            bool: True if successful
        """
        try:
            self.study_path = Path(folder_path)
            if not self.study_path.exists():
                logger.error(f"Folder not found: {folder_path}")
                return False

            # Check for DICOMDIR
            dicomdir_path = self.study_path / "DICOMDIR"
            if dicomdir_path.exists():
                return self._load_from_dicomdir(dicomdir_path)
            else:
                return self._load_from_folder_structure()

        except Exception as e:
            logger.error(f"Error loading folder: {e}")
            return False

    def _load_from_dicomdir(self, dicomdir_path: Path) -> bool:
        """Load study information from DICOMDIR"""
        try:
            # For now, if DICOMDIR exists, just scan the folder structure
            # This is more reliable than parsing DICOMDIR
            logger.info("DICOMDIR found, scanning folder structure instead")
            return self._load_from_folder_structure()

        except Exception as e:
            logger.error(f"Error reading DICOMDIR: {e}")
            return False

    def _load_from_folder_structure(self) -> bool:
        """Load study by scanning folder structure"""
        try:
            self.series_list.clear()

            # Group DICOM files by series
            series_dict: Dict[str, List[Path]] = {}

            # Find all DICOM files
            for file_path in self.study_path.rglob("*"):
                if file_path.is_file():
                    try:
                        # Try to read as DICOM
                        dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                        series_uid = str(dcm.get("SeriesInstanceUID", "unknown"))

                        if series_uid not in series_dict:
                            series_dict[series_uid] = []
                        series_dict[series_uid].append(file_path)

                    except:
                        # Not a DICOM file, skip
                        continue

            # Create series info for each unique series
            for series_uid, files in series_dict.items():
                if files:
                    # Read first file for metadata
                    first_dcm = pydicom.dcmread(str(files[0]), stop_before_pixels=True)

                    # Skip non-image DICOM files
                    # Check if it has pixel data related tags
                    has_pixel_data = any(
                        [
                            hasattr(first_dcm, "PixelData"),
                            hasattr(first_dcm, "FloatPixelData"),
                            hasattr(first_dcm, "DoubleFloatPixelData"),
                            # For multi-frame, NumberOfFrames indicates image data
                            hasattr(first_dcm, "NumberOfFrames")
                            and int(first_dcm.NumberOfFrames) > 0,
                        ]
                    )

                    # Also check SOP Class to filter out non-image types
                    sop_class = str(first_dcm.get("SOPClassUID", ""))
                    non_image_sop_classes = [
                        "1.2.840.10008.5.1.4.1.1.88",  # Structured Report
                        "1.2.840.10008.5.1.4.1.1.11",  # Grayscale Softcopy Presentation State
                        "1.2.840.10008.5.1.4.1.1.104",  # Encapsulated PDF
                    ]

                    if not has_pixel_data or any(
                        sop_class.startswith(prefix) for prefix in non_image_sop_classes
                    ):
                        logger.info(
                            f"Skipping non-image series: {series_uid} - {first_dcm.get('SeriesDescription', 'Unknown')}"
                        )
                        continue

                    series_info = DicomSeries(
                        series_number=int(first_dcm.get("SeriesNumber", 0)),
                        series_description=str(first_dcm.get("SeriesDescription", "Unknown")),
                        series_uid=series_uid,
                        modality=str(first_dcm.get("Modality", "XA")),
                        num_instances=len(files),
                        files=sorted(files),
                    )

                    # Extract projection info
                    if (
                        hasattr(first_dcm, "PositionerPrimaryAngle")
                        and first_dcm.PositionerPrimaryAngle is not None
                    ):
                        try:
                            series_info.primary_angle = float(first_dcm.PositionerPrimaryAngle)
                        except (ValueError, TypeError):
                            pass
                    if (
                        hasattr(first_dcm, "PositionerSecondaryAngle")
                        and first_dcm.PositionerSecondaryAngle is not None
                    ):
                        try:
                            series_info.secondary_angle = float(first_dcm.PositionerSecondaryAngle)
                        except (ValueError, TypeError):
                            pass
                    if hasattr(first_dcm, "ViewPosition"):
                        series_info.view_position = str(first_dcm.ViewPosition)
                    if hasattr(first_dcm, "NumberOfFrames"):
                        series_info.num_frames = int(first_dcm.NumberOfFrames)

                    self.series_list.append(series_info)

            # Sort by series number
            self.series_list.sort(key=lambda x: x.series_number)

            logger.info(f"Found {len(self.series_list)} series in folder")
            return len(self.series_list) > 0

        except Exception as e:
            logger.error(f"Error scanning folder: {e}")
            return False

    def get_series_list(self) -> List[DicomSeries]:
        """Get list of available series"""
        return self.series_list

    def load_series(self, series: DicomSeries) -> Optional[pydicom.Dataset]:
        """
        Load a specific series

        Args:
            series: DicomSeries object to load

        Returns:
            pydicom.Dataset of the first file in the series
        """
        try:
            if not series.files:
                logger.error("No files in series")
                return None

            # For multi-frame DICOM, we only need the first file
            return pydicom.dcmread(str(series.files[0]))

        except Exception as e:
            logger.error(f"Error loading series: {e}")
            return None

    def get_study_info(self) -> Dict[str, Any]:
        """Get general study information"""
        info = {
            "study_path": str(self.study_path),
            "num_series": len(self.series_list),
            "has_dicomdir": self.dicomdir is not None,
        }

        # Try to get patient/study info from first series
        if self.series_list and self.series_list[0].files:
            try:
                dcm = pydicom.dcmread(str(self.series_list[0].files[0]), stop_before_pixels=True)
                info.update(
                    {
                        "patient_name": str(dcm.get("PatientName", "Unknown")),
                        "patient_id": str(dcm.get("PatientID", "Unknown")),
                        "study_date": str(dcm.get("StudyDate", "Unknown")),
                        "study_description": str(dcm.get("StudyDescription", "Unknown")),
                    }
                )
            except:
                pass

        return info
