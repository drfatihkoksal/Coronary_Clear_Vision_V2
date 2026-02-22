import io
import logging
from datetime import date

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

from app.models.domain import PatientInfo, StudyInfo, PixelSpacing, Study
from app.models.enums import CalibrationSource

logger = logging.getLogger(__name__)

# HIPAA identifiers to strip during anonymization
HIPAA_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "InstitutionName", "ReferringPhysicianName", "StudyID",
    "AccessionNumber", "InstitutionAddress", "PhysiciansOfRecord",
    "PerformingPhysicianName", "OperatorsName", "OtherPatientIDs",
    "PatientAddress", "PatientTelephoneNumbers", "MedicalRecordLocator",
    "RequestingPhysician", "ScheduledPerformingPhysicianName",
]


class DicomError(Exception):
    """DICOM processing error."""
    pass


class DicomHandler:
    """Handles DICOM file parsing, frame extraction, and anonymization."""

    def load_from_bytes(self, data: bytes, anonymize: bool = True) -> tuple[Dataset, Study, list[np.ndarray]]:
        """Parse DICOM data, extract metadata and frames.

        Returns: (dataset, study, frames) where frames is list of numpy arrays (HxW uint8 grayscale)
        """
        ds = pydicom.dcmread(io.BytesIO(data))

        # Validate modality
        modality = getattr(ds, "Modality", "")
        if modality != "XA":
            raise DicomError(f"Expected XA modality, got '{modality}'")

        # Extract frames
        frames = self._extract_frames(ds)
        if len(frames) == 0:
            raise DicomError("No frames found in DICOM file")

        # Anonymize if requested (before extracting metadata)
        if anonymize:
            ds = self._anonymize(ds)

        # Extract metadata
        study = self._extract_study(ds, len(frames), frames[0].shape)

        return ds, study, frames

    def _extract_frames(self, ds: Dataset) -> list[np.ndarray]:
        """Extract all frames from multi-frame DICOM."""
        pixel_data = ds.pixel_array

        if pixel_data.ndim == 2:
            # Single frame
            frames = [self._normalize_frame(pixel_data)]
        elif pixel_data.ndim == 3:
            # Multi-frame: (N, H, W) or (H, W, C)
            frames = [self._normalize_frame(pixel_data[i]) for i in range(pixel_data.shape[0])]
        elif pixel_data.ndim == 4:
            # Multi-frame RGB: (N, H, W, C)
            frames = [self._normalize_frame(pixel_data[i]) for i in range(pixel_data.shape[0])]
        else:
            raise DicomError(f"Unexpected pixel array shape: {pixel_data.shape}")

        return frames

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to uint8 grayscale."""
        if frame.ndim == 3:
            # RGB -> grayscale
            frame = np.mean(frame, axis=2)

        # Normalize to uint8
        if frame.dtype != np.uint8:
            fmin, fmax = frame.min(), frame.max()
            if fmax > fmin:
                frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)

        return frame

    def _extract_study(self, ds: Dataset, num_frames: int, frame_shape: tuple) -> Study:
        """Extract Study metadata from DICOM dataset."""
        # Patient info
        patient = PatientInfo(
            patient_id=str(getattr(ds, "PatientID", "")),
            name=str(getattr(ds, "PatientName", "")),
            birth_date=self._parse_date(getattr(ds, "PatientBirthDate", None)),
            sex=str(getattr(ds, "PatientSex", "")),
            age=self._parse_age(getattr(ds, "PatientAge", None)),
        )

        # Study info
        study_info = StudyInfo(
            study_instance_uid=str(getattr(ds, "StudyInstanceUID", "")),
            series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "")),
            study_date=self._parse_date(getattr(ds, "StudyDate", None)),
            study_time=str(getattr(ds, "StudyTime", "")),
            description=str(getattr(ds, "StudyDescription", "")),
            institution=str(getattr(ds, "InstitutionName", "")),
            modality=str(getattr(ds, "Modality", "XA")),
        )

        # Pixel spacing
        pixel_spacing = self._extract_pixel_spacing(ds)

        # Frame rate
        frame_rate = self._extract_frame_rate(ds)

        h, w = frame_shape[:2]

        return Study(
            patient=patient,
            study_info=study_info,
            num_frames=num_frames,
            frame_rate=frame_rate,
            image_width=w,
            image_height=h,
            pixel_spacing=pixel_spacing,
        )

    def _extract_pixel_spacing(self, ds: Dataset) -> PixelSpacing | None:
        """Extract pixel spacing corrected to isocenter plane.

        ImagerPixelSpacing (0018,1164) is at the **detector** plane.
        PixelSpacing (0028,0030) is at the **patient** plane per DICOM XA standard.

        For ImagerPixelSpacing we apply magnification correction:
            isocenter_ps = detector_ps × (SOD / SID)
        """
        sid = getattr(ds, "DistanceSourceToDetector", None)
        sod = getattr(ds, "DistanceSourceToPatient", None)

        # Try ImagerPixelSpacing first (detector plane — needs correction)
        if hasattr(ds, "ImagerPixelSpacing"):
            ps = ds.ImagerPixelSpacing
            row_spacing = float(ps[0])
            col_spacing = float(ps[1])

            # Correct from detector plane to isocenter plane
            if sid is not None and sod is not None:
                sid_f, sod_f = float(sid), float(sod)
                if sid_f > 0 and sod_f > 0 and sid_f > sod_f:
                    mag_correction = sod_f / sid_f
                    row_spacing *= mag_correction
                    col_spacing *= mag_correction

            return PixelSpacing(
                row_spacing=row_spacing,
                col_spacing=col_spacing,
                source=CalibrationSource.DICOM,
            )

        # Fallback to PixelSpacing (already at patient plane per XA standard)
        if hasattr(ds, "PixelSpacing"):
            ps = ds.PixelSpacing
            return PixelSpacing(
                row_spacing=float(ps[0]),
                col_spacing=float(ps[1]),
                source=CalibrationSource.DICOM,
            )

        # Last resort: compute from detector element spacing + geometry
        if sid is not None and sod is not None:
            det_spacing = getattr(ds, "DetectorElementSpacing", None)
            if det_spacing is not None:
                sid_f, sod_f = float(sid), float(sod)
                if sid_f > 0 and sod_f > 0:
                    iso_spacing = float(det_spacing[0]) * sod_f / sid_f
                    return PixelSpacing(
                        row_spacing=iso_spacing,
                        col_spacing=iso_spacing,
                        source=CalibrationSource.DICOM,
                        confidence=0.85,
                    )

        return None

    def _extract_frame_rate(self, ds: Dataset) -> float:
        """Extract frame rate from DICOM tags."""
        # Try RecommendedDisplayFrameRate
        if hasattr(ds, "RecommendedDisplayFrameRate"):
            return float(ds.RecommendedDisplayFrameRate)
        # Try CineRate
        if hasattr(ds, "CineRate"):
            return float(ds.CineRate)
        # Try FrameTimeVector
        if hasattr(ds, "FrameTimeVector") and len(ds.FrameTimeVector) > 0:
            avg_interval = np.mean([float(t) for t in ds.FrameTimeVector])
            if avg_interval > 0:
                return 1000.0 / avg_interval
        # Try FrameTime
        if hasattr(ds, "FrameTime") and float(ds.FrameTime) > 0:
            return 1000.0 / float(ds.FrameTime)
        return 15.0  # Default

    def _anonymize(self, ds: Dataset) -> Dataset:
        """Strip HIPAA identifiers from dataset."""
        for tag_name in HIPAA_TAGS:
            if hasattr(ds, tag_name):
                if tag_name == "PatientName":
                    ds.PatientName = "ANONYMOUS"
                elif tag_name == "PatientID":
                    ds.PatientID = str(generate_uid())[:16]
                else:
                    delattr(ds, tag_name)
        return ds

    def _parse_date(self, value) -> date | None:
        if value is None:
            return None
        try:
            s = str(value).strip()
            if len(s) == 8:
                return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except (ValueError, IndexError):
            pass
        return None

    def _parse_age(self, value) -> int | None:
        if value is None:
            return None
        try:
            s = str(value).strip()
            if s.endswith("Y"):
                return int(s[:-1])
            return int(s)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def frame_to_png(frame: np.ndarray) -> bytes:
        """Convert numpy frame to PNG bytes."""
        img = Image.fromarray(frame, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
