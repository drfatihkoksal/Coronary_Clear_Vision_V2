import io

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.infra.dicom_handler import DicomHandler, DicomError


def create_test_dicom(num_frames=10, width=64, height=64, modality="XA") -> bytes:
    """Create a minimal multi-frame DICOM file in memory."""
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"  # XA IOD
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = modality
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "12345"
    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = num_frames
    ds.ImagerPixelSpacing = [0.308, 0.308]
    ds.CineRate = 15

    # Create pixel data - gradient frames
    frames = np.zeros((num_frames, height, width), dtype=np.uint8)
    for i in range(num_frames):
        frames[i] = np.full((height, width), i * (255 // max(num_frames - 1, 1)), dtype=np.uint8)
    ds.PixelData = frames.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


class TestDicomHandler:
    def setup_method(self):
        self.handler = DicomHandler()

    def test_load_multiframe(self):
        data = create_test_dicom(num_frames=10, width=64, height=64)
        ds, study, frames = self.handler.load_from_bytes(data, anonymize=False)
        assert study.num_frames == 10
        assert study.image_width == 64
        assert study.image_height == 64
        assert len(frames) == 10
        assert frames[0].shape == (64, 64)
        assert frames[0].dtype == np.uint8

    def test_metadata_extraction(self):
        data = create_test_dicom()
        ds, study, frames = self.handler.load_from_bytes(data, anonymize=False)
        assert study.patient.name == "TEST^PATIENT"
        assert study.patient.patient_id == "12345"
        assert study.study_info.modality == "XA"
        assert study.frame_rate == 15.0
        assert study.pixel_spacing is not None
        assert abs(study.pixel_spacing.row_spacing - 0.308) < 0.001

    def test_anonymization(self):
        data = create_test_dicom()
        ds, study, frames = self.handler.load_from_bytes(data, anonymize=True)
        assert study.patient.name == "ANONYMOUS"
        assert study.patient.patient_id != "12345"

    def test_wrong_modality_raises(self):
        data = create_test_dicom(modality="CT")
        with pytest.raises(DicomError, match="Expected XA"):
            self.handler.load_from_bytes(data)

    def test_frame_to_png(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[10:20, 10:20] = 255
        png = DicomHandler.frame_to_png(frame)
        assert png[:4] == b"\x89PNG"
        assert len(png) > 0

    def test_pixel_spacing_extraction(self):
        data = create_test_dicom()
        ds, study, frames = self.handler.load_from_bytes(data, anonymize=False)
        assert study.pixel_spacing is not None
        assert study.pixel_spacing.source == "dicom"

    def test_single_frame(self):
        data = create_test_dicom(num_frames=1, width=32, height=32)
        ds, study, frames = self.handler.load_from_bytes(data, anonymize=False)
        assert study.num_frames == 1
        assert len(frames) == 1
        assert frames[0].shape == (32, 32)

    def test_imager_pixel_spacing_magnification_correction(self):
        """ImagerPixelSpacing (detector plane) should be corrected to isocenter."""
        data = create_test_dicom()
        # Patch the DICOM with SID/SOD for magnification correction
        ds = pydicom.dcmread(io.BytesIO(data))
        ds.ImagerPixelSpacing = [0.30, 0.30]  # detector spacing
        ds.DistanceSourceToDetector = 1200.0   # SID
        ds.DistanceSourceToPatient = 750.0     # SOD
        buf = io.BytesIO()
        ds.save_as(buf)
        patched_data = buf.getvalue()

        _, study, _ = self.handler.load_from_bytes(patched_data, anonymize=False)
        # isocenter_ps = 0.30 * 750/1200 = 0.1875
        expected = 0.30 * 750.0 / 1200.0
        assert study.pixel_spacing is not None
        assert abs(study.pixel_spacing.row_spacing - expected) < 0.001

    def test_pixel_spacing_no_correction(self):
        """PixelSpacing (patient plane) should NOT be magnification-corrected."""
        data = create_test_dicom()
        ds = pydicom.dcmread(io.BytesIO(data))
        # Remove ImagerPixelSpacing, keep only PixelSpacing
        if hasattr(ds, "ImagerPixelSpacing"):
            del ds.ImagerPixelSpacing
        ds.PixelSpacing = [0.20, 0.20]
        ds.DistanceSourceToDetector = 1200.0
        ds.DistanceSourceToPatient = 750.0
        buf = io.BytesIO()
        ds.save_as(buf)
        patched_data = buf.getvalue()

        _, study, _ = self.handler.load_from_bytes(patched_data, anonymize=False)
        # PixelSpacing is already at patient plane — no correction
        assert study.pixel_spacing is not None
        assert abs(study.pixel_spacing.row_spacing - 0.20) < 0.001

    def test_imager_pixel_spacing_no_sid_sod(self):
        """Without SID/SOD, ImagerPixelSpacing used as-is (no correction possible)."""
        data = create_test_dicom()
        ds = pydicom.dcmread(io.BytesIO(data))
        ds.ImagerPixelSpacing = [0.30, 0.30]
        # Remove geometry tags if present
        for tag in ("DistanceSourceToDetector", "DistanceSourceToPatient"):
            if hasattr(ds, tag):
                delattr(ds, tag)
        buf = io.BytesIO()
        ds.save_as(buf)
        patched_data = buf.getvalue()

        _, study, _ = self.handler.load_from_bytes(patched_data, anonymize=False)
        # No SID/SOD → raw detector spacing used
        assert study.pixel_spacing is not None
        assert abs(study.pixel_spacing.row_spacing - 0.30) < 0.001
