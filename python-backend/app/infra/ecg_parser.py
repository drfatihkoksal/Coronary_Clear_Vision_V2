"""Parse ECG waveform data from DICOM WaveformSequence or CurveData.

Supports:
- WaveformSequence (modern DICOM standard)
- Siemens curved ECG (legacy 50xx group) with transition artifact filtering
- Siemens private tags (0019,1010)
"""
import logging

import numpy as np
from pydicom.dataset import Dataset
from pydicom.tag import Tag

from app.core.siemens_ecg_filter import SiemensECGFilter

logger = logging.getLogger(__name__)


def extract_ecg_from_dicom(ds: Dataset) -> dict | None:
    """Extract ECG signal from DICOM WaveformSequence or legacy CurveData.

    Returns:
        Dict with 'signal', 'sample_rate', 'num_samples',
        and optionally 'siemens_filtered', 'transitions_detected'
        or None if no ECG data found.
    """
    # Try standard WaveformSequence first
    result = _extract_from_waveform_sequence(ds)
    if result is not None:
        return result

    # Fall back to legacy Curve Data (Siemens etc.)
    result = _extract_from_curve_data(ds)
    if result is not None:
        return result

    # Try Siemens private tags
    result = _extract_from_siemens_private(ds)
    if result is not None:
        return result

    logger.debug("No ECG data found in DICOM")
    return None


def _extract_from_waveform_sequence(ds: Dataset) -> dict | None:
    """Extract ECG from standard DICOM WaveformSequence."""
    if not hasattr(ds, "WaveformSequence"):
        return None

    for waveform in ds.WaveformSequence:
        num_channels = int(getattr(waveform, "NumberOfWaveformChannels", 0))
        num_samples = int(getattr(waveform, "NumberOfWaveformSamples", 0))
        sample_rate = float(getattr(waveform, "SamplingFrequency", 0))

        if num_samples == 0 or sample_rate == 0:
            continue

        data = getattr(waveform, "WaveformData", None)
        if data is None:
            continue

        bits = int(getattr(waveform, "WaveformBitsAllocated", 16))
        dtype = np.int16 if bits <= 16 else np.int32

        try:
            samples = np.frombuffer(data, dtype=dtype)

            if num_channels > 1:
                samples = samples.reshape(-1, num_channels)
                signal = samples[:, 0].astype(np.float64)
            else:
                signal = samples.astype(np.float64)

            channel_def = None
            if hasattr(waveform, "ChannelDefinitionSequence") and len(waveform.ChannelDefinitionSequence) > 0:
                channel_def = waveform.ChannelDefinitionSequence[0]

            if channel_def:
                sensitivity = float(getattr(channel_def, "ChannelSensitivity", 1.0))
                baseline = float(getattr(channel_def, "ChannelBaseline", 0.0))
                signal = (signal - baseline) * sensitivity

            logger.debug("Extracted ECG from WaveformSequence: %d samples @ %.1f Hz", len(signal), sample_rate)
            return {
                "signal": signal.tolist(),
                "sample_rate": sample_rate,
                "num_samples": len(signal),
            }
        except Exception as e:
            logger.warning("Failed to parse waveform: %s", e)
            continue

    return None


def _extract_from_curve_data(ds: Dataset) -> dict | None:
    """Extract ECG from legacy DICOM Curve Data tags (group 5000-501E).

    Siemens angiography systems store ECG as Curve Data with:
      - 12-bit ADC range (0-4095), baseline around 2048
      - Stored as unsigned 16-bit
      - Curved display with screen transition artifacts
    """
    for group in range(0x5000, 0x5020, 2):
        try:
            curve_data_tag = Tag(group, 0x3000)
            if curve_data_tag not in ds:
                continue

            # Check type - we want ECG curves
            type_tag = Tag(group, 0x0020)
            if type_tag in ds:
                curve_type = str(ds[type_tag].value).strip().upper()
                if curve_type not in ("ECG", "PRESSURE", ""):
                    continue

            # Get number of points
            num_points_tag = Tag(group, 0x0010)
            if num_points_tag not in ds:
                continue
            num_points = int(ds[num_points_tag].value)
            if num_points == 0:
                continue

            # Get dimensions (number of channels)
            dims_tag = Tag(group, 0x0005)
            num_dims = int(ds[dims_tag].value) if dims_tag in ds else 1

            # Get data representation
            repr_tag = Tag(group, 0x0103)
            data_repr = int(ds[repr_tag].value) if repr_tag in ds else 0
            dtype_map = {0: np.uint16, 1: np.int16, 2: np.float32, 3: np.float64}
            dtype = dtype_map.get(data_repr, np.uint16)

            raw = bytes(ds[curve_data_tag].value)
            samples = np.frombuffer(raw, dtype=dtype)

            if num_dims > 1 and num_dims == 2:
                signal = samples[:num_points].astype(np.float64)
            elif num_dims > 1:
                samples = samples.reshape(-1, num_dims)
                signal = samples[:, 0].astype(np.float64)
            else:
                signal = samples.astype(np.float64)

            # Estimate sample rate
            sample_rate = _estimate_curve_sample_rate(ds, len(signal))

            # Apply Siemens-specific processing
            signal = _process_siemens_curve(signal, sample_rate)

            logger.debug(
                "Extracted ECG from CurveData (group %04X): %d samples @ %.1f Hz",
                group, len(signal), sample_rate,
            )
            return {
                "signal": signal.tolist(),
                "sample_rate": sample_rate,
                "num_samples": len(signal),
                "source": f"SiemensCurve_0x{group:04X}",
                "siemens_filtered": True,
            }
        except Exception as e:
            logger.warning("Failed to parse CurveData group %04X: %s", group, e)
            continue

    return None


def _extract_from_siemens_private(ds: Dataset) -> dict | None:
    """Extract ECG from Siemens private tags (0019,1010)."""
    try:
        siemens_tag = Tag(0x0019, 0x1010)
        if siemens_tag not in ds:
            return None

        raw_data = ds[siemens_tag].value
        if len(raw_data) <= 4:
            return None

        # Skip 4-byte header
        signal = np.frombuffer(raw_data[4:], dtype=np.int16).astype(np.float64)
        sample_rate = 1000.0

        signal = _process_siemens_curve(signal, sample_rate)

        logger.debug("Extracted ECG from Siemens private: %d samples", len(signal))
        return {
            "signal": signal.tolist(),
            "sample_rate": sample_rate,
            "num_samples": len(signal),
            "source": "SiemensPrivate",
            "siemens_filtered": True,
        }
    except Exception as e:
        logger.warning("Siemens private extraction failed: %s", e)
        return None


def _process_siemens_curve(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """Process Siemens ECG data: ADC conversion + transition artifact removal.

    Siemens uses 12-bit ADC with baseline at 2048, ~200 ADC units per mV.
    """
    # Remove DC offset (12-bit ADC center) and convert to millivolts
    baseline = 2048.0
    adc_per_mv = 200.0
    signal = (signal - baseline) / adc_per_mv

    # Apply Siemens screen transition filter
    if sample_rate > 0 and len(signal) > 100:
        try:
            siemens_filter = SiemensECGFilter(sample_rate)
            filtered, mask = siemens_filter.filter_signal(signal)
            transitions = siemens_filter.detect_transitions(signal)
            signal = filtered
            if transitions:
                logger.debug("Siemens filter: %d transitions removed", len(transitions))
        except Exception as e:
            logger.warning("Siemens filter failed: %s", e)

    return signal


def _estimate_curve_sample_rate(ds: Dataset, num_samples: int) -> float:
    """Estimate sample rate for curve data from DICOM metadata."""
    num_frames = int(getattr(ds, "NumberOfFrames", 0))
    frame_rate = float(getattr(ds, "CineRate", 0) or getattr(ds, "RecommendedDisplayFrameRate", 0))

    if num_frames > 0 and frame_rate > 0:
        duration = num_frames / frame_rate
        if duration > 0:
            return num_samples / duration

    # Common Siemens cath lab ECG rate
    return 200.0
