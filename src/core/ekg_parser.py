"""
EKG Parser Module
MVP Phase 2: Extract ECG data from DICOM files
Supports multiple vendor formats and legacy/modern curve data
"""

import numpy as np
from typing import Optional, Dict
from pydicom.tag import Tag
from .simple_ekg_processor import SimpleEKGProcessor
from .simple_rpeak_detector import SimpleRPeakDetector
from .cardiac_phase_detector import CardiacPhaseDetector


class EKGParser:
    """Parser for extracting EKG data from DICOM files"""

    # Common EKG-related DICOM tags
    CURVE_DATA_TAG = Tag(0x5000, 0x3000)  # Legacy curve data
    WAVEFORM_SEQUENCE_TAG = Tag(0x5400, 0x0100)  # Modern waveform sequence

    # Siemens private tags
    SIEMENS_ECG_TAG = Tag(0x0019, 0x1010)

    # GE private tags
    GE_ECG_TAG = Tag(0x0009, 0x1010)

    def __init__(self):
        self.ekg_data: Optional[np.ndarray] = None
        self.sampling_rate: float = 1000.0  # Default Hz
        self.num_samples: int = 0
        self.metadata: Dict = {}
        self.processor = SimpleEKGProcessor()
        self.rpeak_detector = None  # Will be initialized when sampling rate is known
        self.phase_detector = None  # Will be initialized when sampling rate is known
        self.quality_metrics: Dict = {}
        self.use_advanced_detection = True  # Flag to use advanced algorithms
        self.cardiac_phases: Optional[Dict] = None

    def extract_ekg_from_dicom(self, dicom_dataset) -> bool:
        """
        Extract EKG data from DICOM dataset
        Returns True if successful
        """
        # Try modern waveform sequence first
        if self._extract_waveform_sequence(dicom_dataset):
            self._post_process_ekg()
            return True

        # Try legacy curve data
        if self._extract_legacy_curve_data(dicom_dataset):
            self._post_process_ekg()
            return True

        # Try Siemens private tags
        if self._extract_siemens_private(dicom_dataset):
            self._post_process_ekg()
            return True

        # Try GE private tags
        if self._extract_ge_private(dicom_dataset):
            self._post_process_ekg()
            return True

        return False

    def _post_process_ekg(self):
        """Apply advanced processing to extracted EKG data"""
        if self.ekg_data is None or len(self.ekg_data) < 10:
            return

        # Apply simple processing
        self.processor.sampling_rate = self.sampling_rate
        self.ekg_data = self.processor.process_signal(self.ekg_data)

    def _extract_waveform_sequence(self, ds) -> bool:
        """Extract from modern waveform sequence (5400,0100)"""
        try:
            if hasattr(ds, "WaveformSequence") and len(ds.WaveformSequence) > 0:
                waveform = ds.WaveformSequence[0]

                # Get sampling frequency
                if hasattr(waveform, "SamplingFrequency"):
                    self.sampling_rate = float(waveform.SamplingFrequency)

                # Get number of samples
                if hasattr(waveform, "NumberOfWaveformSamples"):
                    self.num_samples = int(waveform.NumberOfWaveformSamples)

                # Get waveform data
                if hasattr(waveform, "WaveformData"):
                    raw_data = waveform.WaveformData

                    # Determine data type
                    if hasattr(waveform, "WaveformBitsAllocated"):
                        bits = int(waveform.WaveformBitsAllocated)
                        if bits == 16:
                            self.ekg_data = np.frombuffer(raw_data, dtype=np.int16)
                        elif bits == 8:
                            self.ekg_data = np.frombuffer(raw_data, dtype=np.int8)
                        else:
                            self.ekg_data = np.frombuffer(raw_data, dtype=np.int16)
                    else:
                        self.ekg_data = np.frombuffer(raw_data, dtype=np.int16)

                    # Normalize to millivolts if needed
                    self._normalize_ekg_data(waveform)

                    self.metadata["source"] = "WaveformSequence"
                    return True

        except (AttributeError, KeyError, ValueError):
            pass

        return False

    def _extract_legacy_curve_data(self, ds) -> bool:
        """Extract from legacy curve data (5000-501E group)"""
        try:
            pass
            # Check for curve data in group 0x5000-0x501E
            for group in range(0x5000, 0x5020, 0x0002):
                curve_data_tag = Tag(group, 0x3000)
                curve_dimensions_tag = Tag(group, 0x0005)
                curve_samples_tag = Tag(group, 0x0010)

                if curve_data_tag in ds:
                    # Get curve dimensions (can be 1 or 2 for ECG)
                    dimensions = ds.get(curve_dimensions_tag, 1)

                    # Get number of points
                    num_points = ds.get(curve_samples_tag, 0)
                    if num_points == 0:
                        continue

                    # Get curve data
                    raw_data = ds[curve_data_tag].value

                    # Check the curve type and label
                    curve_label_tag = Tag(group, 0x0040)
                    curve_type_tag = Tag(group, 0x0020)

                    # For Siemens, check if this is ECG data
                    # Type of Data = 'ECG' or label contains ECG
                    is_ecg = False

                    if curve_type_tag in ds:
                        curve_type = str(ds[curve_type_tag].value).upper()
                        if "ECG" in curve_type or "EKG" in curve_type:
                            is_ecg = True

                    if curve_label_tag in ds and not is_ecg:
                        label = str(ds[curve_label_tag].value).lower()
                        if "ecg" in label or "ekg" in label:
                            is_ecg = True

                    # If no clear ECG indicator but we have curve data, assume it's ECG
                    if not is_ecg and curve_data_tag in ds:
                        is_ecg = True  # Assume curve data is ECG for coronary angiography

                    if not is_ecg:
                        continue

                    # Parse based on data representation
                    # Siemens uses unsigned 16-bit for ECG in curve data
                    curve_data_repr_tag = Tag(group, 0x0103)
                    if curve_data_repr_tag in ds:
                        data_repr = ds[curve_data_repr_tag].value
                        if data_repr == 0:  # unsigned
                            raw_values = np.frombuffer(raw_data, dtype=np.uint16)
                        else:  # signed
                            raw_values = np.frombuffer(raw_data, dtype=np.int16)
                    else:
                        # Default to unsigned 16-bit for Siemens
                        raw_values = np.frombuffer(raw_data, dtype=np.uint16)

                    # Handle dimensions
                    if dimensions == 2:
                        # 2D curve: For Siemens, this is NOT alternating pairs
                        # The data still contains only ECG values, dimensions=2 refers to the coordinate system
                        # Number of points matches the actual ECG samples
                        self.ekg_data = raw_values[:num_points].astype(np.float32)
                    else:
                        # 1D curve: all values are ECG samples
                        self.ekg_data = raw_values.astype(np.float32)

                    self.num_samples = len(self.ekg_data)

                    # Get sampling information from coordinate step
                    coord_step_tag = Tag(group, 0x0048)  # Coordinate Step Value (correct tag)
                    coord_start_tag = Tag(group, 0x0032)  # Coordinate Start Value

                    # Calculate sampling rate based on video duration
                    # Get frame rate and number of frames
                    fps = float(ds.get("CineRate", 15))
                    num_frames = int(ds.get("NumberOfFrames", 1))
                    video_duration = num_frames / fps

                    # EKG samples cover the entire video duration
                    self.sampling_rate = self.num_samples / video_duration

                    # Store metadata
                    self.metadata["source"] = f"LegacyCurve_{group:04X}"
                    if curve_label_tag in ds:
                        self.metadata["label"] = str(ds[curve_label_tag].value)

                    # Process the data
                    self._process_siemens_ecg_data()

                    return True

        except (AttributeError, KeyError, ValueError):
            pass

        return False

    def _extract_siemens_private(self, ds) -> bool:
        """Extract from Siemens private tags"""
        try:
            if self.SIEMENS_ECG_TAG in ds:
                raw_data = ds[self.SIEMENS_ECG_TAG].value

                # Siemens typically uses a specific format
                # First 4 bytes: header info
                # Remaining: 16-bit signed ECG samples

                if len(raw_data) > 4:
                    # Skip header and parse ECG data
                    self.ekg_data = np.frombuffer(raw_data[4:], dtype=np.int16)
                    self.num_samples = len(self.ekg_data)

                    # Siemens usually uses 1000Hz sampling
                    self.sampling_rate = 1000.0

                    self.metadata["source"] = "SiemensPrivate"
                    return True

        except (AttributeError, KeyError, ValueError):
            pass

        return False

    def _extract_ge_private(self, ds) -> bool:
        """Extract from GE private tags"""
        try:
            if self.GE_ECG_TAG in ds:
                raw_data = ds[self.GE_ECG_TAG].value

                # GE format varies, but typically:
                # Header (variable length) + 16-bit ECG samples

                # Try to find ECG data pattern
                # Look for repeating waveform pattern
                data_array = np.frombuffer(raw_data, dtype=np.int16)

                if len(data_array) > 100:
                    self.ekg_data = data_array
                    self.num_samples = len(self.ekg_data)

                    # GE typically uses 500Hz or 1000Hz
                    self.sampling_rate = 500.0  # Conservative estimate

                    self.metadata["source"] = "GEPrivate"
                    return True

        except (AttributeError, KeyError, ValueError):
            pass

        return False

    def _normalize_ekg_data(self, waveform_item=None):
        """Normalize EKG data to millivolts"""
        if self.ekg_data is None:
            return

        # If we have waveform metadata, use it for proper scaling
        if waveform_item is not None:
            if hasattr(waveform_item, "ChannelSensitivity"):
                sensitivity = float(waveform_item.ChannelSensitivity)
                self.ekg_data = self.ekg_data * sensitivity
            elif hasattr(waveform_item, "ChannelSensitivityCorrectionFactor"):
                factor = float(waveform_item.ChannelSensitivityCorrectionFactor)
                self.ekg_data = self.ekg_data * factor
        else:
            # Default normalization - assume data is in ADC units
            # Typical ECG range is ±5mV, map to reasonable values
            max_val = np.max(np.abs(self.ekg_data))
            if max_val > 0:
                self.ekg_data = (self.ekg_data / max_val) * 5.0  # Scale to ±5mV range

    def _process_siemens_ecg_data(self):
        """Process Siemens ECG curve data"""
        if self.ekg_data is None:
            return

        # Siemens ECG data characteristics:
        # - Typically 12-bit ADC (0-4095 range)
        # - Baseline around 2048
        # - Stored as unsigned 16-bit

        # Remove DC offset (center around zero)
        baseline = 2048.0  # Typical 12-bit ADC center
        self.ekg_data = self.ekg_data - baseline

        # Convert to millivolts
        # Typical ECG gain: 1 mV = ~200 ADC units
        adc_per_mv = 200.0
        self.ekg_data = self.ekg_data / adc_per_mv

        # Apply smoothing if needed
        if len(self.ekg_data) > 10:
            from scipy.ndimage import uniform_filter1d

            # Light smoothing to reduce noise
            self.ekg_data = uniform_filter1d(self.ekg_data, size=3)

    def _estimate_sampling_rate(self, ds):
        """Estimate sampling rate from DICOM metadata"""
        # Try to get from frame time
        if hasattr(ds, "FrameTime"):
            frame_time_ms = float(ds.FrameTime)
            if hasattr(ds, "NumberOfFrames"):
                num_frames = int(ds.NumberOfFrames)
                total_duration = (frame_time_ms * num_frames) / 1000.0  # seconds
                if self.num_samples > 0 and total_duration > 0:
                    self.sampling_rate = self.num_samples / total_duration
                    return

        # Default to 1000Hz if can't determine
        self.sampling_rate = 1000.0

    def detect_r_peaks(self) -> Optional[np.ndarray]:
        """
        R-peak detection using simple, reliable algorithm
        Returns array of R-peak indices
        """
        if self.ekg_data is None or len(self.ekg_data) < 10:
            return None

        try:
            # Initialize detector if needed
            if self.rpeak_detector is None:
                self.rpeak_detector = SimpleRPeakDetector(self.sampling_rate)

            # Process signal first
            self.processor.sampling_rate = self.sampling_rate
            processed_signal = self.processor.process_signal(self.ekg_data)

            # Check signal quality
            quality = self.processor.detect_signal_quality(processed_signal)

            if not quality["is_good"]:
                pass

            # Detect R-peaks
            r_peaks, metrics = self.rpeak_detector.detect_r_peaks(processed_signal)

            # Store quality metrics
            self.quality_metrics = {
                "quality_score": 0.8 if metrics["quality"] == "good" else 0.5,
                "message": f"Simple detection ({metrics['quality']} quality)",
                "mean_hr": metrics["heart_rate"],
                "num_peaks": metrics["num_peaks"],
                "signal_quality": quality,
            }

            return r_peaks

        except (AttributeError, KeyError, ValueError):
            pass
            # Fallback to simple detection
            return self._detect_r_peaks_simple()

    def _detect_r_peaks_simple(self) -> Optional[np.ndarray]:
        """
        Simple R-peak detection using derivative method (legacy)
        """
        if self.ekg_data is None or len(self.ekg_data) < 10:
            return None

        try:
            from scipy import signal

            # Design bandpass filter
            nyquist = self.sampling_rate / 2
            low = 5 / nyquist
            high = min(15 / nyquist, 0.99)

            if low < high:
                b, a = signal.butter(2, [low, high], btype="band")
                filtered = signal.filtfilt(b, a, self.ekg_data)
            else:
                filtered = self.ekg_data

            # Square the signal
            squared = filtered**2

            # Moving average
            window_size = int(0.15 * self.sampling_rate)
            ma = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

            # Find peaks
            threshold = 0.35 * np.max(ma)
            min_distance = int(0.3 * self.sampling_rate)
            peaks, _ = signal.find_peaks(ma, height=threshold, distance=min_distance)

            # Refine peak locations
            refined_peaks = []
            window = int(0.05 * self.sampling_rate)

            for peak in peaks:
                start = max(0, peak - window)
                end = min(len(self.ekg_data), peak + window)
                local_max = np.argmax(self.ekg_data[start:end])
                refined_peaks.append(start + local_max)

            return np.array(refined_peaks)

        except (AttributeError, KeyError, ValueError):
            return None

    def get_heart_rate(self) -> Optional[float]:
        """Calculate average heart rate from R-peaks"""
        r_peaks = self.detect_r_peaks()
        if r_peaks is None or len(r_peaks) < 2:
            return None

        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / self.sampling_rate  # in seconds

        # Average heart rate
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60.0 / avg_rr  # BPM

        return heart_rate

    def get_hrv_metrics(self) -> Optional[Dict]:
        """Get heart rate variability metrics"""
        r_peaks = self.detect_r_peaks()
        if r_peaks is None or len(r_peaks) < 3:
            return None

        # Simple HRV calculation
        if len(r_peaks) < 2:
            return {}

        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # in ms

        return {
            "mean_rr": np.mean(rr_intervals),
            "std_rr": np.std(rr_intervals),
            "rmssd": np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
            "pnn50": np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
        }

    def get_signal_quality(self) -> Dict:
        """Get signal quality assessment"""
        return self.quality_metrics

    def set_advanced_detection(self, enabled: bool):
        """Enable/disable advanced R-peak detection"""
        self.use_advanced_detection = enabled

    def reset(self):
        """Reset parser state for new DICOM file"""
        self.ekg_data = None
        self.sampling_rate = 1000.0
        self.num_samples = 0
        self.metadata = {}
        self.quality_metrics = {}
        self.cardiac_phases = None
        self.rpeak_detector = None
        self.phase_detector = None

    def detect_cardiac_phases(self) -> Optional[Dict]:
        """
        Detect cardiac phases (D1, D2, S1, S2) for analysis
        Returns phase information or None if detection fails
        """
        if self.ekg_data is None:
            return None

        # First detect R-peaks if not already done
        r_peaks = self.detect_r_peaks()
        if r_peaks is None or len(r_peaks) < 2:
            return None

        # Initialize phase detector if needed
        if self.phase_detector is None:
            self.phase_detector = CardiacPhaseDetector(self.sampling_rate)

        # Detect phases
        self.cardiac_phases = self.phase_detector.detect_phases(self.ekg_data, r_peaks)

        if self.cardiac_phases and "statistics" in self.cardiac_phases:
            self.cardiac_phases["statistics"]

        return self.cardiac_phases

    def get_cardiac_phases(self) -> Optional[Dict]:
        """Get previously detected cardiac phases"""
        return self.cardiac_phases

    def get_phase_at_time(self, time_s: float) -> Optional[str]:
        """Get cardiac phase at specific time"""
        if self.cardiac_phases is None:
            self.detect_cardiac_phases()

        if self.cardiac_phases and "phases" in self.cardiac_phases:
            return self.phase_detector.get_phase_at_time(self.cardiac_phases["phases"], time_s)
        return None
