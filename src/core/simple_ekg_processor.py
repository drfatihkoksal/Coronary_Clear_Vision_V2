"""
Simple EKG Processor
Clean, straightforward EKG processing with focus on accuracy
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from typing import Dict


class SimpleEKGProcessor:
    """Simple and effective EKG processor"""

    def __init__(self, sampling_rate: float = 1000.0):
        self.sampling_rate = sampling_rate

    def process_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Basic signal preprocessing

        Args:
            raw_signal: Raw EKG signal

        Returns:
            Processed signal
        """
        # 1. Remove DC offset
        signal_dc_removed = raw_signal - np.mean(raw_signal)

        # 2. Remove baseline wander using median filter
        window_size = int(0.2 * self.sampling_rate)  # 200ms window
        if window_size % 2 == 0:
            window_size += 1
        baseline = median_filter(signal_dc_removed, size=window_size, mode="reflect")
        signal_detrended = signal_dc_removed - baseline

        # 3. Apply bandpass filter (0.5-40 Hz for ECG)
        nyquist = self.sampling_rate / 2
        low_freq = 0.5 / nyquist
        high_freq = min(40 / nyquist, 0.99)

        if low_freq < high_freq:
            sos = signal.butter(2, [low_freq, high_freq], btype="band", output="sos")
            signal_filtered = signal.sosfiltfilt(sos, signal_detrended)
        else:
            signal_filtered = signal_detrended

        return signal_filtered

    def detect_signal_quality(self, signal: np.ndarray) -> Dict:
        """
        Assess signal quality

        Returns:
            Dictionary with quality metrics
        """
        # Calculate SNR
        noise_estimate = np.std(np.diff(signal)) / np.sqrt(2)
        signal_power = np.std(signal)
        snr = 20 * np.log10(signal_power / (noise_estimate + 1e-10))

        # Check for clipping
        max_val = np.max(np.abs(signal))
        clipping_ratio = np.sum(np.abs(signal) > 0.95 * max_val) / len(signal)

        # Check signal variability
        variability = np.std(signal) / (np.mean(np.abs(signal)) + 1e-10)

        quality = {
            "snr_db": snr,
            "clipping_ratio": clipping_ratio,
            "variability": variability,
            "is_good": snr > 10 and clipping_ratio < 0.01 and variability > 0.1,
        }

        return quality
