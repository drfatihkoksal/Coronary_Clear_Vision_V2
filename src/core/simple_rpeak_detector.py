"""
Simple R-Peak Detector
Robust R-peak detection using proven methods
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Dict

class SimpleRPeakDetector:
    """Simple and reliable R-peak detector"""

    def __init__(self, sampling_rate: float):
        self.fs = sampling_rate

    def detect_r_peaks(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect R-peaks using modified Pan-Tompkins algorithm

        Args:
            ecg_signal: Preprocessed ECG signal

        Returns:
            Tuple of (r_peak_indices, metrics)
        """
        # 1. Bandpass filter for QRS enhancement (5-15 Hz)
        filtered = self._qrs_filter(ecg_signal)

        # 2. Derivative to emphasize slopes
        derivative = np.gradient(filtered)

        # 3. Square to make all values positive
        squared = derivative ** 2

        # 4. Moving window integration
        window_size = int(0.15 * self.fs)  # 150ms window
        integrated = uniform_filter1d(squared, size=window_size)

        # 5. Find peaks with dynamic thresholding
        r_peaks = self._find_peaks_dynamic(integrated, ecg_signal)

        # 6. Post-process to ensure quality
        r_peaks_final = self._post_process_peaks(ecg_signal, r_peaks)

        # 7. Calculate metrics
        metrics = self._calculate_metrics(r_peaks_final)

        return r_peaks_final, metrics

    def _qrs_filter(self, signal: np.ndarray) -> np.ndarray:
        """Bandpass filter optimized for QRS detection"""
        nyquist = self.fs / 2
        low = 5 / nyquist
        high = min(15 / nyquist, 0.99)

        if low < high:
            sos = scipy_signal.butter(2, [low, high], btype='band', output='sos')
            return scipy_signal.sosfiltfilt(sos, signal)
        return signal

    def _find_peaks_dynamic(self, integrated: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Find peaks using dynamic thresholding"""
        # Initial peak detection with high threshold
        threshold = 0.6 * np.max(integrated)
        min_distance = int(0.3 * self.fs)  # 300ms minimum between peaks

        peaks, properties = scipy_signal.find_peaks(
            integrated,
            height=threshold,
            distance=min_distance
        )

        # If too few peaks, lower threshold
        if len(peaks) < 5:
            threshold = 0.4 * np.max(integrated)
            peaks, properties = scipy_signal.find_peaks(
                integrated,
                height=threshold,
                distance=min_distance
            )

        # Refine peak locations to actual R-peak in original signal
        refined_peaks = []
        search_window = int(0.05 * self.fs)  # 50ms window

        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(original), peak + search_window)

            if start < end:
                # Find maximum absolute value in window
                window = original[start:end]
                max_idx = np.argmax(np.abs(window))
                refined_peaks.append(start + max_idx)

        return np.array(refined_peaks)

    def _post_process_peaks(self, signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Post-process to remove false positives"""
        if len(peaks) < 2:
            return peaks

        # Remove peaks that are too close together
        min_rr_ms = 300  # 300ms = 200 BPM maximum
        min_distance = int(min_rr_ms * self.fs / 1000)

        # Sort peaks
        peaks = np.sort(peaks)

        # Keep only peaks with sufficient spacing
        final_peaks = [peaks[0]]

        for peak in peaks[1:]:
            if peak - final_peaks[-1] >= min_distance:
                final_peaks.append(peak)
            else:
                # If too close, keep the one with higher amplitude
                if abs(signal[peak]) > abs(signal[final_peaks[-1]]):
                    final_peaks[-1] = peak

        # Additional check: remove outliers based on amplitude
        final_peaks = np.array(final_peaks)
        if len(final_peaks) > 3:
            amplitudes = np.abs(signal[final_peaks])
            median_amp = np.median(amplitudes)

            # Remove peaks with amplitude less than 30% of median
            valid_peaks = final_peaks[amplitudes >= 0.3 * median_amp]

            if len(valid_peaks) >= 3:
                final_peaks = valid_peaks

        return final_peaks

    def _calculate_metrics(self, r_peaks: np.ndarray) -> Dict:
        """Calculate detection metrics"""
        metrics = {}

        if len(r_peaks) < 2:
            metrics['heart_rate'] = 0
            metrics['num_peaks'] = len(r_peaks)
            metrics['quality'] = 'poor'
            return metrics

        # RR intervals in milliseconds
        rr_intervals = np.diff(r_peaks) * 1000 / self.fs

        # Heart rate
        mean_rr = np.mean(rr_intervals)
        metrics['heart_rate'] = 60000 / mean_rr if mean_rr > 0 else 0
        metrics['num_peaks'] = len(r_peaks)

        # RR variability (for quality assessment)
        rr_std = np.std(rr_intervals)
        rr_cv = rr_std / mean_rr if mean_rr > 0 else 1.0

        # Quality assessment
        if 40 <= metrics['heart_rate'] <= 200 and rr_cv < 0.3:
            metrics['quality'] = 'good'
        elif 30 <= metrics['heart_rate'] <= 220 and rr_cv < 0.5:
            metrics['quality'] = 'fair'
        else:
            metrics['quality'] = 'poor'

        metrics['rr_mean_ms'] = mean_rr
        metrics['rr_std_ms'] = rr_std

        return metrics