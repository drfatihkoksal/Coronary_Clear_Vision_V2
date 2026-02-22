"""ECG Analyzer: R-peak detection with Siemens artifact suppression.

Implements adaptive R-peak detection optimized for angiography ECG,
with Siemens screen transition artifact suppression.
"""
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt, find_peaks

from app.core.siemens_ecg_filter import SiemensECGFilter


def detect_r_peaks(
    signal: np.ndarray,
    sample_rate: float,
    min_peak_distance_ms: float = 300.0,
) -> list[int]:
    """Detect R-peaks in ECG signal using adaptive algorithm with Siemens suppression.

    Args:
        signal: 1D ECG signal array
        sample_rate: Sampling rate in Hz
        min_peak_distance_ms: Minimum distance between peaks in ms

    Returns:
        List of sample indices where R-peaks occur
    """
    if len(signal) < 10 or sample_rate <= 0:
        return []

    r_peaks = _detect_r_peaks_adaptive(signal, sample_rate, min_peak_distance_ms)
    if r_peaks is not None and len(r_peaks) >= 2:
        return r_peaks

    # Fallback to legacy Pan-Tompkins
    return _detect_r_peaks_legacy(signal, sample_rate, min_peak_distance_ms)


def _detect_r_peaks_adaptive(
    signal: np.ndarray,
    sample_rate: float,
    min_peak_distance_ms: float,
) -> list[int] | None:
    """Adaptive R-peak detection optimized for angiography ECG.

    Suppresses Siemens screen transition artifacts before detection.
    """
    try:
        ecg = signal.copy().astype(np.float64)
        nyquist = sample_rate / 2

        # 1. Detect and suppress Siemens transitions
        diff = np.abs(np.diff(ecg))
        median_diff = np.median(diff)
        std_diff = np.std(diff)
        transition_threshold = median_diff + 2.5 * std_diff
        transition_indices = np.where(diff > transition_threshold)[0]

        if len(transition_indices) > 0:
            suppress_window = int(0.05 * sample_rate)  # 50ms
            for idx in transition_indices:
                start = max(0, idx - suppress_window)
                end = min(len(ecg), idx + suppress_window)
                local_median = np.median(
                    ecg[max(0, start - 50):min(len(ecg), end + 50)]
                )
                ecg[start:end] = local_median

        # 2. Highpass filter (0.5 Hz) to remove baseline wander
        if nyquist > 0.5:
            b, a = butter(2, 0.5 / nyquist, btype='high')
            ecg = filtfilt(b, a, ecg)

        # 3. Bandpass filter for QRS (5-20 Hz)
        low = 5.0 / nyquist
        high = min(20.0 / nyquist, 0.95)
        if low < high:
            b, a = butter(2, [low, high], btype='band')
            ecg_filtered = filtfilt(b, a, ecg)
        else:
            ecg_filtered = ecg

        # 4. Square to enhance peaks
        ecg_squared = ecg_filtered ** 2

        # 5. Moving average (150ms window)
        window_size = max(3, int(0.15 * sample_rate))
        ecg_smooth = uniform_filter1d(ecg_squared, size=window_size)

        # 6. Adaptive threshold
        threshold = np.mean(ecg_smooth) + 0.5 * np.std(ecg_smooth)
        min_distance = max(1, int(min_peak_distance_ms * sample_rate / 1000))

        # 7. Find peaks with prominence
        peaks, _ = find_peaks(
            ecg_smooth,
            height=threshold,
            distance=min_distance,
            prominence=0.1 * np.max(ecg_smooth),
        )

        if len(peaks) < 2:
            # Retry with lower threshold
            threshold = np.mean(ecg_smooth) + 0.3 * np.std(ecg_smooth)
            peaks, _ = find_peaks(
                ecg_smooth,
                height=threshold,
                distance=min_distance,
            )

        if len(peaks) < 2:
            return None

        # 8. Refine to local maxima in original signal
        refined = _refine_peaks(signal, peaks, window=int(0.05 * sample_rate))

        # 9. Remove duplicates and enforce minimum distance
        refined_sorted = sorted(set(refined.tolist()))
        final_peaks = [refined_sorted[0]]
        for peak in refined_sorted[1:]:
            if peak - final_peaks[-1] >= min_distance:
                final_peaks.append(peak)

        # 10. Apply Siemens suppression
        final_peaks = _apply_siemens_suppression(
            final_peaks, signal, sample_rate,
        )

        if len(final_peaks) >= 2:
            return final_peaks

        return None
    except Exception:
        return None


def _detect_r_peaks_legacy(
    signal: np.ndarray,
    sample_rate: float,
    min_peak_distance_ms: float,
) -> list[int]:
    """Legacy Pan-Tompkins fallback."""
    ecg = signal.copy().astype(np.float64)
    ecg = ecg - np.mean(ecg)

    # Bandpass filter (5-15 Hz)
    filtered = _bandpass_filter(ecg, sample_rate, low=5.0, high=15.0)

    # Square and moving average
    squared = filtered ** 2
    window_size = max(1, int(0.15 * sample_rate))
    ma = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

    # Find peaks
    min_distance = max(1, int(min_peak_distance_ms * sample_rate / 1000))
    threshold = 0.35 * np.max(ma)
    peaks, _ = find_peaks(ma, height=threshold, distance=min_distance)

    if len(peaks) == 0:
        threshold = 0.2 * np.max(ma)
        peaks, _ = find_peaks(ma, height=threshold, distance=min_distance)

    refined = _refine_peaks(signal, peaks, window=int(0.05 * sample_rate))
    return refined.tolist()


def _apply_siemens_suppression(
    r_peaks: list[int],
    signal: np.ndarray,
    sample_rate: float,
) -> list[int]:
    """Remove R-peaks that fall within Siemens screen transition windows."""
    try:
        siemens_filter = SiemensECGFilter(sample_rate)
        transitions = siemens_filter.detect_transitions(signal)
        if not transitions or len(transitions) >= len(r_peaks):
            return r_peaks

        suppression = siemens_filter.get_suppression_windows(
            transitions, len(signal),
        )
        filtered = [
            p for p in r_peaks
            if not any(s <= p <= e for s, e in suppression)
        ]
        return filtered if len(filtered) >= 2 else r_peaks
    except Exception:
        return r_peaks


def preprocess_ecg(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """Post-process ECG signal for display.

    1. DC offset removal
    2. Baseline wander removal (median filter, 200ms)
    3. Bandpass filter (0.5-40 Hz)
    """
    if len(signal) < 10 or sample_rate <= 0:
        return signal

    processed = signal.copy().astype(np.float64)

    # 1. Remove DC offset
    processed = processed - np.mean(processed)

    # 2. Remove baseline wander
    window_size = int(0.2 * sample_rate)
    if window_size % 2 == 0:
        window_size += 1
    if window_size >= 3:
        baseline = median_filter(processed, size=window_size, mode='reflect')
        processed = processed - baseline

    # 3. Bandpass filter: 0.5-40 Hz
    nyquist = sample_rate / 2
    low = 0.5 / nyquist
    high = min(40 / nyquist, 0.99)
    if 0 < low < high < 1:
        sos = butter(2, [low, high], btype='band', output='sos')
        processed = sosfiltfilt(sos, processed)

    return processed


def compute_heart_rate(r_peaks: list[int], sample_rate: float) -> float | None:
    """Compute heart rate from R-peak intervals."""
    if len(r_peaks) < 2 or sample_rate <= 0:
        return None
    intervals = np.diff(r_peaks) / sample_rate
    mean_rr = np.mean(intervals)
    if mean_rr <= 0:
        return None
    return 60.0 / mean_rr


def compute_beat_boundaries(
    r_peaks: list[int],
    num_frames: int,
    frame_rate: float,
    sample_rate: float,
) -> list[dict]:
    """Compute cardiac beat boundaries as frame ranges."""
    if len(r_peaks) < 2:
        return []

    beats = []
    samples_per_frame = sample_rate / frame_rate if frame_rate > 0 else 1.0

    for i in range(len(r_peaks) - 1):
        start_frame = int(r_peaks[i] / samples_per_frame)
        end_frame = int(r_peaks[i + 1] / samples_per_frame) - 1

        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame, min(end_frame, num_frames - 1))

        beats.append({
            "beat_number": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    return beats


def add_r_peak(
    r_peaks: list[int],
    sample_index: int,
    sample_rate: float,
    ecg_length: int,
) -> list[int]:
    """Add a single R-peak. Returns updated sorted list."""
    if sample_index < 0 or sample_index >= ecg_length:
        raise ValueError(f"Sample index {sample_index} out of range [0, {ecg_length})")

    min_distance = int(0.05 * sample_rate)  # 50ms
    for peak in r_peaks:
        if abs(peak - sample_index) < min_distance:
            raise ValueError(f"Too close to existing peak at {peak}")

    updated = sorted(r_peaks + [sample_index])
    return updated


def remove_r_peak(
    r_peaks: list[int],
    sample_index: int,
    tolerance: int = 50,
) -> list[int]:
    """Remove R-peak nearest to sample_index within tolerance."""
    if not r_peaks:
        raise ValueError("No R-peaks to remove")

    peaks_arr = np.array(r_peaks)
    distances = np.abs(peaks_arr - sample_index)
    min_idx = int(np.argmin(distances))

    if distances[min_idx] > tolerance:
        raise ValueError(f"No R-peak found within {tolerance} samples of {sample_index}")

    updated = list(r_peaks)
    updated.pop(min_idx)
    return updated


def move_r_peak(
    r_peaks: list[int],
    from_index: int,
    to_index: int,
    ecg_length: int,
    tolerance: int = 50,
) -> list[int]:
    """Move R-peak from one position to another."""
    if to_index < 0 or to_index >= ecg_length:
        raise ValueError(f"Target index {to_index} out of range")

    peaks_arr = np.array(r_peaks)
    distances = np.abs(peaks_arr - from_index)
    min_idx = int(np.argmin(distances))

    if distances[min_idx] > tolerance:
        raise ValueError(f"No R-peak found within {tolerance} samples of {from_index}")

    updated = list(r_peaks)
    updated.pop(min_idx)
    updated.append(to_index)
    return sorted(updated)


def _bandpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    low: float = 5.0,
    high: float = 15.0,
    order: int = 2,
) -> np.ndarray:
    """Apply bandpass Butterworth filter."""
    nyquist = sample_rate / 2
    low_norm = min(low / nyquist, 0.99)
    high_norm = min(high / nyquist, 0.99)

    if low_norm >= high_norm or low_norm <= 0:
        return signal

    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))


def _refine_peaks(
    signal: np.ndarray,
    peaks: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """Refine peak locations to local maxima in original signal."""
    refined = []
    for p in peaks:
        start = max(0, p - window)
        end = min(len(signal), p + window + 1)
        local_max = start + np.argmax(np.abs(signal[start:end]))
        refined.append(local_max)
    return np.array(refined, dtype=int)
