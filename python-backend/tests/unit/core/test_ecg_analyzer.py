import numpy as np
from app.core.ecg_analyzer import detect_r_peaks, compute_heart_rate, compute_beat_boundaries


def _create_synthetic_ecg(duration_s=3.0, sample_rate=500, heart_rate_bpm=72):
    """Create a synthetic ECG-like signal with known R-peak locations."""
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    beat_interval = 60.0 / heart_rate_bpm

    signal = np.random.randn(n_samples) * 0.1  # Baseline noise
    expected_peaks = []

    beat_time = 0.2  # First peak at 200ms
    while beat_time < duration_s - 0.1:
        peak_sample = int(beat_time * sample_rate)
        if peak_sample < n_samples:
            # Create QRS complex
            for offset in range(-5, 6):
                idx = peak_sample + offset
                if 0 <= idx < n_samples:
                    signal[idx] += 2.0 * np.exp(-0.5 * (offset / 2.0) ** 2)
            expected_peaks.append(peak_sample)
        beat_time += beat_interval

    return signal, sample_rate, expected_peaks


class TestECGAnalyzer:
    def test_detect_r_peaks_synthetic(self):
        signal, sr, expected = _create_synthetic_ecg()
        detected = detect_r_peaks(np.array(signal), sr)
        assert len(detected) > 0
        # Should detect approximately the right number of peaks
        assert abs(len(detected) - len(expected)) <= 2

    def test_heart_rate(self):
        signal, sr, expected = _create_synthetic_ecg(heart_rate_bpm=72)
        peaks = detect_r_peaks(np.array(signal), sr)
        hr = compute_heart_rate(peaks, sr)
        assert hr is not None
        assert 50 < hr < 100  # Roughly in range

    def test_beat_boundaries(self):
        beats = compute_beat_boundaries([0, 100, 200], num_frames=30, frame_rate=15, sample_rate=500)
        assert len(beats) == 2
        assert beats[0]["beat_number"] == 1
        assert beats[1]["beat_number"] == 2

    def test_empty_signal(self):
        assert detect_r_peaks(np.array([]), 500) == []
        assert compute_heart_rate([], 500) is None
