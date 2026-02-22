import numpy as np
from app.core.motion_analyzer import compute_motion_signal, detect_motion_peaks


def test_static_frames_zero_motion():
    """Identical frames should have zero motion."""
    frame = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    frames = [frame.copy() for _ in range(5)]
    signal = compute_motion_signal(frames)
    assert len(signal) == 5
    assert signal[0] == 0.0
    assert all(s < 0.5 for s in signal)  # Near-zero


def test_moving_frames_nonzero_motion():
    """Frames with shifting content should have non-zero motion."""
    frames = []
    for i in range(10):
        f = np.zeros((64, 64), dtype=np.uint8)
        f[20+i:30+i, 20:40] = 200  # Shifting rectangle
        frames.append(f)
    signal = compute_motion_signal(frames)
    assert len(signal) == 10
    assert any(s > 0 for s in signal[1:])  # Some motion detected


def test_detect_peaks():
    signal = [0, 1, 5, 2, 0, 1, 6, 2, 0, 1, 7, 1, 0]
    peaks = detect_motion_peaks(signal, min_peak_distance=2)
    assert len(peaks) >= 2  # Should detect at least the two big peaks
