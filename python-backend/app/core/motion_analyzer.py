"""Motion Signal Analyzer: Farneback optical flow for cardiac motion detection."""
import numpy as np
import cv2
from scipy.signal import find_peaks


def compute_motion_signal(
    frames: list[np.ndarray],
    roi: tuple[int, int, int, int] | None = None,
) -> list[float]:
    """Compute frame-to-frame motion magnitude using optical flow.

    Args:
        frames: List of grayscale frames (HxW uint8)
        roi: Optional (x, y, w, h) region to analyze

    Returns:
        List of motion magnitudes (one per frame, first frame is 0.0)
    """
    if len(frames) < 2:
        return [0.0] * len(frames)

    motion = [0.0]

    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        if roi is not None:
            x, y, w, h = roi
            prev = prev[y:y+h, x:x+w]
            curr = curr[y:y+h, x:x+w]

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion.append(float(np.mean(magnitude)))

    return motion


def detect_motion_peaks(
    motion_signal: list[float],
    frame_rate: float = 15.0,
    min_distance_ms: float = 300.0,
) -> list[int]:
    """Detect peaks in motion signal (correspond to maximum cardiac motion).

    Args:
        motion_signal: List of motion magnitudes
        frame_rate: Acquisition frame rate in fps (used to scale peak distance)
        min_distance_ms: Minimum distance between peaks in milliseconds
            (300ms ≈ max ~200 BPM)

    Returns:
        List of frame indices where motion peaks occur
    """
    if len(motion_signal) < 3:
        return []

    min_peak_distance = max(1, int(min_distance_ms * frame_rate / 1000))

    signal = np.array(motion_signal)
    threshold = np.mean(signal) + 0.5 * np.std(signal)

    peaks, _ = find_peaks(
        signal,
        height=threshold,
        distance=min_peak_distance,
    )

    return peaks.tolist()
