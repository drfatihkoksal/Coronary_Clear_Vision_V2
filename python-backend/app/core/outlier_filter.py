"""Outlier filters for RWS diameter time series.

Hampel filter: Replaces outliers based on MAD (Median Absolute Deviation).
Double Hampel: Two-pass with different thresholds.
"""
import numpy as np


def hampel_filter(
    data: list[float],
    window_size: int = 5,
    threshold: float = 3.0,
) -> tuple[list[float], list[int]]:
    """Apply Hampel filter to time series data.

    Args:
        data: Input time series
        window_size: Half-window size for local median computation
        threshold: Number of MAD deviations to consider outlier

    Returns:
        (filtered_data, outlier_indices)
    """
    n = len(data)
    if n < 3:
        return list(data), []

    arr = np.array(data, dtype=np.float64)
    filtered = arr.copy()
    outliers = []

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = arr[start:end]

        median = np.median(window)
        mad = 1.4826 * np.median(np.abs(window - median))  # Scale factor for normal distribution

        deviation = np.abs(arr[i] - median)

        if mad < 1e-10:
            # When MAD is ~0 the window is nearly constant.
            # Flag point if it deviates noticeably from that constant.
            if deviation > 1e-10:
                filtered[i] = median
                outliers.append(i)
            continue

        if deviation > threshold * mad:
            filtered[i] = median
            outliers.append(i)

    return filtered.tolist(), outliers


def double_hampel_filter(
    data: list[float],
    window_size_1: int = 5,
    threshold_1: float = 3.0,
    window_size_2: int = 3,
    threshold_2: float = 2.0,
) -> tuple[list[float], list[int]]:
    """Two-pass Hampel filter with different parameters.

    First pass: coarse outlier removal (larger window, higher threshold)
    Second pass: fine outlier removal (smaller window, lower threshold)
    """
    filtered_1, outliers_1 = hampel_filter(data, window_size_1, threshold_1)
    filtered_2, outliers_2 = hampel_filter(filtered_1, window_size_2, threshold_2)

    all_outliers = sorted(set(outliers_1 + outliers_2))
    return filtered_2, all_outliers
