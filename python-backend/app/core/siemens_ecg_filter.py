"""Siemens ECG Filter - Screen Transition Artifact Removal.

Siemens angiography systems display ECG in curved segments that create
systematic artifacts at transition points between segments. This module
detects and removes these artifacts for accurate R-peak detection.

Common Siemens segment durations: 2.0s, 2.5s, 3.0s, 4.0s
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SiemensECGFilter:
    """Filter for Siemens curved ECG display artifacts at screen transitions."""

    COMMON_SEGMENT_DURATIONS = [2.0, 2.5, 3.0, 4.0]

    def __init__(
        self,
        sampling_rate: float,
        outlier_threshold_factor: float = 3.0,
        transition_window_ms: float = 100.0,
    ):
        self.sampling_rate = sampling_rate
        self.outlier_threshold_factor = outlier_threshold_factor
        self.transition_window_ms = transition_window_ms

    def detect_transitions(self, ecg_data: np.ndarray) -> list[int]:
        """Detect screen transition points in Siemens ECG data.

        Uses two methods:
        1. Jump detection: finds sudden signal changes
        2. Periodic boundary detection: checks common segment boundaries
        """
        if len(ecg_data) < 100:
            return []

        transitions = []

        # Method 1: Detect sudden jumps
        diff = np.abs(np.diff(ecg_data))
        median_diff = np.median(diff)
        std_diff = np.std(diff)
        jump_threshold = max(
            median_diff * self.outlier_threshold_factor,
            median_diff + 3 * std_diff,
        )
        jump_indices = np.where(diff > jump_threshold)[0]

        # Method 2: Check periodic segment boundaries
        signal_duration = len(ecg_data) / self.sampling_rate
        for segment_duration in self.COMMON_SEGMENT_DURATIONS:
            n_segments = int(signal_duration / segment_duration)
            if n_segments > 1:
                for i in range(1, n_segments):
                    boundary_idx = int(i * segment_duration * self.sampling_rate)
                    if boundary_idx < len(diff):
                        window = slice(
                            max(0, boundary_idx - 10),
                            min(len(diff), boundary_idx + 10),
                        )
                        if np.max(diff[window]) > jump_threshold:
                            transitions.append(boundary_idx)

        # Combine, deduplicate, merge nearby (within 50ms)
        all_trans = sorted(set(list(jump_indices) + transitions))
        merge_dist = int(0.05 * self.sampling_rate)
        merged: list[int] = []
        for t in all_trans:
            if not merged or t - merged[-1] > merge_dist:
                merged.append(t)

        if merged:
            logger.debug("Detected %d screen transitions in Siemens ECG", len(merged))
        return merged

    def filter_signal(
        self, ecg_data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter ECG signal to remove transition artifacts.

        Returns (filtered_signal, artifact_mask).
        """
        transitions = self.detect_transitions(ecg_data)
        if not transitions:
            return ecg_data.copy(), np.zeros(len(ecg_data), dtype=bool)

        # Create artifact mask
        mask = np.zeros(len(ecg_data), dtype=bool)
        window_samples = int(self.transition_window_ms * self.sampling_rate / 1000)
        for trans_idx in transitions:
            start = max(0, trans_idx - window_samples // 2)
            end = min(len(ecg_data), trans_idx + window_samples // 2)
            mask[start:end] = True

        # Linear interpolation over masked regions
        filtered = ecg_data.copy()
        affected = np.where(mask)[0]
        if len(affected) > 0:
            regions = np.split(affected, np.where(np.diff(affected) != 1)[0] + 1)
            for region in regions:
                if len(region) == 0:
                    continue
                start, end = region[0], region[-1] + 1
                if start > 0 and end < len(filtered):
                    filtered[start:end] = np.linspace(
                        ecg_data[start - 1], ecg_data[end],
                        end - start, endpoint=False,
                    )

        logger.debug("Filtered %d transition regions", len(transitions))
        return filtered, mask

    def get_suppression_windows(
        self, transitions: list[int], length: int,
    ) -> list[tuple[int, int]]:
        """Get windows where R-peak detection should be suppressed."""
        suppression = int(0.05 * self.sampling_rate)  # 50ms
        return [
            (max(0, t - suppression), min(length, t + suppression))
            for t in transitions
        ]
