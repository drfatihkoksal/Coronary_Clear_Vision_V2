"""
Cardiac Phase Detector
Detects cardiac phases (D1, D2, S1, S2) based on R-peaks and ECG morphology
Used for cardiac phase analysis
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import signal as scipy_signal


class CardiacPhaseDetector:
    """Detect cardiac phases for analysis"""

    def __init__(self, sampling_rate: float):
        self.fs = sampling_rate

    def detect_phases(
        self, ecg_signal: np.ndarray, r_peaks: np.ndarray, include_beat_numbers: bool = True
    ) -> Dict:
        """
        Detect cardiac phases based on R-peaks and ECG morphology

        Phases for analysis:
        - D2: End-diastole (just before R wave)
        - S1: Early-systole (peak of R wave)
        - S2: End-systole (end of T wave)
        - D1: Mid-diastole (middle of diastolic period)

        Phase transitions:
        - D2→S1: End-diastole
        - S1→S2: Early-systole
        - S2→D1: End-systole
        - D1→D2: Mid-diastole

        Args:
            ecg_signal: ECG signal data
            r_peaks: Indices of R-peaks

        Returns:
            Dictionary containing phase indices and timing information
        """
        if len(r_peaks) < 2:
            return {"phases": [], "error": "Insufficient R-peaks for phase detection"}

        phases = []
        beat_number = 1  # Start beat numbering from 1

        # Process each cardiac cycle
        for i in range(len(r_peaks)):
            current_r = r_peaks[i]

            if i < len(r_peaks) - 1:
                # Normal case: we have next R-peak
                next_r = r_peaks[i + 1]
                rr_interval = next_r - current_r
            else:
                # Last R-peak: estimate RR interval from previous cycles
                if i > 0:
                    # Use average of previous RR intervals
                    prev_intervals = []
                    for j in range(max(0, i - 2), i):
                        prev_intervals.append(r_peaks[j + 1] - r_peaks[j])
                    avg_rr = np.mean(prev_intervals) if prev_intervals else 0
                    rr_interval = int(avg_rr)
                    # Create virtual next_r for phase calculations
                    next_r = current_r + rr_interval
                else:
                    # Only one R-peak, can't calculate phases
                    continue

            # Ensure we don't go beyond signal length
            if next_r > len(ecg_signal):
                next_r = len(ecg_signal)
                rr_interval = next_r - current_r

            # Detect phases for this cycle
            cycle_phases = self._detect_cycle_phases(ecg_signal, current_r, next_r, rr_interval)

            if cycle_phases:
                # Add beat number if requested
                if include_beat_numbers:
                    cycle_phases["beat_number"] = beat_number
                phases.append(cycle_phases)
                beat_number += 1  # Increment for next beat

        # Calculate average phase timings relative to R-peak
        phase_statistics = self._calculate_phase_statistics(phases, r_peaks)

        return {
            "phases": phases,
            "r_peaks": r_peaks,
            "statistics": phase_statistics,
            "sampling_rate": self.fs,
        }

    def _detect_cycle_phases(
        self, signal: np.ndarray, r_current: int, r_next: int, rr_interval: int
    ) -> Dict:
        """Detect phases within a single cardiac cycle"""

        phases = {}

        # S1: Early-systole (R-peak itself)
        phases["S1"] = {
            "index": r_current,
            "time": r_current / self.fs,
            "phase_name": "Early-systole",
        }

        # D2: End-diastole (just before R-peak)
        # Adjust offset based on heart rate
        # Higher heart rate = shorter phases
        heart_rate = 60000 / ((rr_interval / self.fs) * 1000)

        # Base offset 50ms, adjusted by heart rate
        # Normal HR (60-80): 40-60ms
        # High HR (>100): 30-40ms
        # Low HR (<60): 50-70ms
        base_offset_ms = 50
        hr_factor = 70 / heart_rate  # Normalize to HR=70
        d2_offset_ms = base_offset_ms * hr_factor
        d2_offset_ms = np.clip(d2_offset_ms, 30, 70)  # Keep within reasonable bounds

        d2_offset = int(d2_offset_ms * self.fs / 1000)
        d2_index = max(0, r_current - d2_offset)
        phases["D2"] = {"index": d2_index, "time": d2_index / self.fs, "phase_name": "End-diastole"}

        # S2: End-systole (end of T-wave)
        # Detect T-wave end using derivative method
        s2_index = self._detect_t_wave_end(signal, r_current, rr_interval)
        if s2_index is None:
            # Fallback: estimate based on heart rate
            # Normal HR: 35-40% of RR interval
            # High HR: 40-45% (relatively longer systole)
            # Low HR: 30-35% (relatively shorter systole)
            systole_fraction = 0.35 + (0.10 * (heart_rate - 70) / 70)
            systole_fraction = np.clip(systole_fraction, 0.30, 0.45)
            s2_index = r_current + int(systole_fraction * rr_interval)

        phases["S2"] = {
            "index": min(s2_index, r_next - int(0.1 * rr_interval)),
            "time": s2_index / self.fs,
            "phase_name": "End-systole",
        }

        # D1: Mid-diastole
        # Middle of diastolic period between end-systole and next end-diastole
        diastole_start = phases["S2"]["index"]
        diastole_end = r_next - d2_offset
        d1_index = diastole_start + (diastole_end - diastole_start) // 2

        phases["D1"] = {"index": d1_index, "time": d1_index / self.fs, "phase_name": "Mid-diastole"}

        # Add RR interval info
        phases["rr_interval_ms"] = (rr_interval / self.fs) * 1000
        phases["heart_rate"] = 60000 / phases["rr_interval_ms"]

        return phases

    def _detect_t_wave_end(
        self, signal: np.ndarray, r_peak: int, rr_interval: int
    ) -> Optional[int]:
        """Detect end of T-wave using derivative analysis"""

        # Adjust search window based on heart rate
        heart_rate = 60000 / ((rr_interval / self.fs) * 1000)

        # Normal HR: 200-400ms
        # High HR: 150-350ms (shorter)
        # Low HR: 250-450ms (longer)
        hr_factor = 70 / heart_rate
        start_ms = 200 * hr_factor
        end_ms = 400 * hr_factor

        # Keep within reasonable bounds
        start_ms = np.clip(start_ms, 150, 250)
        end_ms = np.clip(end_ms, 350, 450)

        search_start = r_peak + int(start_ms * self.fs / 1000)
        search_end = min(r_peak + int(end_ms * self.fs / 1000), r_peak + int(0.5 * rr_interval))

        if search_start >= len(signal) or search_end >= len(signal):
            return None

        # Extract search window
        window = signal[search_start:search_end]
        if len(window) < 10:
            return None

        # Smooth the signal
        window_smooth = scipy_signal.savgol_filter(
            window, window_length=min(11, len(window) // 2 * 2 + 1), polyorder=3
        )

        # Calculate derivative
        derivative = np.gradient(window_smooth)

        # Find where derivative crosses zero from positive to negative
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

        if len(zero_crossings) > 0:
            # Take the last significant zero crossing
            t_end_idx = zero_crossings[-1]
            return search_start + t_end_idx

        return None

    def _calculate_phase_statistics(self, phases: List[Dict], r_peaks: np.ndarray) -> Dict:
        """Calculate statistics for phase timings"""

        if not phases:
            return {}

        stats = {
            "D2_offset_ms": [],
            "S2_offset_ms": [],
            "D1_offset_ms": [],
            "systole_duration_ms": [],
            "diastole_duration_ms": [],
        }

        for i, phase in enumerate(phases):
            r_peak_time = phase["S1"]["index"]  # S1 is now the R-peak

            # Calculate offsets relative to R-peak
            d2_offset = (r_peak_time - phase["D2"]["index"]) / self.fs * 1000
            s2_offset = (phase["S2"]["index"] - r_peak_time) / self.fs * 1000
            d1_offset = (phase["D1"]["index"] - r_peak_time) / self.fs * 1000

            stats["D2_offset_ms"].append(d2_offset)
            stats["S2_offset_ms"].append(s2_offset)
            stats["D1_offset_ms"].append(d1_offset)

            # Calculate phase durations
            systole_duration = (phase["S2"]["index"] - phase["S1"]["index"]) / self.fs * 1000

            if i < len(phases) - 1:
                next_d2 = phases[i + 1]["D2"]["index"]
                diastole_duration = (next_d2 - phase["S2"]["index"]) / self.fs * 1000
                stats["diastole_duration_ms"].append(diastole_duration)

            stats["systole_duration_ms"].append(systole_duration)

        # Calculate mean and std for each metric
        result = {}
        for key, values in stats.items():
            if values:
                result[f"{key}_mean"] = np.mean(values)
                result[f"{key}_std"] = np.std(values)

        return result

    def get_phase_at_time(self, phases: List[Dict], time_s: float) -> Optional[str]:
        """
        Get the cardiac phase at a specific time point

        Args:
            phases: List of phase dictionaries from detect_phases
            time_s: Time in seconds

        Returns:
            Phase code at the given time (d1, d2, s1, s2), or None if not found
        """
        if not phases:
            return None

        # Create a sorted list of all phase events with their types
        phase_events = []

        for i, cycle in enumerate(phases):
            cycle_num = i
            phase_events.append((cycle["D1"]["time"], "d1", cycle_num))
            phase_events.append((cycle["D2"]["time"], "d2", cycle_num))
            phase_events.append((cycle["S1"]["time"], "s1", cycle_num))
            phase_events.append((cycle["S2"]["time"], "s2", cycle_num))

        # Sort by time
        phase_events.sort(key=lambda x: x[0])

        # Find which interval the time falls into
        for i in range(len(phase_events) - 1):
            current_time, current_phase, _ = phase_events[i]
            next_time, next_phase, _ = phase_events[i + 1]

            # Check if time falls in this interval
            if current_time <= time_s < next_time:
                # Return the phase code of the starting boundary
                # According to the rule: d2-s1 is "d2" phase, s1-s2 is "s1" phase, etc.
                return current_phase

        # Check if time is after the last phase event
        if phase_events and time_s >= phase_events[-1][0]:
            return phase_events[-1][1]

        return None

    def get_phase_transition_name(self, phase_code: str) -> str:
        """
        Get the human-readable name for a phase transition

        According to EKG processor logic:
        - D2→S1 interval is called "End-diastole"
        - S1→S2 interval is called "Early-systole"
        - S2→D1 interval is called "End-systole"
        - D1→D2 interval is called "Mid-diastole"
        """
        phase_transition_names = {
            "d2": "End-diastole",
            "s1": "Early-systole",
            "s2": "End-systole",
            "d1": "Mid-diastole",
        }
        return phase_transition_names.get(phase_code.lower(), "Unknown")

    def interpolate_phases_to_timestamps(
        self, phases: List[Dict], timestamps: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate phase information to specific timestamps
        Used for synchronizing with angiography frames

        Args:
            phases: Phase detection results
            timestamps: Array of timestamps to interpolate to

        Returns:
            Dictionary with interpolated phase information
        """
        if not phases:
            return {}

        # Extract phase times
        [p["D1"]["time"] for p in phases]
        [p["D2"]["time"] for p in phases]
        [p["S1"]["time"] for p in phases]
        [p["S2"]["time"] for p in phases]

        # Create phase value arrays (0-1 normalized within each cycle)
        result = {
            "phase_names": [],
            "systole_probability": np.zeros(len(timestamps)),
            "diastole_probability": np.zeros(len(timestamps)),
            "cycle_position": np.zeros(len(timestamps)),  # 0-1 within cardiac cycle
        }

        for i, t in enumerate(timestamps):
            phase_name = self.get_phase_at_time(phases, t)
            result["phase_names"].append(phase_name or "Unknown")

            # Find which cycle this timestamp belongs to
            for j, cycle in enumerate(phases):
                if (
                    cycle["D1"]["time"]
                    <= t
                    <= cycle["D1"]["time"] + (cycle["rr_interval_ms"] / 1000)
                ):
                    # Calculate position within cycle (0-1)
                    cycle_start = cycle["D1"]["time"]
                    cycle_duration = cycle["rr_interval_ms"] / 1000
                    result["cycle_position"][i] = (t - cycle_start) / cycle_duration

                    # Calculate systole/diastole probability
                    if t <= cycle["S1"]["time"]:
                        result["systole_probability"][i] = 1.0
                    else:
                        result["diastole_probability"][i] = 1.0
                    break

        return result

    def map_phases_to_frames(
        self, phases: List[Dict], total_frames: int, frame_rate: float, start_time: float = 0.0
    ) -> List[Dict]:
        """
        Map cardiac phases to video frames

        Args:
            phases: List of phase dictionaries from detect_phases
            total_frames: Total number of frames in the video
            frame_rate: Frame rate of the video (fps)
            start_time: Start time offset in seconds

        Returns:
            List of dicts with frame ranges for each phase transition
        """
        if not phases:
            return []

        1.0 / frame_rate
        frame_phases = []

        # Process each cardiac cycle
        for cycle in phases:
            beat_number = cycle.get("beat_number", None)

            # D2→S1: End-diastole
            d2_frame = int((cycle["D2"]["time"] - start_time) * frame_rate)
            s1_frame = int((cycle["S1"]["time"] - start_time) * frame_rate)

            if 0 <= d2_frame < total_frames and 0 <= s1_frame < total_frames:
                phase_info = {
                    "phase": "d2",
                    "phase_name": "End-diastole",
                    "frame_start": d2_frame,
                    "frame_end": s1_frame - 1,
                    "time_start": cycle["D2"]["time"],
                    "time_end": cycle["S1"]["time"],
                }
                if beat_number is not None:
                    phase_info["beat_number"] = beat_number
                frame_phases.append(phase_info)

            # S1→S2: Early-systole
            s2_frame = int((cycle["S2"]["time"] - start_time) * frame_rate)

            if 0 <= s1_frame < total_frames and 0 <= s2_frame < total_frames:
                phase_info = {
                    "phase": "s1",
                    "phase_name": "Early-systole",
                    "frame_start": s1_frame,
                    "frame_end": s2_frame - 1,
                    "time_start": cycle["S1"]["time"],
                    "time_end": cycle["S2"]["time"],
                }
                if beat_number is not None:
                    phase_info["beat_number"] = beat_number
                frame_phases.append(phase_info)

            # S2→D1: End-systole
            d1_frame = int((cycle["D1"]["time"] - start_time) * frame_rate)

            if 0 <= s2_frame < total_frames and 0 <= d1_frame < total_frames:
                phase_info = {
                    "phase": "s2",
                    "phase_name": "End-systole",
                    "frame_start": s2_frame,
                    "frame_end": d1_frame - 1,
                    "time_start": cycle["S2"]["time"],
                    "time_end": cycle["D1"]["time"],
                }
                if beat_number is not None:
                    phase_info["beat_number"] = beat_number
                frame_phases.append(phase_info)

            # D1→D2: Mid-diastole (until next cycle's D2 or end of data)
            # Find next cycle's D2
            next_d2_frame = total_frames  # Default to end
            if phases.index(cycle) < len(phases) - 1:
                next_cycle = phases[phases.index(cycle) + 1]
                next_d2_frame = int((next_cycle["D2"]["time"] - start_time) * frame_rate)

            if 0 <= d1_frame < total_frames:
                phase_info = {
                    "phase": "d1",
                    "phase_name": "Mid-diastole",
                    "frame_start": d1_frame,
                    "frame_end": min(next_d2_frame - 1, total_frames - 1),
                    "time_start": cycle["D1"]["time"],
                    "time_end": cycle["D1"]["time"] + (cycle["rr_interval_ms"] / 1000),
                }
                if beat_number is not None:
                    phase_info["beat_number"] = beat_number
                frame_phases.append(phase_info)

        return frame_phases
