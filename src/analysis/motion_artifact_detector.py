"""
Motion Artifact Detection for Angiography Frames
Implements automatic frame selection with minimal motion artifacts
"""

import numpy as np
import cv2
from typing import List, Dict
import logging
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class MotionArtifactDetector:
    """Detects motion artifacts in angiography frames and selects optimal frames"""

    def __init__(self):
        self.motion_threshold = 0.3  # Normalized motion score threshold

    def detect_motion_artifacts(
        self, frames: List[np.ndarray], cardiac_phases: Dict, frame_timestamps: np.ndarray
    ) -> Dict[str, int]:
        """
        Detect motion artifacts and select optimal frames for each cardiac phase

        Args:
            frames: List of grayscale frames
            cardiac_phases: Dictionary containing cardiac phase information
            frame_timestamps: Timestamps for each frame

        Returns:
            Dictionary mapping phase names to selected frame indices
        """
        if len(frames) < 2:
            logger.error("Need at least 2 frames for motion detection")
            return {}

        # Calculate motion scores for all frames
        motion_scores = self._calculate_motion_scores(frames)

        # Select frames for each cardiac phase
        selected_frames = {}
        phase_names = ["end-diastole", "early-systole", "end-systole", "mid-diastole"]

        # Try to get at least one frame from each phase across all beats
        for phase_name in phase_names:
            frame_indices = self._get_frames_for_phase(
                phase_name, cardiac_phases, frame_timestamps, len(frames)
            )

            if frame_indices:
                # Select frame with minimal motion artifact
                best_frame = self._select_best_frame(frame_indices, motion_scores)
                selected_frames[phase_name] = best_frame
                logger.info(
                    f"Selected frame {best_frame} for {phase_name} "
                    f"(motion score: {motion_scores[best_frame]:.3f})"
                )

        # If we don't have all 4 phases, try to get the best available frames
        if len(selected_frames) < 4:
            logger.warning(f"Only found {len(selected_frames)} phases, looking for alternatives...")

            # Get all phase transitions from frame_phases
            if "frame_phases" in cardiac_phases:
                phase_mapping = {
                    "d2": "end-diastole",
                    "s1": "early-systole",
                    "s2": "end-systole",
                    "d1": "mid-diastole",
                }

                for phase_info in cardiac_phases["frame_phases"]:
                    phase_code = phase_info["phase"]
                    phase_name = phase_mapping.get(phase_code)

                    if phase_name and phase_name not in selected_frames:
                        # Get frames in this phase range
                        start = phase_info["frame_start"]
                        end = min(phase_info["frame_end"] + 1, len(frames))
                        phase_frames = list(range(start, end))

                        if phase_frames:
                            best_frame = self._select_best_frame(phase_frames, motion_scores)
                            selected_frames[phase_name] = best_frame
                            logger.info(
                                f"Alternative frame {best_frame} for {phase_name} "
                                f"(motion score: {motion_scores[best_frame]:.3f})"
                            )

        return selected_frames

    def _calculate_motion_scores(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Calculate motion scores for each frame based on temporal differences

        Lower scores indicate less motion artifact
        """
        motion_scores = np.zeros(len(frames))

        for i in range(len(frames)):
            # Calculate motion relative to neighboring frames
            motion_components = []

            # Compare with previous frame
            if i > 0:
                diff_prev = self._calculate_frame_difference(frames[i - 1], frames[i])
                motion_components.append(diff_prev)

            # Compare with next frame
            if i < len(frames) - 1:
                diff_next = self._calculate_frame_difference(frames[i], frames[i + 1])
                motion_components.append(diff_next)

            # Average motion score
            if motion_components:
                motion_scores[i] = np.mean(motion_components)
            else:
                motion_scores[i] = 0.0

        # Normalize scores to 0-1 range
        if motion_scores.max() > 0:
            motion_scores = motion_scores / motion_scores.max()

        return motion_scores

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate motion between two frames using multiple metrics
        """
        # Ensure frames are same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Apply Gaussian blur to reduce noise
        frame1_blur = gaussian_filter(frame1.astype(np.float32), sigma=2.0)
        frame2_blur = gaussian_filter(frame2.astype(np.float32), sigma=2.0)

        # Calculate absolute difference
        diff = np.abs(frame1_blur - frame2_blur)

        # Focus on vessel regions (higher intensity areas)
        mask = frame1_blur > np.percentile(frame1_blur, 30)

        # Calculate motion metrics
        # 1. Mean absolute difference in vessel regions
        mad = np.mean(diff[mask]) if mask.any() else 0

        # 2. Gradient-based motion (edge movement)
        grad1_x = cv2.Sobel(frame1_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(frame1_blur, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(frame2_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(frame2_blur, cv2.CV_64F, 0, 1, ksize=3)

        grad_diff = np.sqrt((grad2_x - grad1_x) ** 2 + (grad2_y - grad1_y) ** 2)
        grad_motion = np.mean(grad_diff[mask]) if mask.any() else 0

        # 3. Structural similarity degradation
        # Higher motion causes lower structural similarity
        ssim_score = self._calculate_ssim_region(frame1_blur, frame2_blur, mask)
        ssim_motion = 1.0 - ssim_score

        # Combine metrics
        motion_score = 0.4 * mad + 0.4 * grad_motion + 0.2 * ssim_motion

        return motion_score

    def _calculate_ssim_region(
        self, frame1: np.ndarray, frame2: np.ndarray, mask: np.ndarray
    ) -> float:
        """Calculate structural similarity in masked region"""
        if not mask.any():
            return 1.0

        # Extract region of interest
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            return 1.0

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        roi1 = frame1[y_min : y_max + 1, x_min : x_max + 1]
        roi2 = frame2[y_min : y_max + 1, x_min : x_max + 1]

        # Calculate SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = np.mean(roi1)
        mu2 = np.mean(roi2)
        sigma1_sq = np.var(roi1)
        sigma2_sq = np.var(roi2)
        sigma12 = np.mean((roi1 - mu1) * (roi2 - mu2))

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return np.clip(ssim, 0, 1)

    def _get_frames_for_phase(
        self, phase_name: str, cardiac_phases: Dict, frame_timestamps: np.ndarray, total_frames: int
    ) -> List[int]:
        """Get frame indices corresponding to a specific cardiac phase"""
        frame_indices = []

        # Map phase names to phase codes
        phase_mapping = {
            "end-diastole": "d2",
            "early-systole": "s1",
            "end-systole": "s2",
            "mid-diastole": "d1",
        }

        phase_code = phase_mapping.get(phase_name)
        if not phase_code:
            return []

        # Check if we have frame_phases mapping
        if "frame_phases" in cardiac_phases:
            # Use pre-mapped frame phases
            for phase_info in cardiac_phases["frame_phases"]:
                if phase_info["phase"] == phase_code:
                    # Get all frames in this phase range
                    start = phase_info["frame_start"]
                    end = phase_info["frame_end"]
                    frame_indices.extend(range(start, min(end + 1, total_frames)))
        else:
            # Fallback to time-based mapping
            phases = cardiac_phases.get("phases", [])
            if not phases:
                return []

            # Find frames that fall within this phase across all beats
            for cycle in phases:
                phase_data = cycle.get(phase_code.upper(), {})
                if not phase_data:
                    continue

                phase_time = phase_data.get("time", 0)

                # Find closest frame to this phase time
                if len(frame_timestamps) > 0:
                    time_diffs = np.abs(frame_timestamps - phase_time)
                    closest_frame = np.argmin(time_diffs)

                    # Also include neighboring frames for selection
                    candidates = [closest_frame]
                    if closest_frame > 0:
                        candidates.append(closest_frame - 1)
                    if closest_frame < total_frames - 1:
                        candidates.append(closest_frame + 1)

                    frame_indices.extend(candidates)

        # Remove duplicates and sort
        frame_indices = sorted(list(set(frame_indices)))

        return frame_indices

    def _select_best_frame(self, frame_indices: List[int], motion_scores: np.ndarray) -> int:
        """Select frame with minimal motion artifact from candidates"""
        if not frame_indices:
            return 0

        # Get motion scores for candidate frames
        candidate_scores = [
            (idx, motion_scores[idx]) for idx in frame_indices if idx < len(motion_scores)
        ]

        if not candidate_scores:
            return frame_indices[0]

        # Sort by motion score (lower is better)
        candidate_scores.sort(key=lambda x: x[1])

        # Return frame with lowest motion score
        return candidate_scores[0][0]

    def analyze_motion_quality(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze overall motion quality of frame sequence

        Returns:
            Dictionary with quality metrics
        """
        if len(frames) < 2:
            return {"error": "Insufficient frames"}

        motion_scores = self._calculate_motion_scores(frames)

        quality_metrics = {
            "mean_motion": float(np.mean(motion_scores)),
            "std_motion": float(np.std(motion_scores)),
            "max_motion": float(np.max(motion_scores)),
            "min_motion": float(np.min(motion_scores)),
            "motion_range": float(np.max(motion_scores) - np.min(motion_scores)),
            "num_low_motion_frames": int(np.sum(motion_scores < self.motion_threshold)),
            "percent_low_motion": float(
                np.sum(motion_scores < self.motion_threshold) / len(frames) * 100
            ),
        }

        # Classify overall quality
        if quality_metrics["mean_motion"] < 0.2:
            quality_metrics["overall_quality"] = "Excellent"
        elif quality_metrics["mean_motion"] < 0.4:
            quality_metrics["overall_quality"] = "Good"
        elif quality_metrics["mean_motion"] < 0.6:
            quality_metrics["overall_quality"] = "Fair"
        else:
            quality_metrics["overall_quality"] = "Poor"

        return quality_metrics
