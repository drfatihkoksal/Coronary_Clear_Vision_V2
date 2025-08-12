"""
Frame Quality Analyzer for Enhanced RWS Analysis
Mathematical methods for temporal consistency and outlier detection
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class FrameQualityAnalyzer:
    """Analyzes frame quality using mathematical methods"""
    
    def __init__(self):
        self.mad_threshold = 3.5  # Modified Z-score threshold for outliers
        self.smoothness_weight = 0.5
        self.outlier_weight = 0.0  # DISABLED: Frame-level outlier detection not used
        self.motion_weight = 0.5  # Increased motion weight
        self.quality_weight = 0.5  # Quality-based selection weight
        
    def calculate_temporal_smoothness_score(self, diameters_sequence: List[float]) -> float:
        """
        Calculate temporal smoothness using Laplacian operator
        
        S(d) = Σ |d[i-1] - 2*d[i] + d[i+1]|²
        
        Lower score = smoother variation
        
        Args:
            diameters_sequence: Diameter values across time
            
        Returns:
            Smoothness score (lower is better)
        """
        n = len(diameters_sequence)
        if n < 3:
            return float('inf')
        
        smoothness_score = 0
        for i in range(1, n-1):
            # Second derivative (discrete Laplacian)
            second_derivative = (diameters_sequence[i-1] - 
                               2*diameters_sequence[i] + 
                               diameters_sequence[i+1])
            smoothness_score += second_derivative ** 2
        
        # Normalize by sequence length
        return smoothness_score / (n - 2)
    
    def detect_outlier_frames(self, qca_results: Dict[int, Dict], 
                            frame_indices: List[int]) -> Tuple[Dict[int, float], List[int]]:
        """
        Detect outlier frames using Extreme Value Theory (EVT) Peak-over-Threshold method
        EVT naturally preserves extreme minimum and maximum values as statistically valid
        
        Args:
            qca_results: QCA analysis results for each frame
            frame_indices: List of frame indices to analyze
            
        Returns:
            Tuple of (outlier scores dict, list of outlier frame indices)
        """
        # Collect MLD values from each frame (focus on MLD rather than all positions)
        mld_values = []
        frame_mld_map = {}
        
        for idx in frame_indices:
            if idx not in qca_results:
                continue
                
            # Get MLD value directly (more reliable than position-based analysis)
            mld = None
            if 'mld' in qca_results[idx] and qca_results[idx]['mld'] is not None:
                mld = float(qca_results[idx]['mld'])
            elif 'diameters_mm' in qca_results[idx]:
                # Use minimum diameter as MLD
                diameters = qca_results[idx]['diameters_mm']
                if diameters and len(diameters) > 0:
                    mld = float(min(diameters))
            elif 'diameters_pixels' in qca_results[idx]:
                # Convert to mm if calibration available
                if 'calibration_factor' in qca_results[idx]:
                    cal = qca_results[idx]['calibration_factor']
                    diameters = [d * cal for d in qca_results[idx]['diameters_pixels']]
                    if diameters and len(diameters) > 0:
                        mld = float(min(diameters))
            
            if mld is not None:
                mld_values.append(mld)
                frame_mld_map[idx] = mld
        
        if len(mld_values) < 3:
            logger.warning(f"Insufficient MLD values for outlier detection: {len(mld_values)} values found")
            logger.warning(f"Available MLD values: {list(frame_mld_map.values())}")
            return {idx: 0.0 for idx in frame_indices}, []
        
        logger.info(f"EVT MLD outlier detection starting with {len(mld_values)} values: {list(frame_mld_map.values())}")
        
        # Apply Extreme Value Theory (EVT) Peak-over-Threshold approach
        from scipy import stats
        
        mld_array = np.array(mld_values)
        min_mld = np.min(mld_array)
        max_mld = np.max(mld_array)
        
        logger.info(f"EVT Analysis: MLD range [{min_mld:.3f}mm, {max_mld:.3f}mm]")
        
        # Determine thresholds for upper and lower tails (EVT approach)
        # Use 90th percentile for upper tail, 10th percentile for lower tail
        upper_threshold = np.percentile(mld_array, 90)
        lower_threshold = np.percentile(mld_array, 10)
        
        logger.info(f"EVT Thresholds: Lower={lower_threshold:.3f}mm, Upper={upper_threshold:.3f}mm")
        
        # Extract excesses over thresholds
        upper_excesses = mld_array[mld_array > upper_threshold] - upper_threshold
        lower_excesses = lower_threshold - mld_array[mld_array < lower_threshold]
        
        # EVT: Fit Generalized Pareto Distribution to excesses
        upper_outliers = set()
        lower_outliers = set()
        
        if len(upper_excesses) >= 3:
            try:
                # Fit GPD to upper excesses
                shape_upper, loc_upper, scale_upper = stats.genpareto.fit(upper_excesses, floc=0)
                logger.info(f"EVT Upper GPD: shape={shape_upper:.3f}, scale={scale_upper:.3f}")
                
                # Calculate p-values for upper tail values
                for mld_val in mld_array[mld_array > upper_threshold]:
                    excess = mld_val - upper_threshold
                    p_value = 1 - stats.genpareto.cdf(excess, shape_upper, loc=0, scale=scale_upper)
                    # Conservative p-value threshold for EVT (0.01 = very extreme)
                    if p_value < 0.01:
                        upper_outliers.add(mld_val)
                        logger.info(f"EVT Upper outlier: {mld_val:.3f}mm (p={p_value:.4f})")
                        
            except Exception as e:
                logger.warning(f"EVT Upper tail fitting failed: {e}")
        
        if len(lower_excesses) >= 3:
            try:
                # Fit GPD to lower excesses 
                shape_lower, loc_lower, scale_lower = stats.genpareto.fit(lower_excesses, floc=0)
                logger.info(f"EVT Lower GPD: shape={shape_lower:.3f}, scale={scale_lower:.3f}")
                
                # Calculate p-values for lower tail values
                for mld_val in mld_array[mld_array < lower_threshold]:
                    excess = lower_threshold - mld_val
                    p_value = 1 - stats.genpareto.cdf(excess, shape_lower, loc=0, scale=scale_lower)
                    # Conservative p-value threshold for EVT (0.01 = very extreme)
                    if p_value < 0.01:
                        lower_outliers.add(mld_val)
                        logger.info(f"EVT Lower outlier: {mld_val:.3f}mm (p={p_value:.4f})")
                        
            except Exception as e:
                logger.warning(f"EVT Lower tail fitting failed: {e}")
        
        # EVT PRINCIPLE: Extreme minimum and maximum are NEVER outliers by definition
        # They represent the true extreme behavior of the distribution
        if min_mld in upper_outliers or min_mld in lower_outliers:
            logger.info(f"EVT: Minimum MLD {min_mld:.3f}mm protected (true extreme)")
        if max_mld in upper_outliers or max_mld in lower_outliers:
            logger.info(f"EVT: Maximum MLD {max_mld:.3f}mm protected (true extreme)")
            
        upper_outliers.discard(min_mld)
        upper_outliers.discard(max_mld)
        lower_outliers.discard(min_mld)
        lower_outliers.discard(max_mld)
        
        evt_outlier_values = upper_outliers | lower_outliers
        logger.info(f"EVT detected {len(evt_outlier_values)} outlier MLD values: {[f'{v:.3f}mm' for v in evt_outlier_values]}")
        
        # Calculate EVT-based outlier scores for each frame
        frame_outlier_scores = {}
        outlier_frames = []
        tolerance = 0.01  # 0.01mm tolerance for matching outlier values
        
        for idx in frame_indices:
            if idx not in frame_mld_map:
                frame_outlier_scores[idx] = 0.0
                continue
                
            mld = frame_mld_map[idx]
            
            # Check if this MLD value is detected as EVT outlier
            is_evt_outlier = any(abs(mld - outlier_val) <= tolerance for outlier_val in evt_outlier_values)
            
            # EVT PRINCIPLE: Minimum and maximum are NEVER outliers
            is_true_extreme = (abs(mld - min_mld) <= tolerance) or (abs(mld - max_mld) <= tolerance)
            
            logger.info(f"🔍 EVT Frame {idx} - MLD {mld:.3f}mm:")
            logger.info(f"🔍 EVT:   Is EVT outlier: {is_evt_outlier}")
            logger.info(f"🔍 EVT:   Is true extreme (min/max): {is_true_extreme}")
            
            if is_true_extreme:
                # True extremes (min/max) are NEVER outliers by EVT definition
                frame_outlier_scores[idx] = 0.0
                logger.info(f"🔍 EVT:   ✅ TRUE EXTREME PROTECTED - Score: 0.0")
            elif is_evt_outlier:
                # EVT detected this as statistical outlier
                frame_outlier_scores[idx] = 10.0  # High outlier score
                outlier_frames.append(idx)
                logger.info(f"🔍 EVT:   ❌ EVT OUTLIER DETECTED - Score: 10.0")
            else:
                # Normal value according to EVT
                frame_outlier_scores[idx] = 0.0
                logger.info(f"🔍 EVT:   ✅ NORMAL VALUE - Score: 0.0")
        
        logger.info(f"EVT Frame outlier analysis summary:")
        logger.info(f"  Total frames analyzed: {len(frame_indices)}")
        logger.info(f"  Frames with valid MLD: {len(frame_mld_map)}")
        logger.info(f"  EVT outlier values detected: {len(evt_outlier_values)} unique values")
        logger.info(f"  EVT outlier frames: {len(outlier_frames)}")
        logger.info(f"  True extremes protected: min={min_mld:.3f}mm, max={max_mld:.3f}mm")
        
        if outlier_frames:
            logger.info(f"  Outlier frame details:")
            for frame_idx in outlier_frames:
                mld_val = frame_mld_map.get(frame_idx, 'N/A')
                score = frame_outlier_scores.get(frame_idx, 0)
                logger.info(f"    Frame {frame_idx}: MLD={mld_val:.3f}mm, Score={score:.2f}")
        else:
            logger.info(f"  No EVT outlier frames detected - all MLD values within EVT acceptable range")
        
        return frame_outlier_scores, outlier_frames
    
    def evaluate_centerline_quality(self, centerline: np.ndarray) -> Dict[str, float]:
        """
        Evaluate centerline quality using curvature and smoothness metrics
        
        Args:
            centerline: Array of centerline points (N x 2)
            
        Returns:
            Dictionary with quality metrics
        """
        if len(centerline) < 3:
            return {
                'max_curvature': float('inf'),
                'edge_anomaly_score': float('inf'),
                'smoothness': 0.0
            }
        
        # Calculate curvature using discrete derivatives
        # First derivative (tangent)
        dx = np.gradient(centerline[:, 0])
        dy = np.gradient(centerline[:, 1])
        
        # Second derivative
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5)
        
        # Avoid division by zero
        curvature = np.zeros_like(numerator)
        valid_idx = denominator > 1e-6
        curvature[valid_idx] = numerator[valid_idx] / denominator[valid_idx]
        
        # Calculate angle changes between consecutive segments
        angles = []
        for i in range(1, len(centerline) - 1):
            v1 = centerline[i] - centerline[i-1]
            v2 = centerline[i+1] - centerline[i]
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate angle (clip to handle numerical errors)
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return {
                'max_curvature': float('inf'),
                'edge_anomaly_score': float('inf'),
                'smoothness': 0.0
            }
        
        angles = np.array(angles)
        
        # Edge anomaly score: focus on first and last 10% of centerline
        edge_size = max(1, len(angles) // 10)
        edge_angles = np.concatenate([angles[:edge_size], angles[-edge_size:]])
        edge_anomaly_score = np.mean(edge_angles) if len(edge_angles) > 0 else 0
        
        # Overall smoothness (inverse of angle variation)
        smoothness = 1.0 / (1.0 + np.std(angles))
        
        return {
            'max_curvature': float(np.max(curvature)),
            'edge_anomaly_score': float(edge_anomaly_score),
            'smoothness': float(smoothness),
            'mean_curvature': float(np.mean(curvature)),
            'curvature_std': float(np.std(curvature))
        }
    
    def check_diameter_consistency(self, diameter_profile: np.ndarray) -> Dict[str, float]:
        """
        Check diameter profile consistency, especially at edges
        
        Args:
            diameter_profile: Array of diameter values along vessel
            
        Returns:
            Dictionary with consistency metrics
        """
        if len(diameter_profile) < 10:
            return {
                'edge_consistency': 1.0,
                'profile_smoothness': 0.0,
                'is_consistent': True
            }
        
        # Edge vs middle comparison
        edge_size = max(2, len(diameter_profile) // 10)
        edge_avg = (np.mean(diameter_profile[:edge_size]) + 
                   np.mean(diameter_profile[-edge_size:])) / 2
        
        middle_start = len(diameter_profile) // 4
        middle_end = 3 * len(diameter_profile) // 4
        middle_avg = np.mean(diameter_profile[middle_start:middle_end])
        
        # Consistency ratio (should be close to 1.0)
        consistency_ratio = edge_avg / middle_avg if middle_avg > 0 else 0
        
        # Profile smoothness using total variation
        total_variation = np.sum(np.abs(np.diff(diameter_profile)))
        profile_smoothness = 1.0 / (1.0 + total_variation / len(diameter_profile))
        
        # Check if consistent (within 30% variation)
        is_consistent = 0.7 <= consistency_ratio <= 1.3
        
        return {
            'edge_consistency': float(consistency_ratio),
            'profile_smoothness': float(profile_smoothness),
            'is_consistent': bool(is_consistent),
            'edge_avg': float(edge_avg),
            'middle_avg': float(middle_avg)
        }
    
    def select_optimal_frames(self, qca_results: Dict[int, Dict],
                            cardiac_phases: Dict,
                            motion_scores: Dict[int, float],
                            target_phases: List[str] = None) -> Dict[str, int]:
        """
        Select optimal frames using combined mathematical criteria
        
        Args:
            qca_results: QCA analysis results
            cardiac_phases: Cardiac phase information
            motion_scores: Motion artifact scores for each frame
            target_phases: Target cardiac phases (default: S1, S2, D1)
            
        Returns:
            Dictionary mapping phase names to selected frame indices
        """
        if target_phases is None:
            # Exclude end-diastole (D2) as per paper
            target_phases = ['early-systole', 'end-systole', 'mid-diastole']
        
        # Phase code mapping
        phase_mapping = {
            'early-systole': 's1',
            'end-systole': 's2',
            'mid-diastole': 'd1'
        }
        
        # Collect candidate frames for each phase
        phase_candidates = {phase: [] for phase in target_phases}
        
        if 'frame_phases' in cardiac_phases:
            for phase_info in cardiac_phases['frame_phases']:
                phase_code = phase_info['phase']
                phase_name = None
                
                # Find corresponding phase name
                for name, code in phase_mapping.items():
                    if code == phase_code and name in target_phases:
                        phase_name = name
                        break
                
                if phase_name:
                    frames = list(range(phase_info['frame_start'], 
                                      phase_info['frame_end'] + 1))
                    phase_candidates[phase_name].extend(frames)
        
        # Get all candidate frames
        all_frames = []
        for frames in phase_candidates.values():
            all_frames.extend(frames)
        all_frames = list(set(all_frames))  # Remove duplicates
        
        if not all_frames:
            logger.warning("No candidate frames found from cardiac phases")
            return {}
        
        # Calculate outlier scores for information, but don't use them to filter frames
        logger.info(f"Frame outlier detection: INFORMATIONAL ONLY - frames not filtered by outlier status")
        outlier_scores, outlier_frames = self.detect_outlier_frames(qca_results, all_frames)
        
        # Evaluate each frame's quality
        frame_quality_scores = {}
        
        for frame_idx in all_frames:
            if frame_idx not in qca_results:
                continue
            
            # Initialize score components
            scores = {
                'outlier': outlier_scores.get(frame_idx, 0.0),
                'motion': motion_scores.get(frame_idx, 1.0),
                'centerline': 0.0,
                'diameter_consistency': 0.0
            }
            
            # Evaluate centerline quality if available
            if 'centerline' in qca_results[frame_idx]:
                centerline = np.array(qca_results[frame_idx]['centerline'])
                cl_quality = self.evaluate_centerline_quality(centerline)
                # Convert to penalty score (lower is better)
                scores['centerline'] = (1.0 - cl_quality['smoothness'] + 
                                      cl_quality['edge_anomaly_score'])
            
            # Check diameter consistency
            diameters = None
            if 'diameters_mm' in qca_results[frame_idx]:
                diameters = np.array(qca_results[frame_idx]['diameters_mm'])
            elif 'diameters_pixels' in qca_results[frame_idx]:
                diameters = np.array(qca_results[frame_idx]['diameters_pixels'])
            
            if diameters is not None and len(diameters) > 0:
                consistency = self.check_diameter_consistency(diameters)
                scores['diameter_consistency'] = 0.0 if consistency['is_consistent'] else 1.0
            
            # Calculate combined score (lower is better) - NO OUTLIER WEIGHT
            # Only use motion, centerline smoothness, and diameter consistency
            remaining_weight = 1.0 - self.motion_weight - self.smoothness_weight
            combined_score = (self.motion_weight * scores['motion'] +
                            self.smoothness_weight * scores['centerline'] +
                            remaining_weight * scores['diameter_consistency'])
            
            frame_quality_scores[frame_idx] = {
                'combined_score': combined_score,
                'components': scores,
                'is_outlier': frame_idx in outlier_frames  # Outlier info kept for logging
            }
        
        # Collect MLD values for variance optimization
        mld_values_by_frame = {}
        for frame_idx in all_frames:
            if frame_idx in qca_results and 'mld' in qca_results[frame_idx]:
                mld_values_by_frame[frame_idx] = qca_results[frame_idx]['mld']
        
        # FIRST: Find global extremes across ALL frames (not just per phase)
        global_min_mld = float('inf')
        global_max_mld = float('-inf')
        global_min_frame = None
        global_max_frame = None
        
        for frame_idx, mld_value in mld_values_by_frame.items():
            if mld_value < global_min_mld:
                global_min_mld = mld_value
                global_min_frame = frame_idx
            if mld_value > global_max_mld:
                global_max_mld = mld_value
                global_max_frame = frame_idx
        
        logger.info(f"🎯 GLOBAL EXTREMES across all frames:")
        logger.info(f"  MIN: Frame {global_min_frame} = {global_min_mld:.3f}mm")
        logger.info(f"  MAX: Frame {global_max_frame} = {global_max_mld:.3f}mm")
        
        # Select best frame for each phase
        selected_frames = {}
        
        for phase_name, candidates in phase_candidates.items():
            # IMPORTANT: Include ALL frames regardless of outlier status
            # Outlier detection is informational only - frames are NOT filtered
            # MLD extreme values and quality metrics determine selection
            valid_candidates = [f for f in candidates if f in frame_quality_scores]
            
            logger.info(f"Phase {phase_name}: {len(candidates)} candidate frames, "
                       f"{len([f for f in candidates if f in outlier_frames])} marked as outliers "
                       f"(but all candidates kept for selection)")
            
            # If no candidates with quality scores, use all candidates
            if not valid_candidates:
                valid_candidates = candidates
                logger.warning(f"No quality scores available for {phase_name} candidates: {candidates}")
            
            # Enhanced selection: Use GLOBAL extremes if they're in this phase
            if valid_candidates and mld_values_by_frame:
                # Check if global extreme is in this phase
                use_global_extreme = False
                
                if 'systole' in phase_name.lower() and global_min_frame in valid_candidates:
                    best_frame = global_min_frame
                    use_global_extreme = True
                    logger.info(f"✅ Using GLOBAL MIN for {phase_name}: Frame {best_frame} ({global_min_mld:.3f}mm)")
                elif 'diastole' in phase_name.lower() and global_max_frame in valid_candidates:
                    best_frame = global_max_frame
                    use_global_extreme = True
                    logger.info(f"✅ Using GLOBAL MAX for {phase_name}: Frame {best_frame} ({global_max_mld:.3f}mm)")
                
                if not use_global_extreme:
                    # Fallback to phase-specific extremes
                    candidate_mlds = {f: mld_values_by_frame.get(f, 0) for f in valid_candidates 
                                     if f in mld_values_by_frame}
                
                    if candidate_mlds:
                        # PRIORITY: Use extreme values if they appear ≥2 times (physiological)
                        # Count MLD value frequencies 
                        mld_frequency = {}
                        for frame, mld_val in candidate_mlds.items():
                            # Find how many times this MLD appears (with tolerance)
                            count = sum(1 for other_mld in candidate_mlds.values() 
                                      if abs(mld_val - other_mld) <= 0.1)  # 0.1mm tolerance
                            mld_frequency[frame] = count
                        
                        # PRIORITY: Always select the TRUE EXTREME value (min for systole, max for diastole)
                        # If extreme appears ≥2 times: perfect (physiological)
                        # If extreme appears once: still select it (it's the most extreme)
                        
                        if 'systole' in phase_name.lower():
                            # For systole: Find ABSOLUTE minimum MLD
                            sorted_by_mld = sorted(candidate_mlds.items(), key=lambda x: x[1])
                            min_mld_value = sorted_by_mld[0][1]
                            min_mld_frames = [f for f, mld in candidate_mlds.items() 
                                         if abs(mld - min_mld_value) <= 0.01]  # Tighter tolerance
                        
                        # ALWAYS use the absolute minimum, regardless of frequency
                        best_frame = min(min_mld_frames,
                                       key=lambda f: frame_quality_scores.get(f, {}).get('combined_score', float('inf')))
                        
                        freq_info = f"({len(min_mld_frames)} frames)" if len(min_mld_frames) > 1 else "(single occurrence)"
                        logger.info(f"🔍 TRUE EXTREME MIN selected for {phase_name}: MLD={min_mld_value:.3f}mm {freq_info}")
                        logger.info(f"  Verifying: Frame {best_frame} actual MLD = {mld_values_by_frame.get(best_frame, 'N/A')}mm")
                        
                        # Verification: Make sure we got the right frame
                        if abs(mld_values_by_frame.get(best_frame, 0) - min_mld_value) > 0.01:
                            logger.warning(f"  ⚠️ MLD mismatch! Expected {min_mld_value:.3f}mm but got {mld_values_by_frame.get(best_frame, 0):.3f}mm")
                            # Find the correct frame
                            for f in valid_candidates:
                                if abs(mld_values_by_frame.get(f, 0) - min_mld_value) <= 0.01:
                                    best_frame = f
                                    logger.info(f"  ✅ Corrected to frame {best_frame} with MLD {mld_values_by_frame.get(f, 0):.3f}mm")
                                    break
                        
                    else:  # diastole
                        # For diastole: Find ABSOLUTE maximum MLD
                        sorted_by_mld = sorted(candidate_mlds.items(), key=lambda x: x[1], reverse=True)
                        max_mld_value = sorted_by_mld[0][1]
                        max_mld_frames = [f for f, mld in candidate_mlds.items() 
                                         if abs(mld - max_mld_value) <= 0.01]  # Tighter tolerance
                        
                        # ALWAYS use the absolute maximum, regardless of frequency
                        best_frame = min(max_mld_frames,
                                       key=lambda f: frame_quality_scores.get(f, {}).get('combined_score', float('inf')))
                        
                        freq_info = f"({len(max_mld_frames)} frames)" if len(max_mld_frames) > 1 else "(single occurrence)"
                        logger.info(f"🔍 TRUE EXTREME MAX selected for {phase_name}: MLD={max_mld_value:.3f}mm {freq_info}")
                        logger.info(f"  Verifying: Frame {best_frame} actual MLD = {mld_values_by_frame.get(best_frame, 'N/A')}mm")
                        
                        # Verification: Make sure we got the right frame
                        if abs(mld_values_by_frame.get(best_frame, 0) - max_mld_value) > 0.01:
                            logger.warning(f"  ⚠️ MLD mismatch! Expected {max_mld_value:.3f}mm but got {mld_values_by_frame.get(best_frame, 0):.3f}mm")
                            # Find the correct frame
                            for f in valid_candidates:
                                if abs(mld_values_by_frame.get(f, 0) - max_mld_value) <= 0.01:
                                    best_frame = f
                                    logger.info(f"  ✅ Corrected to frame {best_frame} with MLD {mld_values_by_frame.get(f, 0):.3f}mm")
                                    break
                    
                    logger.info(f"Selected frame {best_frame} for {phase_name} "
                              f"(MLD: {mld_values_by_frame.get(best_frame, 'N/A')}mm, "
                              f"score: {frame_quality_scores[best_frame]['combined_score']:.3f})")
                else:
                    # Fallback to quality-only selection
                    best_frame = min(valid_candidates,
                                   key=lambda f: frame_quality_scores.get(f, {}).get('combined_score', float('inf')))
                    logger.info(f"Selected frame {best_frame} for {phase_name} "
                              f"(score: {frame_quality_scores[best_frame]['combined_score']:.3f})")
            elif valid_candidates:
                # Select frame with lowest combined score
                best_frame = min(valid_candidates,
                               key=lambda f: frame_quality_scores.get(f, {}).get('combined_score', float('inf')))
                logger.info(f"Selected frame {best_frame} for {phase_name} "
                          f"(score: {frame_quality_scores[best_frame]['combined_score']:.3f})")
            else:
                continue
                
            selected_frames[phase_name] = best_frame
        
        return selected_frames