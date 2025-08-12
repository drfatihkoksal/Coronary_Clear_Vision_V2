"""
RWS Calculator - Multi-Phase Analysis
Calculates RWS from different cardiac phases with correct formula
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Literal
from .models import MLDInfo, RWSResult
from ...config.app_config import get_config

logger = logging.getLogger(__name__)

# Get configuration
config = get_config()


class RWSCalculator:
    """Calculate RWS from multi-phase QCA MLD values"""
    
    @staticmethod
    def calculate_multi_phase_rws(beat_frames: List[int], 
                                qca_results: Dict[int, Dict],
                                selected_frames: Optional[Dict[str, int]] = None) -> Optional[RWSResult]:
        """
        Calculate RWS from cardiac beat frames with outlier removal
        Removes 2 highest and 2 lowest MLD values from entire beat before calculation
        
        Args:
            beat_frames: List of frame indices for entire cardiac beat
            qca_results: QCA analysis results for each frame  
            selected_frames: Optional phase mapping for result tracking
            
        Returns:
            RWSResult or None if calculation fails
        """
        # Input validation
        if not beat_frames:
            logger.error("No beat frames provided for RWS calculation")
            return None
            
        if not qca_results:
            logger.error("No QCA results provided for RWS calculation")
            return None
        
        # Check minimum frame requirement
        if len(beat_frames) < config.rws.min_frames_for_analysis:  # Absolute minimum for meaningful RWS
            logger.error(f"Need at least {config.rws.min_frames_for_analysis} frames for RWS calculation, got {len(beat_frames)}")
            return None
        
        # Collect all MLD values from beat frames
        mld_data = []
        for frame_idx in beat_frames:
            if frame_idx not in qca_results:
                logger.debug(f"No QCA results for frame {frame_idx}")
                continue
                
            qca_data = qca_results[frame_idx]
            mld = qca_data.get('mld')
            
            if mld is None:
                logger.debug(f"No MLD value in frame {frame_idx}")
                continue
                
            mld_value = float(mld)
            
            # Validate MLD range
            if not (config.rws.min_mld_mm <= mld_value <= config.rws.max_mld_mm):
                logger.warning(f"MLD {mld_value:.3f}mm in frame {frame_idx} is outside valid range "
                             f"({config.rws.min_mld_mm}-{config.rws.max_mld_mm}mm)")
                continue
            
            mld_data.append({
                'frame': frame_idx,
                'mld': mld_value,
                'mld_index': qca_data.get('mld_index')
            })
            
            logger.info(f"MLD Frame {frame_idx+1}: {mld_value:.3f}mm (index: {qca_data.get('mld_index')})")
        
        # Flexible outlier removal based on data size
        if len(mld_data) < config.rws.min_frames_for_analysis:
            logger.error(f"Insufficient valid MLD values for calculation: {len(mld_data)}")
            return None
        
        # Apply IQR outlier removal
        if len(mld_data) >= 6:  # Need enough frames for IQR method (reduced threshold for less conservative filtering)
            filtered_data = RWSCalculator._remove_mld_outliers(mld_data)
        else:
            logger.info(f"Using all {len(mld_data)} frames without outlier removal (too few frames for IQR method)")
            filtered_data = mld_data
        
        if len(filtered_data) < 2:
            logger.error(f"Insufficient data after outlier removal: {len(filtered_data)}")
            return None
        
        # Find min and max MLD from filtered data
        min_data = min(filtered_data, key=lambda x: x['mld'])
        max_data = max(filtered_data, key=lambda x: x['mld'])
        
        min_mld = min_data['mld']
        max_mld = max_data['mld']
        
        # Validate MLD values before calculation
        if min_mld <= 0:
            logger.error(f"Invalid minimum MLD value: {min_mld:.3f}mm (must be > 0)")
            return None
            
        if max_mld <= 0:
            logger.error(f"Invalid maximum MLD value: {max_mld:.3f}mm (must be > 0)")
            return None
            
        if max_mld <= min_mld:
            logger.error(f"Max MLD ({max_mld:.3f}mm) must be greater than min MLD ({min_mld:.3f}mm)")
            return None
        
        # Calculate RWS with correct formula: (max - min) / min × 100%
        logger.info(f"=== RWS FORMULA CALCULATION ===")
        logger.info(f"Formula: RWS = (max_MLD - min_MLD) / min_MLD × 100%")
        logger.info(f"Values: max_MLD = {max_mld:.3f}mm, min_MLD = {min_mld:.3f}mm")
        
        try:
            numerator = max_mld - min_mld
            rws_percentage = (numerator / min_mld) * 100
            
            # Validate result is not NaN or Inf
            if not np.isfinite(rws_percentage):
                logger.error(f"RWS calculation resulted in invalid value: {rws_percentage}")
                return None
                
            logger.info(f"Step 1: (max - min) = ({max_mld:.3f} - {min_mld:.3f}) = {numerator:.3f}mm")
            logger.info(f"Step 2: {numerator:.3f} / {min_mld:.3f} = {numerator/min_mld:.3f}")
            logger.info(f"Step 3: {numerator/min_mld:.3f} × 100% = {rws_percentage:.1f}%")
            
        except (ZeroDivisionError, ValueError) as e:
            logger.error(f"Error calculating RWS: {e}")
            return None
        
        # Try to identify phases if provided
        min_phase = RWSCalculator._identify_phase(min_data['frame'], selected_frames)
        max_phase = RWSCalculator._identify_phase(max_data['frame'], selected_frames)
        
        logger.info(f"=== RWS RESULT SUMMARY ===")
        logger.info(f"  Min MLD: {min_mld:.3f}mm (Frame {min_data['frame']+1}, {min_phase or 'Unknown phase'})")
        logger.info(f"  Max MLD: {max_mld:.3f}mm (Frame {max_data['frame']+1}, {max_phase or 'Unknown phase'})")
        logger.info(f"  FINAL RWS: {rws_percentage}%")
        logger.info(f"================================")
        
        return RWSResult(
            rws_percentage=rws_percentage,
            min_mld=min_mld,
            max_mld=max_mld,
            min_mld_frame=min_data['frame'],
            max_mld_frame=max_data['frame'],
            min_mld_index=min_data['mld_index'],
            max_mld_index=max_data['mld_index'],
            min_phase=min_phase,
            max_phase=max_phase
        )
    
    @staticmethod
    def _remove_mld_outliers(mld_data: List[Dict]) -> List[Dict]:
        """
        Hampel Filter for outlier detection in MLD values
        Optimized for cardiac cycle data with true min/max protection
        
        Args:
            mld_data: List of frame data with MLD values
            
        Returns:
            Filtered list with outliers removed using Hampel filter
        """
        # Need at least 5 frames for meaningful Hampel filter
        if len(mld_data) < 5:
            logger.info(f"Too few frames ({len(mld_data)}) for Hampel filter - no outlier removal")
            return mld_data
        
        # Extract MLD values
        mld_values = np.array([d['mld'] for d in mld_data])
        n = len(mld_values)
        
        # Dynamic window size based on number of frames
        # Window should be odd, typically 5-7 for cardiac data
        window_size = max(3, min(7, n // 4))
        if window_size % 2 == 0:  # Ensure odd window
            window_size += 1
            
        # Hampel filter parameters
        n_sigma = 3.0  # Number of standard deviations (3 is standard)
        k = 1.4826  # Scale factor for MAD to estimate standard deviation
        
        # Initialize outlier flags
        outliers = np.zeros(n, dtype=bool)
        
        # Apply Hampel filter
        for i in range(n):
            # Define window boundaries
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # Extract window values
            window = mld_values[start:end]
            
            # Calculate local median and MAD
            local_median = np.median(window)
            local_mad = np.median(np.abs(window - local_median))
            
            # Avoid division by zero for constant regions
            if local_mad < 1e-6:
                local_mad = 1e-6
            
            # Hampel identifier: check if point deviates too much from local median
            deviation = np.abs(mld_values[i] - local_median)
            threshold = n_sigma * k * local_mad
            
            if deviation > threshold:
                outliers[i] = True
        
        # CRITICAL: Always protect true minimum and maximum values
        # These represent actual physiological extremes, not outliers
        min_idx = np.argmin(mld_values)
        max_idx = np.argmax(mld_values)
        outliers[min_idx] = False
        outliers[max_idx] = False
        
        # Build filtered data
        filtered_data = []
        removed_outliers = []
        
        for i, (data, is_outlier) in enumerate(zip(mld_data, outliers)):
            if is_outlier:
                removed_outliers.append(data)
            else:
                filtered_data.append(data)
        
        # Logging
        if removed_outliers:
            logger.info(f"Hampel filter outlier detection:")
            logger.info(f"  Window size: {window_size}, Sigma threshold: {n_sigma}")
            logger.info(f"  Protected extremes: min={mld_values[min_idx]:.3f}mm (frame {mld_data[min_idx]['frame']+1}), "
                       f"max={mld_values[max_idx]:.3f}mm (frame {mld_data[max_idx]['frame']+1})")
            outlier_str = [f"Frame {d['frame']+1}({d['mld']:.3f}mm)" for d in removed_outliers]
            logger.info(f"  Removed {len(removed_outliers)} outliers: {outlier_str}")
        else:
            logger.info(f"Hampel filter: No outliers detected in {len(mld_data)} frames")
            logger.info(f"  MLD range: [{np.min(mld_values):.3f}, {np.max(mld_values):.3f}]mm")
        
        # Ensure we have enough data after filtering
        if len(filtered_data) < 2:
            logger.warning(f"Too few points after Hampel filter ({len(filtered_data)}), using original data")
            return mld_data
        
        return filtered_data
    
    @staticmethod
    def _identify_phase(frame_idx: int, selected_frames: Optional[Dict[str, int]]) -> Optional[str]:
        """
        Identify which cardiac phase a frame belongs to
        
        Args:
            frame_idx: Frame index to identify
            selected_frames: Dict mapping phase names to frame indices
            
        Returns:
            Phase name or None
        """
        if not selected_frames:
            return None
            
        for phase_name, phase_frame in selected_frames.items():
            if phase_frame == frame_idx:
                return phase_name
                
        return None
    
    @staticmethod
    def validate_result(rws_result: RWSResult) -> bool:
        """
        Validate RWS calculation result
        
        Args:
            rws_result: RWS calculation result
            
        Returns:
            True if results are valid
        """
        # Check RWS percentage range
        if not 0 <= rws_result.rws_percentage <= config.rws.max_rws_percentage:
            logger.error(f"Invalid RWS percentage: {rws_result.rws_percentage}% "
                        f"(max: {config.rws.max_rws_percentage}%)")
            return False
        
        # Validate MLD ranges
        if not (config.rws.min_mld_mm <= rws_result.min_mld <= config.rws.max_mld_mm):
            logger.error(f"Invalid min MLD: {rws_result.min_mld:.3f}mm")
            return False
        
        if not (config.rws.min_mld_mm <= rws_result.max_mld <= config.rws.max_mld_mm):
            logger.error(f"Invalid max MLD: {rws_result.max_mld:.3f}mm")
            return False
        
        # Check max > min
        if rws_result.max_mld <= rws_result.min_mld:
            logger.error("Max MLD must be greater than min MLD")
            return False
        
        return True
    
    @staticmethod
    def calculate_multi_beat_average_rws(beats_data: List[Dict], 
                                        qca_results: Dict[int, Dict],
                                        use_median: bool = True) -> Optional[RWSResult]:
        """
        Calculate average RWS across multiple cardiac beats
        
        Args:
            beats_data: List of dicts containing beat_frames for each beat
            qca_results: QCA analysis results for all frames
            use_median: Use median instead of mean for robustness
            
        Returns:
            RWSResult with averaged values or None
        """
        if not beats_data:
            logger.error("No beat data provided for multi-beat analysis")
            return None
        
        # Calculate RWS for each beat
        beat_results = []
        for beat_info in beats_data:
            beat_frames = beat_info.get('frames', [])
            if not beat_frames:
                continue
                
            result = RWSCalculator.calculate_multi_phase_rws(beat_frames, qca_results)
            if result:
                beat_results.append(result)
        
        if not beat_results:
            logger.error("No valid RWS results from any beat")
            return None
        
        # Average the results
        if use_median:
            avg_rws = np.median([r.rws_percentage for r in beat_results])
            avg_min_mld = np.median([r.min_mld for r in beat_results])
            avg_max_mld = np.median([r.max_mld for r in beat_results])
        else:
            avg_rws = np.mean([r.rws_percentage for r in beat_results])
            avg_min_mld = np.mean([r.min_mld for r in beat_results])
            avg_max_mld = np.mean([r.max_mld for r in beat_results])
        
        # Use most common frames/phases
        from collections import Counter
        min_frames = [r.min_mld_frame for r in beat_results]
        max_frames = [r.max_mld_frame for r in beat_results]
        min_phases = [r.min_phase for r in beat_results if r.min_phase]
        max_phases = [r.max_phase for r in beat_results if r.max_phase]
        
        most_common_min_frame = Counter(min_frames).most_common(1)[0][0] if min_frames else 0
        most_common_max_frame = Counter(max_frames).most_common(1)[0][0] if max_frames else 0
        most_common_min_phase = Counter(min_phases).most_common(1)[0][0] if min_phases else None
        most_common_max_phase = Counter(max_phases).most_common(1)[0][0] if max_phases else None
        
        logger.info(f"Multi-beat average RWS: {avg_rws:.1f}% from {len(beat_results)} beats")
        logger.info(f"  Individual beat RWS values: {[r.rws_percentage for r in beat_results]}")
        
        return RWSResult(
            rws_percentage=avg_rws,
            min_mld=avg_min_mld,
            max_mld=avg_max_mld,
            min_mld_frame=most_common_min_frame,
            max_mld_frame=most_common_max_frame,
            min_mld_index=beat_results[0].min_mld_index,  # Use first beat's index
            max_mld_index=beat_results[0].max_mld_index,
            min_phase=most_common_min_phase,
            max_phase=most_common_max_phase
        )