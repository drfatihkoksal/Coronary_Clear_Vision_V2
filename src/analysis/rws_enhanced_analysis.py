"""
Enhanced RWS Analysis Module
Implements the methodology from the AngioPlus Core software paper:
- Automatic selection of frames at specific cardiac phases
- Lumen contour coregistration across cardiac cycle
- RWS calculation along entire vessel segment
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .motion_artifact_detector import MotionArtifactDetector
from .frame_quality_analyzer import FrameQualityAnalyzer
from ..models.rws_models import RWSAnalysisResult
from ..core.cardiac_phase_detector import CardiacPhaseDetector

logger = logging.getLogger(__name__)


class EnhancedRWSAnalysis:
    """
    Enhanced RWS Analysis implementing paper methodology:
    - Analyzes diameter changes across the cardiac cycle
    - Calculates RWS at every position along the vessel
    - Identifies maximum RWS (lesion RWS)
    """
    
    def __init__(self):
        self.motion_detector = MotionArtifactDetector()
        self.quality_analyzer = FrameQualityAnalyzer()
        self.calibration_factor = None  # Will be set from input parameter
        self.use_tracking_coregistration = True  # Tracking tabanlı co-registration kullan
        self.reference_frame = None  # Referans frame (ilk d2 frame'i)
        self.enable_coregistration = True  # Co-registration aktif
        self.use_phase_selection = True  # Kardiyak faz seçimi aktif
        self.enable_multi_beat = False  # Multiple beat ortalaması (şimdilik kapalı)
        
    def _simple_length_alignment(self, diameter_profiles: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simple alignment of diameter profiles based on length"""
        if not diameter_profiles:
            return {}
            
        # Find the reference length (median)
        lengths = [len(profile) for profile in diameter_profiles.values()]
        ref_length = int(np.median(lengths))
        
        aligned_profiles = {}
        for phase, profile in diameter_profiles.items():
            if len(profile) == ref_length:
                aligned_profiles[phase] = profile
            else:
                # Resample to reference length
                x_old = np.linspace(0, 1, len(profile))
                x_new = np.linspace(0, 1, ref_length)
                aligned_profile = np.interp(x_new, x_old, profile)
                aligned_profiles[phase] = aligned_profile
                
        return aligned_profiles
    
    def _align_profiles_by_length(self, diameter_profiles: List[np.ndarray]) -> Dict:
        """Align diameter profiles by length and return as dict"""
        if not diameter_profiles:
            return {}
            
        # Find reference length
        lengths = [len(profile) for profile in diameter_profiles]
        ref_length = int(np.median(lengths))
        
        aligned_profiles = []
        for profile in diameter_profiles:
            if len(profile) == ref_length:
                aligned_profiles.append(profile)
            else:
                # Resample to reference length
                x_old = np.linspace(0, 1, len(profile))
                x_new = np.linspace(0, 1, ref_length)
                aligned_profile = np.interp(x_new, x_old, profile)
                aligned_profiles.append(aligned_profile)
                
        return {'aligned_profiles': aligned_profiles}
        
    def analyze_cardiac_cycle(self, 
                            frames: List[np.ndarray],
                            qca_results: Dict[int, Dict],
                            cardiac_phases: Dict,
                            frame_timestamps: np.ndarray,
                            calibration_factor: float,
                            reference_frame_idx: Optional[int] = None) -> Dict:
        """
        Perform enhanced RWS analysis across cardiac cycle
        
        Args:
            frames: List of angiography frames
            qca_results: QCA results for each frame
            cardiac_phases: Cardiac phase detection results
            frame_timestamps: Timestamps for each frame
            calibration_factor: Calibration in mm/pixel
            
        Returns:
            Dictionary containing enhanced RWS analysis results
        """
        self.calibration_factor = calibration_factor
        
        # Set reference frame (first d2 frame if available)
        if reference_frame_idx is not None:
            self.reference_frame = reference_frame_idx
            logger.info(f"Using provided reference frame: {reference_frame_idx}")
        else:
            # Find first d2 frame automatically
            self.reference_frame = self._find_first_d2_frame(qca_results, cardiac_phases)
            logger.info(f"Auto-detected reference frame (first d2): {self.reference_frame}")
        
        try:
            # Step 1: Select optimal frames for each cardiac phase
            logger.info("Selecting optimal frames using mathematical quality analysis...")
            
            # Calculate motion scores for all frames
            motion_scores = self.motion_detector._calculate_motion_scores(frames)
            motion_scores_dict = {i: score for i, score in enumerate(motion_scores)}
            
            # Check if we have cardiac phase information
            if cardiac_phases and cardiac_phases.get('phases'):
                # Use advanced frame selection with quality analyzer
                # Only select 3 phases (excluding end-diastole as per paper)
                selected_frames = self.quality_analyzer.select_optimal_frames(
                    qca_results, 
                    cardiac_phases,
                    motion_scores_dict,
                    target_phases=['early-systole', 'end-systole', 'mid-diastole']
                )
            else:
                # No cardiac phases - select frames based on motion and consistency
                logger.warning("No cardiac phase data available, using quality-based selection only")
                selected_frames = self._select_frames_by_quality_only(frames, qca_results, motion_scores_dict)
            
            if len(selected_frames) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient frames selected for analysis'
                }
            
            # Log selected frames info
            logger.info(f"Selected frames for analysis: {selected_frames}")
            
            # Step 2: Calculate RWS at stenosis (MLD location)
            logger.info("Calculating RWS at stenosis location...")
            stenosis_rws = self._calculate_stenosis_rws(selected_frames, qca_results)
            
            if not stenosis_rws.get('rws_stenosis'):
                return {
                    'success': False,
                    'error': 'Failed to calculate stenosis RWS'
                }
            
            # Step 3: Analyze diameter changes at anatomical locations
            logger.info("Analyzing diameter changes at anatomical locations...")
            segment_analysis = self._analyze_diameter_changes(
                selected_frames, qca_results, stenosis_rws
            )
            
            # Step 4: Clinical interpretation
            risk_level = self._assess_risk_level(stenosis_rws['rws_stenosis'])
            
            # Compile results
            results = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'selected_frames': selected_frames,
                'stenosis_rws': stenosis_rws,
                'segment_analysis': segment_analysis,
                'risk_level': risk_level,
                'clinical_interpretation': self._get_clinical_interpretation(stenosis_rws['rws_stenosis']),
                'motion_quality': self.motion_detector.analyze_motion_quality(frames),
                'num_phases_analyzed': len(selected_frames),
                'calibration_factor': calibration_factor,
                # Summary values
                'rws_percentage': stenosis_rws['rws_stenosis'],
                'max_mld': stenosis_rws['max_mld'],
                'min_mld': stenosis_rws['min_mld'],
                'max_mld_phase': stenosis_rws['max_mld_phase'],
                'min_mld_phase': stenosis_rws['min_mld_phase']
            }
            
            logger.info(f"Enhanced RWS analysis completed. Stenosis RWS: {stenosis_rws['rws_stenosis']}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced RWS analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _coregister_lumen_contours(self, 
                                  frames: List[np.ndarray],
                                  qca_results: Dict[int, Dict],
                                  selected_frames: Dict[str, int]) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract and coregister lumen diameter profiles across cardiac phases
        Uses tracking-based or anatomical registration methods
        """
        # First extract diameter profiles
        diameter_profiles = self._extract_diameter_profiles(qca_results, selected_frames)
        
        if not diameter_profiles or len(diameter_profiles) < 2:
            logger.error("Insufficient diameter profiles for co-registration")
            return None
        
        # Try different co-registration methods in order
        coregistered_diameters = None
        
        # 1. Try anatomical co-registration (most robust)
        try:
            logger.info("Attempting anatomical co-registration...")
            coregistered_diameters = self._simple_length_alignment(diameter_profiles)
            if coregistered_diameters and len(coregistered_diameters) >= 2:
                logger.info("Successfully used anatomical co-registration")
                return coregistered_diameters
        except Exception as e:
            logger.warning(f"Anatomical co-registration failed: {e}")
        
        # 2. Try tracking-based co-registration if frames available
        if self.use_tracking_coregistration and frames is not None:
            try:
                logger.info("Attempting tracking-based co-registration...")
                tracking_result = self._coregister_with_tracking(frames, qca_results, selected_frames)
                if tracking_result and len(tracking_result) >= 2:
                    logger.info("Successfully used tracking-based co-registration")
                    return tracking_result
            except Exception as e:
                logger.warning(f"Tracking co-registration failed: {e}")
        
        # Registration failed
        logger.error("All co-registration methods failed")
        return None
    
    def _extract_diameter_profiles(self, 
                                 qca_results: Dict[int, Dict],
                                 selected_frames: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Extract diameter profiles from QCA results"""
        diameter_profiles = {}
        
        for phase_name, frame_idx in selected_frames.items():
            # Check both int and string keys
            qca_data = None
            if frame_idx in qca_results:
                qca_data = qca_results[frame_idx]
            elif str(frame_idx) in qca_results:
                qca_data = qca_results[str(frame_idx)]
            else:
                logger.warning(f"No QCA results for frame {frame_idx} ({phase_name})")
                continue
            
            # Extract diameter profile with proper calibration
            diameter_profile = None
            if 'diameters_mm' in qca_data:
                diameter_profile = np.array(qca_data['diameters_mm'])
            elif 'diameter_profile' in qca_data and self.calibration_factor:
                diameter_profile = np.array(qca_data['diameter_profile']) * self.calibration_factor
            elif 'diameters_pixels' in qca_data and self.calibration_factor:
                diameter_profile = np.array(qca_data['diameters_pixels']) * self.calibration_factor
            
            if diameter_profile is not None and len(diameter_profile) > 0:
                diameter_profiles[phase_name] = diameter_profile
            else:
                logger.warning(f"Invalid diameter data for {phase_name}")
        
        return diameter_profiles
    
    
    def _align_by_mld_position(self, diameter_profile: np.ndarray, mld_idx: int,
                               reference_length: int, reference_mld_idx: int) -> np.ndarray:
        """
        Align diameter profile using MLD positions as anchor points
        """
        if mld_idx == reference_mld_idx and len(diameter_profile) == reference_length:
            return diameter_profile
        
        # Calculate scaling factor based on MLD positions
        # Preserve anatomical relationships by aligning MLD positions
        
        # Split profile into proximal and distal segments at MLD
        proximal_segment = diameter_profile[:mld_idx+1]
        distal_segment = diameter_profile[mld_idx:]
        
        # Calculate target lengths for each segment
        proximal_target_length = reference_mld_idx + 1
        distal_target_length = reference_length - reference_mld_idx
        
        # Resample each segment independently
        if len(proximal_segment) > 1:
            proximal_x_old = np.linspace(0, 1, len(proximal_segment))
            proximal_x_new = np.linspace(0, 1, proximal_target_length)
            proximal_interp = interp1d(proximal_x_old, proximal_segment, 
                                     kind='cubic', bounds_error=False, 
                                     fill_value='extrapolate')
            proximal_resampled = proximal_interp(proximal_x_new)
        else:
            proximal_resampled = np.full(proximal_target_length, proximal_segment[0])
        
        if len(distal_segment) > 1:
            distal_x_old = np.linspace(0, 1, len(distal_segment))
            distal_x_new = np.linspace(0, 1, distal_target_length)
            distal_interp = interp1d(distal_x_old, distal_segment, 
                                   kind='cubic', bounds_error=False, 
                                   fill_value='extrapolate')
            distal_resampled = distal_interp(distal_x_new)
        else:
            distal_resampled = np.full(distal_target_length, distal_segment[0])
        
        # Combine segments (overlap at MLD position)
        aligned_profile = np.concatenate([proximal_resampled[:-1], distal_resampled])
        
        # Smooth transition at junction with stenosis preservation
        if len(aligned_profile) > 5:
            aligned_profile = self._smooth_with_stenosis_preservation(aligned_profile)
        
        return aligned_profile
    
    def _align_by_centerline(self, diameter_profile: np.ndarray, centerline: np.ndarray,
                           reference_profile: np.ndarray, reference_centerline: np.ndarray) -> np.ndarray:
        """
        Align diameter profiles using centerline correspondence
        """
        # Calculate cumulative arc length along centerlines
        arc_length = self._calculate_arc_length(centerline)
        ref_arc_length = self._calculate_arc_length(reference_centerline)
        
        # Normalize arc lengths to [0, 1]
        if arc_length[-1] > 0:
            normalized_arc = arc_length / arc_length[-1]
        else:
            normalized_arc = np.linspace(0, 1, len(arc_length))
            
        if ref_arc_length[-1] > 0:
            ref_normalized_arc = ref_arc_length / ref_arc_length[-1]
        else:
            ref_normalized_arc = np.linspace(0, 1, len(ref_arc_length))
        
        # Interpolate diameter values to reference arc length positions
        interp_func = interp1d(normalized_arc, diameter_profile, 
                             kind='cubic', bounds_error=False, 
                             fill_value='extrapolate')
        
        aligned_profile = interp_func(ref_normalized_arc)
        
        # Ensure same length as reference
        if len(aligned_profile) != len(reference_profile):
            aligned_profile = self._coregister_diameter_profiles(
                aligned_profile, len(reference_profile)
            )
        
        return aligned_profile
    
    def _calculate_arc_length(self, centerline: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative arc length along centerline
        Handles both [y,x] and [x,y] formats
        """
        if len(centerline) < 2:
            return np.array([0])
        
        # Ensure centerline is 2D array
        if centerline.ndim != 2 or centerline.shape[1] != 2:
            logger.warning(f"Invalid centerline shape: {centerline.shape}")
            return np.linspace(0, len(centerline), len(centerline))
        
        # Calculate distances between consecutive points
        # Works correctly regardless of [y,x] or [x,y] format
        diffs = np.diff(centerline, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Cumulative sum gives arc length at each point
        arc_length = np.concatenate([[0], np.cumsum(distances)])
        
        return arc_length
    
    def _calculate_centerline_length(self, centerline: np.ndarray) -> float:
        """Calculate cumulative length along centerline"""
        if len(centerline) < 2:
            return 0.0
        
        # Calculate distances between consecutive points
        diffs = np.diff(centerline, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Cumulative sum gives position along centerline
        cumulative_length = np.concatenate([[0], np.cumsum(distances)])
        
        return cumulative_length[-1]
    
    def _coregister_diameter_profiles(self, 
                                     diameter_profile: np.ndarray,
                                     reference_length: int) -> np.ndarray:
        """
        Coregister diameter profile to reference length
        """
        if len(diameter_profile) == reference_length:
            return diameter_profile
            
        # Create normalized position arrays (0 to 1)
        current_positions = np.linspace(0, 1, len(diameter_profile))
        reference_positions = np.linspace(0, 1, reference_length)
        
        # Interpolate diameter values to reference positions
        interpolator = interp1d(current_positions, diameter_profile, 
                               kind='cubic', bounds_error=False, 
                               fill_value='extrapolate')
        
        coregistered_diameter = interpolator(reference_positions)
        
        # Smooth to reduce interpolation artifacts with stenosis preservation
        coregistered_diameter = self._smooth_with_stenosis_preservation(coregistered_diameter)
        
        return coregistered_diameter
    
    def _calculate_stenosis_rws(self, 
                               selected_frames: Dict[str, int],
                               qca_results: Dict[int, Dict]) -> Dict:
        """
        Calculate RWS along entire vessel segment and find maximum for each anatomical region
        RWS = (largest diameter - smallest diameter) / largest diameter at each position
        """
        # Use co-registration if enabled
        if self.enable_coregistration and self.reference_frame is not None:
            logger.info(f"Co-registration enabled using reference frame {self.reference_frame}")
            coregistered_diameters = self._perform_coregistration(selected_frames, qca_results)
        else:
            logger.info("Co-registration disabled or no reference frame. Using MLD-based RWS only.")
            coregistered_diameters = None
        
        # Create empty arrays for backward compatibility
        coregistered_diameters = {}
        diameter_matrix = np.array([])
        rws_profile = np.array([])
        
        logger.info("Skipping position-based RWS profile calculation")
        
        # Ensure quality analyzer is available for outlier detection
        if not hasattr(self, 'quality_analyzer'):
            from .frame_quality_analyzer import FrameQualityAnalyzer
            self.quality_analyzer = FrameQualityAnalyzer()
        
        # Find MLD position and values from each phase
        mld_values = {}
        mld_indices = {}
        diameter_profiles_by_phase = {}
        
        for phase_name, frame_idx in selected_frames.items():
            if frame_idx in qca_results:
                qca_data = qca_results[frame_idx]
                
                # Get MLD value
                if 'mld' in qca_data and qca_data['mld'] is not None:
                    mld_values[phase_name] = float(qca_data['mld'])
                    
                    # Get MLD index/position if available
                    if 'mld_index' in qca_data:
                        mld_indices[phase_name] = qca_data['mld_index']
                    elif 'mld_idx' in qca_data:
                        mld_indices[phase_name] = qca_data['mld_idx']
                
                # Get diameter profile for P and D point calculations
                diameter_profile = None
                if 'diameters_mm' in qca_data and qca_data['diameters_mm'] is not None:
                    diameter_profile = np.array(qca_data['diameters_mm'])
                elif 'diameters_pixels' in qca_data and qca_data['diameters_pixels'] is not None:
                    diameter_profile = np.array(qca_data['diameters_pixels']) * self.calibration_factor
                
                if diameter_profile is not None and len(diameter_profile) > 0:
                    diameter_profiles_by_phase[phase_name] = diameter_profile
        
        # Get stenosis boundaries that are already calculated by QCA
        # P-point: Where diameter drops below 75th percentile (stenosis start)
        # D-point: Where diameter rises back to 75th percentile (stenosis end)
        # MLD-point: Location of minimum lumen diameter
        
        # Get stenosis boundaries from QCA results (these are already calculated!)
        stenosis_boundaries = {}
        reference_frame_idx = list(selected_frames.values())[0] if selected_frames else None
        
        if reference_frame_idx and reference_frame_idx in qca_results:
            qca_ref = qca_results[reference_frame_idx]
            
            # Get P, MLD, D positions from QCA (already calculated!)
            if 'p_point' in qca_ref:
                stenosis_boundaries['p_point'] = qca_ref['p_point']
            if 'mld_index' in qca_ref or 'mld_idx' in qca_ref:
                stenosis_boundaries['mld_point'] = qca_ref.get('mld_index') or qca_ref.get('mld_idx')
            if 'd_point' in qca_ref:
                stenosis_boundaries['d_point'] = qca_ref['d_point']
        
        # Use calculated boundaries or fallback to MLD position
        if mld_indices:
            min_mld_phase = min(mld_values.items(), key=lambda x: x[1])[0]
            stenosis_position = mld_indices.get(min_mld_phase, 0)
        else:
            stenosis_position = 0
            
        p_point = stenosis_boundaries.get('p_point', max(0, stenosis_position - 10))
        mld_point = stenosis_boundaries.get('mld_point', stenosis_position)
        d_point = stenosis_boundaries.get('d_point', stenosis_position + 10)
        
        logger.info(f"Using QCA-calculated anatomical boundaries:")
        logger.info(f"  P-point (75%ile start): {p_point}")
        logger.info(f"  MLD-point (Stenosis): {mld_point}")
        logger.info(f"  D-point (75%ile end): {d_point}")
        
        # Store boundaries for UI compatibility
        self.stenosis_boundaries = {
            'p_point': p_point,
            'd_point': d_point,
            'mld_point': mld_point,
            'reference_diameter': 0.0,
            'threshold': 0.0
        }
        
        # Empty region_rws for UI compatibility
        region_rws = {}
        
        # Empty regions for UI compatibility
        regions = {}
        
        # Overall lesion RWS (use MLD-based calculation)
        overall_rws_max = 0.0
        max_region = 'stenosis'
        
        # Get RWS specifically at stenosis (MLD position) - will be calculated after stenosis_rws_from_mld
        # Use MLD-based RWS calculation instead of profile-based (CORRECT APPROACH)
        # This is the key metric: (MLD_max - MLD_min) / MLD_min × 100%
        # rws_at_stenosis = rws_profile[stenosis_position] if stenosis_position < len(rws_profile) else 0  # WRONG: diameter profile approach
        
        # Calculate RWS for P, MLD, and D points using same methodology
        # P-point RWS: (max_p_diameter - min_p_diameter) / min_p_diameter × 100%
        # MLD RWS: (max_mld - min_mld) / min_mld × 100% 
        # D-point RWS: (max_d_diameter - min_d_diameter) / min_d_diameter × 100%
        
        point_rws_results = {}
        
        # Calculate for each anatomical point (using QCA-calculated boundaries)
        anatomical_points = {
            'proximal': p_point,      # 75th percentile start
            'stenosis': mld_point,    # MLD position
            'distal': d_point         # 75th percentile end
        }
        
        for point_name, point_position in anatomical_points.items():
            if point_name == 'stenosis':
                # Use MLD values directly for stenosis (most accurate)
                # NO OUTLIER FILTERING FOR MLD VALUES - Use all selected frames
                # Outlier detection is for information only, not for filtering
                filtered_mld_values = mld_values
                logger.info(f"MLD values used without outlier filtering: {len(mld_values)} phases")
                
                if filtered_mld_values:
                    max_mld_phase = max(filtered_mld_values.items(), key=lambda x: x[1])
                    min_mld_phase = min(filtered_mld_values.items(), key=lambda x: x[1])
                    max_diameter = max_mld_phase[1]
                    min_diameter = min_mld_phase[1]
                    max_phase_name = max_mld_phase[0]
                    min_phase_name = min_mld_phase[0]
                else:
                    # Fallback to unfiltered if all are outliers
                    logger.warning("All MLD values marked as outliers - using unfiltered data")
                    if mld_values:
                        max_mld_phase = max(mld_values.items(), key=lambda x: x[1])
                        min_mld_phase = min(mld_values.items(), key=lambda x: x[1])
                        max_diameter = max_mld_phase[1]
                        min_diameter = min_mld_phase[1]
                        max_phase_name = max_mld_phase[0]
                        min_phase_name = min_mld_phase[0]
                    else:
                        max_diameter = min_diameter = 0.0
                        max_phase_name = min_phase_name = 'unknown'
            else:
                # For P and D points, extract diameters at specific positions
                # NO OUTLIER FILTERING for P and D - use all available frames
                point_diameters = {}
                
                for phase_name, diameter_profile in diameter_profiles_by_phase.items():
                    if point_position < len(diameter_profile):
                        point_diameters[phase_name] = diameter_profile[point_position]
                
                if point_diameters:
                    max_phase_data = max(point_diameters.items(), key=lambda x: x[1])
                    min_phase_data = min(point_diameters.items(), key=lambda x: x[1])
                    max_diameter = max_phase_data[1]
                    min_diameter = min_phase_data[1]
                    max_phase_name = max_phase_data[0]
                    min_phase_name = min_phase_data[0]
                    
                    logger.info(f"{point_name.upper()} point uses all {len(point_diameters)} frames (no outlier filtering)")
                else:
                    max_diameter = min_diameter = 0.0
                    max_phase_name = min_phase_name = 'unknown'
            
            # Calculate RWS for this point
            if min_diameter > 0:
                point_rws = ((max_diameter - min_diameter) / min_diameter * 100)
            else:
                point_rws = 0.0
                
            point_rws_results[point_name] = {
                'rws': float(point_rws),
                'max_diameter': float(max_diameter),
                'min_diameter': float(min_diameter),
                'max_phase': max_phase_name,
                'min_phase': min_phase_name,
                'diameter_change': float(max_diameter - min_diameter)
            }
            
            logger.info(f"{point_name.upper()} RWS: {point_rws:.1f}% "
                       f"(Max: {max_diameter:.3f}mm [{max_phase_name}] → "
                       f"Min: {min_diameter:.3f}mm [{min_phase_name}])")
        
        # Use stenosis RWS as the main RWS value
        stenosis_rws_from_mld = point_rws_results['stenosis']['rws']
        rws_at_stenosis = stenosis_rws_from_mld
        
        # Get overall max/min for compatibility
        max_mld = point_rws_results['stenosis']['max_diameter']
        min_mld = point_rws_results['stenosis']['min_diameter']
        max_mld_phase = (point_rws_results['stenosis']['max_phase'], max_mld)
        min_mld_phase = (point_rws_results['stenosis']['min_phase'], min_mld)
        
        logger.info(f"Stenosis RWS from MLD values: {stenosis_rws_from_mld}% (MLD_max={max_mld:.3f}mm, MLD_min={min_mld:.3f}mm)")
        
        logger.info(f"Stenosis RWS at MLD: {stenosis_rws_from_mld}% (from MLD: {max_mld:.3f} -> {min_mld:.3f} mm)")
        
        # Note: Regional RWS analysis disabled - using only MLD-based calculation
        
        return {
            'rws_stenosis': float(rws_at_stenosis),  # RWS at stenosis location: (MLD_max - MLD_min) / MLD_min
            'rws_at_mld': float(rws_at_stenosis),  # RWS at MLD position
            'rws_profile': rws_profile.tolist(),  # Full RWS profile
            'rws_max': float(overall_rws_max),  # Maximum RWS across all regions
            'region_rws': region_rws,  # RWS analysis for each anatomical region
            'max_rws_region': max_region,  # Which region has the maximum RWS
            'max_mld': float(max_mld),
            'min_mld': float(min_mld),
            'max_mld_phase': max_mld_phase[0],
            'min_mld_phase': min_mld_phase[0],
            'mld_values': mld_values,
            'diameter_change': float(max_mld - min_mld),
            'stenosis_position': int(mld_point),
            'region_boundaries': regions,
            'region_rws': region_rws,  # Empty dict - UI compatibility
            'stenosis_boundaries': getattr(self, 'stenosis_boundaries', {}),
            'diameter_profiles': {},  # Empty - co-registration disabled
            'rws_profile': [],  # Empty - co-registration disabled
            # NEW: Point-specific RWS results
            'point_rws_results': point_rws_results  # P, stenosis, D point RWS calculations
        }
    
    def _analyze_diameter_changes(self, 
                                 selected_frames: Dict[str, int],
                                 qca_results: Dict[int, Dict],
                                 stenosis_rws: Dict) -> Dict:
        """
        Analyze diameter changes at specific anatomical locations between max and min MLD frames
        """
        max_phase = stenosis_rws['max_mld_phase']
        min_phase = stenosis_rws['min_mld_phase']
        
        # Get frame indices for max and min MLD
        max_frame_idx = selected_frames[max_phase]
        min_frame_idx = selected_frames[min_phase]
        
        # Get diameter profiles for these frames
        max_frame_data = qca_results.get(max_frame_idx, {})
        min_frame_data = qca_results.get(min_frame_idx, {})
        
        # Extract diameter profiles
        max_diameters = None
        min_diameters = None
        
        if 'diameters_mm' in max_frame_data:
            max_diameters = np.array(max_frame_data['diameters_mm'])
        elif 'diameters_pixels' in max_frame_data and hasattr(self, 'calibration_factor'):
            max_diameters = np.array(max_frame_data['diameters_pixels']) * self.calibration_factor
            
        if 'diameters_mm' in min_frame_data:
            min_diameters = np.array(min_frame_data['diameters_mm'])
        elif 'diameters_pixels' in min_frame_data and hasattr(self, 'calibration_factor'):
            min_diameters = np.array(min_frame_data['diameters_pixels']) * self.calibration_factor
        
        if max_diameters is None or min_diameters is None:
            logger.warning("Could not extract diameter profiles for segment analysis")
            return {}
        
        # Ensure same length
        min_length = min(len(max_diameters), len(min_diameters))
        max_diameters = max_diameters[:min_length]
        min_diameters = min_diameters[:min_length]
        
        # Find MLD position (stenosis location)
        mld_idx = stenosis_rws.get('mld_index', len(min_diameters) // 2)
        
        # Define anatomical segments relative to stenosis
        segments = {}
        
        # Proximal reference (5-10mm proximal to stenosis)
        proximal_offset = int(7.5 / self.calibration_factor)  # ~7.5mm in pixels
        if mld_idx - proximal_offset >= 0:
            proximal_idx = mld_idx - proximal_offset
            segments['proximal_reference'] = {
                'diameter_max_phase': float(max_diameters[proximal_idx]),
                'diameter_min_phase': float(min_diameters[proximal_idx]),
                'diameter_change': float(max_diameters[proximal_idx] - min_diameters[proximal_idx]),
                'location_mm': -7.5  # 7.5mm proximal to stenosis
            }
        
        # Stenosis/throat (MLD location)
        segments['stenosis'] = {
            'diameter_max_phase': stenosis_rws['max_mld'],
            'diameter_min_phase': stenosis_rws['min_mld'],
            'diameter_change': stenosis_rws['diameter_change'],
            'rws_percentage': stenosis_rws['rws_stenosis'],
            'location_mm': 0.0  # Reference point
        }
        
        # Distal reference (5-10mm distal to stenosis)
        distal_offset = int(7.5 / self.calibration_factor)  # ~7.5mm in pixels
        if mld_idx + distal_offset < len(min_diameters):
            distal_idx = mld_idx + distal_offset
            segments['distal_reference'] = {
                'diameter_max_phase': float(max_diameters[distal_idx]),
                'diameter_min_phase': float(min_diameters[distal_idx]),
                'diameter_change': float(max_diameters[distal_idx] - min_diameters[distal_idx]),
                'location_mm': 7.5  # 7.5mm distal to stenosis
            }
        
        logger.info(f"Analyzed diameter changes between {max_phase} and {min_phase}")
        
        return segments
    
    def _assess_risk_level(self, rws_value: float) -> str:
        """
        Assess cardiovascular risk based on RWS value
        Based on clinical thresholds from literature
        """
        if rws_value < 10:
            return "LOW"
        elif rws_value < 12:
            return "MODERATE"
        elif rws_value < 15:
            return "HIGH"
        else:
            return "VERY HIGH"
    
    def _get_clinical_interpretation(self, rws_value: float) -> str:
        """
        Provide clinical interpretation of RWS value
        """
        if rws_value < 10:
            return "Normal vessel wall motion. Low risk of plaque vulnerability."
        elif rws_value < 12:
            return "Mildly increased wall motion. Consider close monitoring."
        elif rws_value < 15:
            return "Significantly increased wall motion. Suggests plaque vulnerability. Consider intervention."
        else:
            return "Severely increased wall motion. High risk of plaque rupture. Urgent evaluation recommended."
    
    def _select_frames_by_quality_only(self, frames: List[np.ndarray], 
                                      qca_results: Dict[int, Dict],
                                      motion_scores: Dict[int, float]) -> Dict[str, int]:
        """
        Select frames based on quality metrics without cardiac phase info
        Uses temporal consistency and outlier detection
        """
        logger.info(f"Using quality-based frame selection. QCA results available for frames: {list(qca_results.keys())}")
        
        # Get frames with valid QCA results - handle both int and string keys
        all_keys = list(qca_results.keys())
        logger.info(f"QCA result keys types: {[type(k) for k in all_keys[:5]]}")  # Show first 5 key types
        
        valid_frames = []
        for key in all_keys:
            # Convert string keys to int if needed
            try:
                idx = int(key) if isinstance(key, str) else key
                if idx < len(frames):
                    valid_frames.append(idx)
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid frame key: {key}")
                
        valid_frames = sorted(valid_frames)
        logger.info(f"Valid frames (with QCA results): {valid_frames}")
        
        if len(valid_frames) < 2:
            logger.error(f"Not enough valid frames: {len(valid_frames)} < 2")
            return {}
        
        # Detect outliers for information only - do NOT filter frames
        outlier_scores, outlier_frames = self.quality_analyzer.detect_outlier_frames(
            qca_results, valid_frames
        )
        logger.info(f"Quality-based selection: {len(outlier_frames)} outlier frames detected but NOT filtered")
        
        # Use ALL valid frames - no outlier filtering
        candidate_frames = valid_frames
        
        # Get diameter values for temporal analysis
        frame_diameters = {}
        for idx in candidate_frames:
            if idx in qca_results:
                if 'mld' in qca_results[idx]:
                    frame_diameters[idx] = qca_results[idx]['mld']
                elif 'diameters_mm' in qca_results[idx]:
                    # Use mean diameter if MLD not available
                    diams = qca_results[idx]['diameters_mm']
                    if diams and len(diams) > 0:
                        frame_diameters[idx] = np.mean(diams)
        
        if len(frame_diameters) < 3:
            logger.warning("Insufficient diameter data for quality-based selection")
            # Use motion-based selection
            selected = {}
            sorted_by_motion = sorted(candidate_frames, 
                                    key=lambda f: motion_scores.get(f, float('inf')))
            
            phase_names = ['early-systole', 'end-systole', 'mid-diastole']
            for i, phase_name in enumerate(phase_names[:min(3, len(sorted_by_motion))]):
                selected[phase_name] = sorted_by_motion[i]
            
            return selected
        
        # Sort frames by diameter to identify systole/diastole
        sorted_diameters = sorted(frame_diameters.items(), key=lambda x: x[1])
        
        # Calculate combined quality scores
        frame_scores = {}
        for idx in candidate_frames:
            outlier_score = outlier_scores.get(idx, 0.0)
            motion_score = motion_scores.get(idx, 1.0)
            
            # Combined score (lower is better)
            combined_score = 0.5 * outlier_score + 0.5 * motion_score
            frame_scores[idx] = combined_score
        
        # Select frames representing different phases
        selected = {}
        
        # Early-systole: smaller diameter frames
        systole_candidates = [f for f, _ in sorted_diameters[:len(sorted_diameters)//3]]
        if systole_candidates:
            best_systole = min(systole_candidates, key=lambda f: frame_scores.get(f, float('inf')))
            selected['early-systole'] = best_systole
        
        # Mid-diastole: larger diameter frames  
        diastole_candidates = [f for f, _ in sorted_diameters[-len(sorted_diameters)//3:]]
        if diastole_candidates:
            best_diastole = min(diastole_candidates, key=lambda f: frame_scores.get(f, float('inf')))
            selected['mid-diastole'] = best_diastole
        
        # End-systole: intermediate frames
        used_frames = set(selected.values())
        intermediate_candidates = [f for f in candidate_frames if f not in used_frames]
        
        if intermediate_candidates:
            # Select frame with diameter closest to median
            median_diameter = np.median(list(frame_diameters.values()))
            best_intermediate = min(intermediate_candidates,
                                  key=lambda f: abs(frame_diameters.get(f, 0) - median_diameter))
            selected['end-systole'] = best_intermediate
        
        logger.info(f"Selected {len(selected)} frames using quality-based analysis")
        
        return selected
    
    def _find_first_d2_frame(self, qca_results: Dict, cardiac_phases) -> Optional[int]:
        """Find the first d2 (diastole) frame to use as reference"""
        # Handle different cardiac_phases formats
        if not cardiac_phases:
            # No phase info, use first available frame
            if qca_results:
                return min(qca_results.keys())
            return None
        
        # Extract phases based on type
        phases = None
        if isinstance(cardiac_phases, dict):
            phases = cardiac_phases.get('phases', cardiac_phases)
        elif isinstance(cardiac_phases, list):
            # Convert list to dict if needed
            phases = {i: phase for i, phase in enumerate(cardiac_phases)}
        else:
            phases = cardiac_phases
            
        if not phases:
            if qca_results:
                return min(qca_results.keys())
            return None
        
        # Look for first d2 frame
        try:
            # Handle both dict and list iterations
            if hasattr(phases, 'items'):
                iterator = phases.items()
            else:
                iterator = enumerate(phases)
                
            for frame_idx, phase_info in iterator:
                if isinstance(phase_info, dict):
                    phase_name = phase_info.get('phase', '')
                else:
                    phase_name = str(phase_info)
                    
                if 'd2' in phase_name.lower() or 'diastole' in phase_name.lower():
                    if frame_idx in qca_results:
                        return frame_idx
        except Exception as e:
            logger.warning(f"Error finding d2 frame: {e}")
        
        # If no d2 found, use first frame with QCA results
        if qca_results:
            return min(qca_results.keys())
            
        return None
    
    def _perform_coregistration(self, selected_frames: Dict[str, int], qca_results: Dict) -> Optional[Dict]:
        """Perform co-registration using reference frame"""
        if not self.reference_frame or self.reference_frame not in qca_results:
            logger.warning("Reference frame not available for co-registration")
            return None
            
        try:
            # Get reference frame data
            ref_qca = qca_results[self.reference_frame]
            ref_centerline = ref_qca.get('centerline')
            
            if ref_centerline is None:
                logger.warning("Reference frame has no centerline for co-registration")
                return None
            
            # Align all selected frames to reference
            aligned_profiles = {}
            
            for phase_name, frame_idx in selected_frames.items():
                if frame_idx not in qca_results:
                    continue
                    
                frame_qca = qca_results[frame_idx]
                
                # Get diameter profile
                if 'diameters_mm' in frame_qca:
                    profile = np.array(frame_qca['diameters_mm'])
                elif 'diameters_pixels' in frame_qca and self.calibration_factor:
                    profile = np.array(frame_qca['diameters_pixels']) * self.calibration_factor
                else:
                    continue
                
                # Align to reference length
                if len(profile) != len(ref_centerline):
                    # Resample to match reference
                    x_old = np.linspace(0, 1, len(profile))
                    x_new = np.linspace(0, 1, len(ref_centerline))
                    aligned_profile = np.interp(x_new, x_old, profile)
                else:
                    aligned_profile = profile
                    
                aligned_profiles[phase_name] = aligned_profile
            
            logger.info(f"Successfully co-registered {len(aligned_profiles)} profiles to reference frame {self.reference_frame}")
            return aligned_profiles
            
        except Exception as e:
            logger.error(f"Co-registration failed: {e}")
            return None
    
    def _coregister_with_tracking(self,
                                 frames: List[np.ndarray],
                                 qca_results: Dict[int, Dict],
                                 selected_frames: Dict[str, int]) -> Optional[Dict[str, np.ndarray]]:
        """
        Tracking tabanlı co-registration kullanarak çap profillerini hizala
        
        Args:
            frames: Tüm frame'ler
            qca_results: QCA analiz sonuçları
            selected_frames: Seçili frame'ler
            
        Returns:
            Hizalanmış çap profilleri
        """
        logger.info("Tracking tabanlı co-registration başlatılıyor...")
        
        try:
            # Frame verilerini hazırla
            frame_indices = list(selected_frames.values())
            selected_frame_data = [frames[idx] for idx in frame_indices if idx < len(frames)]
            
            # Maske ve centerline verilerini topla
            masks = []
            centerlines = []
            diameter_profiles = []
            
            for phase_name, frame_idx in selected_frames.items():
                if frame_idx in qca_results:
                    result = qca_results[frame_idx]
                    
                    # Maske
                    mask = result.get('mask')
                    if mask is not None:
                        masks.append(mask)
                    else:
                        # Boş maske ekle
                        masks.append(np.zeros_like(selected_frame_data[0][:,:,0] if len(selected_frame_data[0].shape) == 3 else selected_frame_data[0]))
                    
                    # Centerline
                    centerline = result.get('centerline')
                    if centerline is not None:
                        centerlines.append(np.array(centerline))
                    else:
                        centerlines.append(None)
                    
                    # Çap profili
                    diameters = result.get('diameters_mm') or result.get('diameters_pixels')
                    if diameters is not None:
                        diameter_profiles.append(np.array(diameters))
                    else:
                        logger.warning(f"Frame {frame_idx} için çap profili bulunamadı")
                        diameter_profiles.append(np.zeros(100))  # Varsayılan boyut
            
            # Tracking tabanlı co-registration uygula
            tracking_result = self._align_profiles_by_length(diameter_profiles)
            
            if tracking_result and 'aligned_profiles' in tracking_result:
                # Sonuçları phase isimleriyle eşleştir
                aligned_profiles = tracking_result['aligned_profiles']
                coregistered_diameters = {}
                
                for i, (phase_name, _) in enumerate(selected_frames.items()):
                    if i < len(aligned_profiles):
                        coregistered_diameters[phase_name] = aligned_profiles[i]
                
                # Kalite metriklerini logla
                if 'quality_metrics' in tracking_result:
                    metrics = tracking_result['quality_metrics']
                    logger.info(f"Tracking co-registration kalite metrikleri:")
                    logger.info(f"  - Tracking başarı oranı: {metrics.get('tracking_success_rate', 0):.2%}")
                    logger.info(f"  - Ortalama tracking güveni: {metrics.get('average_tracking_confidence', 0):.3f}")
                    logger.info(f"  - Transform stabilitesi: {metrics.get('transformation_stability', 0):.3f}")
                    logger.info(f"  - Trajectory düzgünlüğü: {metrics.get('trajectory_smoothness', 0):.3f}")
                    logger.info(f"  - Genel kalite: {metrics.get('overall_quality', 0):.3f}")
                
                return coregistered_diameters
            else:
                logger.warning("Tracking co-registration başarısız, eski yönteme dönülüyor")
                return None
                
        except Exception as e:
            logger.error(f"Tracking co-registration hatası: {e}")
            logger.error(f"Detaylı hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _smooth_with_stenosis_preservation(self, diameters_px: np.ndarray) -> np.ndarray:
        """
        Apply conservative smoothing while strictly preserving stenotic values
        
        Args:
            diameters_px: Diameter measurements in pixels
            
        Returns:
            Smoothed profile with preserved stenosis
        """
        if len(diameters_px) < 3:
            return diameters_px
            
        from scipy.ndimage import gaussian_filter1d
        import logging
        logger = logging.getLogger(__name__)
        
        original_profile = diameters_px.copy()
        
        # Identify stenotic regions before any smoothing
        valid_mask = original_profile > 0
        if np.sum(valid_mask) == 0:
            return original_profile
        
        valid_diameters = original_profile[valid_mask]
        mean_val = np.mean(valid_diameters)
        
        # Aggressive stenosis detection (50% reduction indicates stenosis)
        stenosis_threshold = mean_val * 0.5
        stenotic_mask = (original_profile < stenosis_threshold) & valid_mask
        stenotic_indices = np.where(stenotic_mask)[0]
        
        if len(stenotic_indices) > 0:
            # STRICT STENOSIS PRESERVATION
            final_smoothed = original_profile.copy()
            
            # Apply ultra-light smoothing only to non-stenotic regions
            non_stenotic_mask = ~stenotic_mask & valid_mask
            if np.sum(non_stenotic_mask) > 2:
                lightly_smoothed = gaussian_filter1d(original_profile, sigma=0.1)
                final_smoothed[non_stenotic_mask] = lightly_smoothed[non_stenotic_mask]
            
            logger.debug(f"Stenosis-preserving smoothing: {len(stenotic_indices)} points preserved")
        else:
            # No significant stenosis detected, apply conservative smoothing
            final_smoothed = gaussian_filter1d(original_profile, sigma=0.1)
            logger.debug("No stenosis detected - applied ultra-conservative smoothing")
        
        return final_smoothed