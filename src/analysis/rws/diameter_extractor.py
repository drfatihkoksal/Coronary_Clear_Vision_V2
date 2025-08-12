"""
Diameter extraction logic for RWS analysis
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DiameterExtractor:
    """Extract diameter profiles from QCA results"""
    
    def __init__(self, calibration_factor: float):
        self.calibration_factor = calibration_factor
    
    def extract_profiles(self, qca_results: Dict[int, Dict]) -> Dict[int, np.ndarray]:
        """
        Extract diameter measurements along the vessel for each frame
        
        Args:
            qca_results: QCA results dictionary keyed by frame index
            
        Returns:
            Dictionary of diameter arrays keyed by frame index
        """
        diameter_profiles = {}
        
        for frame_idx, qca_data in qca_results.items():
            diameters = self._extract_frame_diameters(frame_idx, qca_data)
            if diameters is not None and len(diameters) > 0:
                diameter_profiles[frame_idx] = diameters
            else:
                logger.warning(f"Frame {frame_idx}: No diameter data found")
                
        logger.info(f"Extracted diameter profiles for {len(diameter_profiles)} frames")
        return diameter_profiles
    
    def _extract_frame_diameters(self, frame_idx: int, qca_data: Dict) -> Optional[np.ndarray]:
        """Extract diameters from a single frame's QCA data"""
        diameters = None
        
        # Try 'diameters_mm' first (already in mm)
        if 'diameters_mm' in qca_data and qca_data['diameters_mm'] is not None:
            diameters = np.array(qca_data['diameters_mm'])
            logger.debug(f"Frame {frame_idx}: Found diameters_mm with {len(diameters)} points")
            
        # Try 'diameters_pixels' and convert to mm
        elif 'diameters_pixels' in qca_data and qca_data['diameters_pixels'] is not None:
            diameters = np.array(qca_data['diameters_pixels'])
            if self.calibration_factor:
                diameters = diameters * self.calibration_factor
                logger.debug(f"Frame {frame_idx}: Found diameters_pixels, converted to mm with {len(diameters)} points")
            else:
                logger.warning(f"Frame {frame_idx}: Found diameters_pixels but no calibration factor")
                return None
                
        # Try 'diameters' (might be in pixels or mm)
        elif 'diameters' in qca_data and qca_data['diameters'] is not None:
            diameters = np.array(qca_data['diameters'])
            # If values are too large (>50), assume they're in pixels and need conversion
            if np.max(diameters) > 50 and self.calibration_factor:
                logger.debug(f"Frame {frame_idx}: Converting pixel diameters to mm")
                diameters = diameters * self.calibration_factor
            logger.debug(f"Frame {frame_idx}: Found diameters with {len(diameters)} points")
            
        # Try 'profile_data' which contains diameter information
        elif 'profile_data' in qca_data and qca_data['profile_data'] is not None:
            profile = qca_data['profile_data']
            if 'diameters' in profile:
                diameters = np.array(profile['diameters'])
                logger.debug(f"Frame {frame_idx}: Found diameters in profile_data with {len(diameters)} points")
        
        return diameters
    
    def validate_mld_position(self, mld_idx: int, proximal_ref: Optional[int], 
                            distal_ref: Optional[int], min_distance: int = 20) -> bool:
        """
        Validate MLD position relative to reference points
        
        Args:
            mld_idx: MLD position index
            proximal_ref: Proximal reference position
            distal_ref: Distal reference position
            min_distance: Minimum required distance from references
            
        Returns:
            True if MLD position is valid
        """
        # If no reference points, accept any MLD
        if proximal_ref is None or distal_ref is None:
            return True
            
        # Ensure correct order (handle if they're reversed)
        min_ref = min(proximal_ref, distal_ref)
        max_ref = max(proximal_ref, distal_ref)
        
        # Check if MLD is between proximal and distal references
        if not (min_ref < mld_idx < max_ref):
            logger.debug(f"MLD at {mld_idx} is not between references ({min_ref}, {max_ref})")
            return False
            
        # Check minimum distance from proximal reference
        if abs(mld_idx - proximal_ref) < min_distance:
            logger.debug(f"MLD at {mld_idx} is too close to proximal reference {proximal_ref}")
            return False
            
        # Check minimum distance from distal reference
        if abs(mld_idx - distal_ref) < min_distance:
            logger.debug(f"MLD at {mld_idx} is too close to distal reference {distal_ref}")
            return False
            
        return True