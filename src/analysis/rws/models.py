"""
RWS Analysis Data Models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from enum import Enum


class RiskLevel(Enum):
    """Risk level classification for RWS values"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class RWSAnalysisProgress:
    """Progress information for RWS analysis"""
    status: str
    current_step: int
    total_steps: int
    message: str
    percentage: float = 0.0


@dataclass
class RWSAnalysisRequest:
    """Request parameters for RWS analysis"""
    vessel_name: str
    frames: List[int]
    calibration_factor: float
    patient_info: Optional[Dict] = None


@dataclass
class MLDInfo:
    """Information about MLD at a specific frame"""
    frame_index: int
    mld_value: float
    mld_index: Optional[int] = None
    
    @property
    def frame_number(self) -> int:
        """Get 1-based frame number for UI display"""
        return self.frame_index + 1


@dataclass
class RWSResult:
    """Multi-phase RWS calculation result"""
    rws_percentage: float
    min_mld: float
    max_mld: float
    min_mld_frame: int
    max_mld_frame: int
    min_mld_index: Optional[int] = None
    max_mld_index: Optional[int] = None
    min_phase: Optional[str] = None  # Phase name where min MLD occurs
    max_phase: Optional[str] = None  # Phase name where max MLD occurs
    
    @property
    def mld_variation(self) -> float:
        """Absolute MLD variation between phases"""
        return self.max_mld - self.min_mld
    
    @property
    def is_vulnerable(self) -> bool:
        """Check if RWS indicates vulnerable plaque (>12%)"""
        return self.rws_percentage > 12.0
    
    def get_clinical_interpretation(self) -> str:
        """Get clinical interpretation of RWS result"""
        if self.is_vulnerable:
            return f"HIGH RWS ({self.rws_percentage}%): Indicates potential plaque vulnerability"
        return f"NORMAL RWS ({self.rws_percentage}%): Indicates stable plaque characteristics"
    
    def get_phase_summary(self) -> str:
        """Get phase-based summary"""
        if self.min_phase and self.max_phase:
            return f"RWS from {self.min_phase} ({self.min_mld:.2f}mm) to {self.max_phase} ({self.max_mld:.2f}mm)"
        return f"RWS: {self.min_mld:.2f}mm → {self.max_mld:.2f}mm ({self.rws_percentage}%)"


@dataclass
class PatientInfo:
    """Patient information for reports"""
    patient_id: str
    patient_name: str
    study_date: str
    study_description: Optional[str] = None
    physician: Optional[str] = None


@dataclass
class RWSAnalysisResult:
    """RWS analysis result with risk assessment"""
    success: bool
    timestamp: datetime
    rws_max: float = 0.0
    mld_min: float = 0.0
    mld_max: float = 0.0
    min_frame: int = 0
    max_frame: int = 0
    beat_frames: List[int] = None
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    interpretation: str = ""
    error: Optional[str] = None
    raw_data: Optional[Dict] = None


@dataclass
class RWSAnalysisData:
    """Complete RWS analysis data"""
    success: bool
    timestamp: datetime
    beat_frames: List[int]
    num_frames_analyzed: int
    calibration_factor: float
    rws_result: Optional[RWSResult] = None
    mld_info_by_frame: Dict[int, MLDInfo] = None
    diameter_profiles: Dict[int, np.ndarray] = None
    patient_info: Optional[PatientInfo] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility"""
        if not self.success:
            return {
                'success': False,
                'error': self.error
            }
        
        return {
            'success': True,
            'timestamp': self.timestamp.isoformat(),
            'beat_frames': self.beat_frames,
            'num_frames_analyzed': self.num_frames_analyzed,
            'calibration_factor': self.calibration_factor,
            'rws_at_mld': self.rws_result.rws_percentage,
            'mld_min_value': self.rws_result.min_mld,
            'mld_max_value': self.rws_result.max_mld,
            'mld_min_frame': self.rws_result.min_mld_frame,
            'mld_max_frame': self.rws_result.max_mld_frame,
            'mld_min_index': self.rws_result.min_mld_index,
            'mld_max_index': self.rws_result.max_mld_index,
            'mld_values_by_frame': {
                frame: {
                    'mld_value': info.mld_value,
                    'mld_index': info.mld_index
                }
                for frame, info in self.mld_info_by_frame.items()
            },
            'diameter_profiles': self.diameter_profiles,
            'patient_info': {
                'id': self.patient_info.patient_id,
                'name': self.patient_info.patient_name,
                'study_date': self.patient_info.study_date
            } if self.patient_info else None
        }