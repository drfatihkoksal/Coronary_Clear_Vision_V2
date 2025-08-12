"""
Database Service for Coronary Analysis Application
Provides high-level database operations for the UI layer
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from ..database import (
    DatabaseManager, CoronaryAnalysisRepository,
    Patient, Study, Analysis, CalibrationData, RWSData, QCAData,
    CoronaryVessel, AnalysisType
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database service for coronary analysis application"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database service
        
        Args:
            db_path: Optional custom database path
        """
        self.db_manager = DatabaseManager(db_path or "coronary_analysis.db")
        self.repository = CoronaryAnalysisRepository(self.db_manager)
        self._current_patient_id: Optional[str] = None
        self._current_study_id: Optional[str] = None
        
        logger.info("Database service initialized")
    
    def set_current_patient(self, patient_name: str, 
                          medical_record_number: Optional[str] = None,
                          date_of_birth: Optional[str] = None,
                          gender: Optional[str] = None) -> str:
        """
        Set or create current patient for analysis session
        
        Args:
            patient_name: Patient name
            medical_record_number: Medical record number
            date_of_birth: Date of birth
            gender: Patient gender
            
        Returns:
            Patient ID
        """
        try:
            # Try to find existing patient by MRN or name
            if medical_record_number:
                with self.db_manager.get_connection() as conn:
                    row = conn.execute(
                        "SELECT patient_id FROM patients WHERE medical_record_number = ?", 
                        (medical_record_number,)
                    ).fetchone()
                    if row:
                        self._current_patient_id = row['patient_id']
                        logger.info(f"Found existing patient by MRN: {self._current_patient_id}")
                        return self._current_patient_id
            
            # Create new patient
            patient = Patient(
                patient_id=str(uuid.uuid4()),
                name=patient_name,
                medical_record_number=medical_record_number,
                date_of_birth=date_of_birth,
                gender=gender
            )
            
            self._current_patient_id = self.repository.create_patient(patient)
            logger.info(f"Created new patient: {self._current_patient_id}")
            return self._current_patient_id
            
        except Exception as e:
            logger.error(f"Failed to set current patient: {e}")
            raise
    
    def set_current_study(self, study_description: str = "Coronary Angiography",
                         accession_number: Optional[str] = None,
                         study_instance_uid: Optional[str] = None,
                         study_date: Optional[datetime] = None) -> str:
        """
        Set or create current study for analysis session
        
        Args:
            study_description: Description of the study
            accession_number: Study accession number
            study_instance_uid: DICOM study instance UID
            study_date: Study date
            
        Returns:
            Study ID
        """
        if not self._current_patient_id:
            raise ValueError("No current patient set. Call set_current_patient first.")
        
        try:
            # Try to find existing study by UID
            if study_instance_uid:
                with self.db_manager.get_connection() as conn:
                    row = conn.execute(
                        "SELECT study_id FROM studies WHERE study_instance_uid = ?", 
                        (study_instance_uid,)
                    ).fetchone()
                    if row:
                        self._current_study_id = row['study_id']
                        logger.info(f"Found existing study by UID: {self._current_study_id}")
                        return self._current_study_id
            
            # Create new study
            study = Study(
                study_id=str(uuid.uuid4()),
                patient_id=self._current_patient_id,
                study_date=study_date or datetime.now(),
                study_description=study_description,
                accession_number=accession_number,
                study_instance_uid=study_instance_uid,
                modality="XA"
            )
            
            self._current_study_id = self.repository.create_study(study)
            logger.info(f"Created new study: {self._current_study_id}")
            return self._current_study_id
            
        except Exception as e:
            logger.error(f"Failed to set current study: {e}")
            raise
    
    def save_calibration_data(self, calibration_factor: float,
                            catheter_size_french: Optional[int] = None,
                            catheter_diameter_mm: Optional[float] = None,
                            point1_x: Optional[float] = None,
                            point1_y: Optional[float] = None,
                            point2_x: Optional[float] = None,
                            point2_y: Optional[float] = None,
                            method: str = "manual",
                            confidence_score: float = 1.0) -> str:
        """
        Save calibration data for current session
        
        Args:
            calibration_factor: Pixels per mm calibration factor
            catheter_size_french: Catheter size in French
            catheter_diameter_mm: Catheter diameter in mm
            point1_x, point1_y: First calibration point
            point2_x, point2_y: Second calibration point
            method: Calibration method
            confidence_score: Confidence score
            
        Returns:
            Analysis ID for the calibration
        """
        if not self._current_study_id or not self._current_patient_id:
            raise ValueError("No current study/patient set. Call set_current_study first.")
        
        try:
            # Create calibration analysis record
            analysis = Analysis(
                analysis_id=str(uuid.uuid4()),
                study_id=self._current_study_id,
                patient_id=self._current_patient_id,
                analysis_type=AnalysisType.CALIBRATION,
                vessel=CoronaryVessel.UNKNOWN,
                analysis_date=datetime.now(),
                frame_numbers=[0],  # Calibration typically on a single frame
                operator="System",
                notes=f"Calibration method: {method}"
            )
            
            analysis_id = self.repository.create_analysis(analysis)
            
            # Create calibration data record
            calibration_data = CalibrationData(
                calibration_id=str(uuid.uuid4()),
                analysis_id=analysis_id,
                calibration_factor=calibration_factor,
                catheter_size_french=catheter_size_french,
                catheter_diameter_mm=catheter_diameter_mm,
                calibration_method=method,
                confidence_score=confidence_score,
                point1_x=point1_x,
                point1_y=point1_y,
                point2_x=point2_x,
                point2_y=point2_y,
                distance_pixels=None,  # Can be calculated if points provided
                distance_mm=None
            )
            
            # Calculate distance if points provided
            if all(v is not None for v in [point1_x, point1_y, point2_x, point2_y]):
                import math
                calibration_data.distance_pixels = math.sqrt(
                    (point2_x - point1_x)**2 + (point2_y - point1_y)**2
                )
                if calibration_factor > 0:
                    calibration_data.distance_mm = calibration_data.distance_pixels / calibration_factor
            
            self.repository.create_calibration_data(calibration_data)
            
            logger.info(f"Saved calibration data for analysis: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")
            raise
    
    def save_rws_analysis(self, vessel: CoronaryVessel,
                         rws_percentage: float,
                         mld_min_mm: float,
                         mld_max_mm: float,
                         min_frame_number: int,
                         max_frame_number: int,
                         frame_numbers: List[int],
                         cardiac_phase_min: Optional[str] = None,
                         cardiac_phase_max: Optional[str] = None,
                         raw_diameter_data: Optional[Dict] = None,
                         clinical_interpretation: Optional[str] = None,
                         projection_angle: Optional[str] = None,
                         operator: Optional[str] = None) -> str:
        """
        Save RWS analysis results
        
        Args:
            vessel: Analyzed vessel
            rws_percentage: RWS percentage
            mld_min_mm: Minimum lumen diameter
            mld_max_mm: Maximum lumen diameter
            min_frame_number: Frame with minimum diameter
            max_frame_number: Frame with maximum diameter
            frame_numbers: List of analyzed frames
            cardiac_phase_min: Cardiac phase at minimum
            cardiac_phase_max: Cardiac phase at maximum
            raw_diameter_data: Raw diameter measurements
            clinical_interpretation: Clinical interpretation
            projection_angle: Projection angle
            operator: Analysis operator
            
        Returns:
            Analysis ID
        """
        if not self._current_study_id or not self._current_patient_id:
            raise ValueError("No current study/patient set. Call set_current_study first.")
        
        try:
            # Create RWS analysis record
            analysis = Analysis(
                analysis_id=str(uuid.uuid4()),
                study_id=self._current_study_id,
                patient_id=self._current_patient_id,
                analysis_type=AnalysisType.RWS,
                vessel=vessel,
                analysis_date=datetime.now(),
                frame_numbers=frame_numbers,
                projection_angle=projection_angle,
                operator=operator or "User",
                notes=f"RWS Analysis - {vessel.value}"
            )
            
            analysis_id = self.repository.create_analysis(analysis)
            
            # Determine risk level
            risk_level = "LOW"
            if rws_percentage > 15:
                risk_level = "HIGH"
            elif rws_percentage > 10:
                risk_level = "MODERATE"
            
            # Create RWS data record
            rws_data = RWSData(
                rws_id=str(uuid.uuid4()),
                analysis_id=analysis_id,
                rws_percentage=rws_percentage,
                mld_min_mm=mld_min_mm,
                mld_max_mm=mld_max_mm,
                mld_variation_mm=mld_max_mm - mld_min_mm,
                min_frame_number=min_frame_number,
                max_frame_number=max_frame_number,
                cardiac_phase_min=cardiac_phase_min,
                cardiac_phase_max=cardiac_phase_max,
                risk_level=risk_level,
                clinical_interpretation=clinical_interpretation,
                raw_diameter_data=raw_diameter_data,
                outliers_included=True
            )
            
            self.repository.create_rws_data(rws_data)
            
            logger.info(f"Saved RWS analysis for vessel {vessel.value}: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to save RWS analysis: {e}")
            raise
    
    def save_qca_analysis(self, vessel: CoronaryVessel,
                         frame_number: int,
                         vessel_length_mm: float,
                         mean_diameter_mm: float,
                         min_diameter_mm: float,
                         max_diameter_mm: float,
                         stenosis_percentage: Optional[float] = None,
                         centerline_points: Optional[List] = None,
                         diameter_profile: Optional[List] = None,
                         projection_angle: Optional[str] = None,
                         operator: Optional[str] = None) -> str:
        """
        Save QCA analysis results
        
        Args:
            vessel: Analyzed vessel
            frame_number: Analyzed frame number
            vessel_length_mm: Vessel length in mm
            mean_diameter_mm: Mean diameter in mm
            min_diameter_mm: Minimum diameter in mm
            max_diameter_mm: Maximum diameter in mm
            stenosis_percentage: Stenosis percentage if applicable
            centerline_points: Centerline points
            diameter_profile: Diameter profile along vessel
            projection_angle: Projection angle
            operator: Analysis operator
            
        Returns:
            Analysis ID
        """
        if not self._current_study_id or not self._current_patient_id:
            raise ValueError("No current study/patient set. Call set_current_study first.")
        
        try:
            # Create QCA analysis record
            analysis = Analysis(
                analysis_id=str(uuid.uuid4()),
                study_id=self._current_study_id,
                patient_id=self._current_patient_id,
                analysis_type=AnalysisType.QCA,
                vessel=vessel,
                analysis_date=datetime.now(),
                frame_numbers=[frame_number],
                projection_angle=projection_angle,
                operator=operator or "User",
                notes=f"QCA Analysis - {vessel.value} Frame {frame_number}"
            )
            
            analysis_id = self.repository.create_analysis(analysis)
            
            # Create QCA data record
            qca_data = QCAData(
                qca_id=str(uuid.uuid4()),
                analysis_id=analysis_id,
                frame_number=frame_number,
                vessel_length_mm=vessel_length_mm,
                mean_diameter_mm=mean_diameter_mm,
                min_diameter_mm=min_diameter_mm,
                max_diameter_mm=max_diameter_mm,
                stenosis_percentage=stenosis_percentage,
                centerline_points=centerline_points,
                diameter_profile=diameter_profile
            )
            
            self.repository.create_qca_data(qca_data)
            
            logger.info(f"Saved QCA analysis for vessel {vessel.value} frame {frame_number}: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to save QCA analysis: {e}")
            raise
    
    def get_patient_analyses(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all analyses for a patient
        
        Args:
            patient_id: Patient ID (uses current if not provided)
            
        Returns:
            List of analysis summaries
        """
        target_patient_id = patient_id or self._current_patient_id
        if not target_patient_id:
            return []
        
        try:
            studies = self.repository.get_studies_by_patient(target_patient_id)
            all_analyses = []
            
            for study in studies:
                analyses = self.repository.get_analyses_by_study(study.study_id)
                for analysis in analyses:
                    summary = {
                        'analysis_id': analysis.analysis_id,
                        'study_id': analysis.study_id,
                        'analysis_type': analysis.analysis_type.value,
                        'vessel': analysis.vessel.value,
                        'analysis_date': analysis.analysis_date.isoformat(),
                        'study_description': study.study_description,
                        'study_date': study.study_date.isoformat()
                    }
                    all_analyses.append(summary)
            
            return sorted(all_analyses, key=lambda x: x['analysis_date'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get patient analyses: {e}")
            return []
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            return self.repository.get_database_statistics()
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {}
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create database backup"""
        try:
            return self.repository.backup_database(backup_path)
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise
    
    @property
    def current_patient_id(self) -> Optional[str]:
        """Get current patient ID"""
        return self._current_patient_id
    
    @property
    def current_study_id(self) -> Optional[str]:
        """Get current study ID"""
        return self._current_study_id