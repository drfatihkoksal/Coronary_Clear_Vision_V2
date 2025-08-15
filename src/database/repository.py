"""
Repository layer for coronary analysis data access
Provides CRUD operations and query methods for all database entities
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from .database import DatabaseManager
from .models import (
    Patient,
    Study,
    Analysis,
    CalibrationData,
    RWSData,
    QCAData,
    CoronaryVessel,
    AnalysisType,
)

logger = logging.getLogger(__name__)


class CoronaryAnalysisRepository:
    """Repository for coronary analysis data operations"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize repository with database manager

        Args:
            db_manager: Database manager instance. If None, creates new one.
        """
        self.db_manager = db_manager if db_manager else DatabaseManager()

    # ==================== PATIENT OPERATIONS ====================

    def create_patient(self, patient: Patient) -> str:
        """Create new patient record"""
        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO patients (
                    patient_id, name, date_of_birth, gender, 
                    medical_record_number, phone, email, address, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    patient.patient_id,
                    patient.name,
                    patient.date_of_birth,
                    patient.gender,
                    patient.medical_record_number,
                    patient.phone,
                    patient.email,
                    patient.address,
                    patient.notes,
                ),
            )
            logger.info(f"Created patient: {patient.patient_id}")
            return patient.patient_id

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get patient by ID"""
        with self.db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()

            if row:
                return Patient(
                    patient_id=row["patient_id"],
                    name=row["name"],
                    date_of_birth=row["date_of_birth"],
                    gender=row["gender"],
                    medical_record_number=row["medical_record_number"],
                    phone=row["phone"],
                    email=row["email"],
                    address=row["address"],
                    notes=row["notes"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
            return None

    def get_patients_by_name(self, name_pattern: str) -> List[Patient]:
        """Search patients by name pattern"""
        with self.db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM patients WHERE name LIKE ? ORDER BY name", (f"%{name_pattern}%",)
            ).fetchall()

            return [
                Patient(
                    patient_id=row["patient_id"],
                    name=row["name"],
                    date_of_birth=row["date_of_birth"],
                    gender=row["gender"],
                    medical_record_number=row["medical_record_number"],
                    phone=row["phone"],
                    email=row["email"],
                    address=row["address"],
                    notes=row["notes"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

    def update_patient(self, patient: Patient) -> bool:
        """Update patient record"""
        patient.updated_at = datetime.now()

        with self.db_manager.get_connection() as conn:
            result = conn.execute(
                """
                UPDATE patients SET 
                    name = ?, date_of_birth = ?, gender = ?,
                    medical_record_number = ?, phone = ?, email = ?,
                    address = ?, notes = ?, updated_at = ?
                WHERE patient_id = ?
            """,
                (
                    patient.name,
                    patient.date_of_birth,
                    patient.gender,
                    patient.medical_record_number,
                    patient.phone,
                    patient.email,
                    patient.address,
                    patient.notes,
                    patient.updated_at.isoformat(),
                    patient.patient_id,
                ),
            )

            success = result.rowcount > 0
            if success:
                logger.info(f"Updated patient: {patient.patient_id}")
            return success

    # ==================== STUDY OPERATIONS ====================

    def create_study(self, study: Study) -> str:
        """Create new study record"""
        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO studies (
                    study_id, patient_id, study_date, study_description,
                    accession_number, referring_physician, performing_physician,
                    institution, department, modality, study_instance_uid,
                    series_instance_uid, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    study.study_id,
                    study.patient_id,
                    study.study_date.isoformat(),
                    study.study_description,
                    study.accession_number,
                    study.referring_physician,
                    study.performing_physician,
                    study.institution,
                    study.department,
                    study.modality,
                    study.study_instance_uid,
                    study.series_instance_uid,
                    study.notes,
                ),
            )
            logger.info(f"Created study: {study.study_id}")
            return study.study_id

    def get_studies_by_patient(self, patient_id: str) -> List[Study]:
        """Get all studies for a patient"""
        with self.db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM studies WHERE patient_id = ? ORDER BY study_date DESC", (patient_id,)
            ).fetchall()

            return [
                Study(
                    study_id=row["study_id"],
                    patient_id=row["patient_id"],
                    study_date=datetime.fromisoformat(row["study_date"]),
                    study_description=row["study_description"],
                    accession_number=row["accession_number"],
                    referring_physician=row["referring_physician"],
                    performing_physician=row["performing_physician"],
                    institution=row["institution"],
                    department=row["department"],
                    modality=row["modality"],
                    study_instance_uid=row["study_instance_uid"],
                    series_instance_uid=row["series_instance_uid"],
                    notes=row["notes"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    # ==================== ANALYSIS OPERATIONS ====================

    def create_analysis(self, analysis: Analysis) -> str:
        """Create new analysis record"""
        frame_numbers_json = json.dumps(analysis.frame_numbers)

        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    analysis_id, study_id, patient_id, analysis_type, vessel,
                    analysis_date, frame_numbers, projection_angle,
                    contrast_volume_ml, radiation_dose_mgy, operator, notes, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    analysis.analysis_id,
                    analysis.study_id,
                    analysis.patient_id,
                    analysis.analysis_type.value,
                    analysis.vessel.value,
                    analysis.analysis_date.isoformat(),
                    frame_numbers_json,
                    analysis.projection_angle,
                    analysis.contrast_volume_ml,
                    analysis.radiation_dose_mgy,
                    analysis.operator,
                    analysis.notes,
                    analysis.status,
                ),
            )
            logger.info(f"Created analysis: {analysis.analysis_id}")
            return analysis.analysis_id

    def get_analyses_by_study(self, study_id: str) -> List[Analysis]:
        """Get all analyses for a study"""
        with self.db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses WHERE study_id = ? ORDER BY analysis_date DESC", (study_id,)
            ).fetchall()

            return [
                Analysis(
                    analysis_id=row["analysis_id"],
                    study_id=row["study_id"],
                    patient_id=row["patient_id"],
                    analysis_type=AnalysisType(row["analysis_type"]),
                    vessel=CoronaryVessel(row["vessel"]),
                    analysis_date=datetime.fromisoformat(row["analysis_date"]),
                    frame_numbers=json.loads(row["frame_numbers"]),
                    projection_angle=row["projection_angle"],
                    contrast_volume_ml=row["contrast_volume_ml"],
                    radiation_dose_mgy=row["radiation_dose_mgy"],
                    operator=row["operator"],
                    notes=row["notes"],
                    status=row["status"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    # ==================== RWS DATA OPERATIONS ====================

    def create_rws_data(self, rws_data: RWSData) -> str:
        """Create new RWS data record"""
        metadata_json = json.dumps(rws_data.metadata) if rws_data.metadata else None

        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO rws_data (
                    rws_id, analysis_id, rws_percentage, mld_min_mm, mld_max_mm,
                    mld_variation_mm, min_frame_number, max_frame_number,
                    min_mld_position, max_mld_position, cardiac_phase_min,
                    cardiac_phase_max, risk_level, clinical_interpretation,
                    raw_diameter_data, outliers_included, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rws_data.rws_id,
                    rws_data.analysis_id,
                    rws_data.rws_percentage,
                    rws_data.mld_min_mm,
                    rws_data.mld_max_mm,
                    rws_data.mld_variation_mm,
                    rws_data.min_frame_number,
                    rws_data.max_frame_number,
                    rws_data.min_mld_position,
                    rws_data.max_mld_position,
                    rws_data.cardiac_phase_min,
                    rws_data.cardiac_phase_max,
                    rws_data.risk_level,
                    rws_data.clinical_interpretation,
                    rws_data.raw_diameter_data,
                    rws_data.outliers_included,
                    metadata_json,
                ),
            )
            logger.info(f"Created RWS data: {rws_data.rws_id}")
            return rws_data.rws_id

    def get_rws_data_by_analysis(self, analysis_id: str) -> Optional[RWSData]:
        """Get RWS data for an analysis"""
        with self.db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM rws_data WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()

            if row:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None

                return RWSData(
                    rws_id=row["rws_id"],
                    analysis_id=row["analysis_id"],
                    rws_percentage=row["rws_percentage"],
                    mld_min_mm=row["mld_min_mm"],
                    mld_max_mm=row["mld_max_mm"],
                    mld_variation_mm=row["mld_variation_mm"],
                    min_frame_number=row["min_frame_number"],
                    max_frame_number=row["max_frame_number"],
                    min_mld_position=row["min_mld_position"],
                    max_mld_position=row["max_mld_position"],
                    cardiac_phase_min=row["cardiac_phase_min"],
                    cardiac_phase_max=row["cardiac_phase_max"],
                    risk_level=row["risk_level"],
                    clinical_interpretation=row["clinical_interpretation"],
                    raw_diameter_data=row["raw_diameter_data"],
                    outliers_included=row["outliers_included"],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            return None

    # ==================== CALIBRATION OPERATIONS ====================

    def create_calibration_data(self, calibration: CalibrationData) -> str:
        """Create new calibration data record"""
        metadata_json = json.dumps(calibration.metadata) if calibration.metadata else None

        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO calibration_data (
                    calibration_id, analysis_id, calibration_factor,
                    catheter_size_french, catheter_diameter_mm, calibration_method,
                    confidence_score, point1_x, point1_y, point2_x, point2_y,
                    distance_pixels, distance_mm, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    calibration.calibration_id,
                    calibration.analysis_id,
                    calibration.calibration_factor,
                    calibration.catheter_size_french,
                    calibration.catheter_diameter_mm,
                    calibration.calibration_method,
                    calibration.confidence_score,
                    calibration.point1_x,
                    calibration.point1_y,
                    calibration.point2_x,
                    calibration.point2_y,
                    calibration.distance_pixels,
                    calibration.distance_mm,
                    metadata_json,
                ),
            )
            logger.info(f"Created calibration data: {calibration.calibration_id}")
            return calibration.calibration_id

    def get_calibration_by_analysis(self, analysis_id: str) -> Optional[CalibrationData]:
        """Get calibration data for an analysis"""
        with self.db_manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM calibration_data WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()

            if row:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None

                return CalibrationData(
                    calibration_id=row["calibration_id"],
                    analysis_id=row["analysis_id"],
                    calibration_factor=row["calibration_factor"],
                    catheter_size_french=row["catheter_size_french"],
                    catheter_diameter_mm=row["catheter_diameter_mm"],
                    calibration_method=row["calibration_method"],
                    confidence_score=row["confidence_score"],
                    point1_x=row["point1_x"],
                    point1_y=row["point1_y"],
                    point2_x=row["point2_x"],
                    point2_y=row["point2_y"],
                    distance_pixels=row["distance_pixels"],
                    distance_mm=row["distance_mm"],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            return None

    # ==================== QCA DATA OPERATIONS ====================

    def create_qca_data(self, qca_data: QCAData) -> str:
        """Create new QCA data record"""
        centerline_json = (
            json.dumps(qca_data.centerline_points) if qca_data.centerline_points else None
        )
        diameter_profile_json = (
            json.dumps(qca_data.diameter_profile) if qca_data.diameter_profile else None
        )
        metadata_json = json.dumps(qca_data.metadata) if qca_data.metadata else None

        with self.db_manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO qca_data (
                    qca_id, analysis_id, frame_number, vessel_length_mm,
                    mean_diameter_mm, min_diameter_mm, max_diameter_mm,
                    stenosis_percentage, stenosis_length_mm, stenosis_location_mm,
                    reference_diameter_mm, lesion_diameter_mm, centerline_points,
                    diameter_profile, area_stenosis_percentage, flow_reserve, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    qca_data.qca_id,
                    qca_data.analysis_id,
                    qca_data.frame_number,
                    qca_data.vessel_length_mm,
                    qca_data.mean_diameter_mm,
                    qca_data.min_diameter_mm,
                    qca_data.max_diameter_mm,
                    qca_data.stenosis_percentage,
                    qca_data.stenosis_length_mm,
                    qca_data.stenosis_location_mm,
                    qca_data.reference_diameter_mm,
                    qca_data.lesion_diameter_mm,
                    centerline_json,
                    diameter_profile_json,
                    qca_data.area_stenosis_percentage,
                    qca_data.flow_reserve,
                    metadata_json,
                ),
            )
            logger.info(f"Created QCA data: {qca_data.qca_id}")
            return qca_data.qca_id

    def get_qca_data_by_analysis(self, analysis_id: str) -> List[QCAData]:
        """Get QCA data for an analysis"""
        with self.db_manager.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM qca_data WHERE analysis_id = ? ORDER BY frame_number", (analysis_id,)
            ).fetchall()

            qca_data_list = []
            for row in rows:
                centerline_points = (
                    json.loads(row["centerline_points"]) if row["centerline_points"] else None
                )
                diameter_profile = (
                    json.loads(row["diameter_profile"]) if row["diameter_profile"] else None
                )
                metadata = json.loads(row["metadata"]) if row["metadata"] else None

                qca_data = QCAData(
                    qca_id=row["qca_id"],
                    analysis_id=row["analysis_id"],
                    frame_number=row["frame_number"],
                    vessel_length_mm=row["vessel_length_mm"],
                    mean_diameter_mm=row["mean_diameter_mm"],
                    min_diameter_mm=row["min_diameter_mm"],
                    max_diameter_mm=row["max_diameter_mm"],
                    stenosis_percentage=row["stenosis_percentage"],
                    stenosis_length_mm=row["stenosis_length_mm"],
                    stenosis_location_mm=row["stenosis_location_mm"],
                    reference_diameter_mm=row["reference_diameter_mm"],
                    lesion_diameter_mm=row["lesion_diameter_mm"],
                    centerline_points=centerline_points,
                    diameter_profile=diameter_profile,
                    area_stenosis_percentage=row["area_stenosis_percentage"],
                    flow_reserve=row["flow_reserve"],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                qca_data_list.append(qca_data)

            return qca_data_list

    # ==================== UTILITY METHODS ====================

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = self.db_manager.get_statistics()

        # Add additional analysis statistics
        with self.db_manager.get_connection() as conn:
            # Analysis type distribution
            analysis_types = conn.execute(
                """
                SELECT analysis_type, COUNT(*) as count 
                FROM analyses 
                GROUP BY analysis_type
            """
            ).fetchall()
            stats["analysis_types"] = {row["analysis_type"]: row["count"] for row in analysis_types}

            # Vessel distribution
            vessels = conn.execute(
                """
                SELECT vessel, COUNT(*) as count 
                FROM analyses 
                GROUP BY vessel
            """
            ).fetchall()
            stats["vessels"] = {row["vessel"]: row["count"] for row in vessels}

            # Recent activity (last 30 days)
            recent_analyses = conn.execute(
                """
                SELECT COUNT(*) as count 
                FROM analyses 
                WHERE analysis_date > datetime('now', '-30 days')
            """
            ).fetchone()
            stats["recent_analyses"] = recent_analyses["count"]

        return stats

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create database backup"""
        return self.db_manager.backup_database(backup_path)

    def vacuum_database(self):
        """Vacuum database to optimize performance"""
        self.db_manager.vacuum()
