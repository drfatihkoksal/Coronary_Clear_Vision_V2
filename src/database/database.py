"""
Database Management and Schema Creation

This module handles SQLite database connection, table creation,
and schema management for the coronary analysis system.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and schema"""

    def __init__(self, db_path: str = "coronary_analysis.db"):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database and create tables if needed"""
        try:
            with self.get_connection() as conn:
                self._create_tables(conn)
                self._create_indexes(conn)
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _create_tables(self, conn: sqlite3.Connection):
        """Create all database tables"""

        # Patients table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                date_of_birth TEXT,
                gender TEXT,
                medical_record_number TEXT UNIQUE,
                phone TEXT,
                email TEXT,
                address TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Studies table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS studies (
                study_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                study_date TIMESTAMP NOT NULL,
                study_description TEXT,
                accession_number TEXT,
                referring_physician TEXT,
                performing_physician TEXT,
                institution TEXT,
                department TEXT,
                modality TEXT DEFAULT 'XA',
                study_instance_uid TEXT UNIQUE,
                series_instance_uid TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
            )
        """
        )

        # Analyses table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                vessel TEXT NOT NULL,
                analysis_date TIMESTAMP NOT NULL,
                frame_numbers TEXT,  -- JSON array
                projection_angle TEXT,
                contrast_volume_ml REAL,
                radiation_dose_mgy REAL,
                operator TEXT,
                notes TEXT,
                status TEXT DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
            )
        """
        )

        # Calibration data table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_data (
                calibration_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                calibration_factor REAL NOT NULL,
                catheter_size_french INTEGER,
                catheter_diameter_mm REAL,
                calibration_method TEXT DEFAULT 'manual',
                confidence_score REAL DEFAULT 1.0,
                point1_x REAL,
                point1_y REAL,
                point2_x REAL,
                point2_y REAL,
                distance_pixels REAL,
                distance_mm REAL,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            )
        """
        )

        # RWS data table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rws_data (
                rws_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                rws_percentage REAL NOT NULL,
                mld_min_mm REAL NOT NULL,
                mld_max_mm REAL NOT NULL,
                mld_variation_mm REAL NOT NULL,
                min_frame_number INTEGER NOT NULL,
                max_frame_number INTEGER NOT NULL,
                min_mld_position INTEGER,
                max_mld_position INTEGER,
                cardiac_phase_min TEXT,
                cardiac_phase_max TEXT,
                risk_level TEXT DEFAULT 'UNKNOWN',
                clinical_interpretation TEXT,
                raw_diameter_data TEXT,  -- JSON string
                outliers_included BOOLEAN DEFAULT 1,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            )
        """
        )

        # QCA data table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qca_data (
                qca_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                vessel_length_mm REAL NOT NULL,
                mean_diameter_mm REAL NOT NULL,
                min_diameter_mm REAL NOT NULL,
                max_diameter_mm REAL NOT NULL,
                stenosis_percentage REAL,
                stenosis_length_mm REAL,
                stenosis_location_mm REAL,
                reference_diameter_mm REAL,
                lesion_diameter_mm REAL,
                centerline_points TEXT,  -- JSON string
                diameter_profile TEXT,  -- JSON string
                area_stenosis_percentage REAL,
                flow_reserve REAL,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            )
        """
        )

        # Frame measurements table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_measurements (
                measurement_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                cardiac_phase TEXT,
                heart_rate REAL,
                diameter_measurements TEXT NOT NULL,  -- JSON array
                mld_value_mm REAL,
                mld_position INTEGER,
                mean_diameter_mm REAL,
                vessel_area_mm2 REAL,
                quality_score REAL DEFAULT 1.0,
                outlier_detected BOOLEAN DEFAULT 0,
                metadata TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            )
        """
        )

        # Analysis snapshots table for versioning
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                snapshot_data TEXT NOT NULL,  -- JSON string
                snapshot_type TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            )
        """
        )

        # Export history table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS export_history (
                export_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                export_format TEXT NOT NULL,
                export_path TEXT NOT NULL,
                included_data TEXT NOT NULL,  -- JSON array
                exported_by TEXT,
                export_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
            )
        """
        )

        # Enhanced Calibration table with all measurement details
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_measurements (
                measurement_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                study_id TEXT NOT NULL,
                measurement_type TEXT NOT NULL,  -- 'manual', 'catheter', 'sphere', 'grid', 'auto_detect', 'dicom'
                
                -- Calibration results
                calibration_factor REAL NOT NULL,  -- pixels per mm
                pixels_per_mm REAL NOT NULL,
                mm_per_pixel REAL NOT NULL,
                
                -- Catheter information
                catheter_size_french INTEGER,
                catheter_diameter_mm REAL,
                catheter_manufacturer TEXT,
                catheter_model TEXT,
                
                -- Manual calibration points
                point1_x REAL,
                point1_y REAL,
                point2_x REAL,
                point2_y REAL,
                measured_distance_pixels REAL,
                known_distance_mm REAL,
                
                -- Calibration object details
                calibration_object_type TEXT,  -- 'catheter', 'sphere', 'grid', 'ruler'
                object_size_mm REAL,
                object_manufacturer TEXT,
                
                -- Quality metrics
                confidence_score REAL DEFAULT 1.0,
                measurement_error_percentage REAL,
                validation_status TEXT,  -- 'validated', 'pending', 'failed'
                validated_by TEXT,
                validation_date TIMESTAMP,
                
                -- Image conditions
                magnification_factor REAL,
                source_to_image_distance_mm REAL,
                source_to_object_distance_mm REAL,
                field_of_view_mm REAL,
                
                -- Method-specific data
                detection_algorithm TEXT,
                edge_detection_method TEXT,
                threshold_value REAL,
                roi_x INTEGER,
                roi_y INTEGER,
                roi_width INTEGER,
                roi_height INTEGER,
                
                -- Metadata
                notes TEXT,
                operator TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
            )
        """
        )

        # DICOM Metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dicom_metadata (
                metadata_id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                
                -- Patient Information (anonymizable)
                patient_name TEXT,
                patient_id TEXT,
                patient_birth_date TEXT,
                patient_sex TEXT,
                patient_age TEXT,
                patient_weight REAL,
                patient_height REAL,
                patient_bmi REAL,
                
                -- Study Information
                study_instance_uid TEXT UNIQUE,
                study_date TEXT,
                study_time TEXT,
                study_description TEXT,
                accession_number TEXT,
                referring_physician TEXT,
                performing_physician TEXT,
                operators_name TEXT,
                institution_name TEXT,
                institution_address TEXT,
                department_name TEXT,
                station_name TEXT,
                
                -- Series Information
                series_instance_uid TEXT,
                series_number INTEGER,
                series_date TEXT,
                series_time TEXT,
                series_description TEXT,
                modality TEXT,
                protocol_name TEXT,
                body_part_examined TEXT,
                patient_position TEXT,
                laterality TEXT,
                
                -- Equipment Information
                manufacturer TEXT,
                manufacturer_model TEXT,
                device_serial_number TEXT,
                software_version TEXT,
                
                -- Image Acquisition Parameters
                kvp REAL,  -- Peak kilovoltage
                exposure_time REAL,
                xray_tube_current REAL,
                exposure REAL,
                exposure_in_microas REAL,
                distance_source_to_detector REAL,
                distance_source_to_patient REAL,
                image_area_dose_product REAL,
                filter_type TEXT,
                focal_spot TEXT,
                collimator_type TEXT,
                collimator_left_edge REAL,
                collimator_right_edge REAL,
                collimator_upper_edge REAL,
                collimator_lower_edge REAL,
                
                -- Image Information
                rows INTEGER,
                columns INTEGER,
                bits_allocated INTEGER,
                bits_stored INTEGER,
                high_bit INTEGER,
                pixel_representation INTEGER,
                samples_per_pixel INTEGER,
                photometric_interpretation TEXT,
                planar_configuration INTEGER,
                
                -- Pixel Spacing and Calibration
                pixel_spacing_row REAL,
                pixel_spacing_column REAL,
                imager_pixel_spacing_row REAL,
                imager_pixel_spacing_column REAL,
                nominal_scanned_pixel_spacing REAL,
                pixel_aspect_ratio REAL,
                
                -- Multi-frame Information
                number_of_frames INTEGER,
                frame_time REAL,
                frame_time_vector TEXT,  -- JSON array
                frame_delay REAL,
                frame_acquisition_datetime TEXT,
                recommended_display_frame_rate REAL,
                cine_rate INTEGER,
                
                -- Angiography Specific
                positioner_primary_angle REAL,
                positioner_secondary_angle REAL,
                positioner_primary_angle_increment REAL,
                positioner_secondary_angle_increment REAL,
                detector_primary_angle REAL,
                detector_secondary_angle REAL,
                table_height REAL,
                table_traverse REAL,
                table_motion TEXT,
                table_vertical_increment REAL,
                table_lateral_increment REAL,
                table_longitudinal_increment REAL,
                
                -- Contrast/Bolus
                contrast_bolus_agent TEXT,
                contrast_bolus_volume REAL,
                contrast_bolus_start_time TEXT,
                contrast_bolus_stop_time TEXT,
                contrast_bolus_total_dose REAL,
                contrast_flow_rate REAL,
                contrast_flow_duration REAL,
                
                -- Window/Level
                window_center REAL,
                window_width REAL,
                window_center_width_explanation TEXT,
                rescale_intercept REAL,
                rescale_slope REAL,
                rescale_type TEXT,
                
                -- Radiation Dose
                dose_area_product REAL,
                entrance_dose REAL,
                entrance_dose_mgy REAL,
                exposed_area REAL,
                radiation_dose REAL,
                radiation_mode TEXT,
                
                -- Cardiac Specific
                heart_rate REAL,
                cardiac_number_of_images INTEGER,
                trigger_time REAL,
                nominal_interval REAL,
                beat_rejection_flag TEXT,
                skip_beats INTEGER,
                heart_rate_measured REAL,
                
                -- View Information
                view_position TEXT,
                view_code_sequence TEXT,
                view_modifier_code_sequence TEXT,
                
                -- ECG Data
                ecg_data_present BOOLEAN DEFAULT 0,
                number_of_waveform_samples INTEGER,
                sampling_frequency REAL,
                
                -- Image Comments and Annotations
                image_comments TEXT,
                lossy_image_compression TEXT,
                lossy_image_compression_ratio REAL,
                derivation_description TEXT,
                
                -- Additional metadata as JSON
                additional_metadata TEXT,  -- JSON for any extra DICOM tags
                
                -- Timestamps
                import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE
            )
        """
        )

        logger.info("Database tables created successfully")

    def _create_indexes(self, conn: sqlite3.Connection):
        """Create database indexes for performance"""

        # Patient indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patients_mrn ON patients(medical_record_number)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(name)")

        # Study indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_patient ON studies(patient_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_date ON studies(study_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_studies_uid ON studies(study_instance_uid)")

        # Analysis indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_study ON analyses(study_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_patient ON analyses(patient_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_type ON analyses(analysis_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_vessel ON analyses(vessel)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_date ON analyses(analysis_date)")

        # Data table indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibration_analysis ON calibration_data(analysis_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rws_analysis ON rws_data(analysis_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rws_risk_level ON rws_data(risk_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rws_percentage ON rws_data(rws_percentage)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qca_analysis ON qca_data(analysis_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qca_frame ON qca_data(analysis_id, frame_number)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qca_stenosis ON qca_data(stenosis_percentage)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_analysis ON frame_measurements(analysis_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_frame ON frame_measurements(analysis_id, frame_number)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_cardiac_phase ON frame_measurements(cardiac_phase)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measurements_outlier ON frame_measurements(outlier_detected)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_analysis ON analysis_snapshots(analysis_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_type ON analysis_snapshots(snapshot_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_export_analysis ON export_history(analysis_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_export_date ON export_history(export_date)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_export_format ON export_history(export_format)"
        )

        # Enhanced calibration table indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibration_measurements_analysis ON calibration_measurements(analysis_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibration_measurements_study ON calibration_measurements(study_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibration_measurements_type ON calibration_measurements(measurement_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_calibration_measurements_date ON calibration_measurements(created_at)"
        )

        # DICOM metadata indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dicom_metadata_study ON dicom_metadata(study_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dicom_metadata_patient ON dicom_metadata(patient_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dicom_metadata_study_uid ON dicom_metadata(study_instance_uid)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dicom_metadata_series_uid ON dicom_metadata(series_instance_uid)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dicom_metadata_modality ON dicom_metadata(modality)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dicom_metadata_date ON dicom_metadata(study_date)"
        )

        logger.info("Database indexes created successfully")

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database

        Args:
            backup_path: Optional custom backup path

        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"coronary_analysis_backup_{timestamp}.db"

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        with self.get_connection() as conn:
            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()

        logger.info(f"Database backed up to {backup_path}")
        return str(backup_path)

    def vacuum(self):
        """Vacuum database to reclaim space and optimize"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed successfully")

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}
        with self.get_connection() as conn:
            cursor = conn.cursor()

            tables = [
                "patients",
                "studies",
                "analyses",
                "calibration_data",
                "rws_data",
                "qca_data",
                "frame_measurements",
                "analysis_snapshots",
                "export_history",
                "calibration_measurements",
                "dicom_metadata",
            ]

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

        return stats
