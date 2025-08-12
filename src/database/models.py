"""
Database Models for Coronary Analysis Data Storage

This module defines the database schema for storing patient, analysis, 
and measurement data with full support for multiple analyses per patient,
different coronary vessels, and comprehensive data tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json
import uuid


class CoronaryVessel(Enum):
    """Coronary vessel types"""
    LMCA = "LMCA"  # Left Main Coronary Artery
    LAD = "LAD"    # Left Anterior Descending
    CX = "CX"      # Circumflex
    RCA = "RCA"    # Right Coronary Artery
    DIAGONAL = "DIAGONAL"
    OBTUSE_MARGINAL = "OBTUSE_MARGINAL"
    PDA = "PDA"    # Posterior Descending Artery
    PLV = "PLV"    # Posterior Left Ventricular
    RAMUS = "RAMUS"
    UNKNOWN = "UNKNOWN"


class AnalysisType(Enum):
    """Types of analysis performed"""
    RWS = "RWS"
    QCA = "QCA"
    CALIBRATION = "CALIBRATION"
    COMBINED = "COMBINED"


@dataclass
class Patient:
    """Patient information model"""
    patient_id: str
    name: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    medical_record_number: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.patient_id:
            self.patient_id = str(uuid.uuid4())


@dataclass
class Study:
    """Study/Examination information"""
    study_id: str
    patient_id: str
    study_date: datetime
    study_description: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None
    performing_physician: Optional[str] = None
    institution: Optional[str] = None
    department: Optional[str] = None
    modality: str = "XA"  # X-Ray Angiography
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.study_id:
            self.study_id = str(uuid.uuid4())


@dataclass
class Analysis:
    """Individual analysis record"""
    analysis_id: str
    study_id: str
    patient_id: str
    analysis_type: AnalysisType
    vessel: CoronaryVessel
    analysis_date: datetime
    frame_numbers: List[int]
    projection_angle: Optional[str] = None
    contrast_volume_ml: Optional[float] = None
    radiation_dose_mgy: Optional[float] = None
    operator: Optional[str] = None
    notes: Optional[str] = None
    status: str = "completed"
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.analysis_id:
            self.analysis_id = str(uuid.uuid4())


@dataclass
class CalibrationData:
    """Calibration data storage"""
    calibration_id: str
    analysis_id: str
    calibration_factor: float  # pixels per mm
    catheter_size_french: Optional[int] = None
    catheter_diameter_mm: Optional[float] = None
    calibration_method: str = "manual"
    confidence_score: float = 1.0
    point1_x: Optional[float] = None
    point1_y: Optional[float] = None
    point2_x: Optional[float] = None
    point2_y: Optional[float] = None
    distance_pixels: Optional[float] = None
    distance_mm: Optional[float] = None
    metadata: Optional[str] = None  # JSON string
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.calibration_id:
            self.calibration_id = str(uuid.uuid4())
        if self.metadata and isinstance(self.metadata, dict):
            self.metadata = json.dumps(self.metadata)


@dataclass
class RWSData:
    """RWS (Radial Wall Strain) analysis data"""
    rws_id: str
    analysis_id: str
    rws_percentage: float
    mld_min_mm: float  # Minimum Lumen Diameter
    mld_max_mm: float  # Maximum Lumen Diameter
    mld_variation_mm: float
    min_frame_number: int
    max_frame_number: int
    min_mld_position: Optional[int] = None
    max_mld_position: Optional[int] = None
    cardiac_phase_min: Optional[str] = None  # systole/diastole
    cardiac_phase_max: Optional[str] = None
    risk_level: str = "UNKNOWN"  # LOW/MODERATE/HIGH
    clinical_interpretation: Optional[str] = None
    raw_diameter_data: Optional[str] = None  # JSON string of diameter profiles
    outliers_included: bool = True
    metadata: Optional[str] = None  # JSON string
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.rws_id:
            self.rws_id = str(uuid.uuid4())
        if self.raw_diameter_data and not isinstance(self.raw_diameter_data, str):
            self.raw_diameter_data = json.dumps(self.raw_diameter_data)
        if self.metadata and isinstance(self.metadata, dict):
            self.metadata = json.dumps(self.metadata)


@dataclass
class QCAData:
    """QCA (Quantitative Coronary Analysis) data"""
    qca_id: str
    analysis_id: str
    frame_number: int
    vessel_length_mm: float
    mean_diameter_mm: float
    min_diameter_mm: float
    max_diameter_mm: float
    stenosis_percentage: Optional[float] = None
    stenosis_length_mm: Optional[float] = None
    stenosis_location_mm: Optional[float] = None
    reference_diameter_mm: Optional[float] = None
    lesion_diameter_mm: Optional[float] = None
    centerline_points: Optional[str] = None  # JSON string of points
    diameter_profile: Optional[str] = None  # JSON string of measurements
    area_stenosis_percentage: Optional[float] = None
    flow_reserve: Optional[float] = None
    metadata: Optional[str] = None  # JSON string
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.qca_id:
            self.qca_id = str(uuid.uuid4())
        if self.centerline_points and not isinstance(self.centerline_points, str):
            self.centerline_points = json.dumps(self.centerline_points)
        if self.diameter_profile and not isinstance(self.diameter_profile, str):
            self.diameter_profile = json.dumps(self.diameter_profile)
        if self.metadata and isinstance(self.metadata, dict):
            self.metadata = json.dumps(self.metadata)


@dataclass
class FrameMeasurement:
    """Individual frame measurements"""
    measurement_id: str
    analysis_id: str
    frame_number: int
    diameter_measurements: str  # JSON array of diameter values along vessel
    cardiac_phase: Optional[str] = None  # systole/diastole
    heart_rate: Optional[float] = None
    mld_value_mm: Optional[float] = None
    mld_position: Optional[int] = None
    mean_diameter_mm: Optional[float] = None
    vessel_area_mm2: Optional[float] = None
    quality_score: float = 1.0
    outlier_detected: bool = False
    metadata: Optional[str] = None  # JSON string
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.measurement_id:
            self.measurement_id = str(uuid.uuid4())
        if not isinstance(self.diameter_measurements, str):
            self.diameter_measurements = json.dumps(self.diameter_measurements)
        if self.metadata and isinstance(self.metadata, dict):
            self.metadata = json.dumps(self.metadata)


@dataclass
class AnalysisSnapshot:
    """Snapshot of analysis for data versioning and recovery"""
    snapshot_id: str
    analysis_id: str
    snapshot_data: str  # JSON string of complete analysis state
    snapshot_type: str  # 'auto' or 'manual'
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.snapshot_id:
            self.snapshot_id = str(uuid.uuid4())
        if not isinstance(self.snapshot_data, str):
            self.snapshot_data = json.dumps(self.snapshot_data)


@dataclass
class ExportHistory:
    """Track data exports for audit trail"""
    export_id: str
    analysis_id: str
    export_format: str  # 'xlsx', 'csv', 'json', 'pdf'
    export_path: str
    included_data: str  # JSON array of data types included
    exported_by: Optional[str] = None
    export_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.export_id:
            self.export_id = str(uuid.uuid4())
        if not isinstance(self.included_data, str):
            self.included_data = json.dumps(self.included_data)


@dataclass
class CalibrationMeasurement:
    """Enhanced calibration measurement data"""
    measurement_id: str
    analysis_id: str
    study_id: str
    measurement_type: str  # 'manual', 'catheter', 'sphere', 'grid', 'auto_detect', 'dicom'
    
    # Calibration results
    calibration_factor: float  # pixels per mm
    pixels_per_mm: Optional[float] = None  # Will be set from calibration_factor if None
    mm_per_pixel: Optional[float] = None   # Will be set from calibration_factor if None
    
    # Catheter information
    catheter_size_french: Optional[int] = None
    catheter_diameter_mm: Optional[float] = None
    catheter_manufacturer: Optional[str] = None
    catheter_model: Optional[str] = None
    
    # Manual calibration points
    point1_x: Optional[float] = None
    point1_y: Optional[float] = None
    point2_x: Optional[float] = None
    point2_y: Optional[float] = None
    measured_distance_pixels: Optional[float] = None
    known_distance_mm: Optional[float] = None
    
    # Calibration object details
    calibration_object_type: Optional[str] = None  # 'catheter', 'sphere', 'grid', 'ruler'
    object_size_mm: Optional[float] = None
    object_manufacturer: Optional[str] = None
    
    # Quality metrics
    confidence_score: float = 1.0
    measurement_error_percentage: Optional[float] = None
    validation_status: str = 'pending'  # 'validated', 'pending', 'failed'
    validated_by: Optional[str] = None
    validation_date: Optional[datetime] = None
    
    # Image conditions
    magnification_factor: Optional[float] = None
    source_to_image_distance_mm: Optional[float] = None
    source_to_object_distance_mm: Optional[float] = None
    field_of_view_mm: Optional[float] = None
    
    # Method-specific data
    detection_algorithm: Optional[str] = None
    edge_detection_method: Optional[str] = None
    threshold_value: Optional[float] = None
    roi_x: Optional[int] = None
    roi_y: Optional[int] = None
    roi_width: Optional[int] = None
    roi_height: Optional[int] = None
    
    # Metadata
    notes: Optional[str] = None
    operator: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.measurement_id:
            self.measurement_id = str(uuid.uuid4())
        
        # Ensure consistency between calibration_factor, pixels_per_mm, and mm_per_pixel
        if self.calibration_factor:
            if self.pixels_per_mm is None:
                self.pixels_per_mm = self.calibration_factor
            if self.mm_per_pixel is None:
                self.mm_per_pixel = 1.0 / self.calibration_factor if self.calibration_factor != 0 else 0.0


@dataclass
class DicomMetadata:
    """Complete DICOM metadata storage"""
    metadata_id: str
    study_id: str
    file_path: str
    file_name: str
    
    # Patient Information
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    patient_age: Optional[str] = None
    patient_weight: Optional[float] = None
    patient_height: Optional[float] = None
    patient_bmi: Optional[float] = None
    
    # Study Information
    study_instance_uid: Optional[str] = None
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None
    performing_physician: Optional[str] = None
    operators_name: Optional[str] = None
    institution_name: Optional[str] = None
    institution_address: Optional[str] = None
    department_name: Optional[str] = None
    station_name: Optional[str] = None
    
    # Series Information
    series_instance_uid: Optional[str] = None
    series_number: Optional[int] = None
    series_date: Optional[str] = None
    series_time: Optional[str] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    protocol_name: Optional[str] = None
    body_part_examined: Optional[str] = None
    patient_position: Optional[str] = None
    laterality: Optional[str] = None
    
    # Equipment Information
    manufacturer: Optional[str] = None
    manufacturer_model: Optional[str] = None
    device_serial_number: Optional[str] = None
    software_version: Optional[str] = None
    
    # Image Acquisition Parameters
    kvp: Optional[float] = None  # Peak kilovoltage
    exposure_time: Optional[float] = None
    xray_tube_current: Optional[float] = None
    exposure: Optional[float] = None
    exposure_in_microas: Optional[float] = None
    distance_source_to_detector: Optional[float] = None
    distance_source_to_patient: Optional[float] = None
    image_area_dose_product: Optional[float] = None
    filter_type: Optional[str] = None
    focal_spot: Optional[str] = None
    collimator_type: Optional[str] = None
    collimator_left_edge: Optional[float] = None
    collimator_right_edge: Optional[float] = None
    collimator_upper_edge: Optional[float] = None
    collimator_lower_edge: Optional[float] = None
    
    # Image Information
    rows: Optional[int] = None
    columns: Optional[int] = None
    bits_allocated: Optional[int] = None
    bits_stored: Optional[int] = None
    high_bit: Optional[int] = None
    pixel_representation: Optional[int] = None
    samples_per_pixel: Optional[int] = None
    photometric_interpretation: Optional[str] = None
    planar_configuration: Optional[int] = None
    
    # Pixel Spacing and Calibration
    pixel_spacing_row: Optional[float] = None
    pixel_spacing_column: Optional[float] = None
    imager_pixel_spacing_row: Optional[float] = None
    imager_pixel_spacing_column: Optional[float] = None
    nominal_scanned_pixel_spacing: Optional[float] = None
    pixel_aspect_ratio: Optional[float] = None
    
    # Multi-frame Information
    number_of_frames: Optional[int] = None
    frame_time: Optional[float] = None
    frame_time_vector: Optional[str] = None  # JSON array
    frame_delay: Optional[float] = None
    frame_acquisition_datetime: Optional[str] = None
    recommended_display_frame_rate: Optional[float] = None
    cine_rate: Optional[int] = None
    
    # Angiography Specific
    positioner_primary_angle: Optional[float] = None
    positioner_secondary_angle: Optional[float] = None
    positioner_primary_angle_increment: Optional[float] = None
    positioner_secondary_angle_increment: Optional[float] = None
    detector_primary_angle: Optional[float] = None
    detector_secondary_angle: Optional[float] = None
    table_height: Optional[float] = None
    table_traverse: Optional[float] = None
    table_motion: Optional[str] = None
    table_vertical_increment: Optional[float] = None
    table_lateral_increment: Optional[float] = None
    table_longitudinal_increment: Optional[float] = None
    
    # Contrast/Bolus
    contrast_bolus_agent: Optional[str] = None
    contrast_bolus_volume: Optional[float] = None
    contrast_bolus_start_time: Optional[str] = None
    contrast_bolus_stop_time: Optional[str] = None
    contrast_bolus_total_dose: Optional[float] = None
    contrast_flow_rate: Optional[float] = None
    contrast_flow_duration: Optional[float] = None
    
    # Window/Level
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    window_center_width_explanation: Optional[str] = None
    rescale_intercept: Optional[float] = None
    rescale_slope: Optional[float] = None
    rescale_type: Optional[str] = None
    
    # Radiation Dose
    dose_area_product: Optional[float] = None
    entrance_dose: Optional[float] = None
    entrance_dose_mgy: Optional[float] = None
    exposed_area: Optional[float] = None
    radiation_dose: Optional[float] = None
    radiation_mode: Optional[str] = None
    
    # Cardiac Specific
    heart_rate: Optional[float] = None
    cardiac_number_of_images: Optional[int] = None
    trigger_time: Optional[float] = None
    nominal_interval: Optional[float] = None
    beat_rejection_flag: Optional[str] = None
    skip_beats: Optional[int] = None
    heart_rate_measured: Optional[float] = None
    
    # View Information
    view_position: Optional[str] = None
    view_code_sequence: Optional[str] = None
    view_modifier_code_sequence: Optional[str] = None
    
    # ECG Data
    ecg_data_present: bool = False
    number_of_waveform_samples: Optional[int] = None
    sampling_frequency: Optional[float] = None
    
    # Image Comments and Annotations
    image_comments: Optional[str] = None
    lossy_image_compression: Optional[str] = None
    lossy_image_compression_ratio: Optional[float] = None
    derivation_description: Optional[str] = None
    
    # Additional metadata as JSON
    additional_metadata: Optional[str] = None  # JSON string
    
    # Timestamps
    import_date: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.metadata_id:
            self.metadata_id = str(uuid.uuid4())
        if self.additional_metadata and isinstance(self.additional_metadata, dict):
            self.additional_metadata = json.dumps(self.additional_metadata)
        if self.frame_time_vector and not isinstance(self.frame_time_vector, str):
            self.frame_time_vector = json.dumps(self.frame_time_vector)