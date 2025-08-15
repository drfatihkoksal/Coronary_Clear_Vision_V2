"""
DICOM Domain Models

DICOM verileri için domain modelleri.
Clean Architecture prensiplerine uygun olarak tasarlanmıştır.
Immutable veri yapıları kullanır.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import date
from enum import Enum
import numpy as np
from pathlib import Path


class DicomModality(Enum):
    """
    DICOM modalite türleri.
    
    Anjiyografi için önemli modaliteler.
    """
    XA = "XA"  # X-Ray Angiography
    XRF = "XRF"  # X-Ray Fluoroscopy  
    CT = "CT"  # Computed Tomography
    MR = "MR"  # Magnetic Resonance
    US = "US"  # Ultrasound
    SC = "SC"  # Secondary Capture
    OTHER = "OTHER"  # Diğer
    
    @classmethod
    def from_string(cls, value: str) -> 'DicomModality':
        """String değerden enum oluştur."""
        try:
            return cls(value.upper())
        except ValueError:
            return cls.OTHER


class ProjectionType(Enum):
    """
    Anjiyografi projeksiyon türleri.
    
    Koroner anjiyografide kullanılan standart görüntüleme açıları.
    """
    # Sol koroner projeksiyonları
    LAO_CRANIAL = "LAO_CRANIAL"  # Left Anterior Oblique + Cranial
    LAO_CAUDAL = "LAO_CAUDAL"  # Left Anterior Oblique + Caudal
    RAO_CRANIAL = "RAO_CRANIAL"  # Right Anterior Oblique + Cranial
    RAO_CAUDAL = "RAO_CAUDAL"  # Right Anterior Oblique + Caudal
    
    # Düz projeksiyonlar
    AP = "AP"  # Anterior-Posterior
    LATERAL = "LATERAL"  # Lateral
    
    # Özel projeksiyonlar
    SPIDER = "SPIDER"  # Spider view
    LEFT_LATERAL = "LEFT_LATERAL"
    RIGHT_LATERAL = "RIGHT_LATERAL"
    
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_angles(cls, primary_angle: float, secondary_angle: float) -> 'ProjectionType':
        """
        Açılardan projeksiyon tipini belirle.
        
        Args:
            primary_angle: Birincil açı (LAO/RAO)
            secondary_angle: İkincil açı (Cranial/Caudal)
            
        Returns:
            ProjectionType: Projeksiyon tipi
        """
        # Basit bir yaklaşım - gerçek uygulamada daha detaylı olmalı
        if abs(primary_angle) < 10 and abs(secondary_angle) < 10:
            return cls.AP
        elif abs(primary_angle) > 80:
            return cls.LATERAL
        elif primary_angle > 0:  # RAO
            if secondary_angle > 0:
                return cls.RAO_CRANIAL
            else:
                return cls.RAO_CAUDAL
        else:  # LAO
            if secondary_angle > 0:
                return cls.LAO_CRANIAL
            else:
                return cls.LAO_CAUDAL


@dataclass(frozen=True)
class DicomPixelSpacing:
    """
    DICOM piksel aralığı bilgisi.
    
    Fiziksel ölçüm için kritik.
    
    Attributes:
        row_spacing (float): Satır aralığı (mm/piksel)
        column_spacing (float): Sütun aralığı (mm/piksel)
        unit (str): Birim (genellikle 'mm')
    """
    row_spacing: float
    column_spacing: float
    unit: str = "mm"
    
    @property
    def is_isotropic(self) -> bool:
        """İzotropik mi? (eşit aralık)"""
        return abs(self.row_spacing - self.column_spacing) < 0.0001
    
    @property
    def average_spacing(self) -> float:
        """Ortalama piksel aralığı"""
        return (self.row_spacing + self.column_spacing) / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'row_spacing': self.row_spacing,
            'column_spacing': self.column_spacing,
            'unit': self.unit,
            'is_isotropic': self.is_isotropic
        }


@dataclass(frozen=True)
class DicomWindowLevel:
    """
    DICOM pencere/seviye ayarları.
    
    Görüntü kontrastını kontrol eder.
    
    Attributes:
        center (float): Pencere merkezi
        width (float): Pencere genişliği
        name (str): Preset adı
    """
    center: float
    width: float
    name: str = "Custom"
    
    @property
    def min_value(self) -> float:
        """Minimum görüntü değeri"""
        return self.center - self.width / 2
    
    @property
    def max_value(self) -> float:
        """Maksimum görüntü değeri"""
        return self.center + self.width / 2
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Görüntüye pencere/seviye uygula.
        
        Args:
            image: Kaynak görüntü
            
        Returns:
            np.ndarray: İşlenmiş görüntü (0-255)
        """
        # Pencere sınırları
        lower = self.min_value
        upper = self.max_value
        
        # Kırpma
        windowed = np.clip(image, lower, upper)
        
        # 0-255'e ölçekle
        if upper > lower:
            scaled = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(image, dtype=np.uint8)
            
        return scaled


@dataclass(frozen=True)
class DicomPatientInfo:
    """
    DICOM hasta bilgileri.
    
    Anonimleştirilmiş hasta verileri.
    
    Attributes:
        patient_id (str): Hasta ID
        patient_name (str): Hasta adı
        birth_date (Optional[date]): Doğum tarihi
        sex (str): Cinsiyet (M/F/O)
        age (Optional[str]): Yaş
    """
    patient_id: str
    patient_name: str = "Anonymous"
    birth_date: Optional[date] = None
    sex: str = "O"  # Other
    age: Optional[str] = None
    
    @property
    def is_anonymous(self) -> bool:
        """Anonim hasta mı?"""
        return self.patient_name.lower() in ["anonymous", ""]
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'patient_id': self.patient_id,
            'patient_name': self.patient_name,
            'birth_date': self.birth_date.isoformat() if self.birth_date else None,
            'sex': self.sex,
            'age': self.age
        }


@dataclass(frozen=True)
class DicomStudyInfo:
    """
    DICOM çalışma bilgileri.
    
    Attributes:
        study_instance_uid (str): Çalışma UID
        study_date (Optional[date]): Çalışma tarihi
        study_time (Optional[str]): Çalışma saati
        study_description (str): Çalışma açıklaması
        accession_number (str): Erişim numarası
        referring_physician (str): Yönlendiren doktor
    """
    study_instance_uid: str
    study_date: Optional[date] = None
    study_time: Optional[str] = None
    study_description: str = ""
    accession_number: str = ""
    referring_physician: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'study_instance_uid': self.study_instance_uid,
            'study_date': self.study_date.isoformat() if self.study_date else None,
            'study_time': self.study_time,
            'study_description': self.study_description,
            'accession_number': self.accession_number,
            'referring_physician': self.referring_physician
        }


@dataclass(frozen=True)
class DicomSeriesInfo:
    """
    DICOM seri bilgileri.
    
    Attributes:
        series_instance_uid (str): Seri UID
        series_number (int): Seri numarası
        series_description (str): Seri açıklaması
        modality (DicomModality): Modalite
        body_part (str): Vücut bölgesi
        patient_position (str): Hasta pozisyonu
    """
    series_instance_uid: str
    series_number: int = 0
    series_description: str = ""
    modality: DicomModality = DicomModality.OTHER
    body_part: str = ""
    patient_position: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'series_instance_uid': self.series_instance_uid,
            'series_number': self.series_number,
            'series_description': self.series_description,
            'modality': self.modality.value,
            'body_part': self.body_part,
            'patient_position': self.patient_position
        }


@dataclass(frozen=True)
class DicomImageInfo:
    """
    DICOM görüntü bilgileri.
    
    Attributes:
        rows (int): Satır sayısı
        columns (int): Sütun sayısı
        bits_allocated (int): Ayrılan bit
        bits_stored (int): Saklanan bit
        pixel_representation (int): Piksel gösterimi
        photometric_interpretation (str): Fotometrik yorumlama
        samples_per_pixel (int): Piksel başına örnek
    """
    rows: int
    columns: int
    bits_allocated: int = 16
    bits_stored: int = 12
    pixel_representation: int = 0
    photometric_interpretation: str = "MONOCHROME2"
    samples_per_pixel: int = 1
    
    @property
    def is_color(self) -> bool:
        """Renkli görüntü mü?"""
        return self.samples_per_pixel > 1
    
    @property
    def is_signed(self) -> bool:
        """İşaretli piksel değerleri mi?"""
        return self.pixel_representation == 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'rows': self.rows,
            'columns': self.columns,
            'bits_allocated': self.bits_allocated,
            'bits_stored': self.bits_stored,
            'pixel_representation': self.pixel_representation,
            'photometric_interpretation': self.photometric_interpretation,
            'samples_per_pixel': self.samples_per_pixel
        }


@dataclass(frozen=True)
class DicomProjectionInfo:
    """
    Anjiyografi projeksiyon bilgileri.
    
    Attributes:
        primary_angle (float): Birincil açı (LAO/RAO)
        secondary_angle (float): İkincil açı (Cranial/Caudal)
        projection_type (ProjectionType): Projeksiyon tipi
        view_position (str): Görüntü pozisyonu
        table_height (Optional[float]): Masa yüksekliği
        distance_source_to_detector (Optional[float]): Kaynak-detektör mesafesi
        distance_source_to_patient (Optional[float]): Kaynak-hasta mesafesi
    """
    primary_angle: float = 0.0
    secondary_angle: float = 0.0
    projection_type: ProjectionType = ProjectionType.UNKNOWN
    view_position: str = ""
    table_height: Optional[float] = None
    distance_source_to_detector: Optional[float] = None
    distance_source_to_patient: Optional[float] = None
    
    @property
    def angle_description(self) -> str:
        """Açı açıklaması"""
        desc = []
        
        # Birincil açı
        if abs(self.primary_angle) > 5:
            if self.primary_angle > 0:
                desc.append(f"RAO {abs(self.primary_angle):.0f}°")
            else:
                desc.append(f"LAO {abs(self.primary_angle):.0f}°")
                
        # İkincil açı
        if abs(self.secondary_angle) > 5:
            if self.secondary_angle > 0:
                desc.append(f"CRA {abs(self.secondary_angle):.0f}°")
            else:
                desc.append(f"CAU {abs(self.secondary_angle):.0f}°")
                
        return " / ".join(desc) if desc else "AP"
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'primary_angle': self.primary_angle,
            'secondary_angle': self.secondary_angle,
            'projection_type': self.projection_type.value,
            'view_position': self.view_position,
            'angle_description': self.angle_description,
            'table_height': self.table_height,
            'distance_source_to_detector': self.distance_source_to_detector,
            'distance_source_to_patient': self.distance_source_to_patient
        }


@dataclass
class DicomFrame:
    """
    DICOM frame verisi.
    
    Tek bir frame'in verisi ve meta bilgileri.
    
    Attributes:
        index (int): Frame indeksi
        pixel_array (np.ndarray): Piksel verisi
        timestamp (Optional[float]): Zaman damgası (saniye)
        has_ekg (bool): EKG verisi var mı?
        ekg_sample_index (Optional[int]): EKG örnek indeksi
    """
    index: int
    pixel_array: np.ndarray
    timestamp: Optional[float] = None
    has_ekg: bool = False
    ekg_sample_index: Optional[int] = None
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Frame boyutu"""
        return self.pixel_array.shape
    
    def apply_window_level(self, window_level: DicomWindowLevel) -> np.ndarray:
        """
        Pencere/seviye uygula.
        
        Args:
            window_level: Pencere/seviye ayarları
            
        Returns:
            np.ndarray: İşlenmiş görüntü
        """
        return window_level.apply(self.pixel_array)


@dataclass
class DicomSeries:
    """
    DICOM serisi.
    
    Bir seriye ait tüm frame'ler ve meta veriler.
    
    Attributes:
        info (DicomSeriesInfo): Seri bilgileri
        frames (List[DicomFrame]): Frame listesi
        pixel_spacing (Optional[DicomPixelSpacing]): Piksel aralığı
        window_level (DicomWindowLevel): Varsayılan pencere/seviye
        projection_info (Optional[DicomProjectionInfo]): Projeksiyon bilgileri
        frame_rate (float): Frame hızı (fps)
    """
    info: DicomSeriesInfo
    frames: List[DicomFrame] = field(default_factory=list)
    pixel_spacing: Optional[DicomPixelSpacing] = None
    window_level: DicomWindowLevel = field(
        default_factory=lambda: DicomWindowLevel(128, 256, "Default")
    )
    projection_info: Optional[DicomProjectionInfo] = None
    frame_rate: float = 30.0
    
    @property
    def num_frames(self) -> int:
        """Frame sayısı"""
        return len(self.frames)
    
    @property
    def duration(self) -> float:
        """Süre (saniye)"""
        if self.num_frames > 0 and self.frame_rate > 0:
            return self.num_frames / self.frame_rate
        return 0.0
    
    @property
    def is_multi_frame(self) -> bool:
        """Çoklu frame mi?"""
        return self.num_frames > 1
    
    def get_frame(self, index: int) -> Optional[DicomFrame]:
        """
        Frame al.
        
        Args:
            index: Frame indeksi
            
        Returns:
            Optional[DicomFrame]: Frame verisi
        """
        if 0 <= index < self.num_frames:
            return self.frames[index]
        return None
    
    def get_frame_at_time(self, time: float) -> Optional[DicomFrame]:
        """
        Zamana göre frame al.
        
        Args:
            time: Zaman (saniye)
            
        Returns:
            Optional[DicomFrame]: Frame verisi
        """
        if time < 0 or time > self.duration:
            return None
            
        frame_index = int(time * self.frame_rate)
        return self.get_frame(min(frame_index, self.num_frames - 1))


@dataclass
class DicomStudy:
    """
    DICOM çalışması.
    
    Bir çalışmaya ait tüm seriler.
    
    Attributes:
        patient_info (DicomPatientInfo): Hasta bilgileri
        study_info (DicomStudyInfo): Çalışma bilgileri
        series_list (List[DicomSeries]): Seri listesi
        file_path (Path): Dosya yolu
        is_dicomdir (bool): DICOMDIR mi?
    """
    patient_info: DicomPatientInfo
    study_info: DicomStudyInfo
    series_list: List[DicomSeries] = field(default_factory=list)
    file_path: Optional[Path] = None
    is_dicomdir: bool = False
    
    @property
    def num_series(self) -> int:
        """Seri sayısı"""
        return len(self.series_list)
    
    @property
    def total_frames(self) -> int:
        """Toplam frame sayısı"""
        return sum(series.num_frames for series in self.series_list)
    
    def get_series_by_uid(self, uid: str) -> Optional[DicomSeries]:
        """
        UID'ye göre seri al.
        
        Args:
            uid: Seri UID
            
        Returns:
            Optional[DicomSeries]: Seri verisi
        """
        for series in self.series_list:
            if series.info.series_instance_uid == uid:
                return series
        return None
    
    def get_series_by_projection(self, projection_type: ProjectionType) -> List[DicomSeries]:
        """
        Projeksiyon tipine göre serileri al.
        
        Args:
            projection_type: Projeksiyon tipi
            
        Returns:
            List[DicomSeries]: Eşleşen seriler
        """
        matching_series = []
        for series in self.series_list:
            if (series.projection_info and 
                series.projection_info.projection_type == projection_type):
                matching_series.append(series)
        return matching_series
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir."""
        return {
            'patient_info': self.patient_info.to_dict(),
            'study_info': self.study_info.to_dict(),
            'num_series': self.num_series,
            'total_frames': self.total_frames,
            'file_path': str(self.file_path) if self.file_path else None,
            'is_dicomdir': self.is_dicomdir
        }


@dataclass
class DicomLoadRequest:
    """
    DICOM yükleme isteği.
    
    Attributes:
        file_path (Path): Dosya yolu
        load_pixel_data (bool): Piksel verisi yüklensin mi?
        specific_frames (Optional[List[int]]): Belirli frame'ler
        window_level_preset (Optional[str]): Pencere/seviye preset'i
    """
    file_path: Path
    load_pixel_data: bool = True
    specific_frames: Optional[List[int]] = None
    window_level_preset: Optional[str] = None


@dataclass
class DicomLoadResult:
    """
    DICOM yükleme sonucu.
    
    Attributes:
        success (bool): Başarılı mı?
        study (Optional[DicomStudy]): Yüklenen çalışma
        error_message (Optional[str]): Hata mesajı
        warnings (List[str]): Uyarılar
        load_time_ms (float): Yükleme süresi (ms)
    """
    success: bool
    study: Optional[DicomStudy] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    load_time_ms: float = 0.0
    
    @property
    def has_warnings(self) -> bool:
        """Uyarı var mı?"""
        return len(self.warnings) > 0