"""
QCA (Quantitative Coronary Analysis) Data Models

Bu modül QCA analizi için kullanılan veri yapılarını tanımlar.
Domain-driven design prensipleri uygulanmıştır.
Business logic'ten bağımsız, immutable veri yapıları içerir.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np


class AnalysisStatus(Enum):
    """
    QCA analiz durumu enumerasyonu.
    
    Analizin hangi aşamada olduğunu belirtir.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VesselType(Enum):
    """
    Damar tipi enumerasyonu.
    
    Koroner arterlerin sınıflandırması.
    """
    LAD = "LAD"  # Left Anterior Descending
    LCX = "LCX"  # Left Circumflex
    RCA = "RCA"  # Right Coronary Artery
    LM = "LM"    # Left Main
    DIAGONAL = "DIAGONAL"
    OBTUSE_MARGINAL = "OBTUSE_MARGINAL"
    UNKNOWN = "UNKNOWN"


class StenosisGrade(Enum):
    """
    Stenoz derecesi sınıflandırması.
    
    Klinik öneme göre darlık seviyeleri.
    """
    MINIMAL = "minimal"          # < 25%
    MILD = "mild"               # 25-49%
    MODERATE = "moderate"       # 50-69%
    SEVERE = "severe"           # 70-99%
    TOTAL_OCCLUSION = "total"   # 100%


@dataclass(frozen=True)
class CalibrationData:
    """
    Kalibrasyon verilerini tutan immutable yapı.
    
    Attributes:
        factor (float): Piksel/mm dönüşüm faktörü
        catheter_size_mm (float): Kateter çapı (mm)
        method (str): Kullanılan kalibrasyon metodu
        confidence (float): Kalibrasyon güven skoru (0-1)
        timestamp (datetime): Kalibrasyon zamanı
    """
    factor: float
    catheter_size_mm: float
    method: str = "manual"
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Değer doğrulaması"""
        if self.factor <= 0:
            raise ValueError("Calibration factor must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class VesselMeasurement:
    """
    Tek bir noktadaki damar ölçümü.
    
    Attributes:
        position_mm (float): Referans noktasından uzaklık (mm)
        diameter_mm (float): Damar çapı (mm)
        area_mm2 (float): Kesit alanı (mm²)
        confidence (float): Ölçüm güvenilirliği (0-1)
    """
    position_mm: float
    diameter_mm: float
    area_mm2: Optional[float] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Alan hesaplaması ve doğrulama"""
        if self.area_mm2 is None:
            # Dairesel kesit varsayımı ile alan hesapla
            object.__setattr__(self, 'area_mm2', 
                             np.pi * (self.diameter_mm / 2) ** 2)


@dataclass(frozen=True)
class StenosisData:
    """
    Stenoz (darlık) analiz sonuçları.
    
    Attributes:
        percent_diameter (float): Çap bazlı darlık yüzdesi
        percent_area (float): Alan bazlı darlık yüzdesi
        location_mm (float): Stenoz konumu (mm)
        length_mm (float): Stenoz uzunluğu (mm)
        grade (StenosisGrade): Klinik darlık derecesi
        reference_diameter_mm (float): Referans damar çapı
        minimal_lumen_diameter_mm (float): Minimum lümen çapı
    """
    percent_diameter: float
    percent_area: float
    location_mm: float
    length_mm: float
    grade: StenosisGrade
    reference_diameter_mm: float
    minimal_lumen_diameter_mm: float
    
    def __post_init__(self):
        """Stenoz derecesini otomatik hesapla"""
        if self.grade is None:
            grade = self._calculate_grade()
            object.__setattr__(self, 'grade', grade)
            
    def _calculate_grade(self) -> StenosisGrade:
        """Yüzdeye göre stenoz derecesini belirle"""
        if self.percent_diameter < 25:
            return StenosisGrade.MINIMAL
        elif self.percent_diameter < 50:
            return StenosisGrade.MILD
        elif self.percent_diameter < 70:
            return StenosisGrade.MODERATE
        elif self.percent_diameter < 100:
            return StenosisGrade.SEVERE
        else:
            return StenosisGrade.TOTAL_OCCLUSION


@dataclass
class QCAAnalysisResult:
    """
    QCA analiz sonuçlarını içeren ana veri yapısı.
    
    Attributes:
        frame_number (int): Analiz yapılan frame numarası
        vessel_type (VesselType): Analiz edilen damar tipi
        measurements (List[VesselMeasurement]): Çap ölçümleri
        stenosis_data (Optional[StenosisData]): Stenoz bilgileri
        centerline (List[Tuple[float, float]]): Damar merkez hattı
        calibration (CalibrationData): Kullanılan kalibrasyon
        analysis_time_ms (float): Analiz süresi
        status (AnalysisStatus): Analiz durumu
        error_message (Optional[str]): Hata mesajı
        metadata (Dict[str, Any]): Ek bilgiler
    """
    frame_number: int
    vessel_type: VesselType = VesselType.UNKNOWN
    measurements: List[VesselMeasurement] = field(default_factory=list)
    stenosis_data: Optional[StenosisData] = None
    centerline: List[Tuple[float, float]] = field(default_factory=list)
    calibration: Optional[CalibrationData] = None
    analysis_time_ms: float = 0.0
    status: AnalysisStatus = AnalysisStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Analiz başarılı mı?"""
        return self.status == AnalysisStatus.COMPLETED
    
    @property
    def has_stenosis(self) -> bool:
        """Stenoz tespit edildi mi?"""
        return self.stenosis_data is not None
    
    @property
    def mean_diameter_mm(self) -> Optional[float]:
        """Ortalama damar çapı (mm)"""
        if not self.measurements:
            return None
        diameters = [m.diameter_mm for m in self.measurements]
        return np.mean(diameters)
    
    @property
    def min_diameter_mm(self) -> Optional[float]:
        """Minimum damar çapı (mm)"""
        if not self.measurements:
            return None
        diameters = [m.diameter_mm for m in self.measurements]
        return np.min(diameters)
    
    @property
    def max_diameter_mm(self) -> Optional[float]:
        """Maximum damar çapı (mm)"""
        if not self.measurements:
            return None
        diameters = [m.diameter_mm for m in self.measurements]
        return np.max(diameters)


@dataclass
class QCAAnalysisRequest:
    """
    QCA analiz isteği parametreleri.
    
    Attributes:
        segmentation_result: Segmentasyon sonucu
        calibration: Kalibrasyon bilgisi
        frame_number: Analiz edilecek frame
        analysis_options: Analiz seçenekleri
    """
    segmentation_result: Any  # SegmentationResult tipinde olacak
    calibration: CalibrationData
    frame_number: int
    analysis_options: Dict[str, Any] = field(default_factory=lambda: {
        'smooth_centerline': True,
        'detect_stenosis': True,
        'edge_detection_method': 'gradient',
        'diameter_method': 'ribbon'
    })


@dataclass
class QCASequentialResult:
    """
    Ardışık frame'ler üzerinde QCA analiz sonuçları.
    
    Attributes:
        results_by_frame (Dict[int, QCAAnalysisResult]): Frame bazlı sonuçlar
        summary_statistics (Dict[str, float]): Özet istatistikler
        temporal_analysis (Dict[str, Any]): Zamansal analiz verileri
    """
    results_by_frame: Dict[int, QCAAnalysisResult] = field(default_factory=dict)
    summary_statistics: Dict[str, float] = field(default_factory=dict)
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def frame_count(self) -> int:
        """Analiz edilen frame sayısı"""
        return len(self.results_by_frame)
    
    @property
    def successful_count(self) -> int:
        """Başarılı analiz sayısı"""
        return sum(1 for r in self.results_by_frame.values() if r.is_successful)
    
    def get_diameter_curve(self) -> Tuple[List[int], List[float]]:
        """
        Frame bazlı ortalama çap eğrisini döndür.
        
        Returns:
            Tuple[List[int], List[float]]: (frame_numbers, mean_diameters)
        """
        frames = []
        diameters = []
        
        for frame, result in sorted(self.results_by_frame.items()):
            if result.is_successful and result.mean_diameter_mm:
                frames.append(frame)
                diameters.append(result.mean_diameter_mm)
                
        return frames, diameters