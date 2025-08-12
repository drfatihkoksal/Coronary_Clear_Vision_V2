"""
Calibration Data Models

Kalibrasyon işlemleri için kullanılan veri yapılarını tanımlar.
Domain-driven design prensipleri uygulanmıştır.
İmmutable veri yapıları ile güvenli veri yönetimi sağlanır.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import numpy as np


class CatheterSize(Enum):
    """
    Standart kateter boyutları (French).
    
    1 French = 0.33 mm
    """
    FR_4 = (4, 1.33)   # 4 French = 1.33 mm
    FR_5 = (5, 1.67)   # 5 French = 1.67 mm
    FR_6 = (6, 2.00)   # 6 French = 2.00 mm
    FR_7 = (7, 2.33)   # 7 French = 2.33 mm
    FR_8 = (8, 2.67)   # 8 French = 2.67 mm
    
    def __init__(self, french_size: int, diameter_mm: float):
        self.french_size = french_size
        self.diameter_mm = diameter_mm
        
    @classmethod
    def from_french(cls, french_size: int) -> 'CatheterSize':
        """
        French boyutundan CatheterSize enum'u döndürür.
        
        Args:
            french_size: French cinsinden kateter boyutu
            
        Returns:
            CatheterSize: İlgili enum değeri
            
        Raises:
            ValueError: Geçersiz French boyutu
        """
        for size in cls:
            if size.french_size == french_size:
                return size
        raise ValueError(f"Invalid French size: {french_size}")
        
    def __str__(self) -> str:
        return f"{self.french_size}F ({self.diameter_mm:.2f}mm)"


class CalibrationMethod(Enum):
    """
    Kalibrasyon yöntemleri.
    
    Her yöntemin kendine özgü avantaj ve dezavantajları vardır.
    """
    MANUAL = "manual"                # Manuel iki nokta seçimi
    CATHETER = "catheter"           # Kateter bazlı otomatik
    SPHERE = "sphere"               # Kalibrasyon küresi
    GRID = "grid"                   # Kalibrasyon ızgarası
    AUTO_DETECT = "auto_detect"     # Otomatik tespit
    DICOM_METADATA = "dicom"        # DICOM metadata'dan


class MeasurementMethod(Enum):
    """
    Kateter genişliği ölçüm yöntemleri.
    
    Farklı görüntü kalitelerinde farklı yöntemler daha iyi çalışır.
    """
    DISTANCE_TRANSFORM = "distance_transform"    # Mesafe dönüşümü
    MIN_AREA_RECT = "min_area_rect"            # Minimum alan dikdörtgeni
    DIRECT_CONTOUR = "direct_contour"          # Doğrudan kontur
    SKELETON_BASED = "skeleton_based"          # İskelet bazlı


@dataclass(frozen=True)
class CalibrationPoint:
    """
    Kalibrasyon noktası.
    
    Görüntü üzerinde seçilen bir noktayı temsil eder.
    
    Attributes:
        x (float): X koordinatı (piksel)
        y (float): Y koordinatı (piksel)
        confidence (float): Nokta güvenilirliği (0-1)
    """
    x: float
    y: float
    confidence: float = 1.0
    
    def __post_init__(self):
        """Değer doğrulaması"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
            
    def distance_to(self, other: 'CalibrationPoint') -> float:
        """
        Başka bir noktaya olan Euclidean mesafeyi hesaplar.
        
        Args:
            other: Diğer nokta
            
        Returns:
            float: Mesafe (piksel)
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        
    def as_tuple(self) -> Tuple[float, float]:
        """Tuple olarak döndürür."""
        return (self.x, self.y)


@dataclass(frozen=True)
class CatheterMeasurement:
    """
    Kateter ölçüm sonucu.
    
    Attributes:
        width_pixels (float): Ölçülen genişlik (piksel)
        method (MeasurementMethod): Kullanılan ölçüm yöntemi
        confidence (float): Ölçüm güvenilirliği (0-1)
        contour_points (List): Kateter konturu noktaları
        measurement_line (Tuple): Ölçüm hattı başlangıç ve bitiş noktaları
    """
    width_pixels: float
    method: MeasurementMethod
    confidence: float = 1.0
    contour_points: List[Tuple[float, float]] = field(default_factory=list)
    measurement_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    
    def __post_init__(self):
        """Değer doğrulaması"""
        if self.width_pixels <= 0:
            raise ValueError("Width must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class CalibrationRequest:
    """
    Kalibrasyon isteği parametreleri.
    
    Attributes:
        image (np.ndarray): Kalibrasyon yapılacak görüntü
        method (CalibrationMethod): Kullanılacak kalibrasyon yöntemi
        catheter_size (Optional[CatheterSize]): Kateter boyutu (catheter method için)
        manual_points (Optional[List[CalibrationPoint]]): Manuel seçilen noktalar
        roi (Optional[Tuple]): İlgi alanı (x, y, width, height)
        options (Dict): Ek seçenekler
    """
    image: np.ndarray
    method: CalibrationMethod
    catheter_size: Optional[CatheterSize] = None
    manual_points: Optional[List[CalibrationPoint]] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Parametreleri doğrula"""
        if self.method == CalibrationMethod.CATHETER and not self.catheter_size:
            raise ValueError("Catheter size required for catheter calibration")
            
        if self.method == CalibrationMethod.MANUAL and not self.manual_points:
            raise ValueError("Manual points required for manual calibration")
            
        if self.manual_points and len(self.manual_points) < 2:
            raise ValueError("At least 2 points required for manual calibration")


@dataclass
class CalibrationResult:
    """
    Kalibrasyon sonucu.
    
    Attributes:
        success (bool): Başarılı mı?
        factor (float): Piksel/mm dönüşüm faktörü
        method (CalibrationMethod): Kullanılan yöntem
        confidence (float): Kalibrasyon güvenilirliği (0-1)
        timestamp (datetime): Kalibrasyon zamanı
        measurement (Optional[CatheterMeasurement]): Kateter ölçümü (varsa)
        error_message (Optional[str]): Hata mesajı (başarısızsa)
        metadata (Dict): Ek bilgiler
        visualization (Optional[np.ndarray]): Görselleştirme için işlenmiş görüntü
    """
    success: bool
    factor: float = 0.0
    method: CalibrationMethod = CalibrationMethod.MANUAL
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    measurement: Optional[CatheterMeasurement] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    visualization: Optional[np.ndarray] = None
    
    @property
    def is_valid(self) -> bool:
        """Kalibrasyon geçerli mi?"""
        return self.success and self.factor > 0
        
    @property
    def pixels_per_mm(self) -> float:
        """Piksel/mm oranı"""
        return self.factor
        
    @property
    def mm_per_pixel(self) -> float:
        """mm/piksel oranı"""
        return 1.0 / self.factor if self.factor > 0 else 0.0
        
    def convert_to_mm(self, pixels: float) -> float:
        """
        Piksel değerini mm'ye çevirir.
        
        Args:
            pixels: Piksel değeri
            
        Returns:
            float: mm değeri
        """
        return pixels * self.mm_per_pixel
        
    def convert_to_pixels(self, mm: float) -> float:
        """
        mm değerini piksele çevirir.
        
        Args:
            mm: mm değeri
            
        Returns:
            float: Piksel değeri
        """
        return mm * self.pixels_per_mm


@dataclass
class CalibrationValidation:
    """
    Kalibrasyon doğrulama sonucu.
    
    Attributes:
        is_valid (bool): Geçerli mi?
        issues (List[str]): Tespit edilen sorunlar
        recommendations (List[str]): İyileştirme önerileri
        confidence_score (float): Genel güven skoru (0-1)
    """
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    
    def add_issue(self, issue: str):
        """Sorun ekler ve geçerliliği günceller."""
        self.issues.append(issue)
        self.is_valid = False
        
    def add_recommendation(self, recommendation: str):
        """Öneri ekler."""
        self.recommendations.append(recommendation)