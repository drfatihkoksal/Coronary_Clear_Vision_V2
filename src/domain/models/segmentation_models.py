"""
Segmentation Data Models

Segmentasyon işlemleri için kullanılan veri yapılarını tanımlar.
Domain-driven design prensipleri uygulanmıştır.
İmmutable veri yapıları ile güvenli veri yönetimi sağlanır.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import numpy as np


class SegmentationMethod(Enum):
    """
    Segmentasyon yöntemleri.
    
    Farklı yöntemler farklı durumlarda daha iyi sonuç verir.
    """
    AI_ANGIOPY = "ai_angiopy"          # AngioPy AI modeli
    TRADITIONAL = "traditional"         # Geleneksel görüntü işleme
    SEMI_AUTOMATIC = "semi_automatic"   # Yarı otomatik (kullanıcı destekli)
    MANUAL = "manual"                   # Manuel çizim
    HYBRID = "hybrid"                   # AI + geleneksel kombinasyonu


class VesselType(Enum):
    """
    Damar tipi sınıflandırması.
    
    Segmentasyon sonrası tespit edilen damar tipi.
    """
    ARTERY = "artery"       # Arter
    VEIN = "vein"          # Ven
    CAPILLARY = "capillary" # Kılcal damar
    UNKNOWN = "unknown"     # Belirsiz


class SegmentationQuality(Enum):
    """
    Segmentasyon kalite seviyeleri.
    
    Otomatik kalite değerlendirmesi için.
    """
    EXCELLENT = "excellent"  # Mükemmel (>90% güven)
    GOOD = "good"           # İyi (70-90% güven)
    FAIR = "fair"           # Orta (50-70% güven)
    POOR = "poor"           # Kötü (<50% güven)


@dataclass(frozen=True)
class UserPoint:
    """
    Kullanıcı tarafından seçilen nokta.
    
    Segmentasyon için başlangıç/rehber noktası.
    
    Attributes:
        x (float): X koordinatı (piksel)
        y (float): Y koordinatı (piksel)
        point_type (str): Nokta tipi (start, end, guide)
        timestamp (datetime): Seçim zamanı
    """
    x: float
    y: float
    point_type: str = "guide"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def as_tuple(self) -> Tuple[float, float]:
        """Tuple olarak döndürür."""
        return (self.x, self.y)
        
    def distance_to(self, other: 'UserPoint') -> float:
        """
        Başka bir noktaya olan mesafeyi hesaplar.
        
        Args:
            other: Diğer nokta
            
        Returns:
            float: Euclidean mesafe
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass(frozen=True)
class VesselFeatures:
    """
    Damar özellikleri.
    
    Segmentasyon sonrası çıkarılan özellikler.
    
    Attributes:
        centerline (np.ndarray): Damar merkez hattı
        boundaries (List[np.ndarray]): Damar sınırları
        skeleton (np.ndarray): Morfolojik iskelet
        branch_points (List[Tuple]): Dallanma noktaları
        end_points (List[Tuple]): Uç noktalar
        mean_diameter (float): Ortalama çap
        length (float): Toplam uzunluk
        tortuosity (float): Kıvrımlılık indeksi
        area (float): Toplam alan
    """
    centerline: np.ndarray
    boundaries: List[np.ndarray] = field(default_factory=list)
    skeleton: Optional[np.ndarray] = None
    branch_points: List[Tuple[int, int]] = field(default_factory=list)
    end_points: List[Tuple[int, int]] = field(default_factory=list)
    mean_diameter: float = 0.0
    length: float = 0.0
    tortuosity: float = 0.0
    area: float = 0.0
    
    def get_main_orientation(self) -> float:
        """
        Ana damar yönünü hesaplar.
        
        Returns:
            float: Açı (derece)
        """
        if len(self.centerline) < 2:
            return 0.0
            
        # İlk ve son nokta arasındaki açı
        start = self.centerline[0]
        end = self.centerline[-1]
        
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        return np.degrees(angle)


@dataclass
class SegmentationRequest:
    """
    Segmentasyon isteği parametreleri.
    
    Attributes:
        image (np.ndarray): Segmente edilecek görüntü
        method (SegmentationMethod): Kullanılacak yöntem
        user_points (List[UserPoint]): Kullanıcı noktaları
        roi (Optional[Tuple]): İlgi alanı (x, y, width, height)
        options (Dict): Ek seçenekler
        previous_mask (Optional[np.ndarray]): Önceki segmentasyon (refinement için)
    """
    image: np.ndarray
    method: SegmentationMethod = SegmentationMethod.AI_ANGIOPY
    user_points: List[UserPoint] = field(default_factory=list)
    roi: Optional[Tuple[int, int, int, int]] = None
    options: Dict[str, Any] = field(default_factory=dict)
    previous_mask: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Parametreleri doğrula"""
        if self.image is None or self.image.size == 0:
            raise ValueError("Image cannot be empty")
            
        # Varsayılan seçenekler
        default_options = {
            'extract_features': True,
            'post_process': True,
            'min_vessel_size': 50,
            'smoothing_iterations': 2
        }
        
        # Varsayılanları kullanıcı seçenekleriyle birleştir
        for key, value in default_options.items():
            if key not in self.options:
                self.options[key] = value


@dataclass
class SegmentationResult:
    """
    Segmentasyon sonucu.
    
    Attributes:
        success (bool): Başarılı mı?
        mask (np.ndarray): Segmentasyon maskesi
        confidence (float): Güven skoru (0-1)
        method (SegmentationMethod): Kullanılan yöntem
        features (Optional[VesselFeatures]): Çıkarılan özellikler
        quality (SegmentationQuality): Kalite değerlendirmesi
        processing_time_ms (float): İşlem süresi
        error_message (Optional[str]): Hata mesajı
        metadata (Dict): Ek bilgiler
        visualization (Optional[np.ndarray]): Görselleştirme
    """
    success: bool
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    method: SegmentationMethod = SegmentationMethod.AI_ANGIOPY
    features: Optional[VesselFeatures] = None
    quality: SegmentationQuality = SegmentationQuality.POOR
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    visualization: Optional[np.ndarray] = None
    
    @property
    def is_valid(self) -> bool:
        """Segmentasyon geçerli mi?"""
        return self.success and self.mask is not None
        
    @property
    def vessel_pixels(self) -> int:
        """Damar piksel sayısı"""
        if self.mask is None:
            return 0
        return np.sum(self.mask > 0)
        
    @property
    def has_features(self) -> bool:
        """Özellikler çıkarılmış mı?"""
        return self.features is not None
        
    def get_quality_score(self) -> float:
        """
        Kalite skorunu sayısal olarak döndürür.
        
        Returns:
            float: 0-100 arası kalite skoru
        """
        quality_scores = {
            SegmentationQuality.EXCELLENT: 95.0,
            SegmentationQuality.GOOD: 80.0,
            SegmentationQuality.FAIR: 60.0,
            SegmentationQuality.POOR: 30.0
        }
        return quality_scores.get(self.quality, 0.0)


@dataclass
class SegmentationValidation:
    """
    Segmentasyon doğrulama sonucu.
    
    Attributes:
        is_valid (bool): Geçerli mi?
        quality_metrics (Dict): Kalite metrikleri
        issues (List[str]): Tespit edilen sorunlar
        suggestions (List[str]): İyileştirme önerileri
    """
    is_valid: bool = True
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str):
        """Sorun ekler ve geçerliliği günceller."""
        self.issues.append(issue)
        self.is_valid = False
        
    def add_suggestion(self, suggestion: str):
        """İyileştirme önerisi ekler."""
        self.suggestions.append(suggestion)
        
    def set_metric(self, name: str, value: float):
        """Kalite metriği ekler."""
        self.quality_metrics[name] = value


@dataclass
class SegmentationHistory:
    """
    Segmentasyon geçmişi.
    
    Undo/redo işlevselliği için.
    
    Attributes:
        results (List[SegmentationResult]): Sonuç geçmişi
        current_index (int): Mevcut indeks
        max_history (int): Maksimum geçmiş sayısı
    """
    results: List[SegmentationResult] = field(default_factory=list)
    current_index: int = -1
    max_history: int = 10
    
    def add_result(self, result: SegmentationResult):
        """
        Yeni sonuç ekler.
        
        Args:
            result: Eklenecek sonuç
        """
        # Mevcut indeksten sonraki geçmişi sil (redo'ları)
        if self.current_index < len(self.results) - 1:
            self.results = self.results[:self.current_index + 1]
            
        # Yeni sonucu ekle
        self.results.append(result)
        self.current_index += 1
        
        # Maksimum geçmiş kontrolü
        if len(self.results) > self.max_history:
            self.results.pop(0)
            self.current_index -= 1
            
    def can_undo(self) -> bool:
        """Undo yapılabilir mi?"""
        return self.current_index > 0
        
    def can_redo(self) -> bool:
        """Redo yapılabilir mi?"""
        return self.current_index < len(self.results) - 1
        
    def undo(self) -> Optional[SegmentationResult]:
        """
        Bir önceki sonuca döner.
        
        Returns:
            Optional[SegmentationResult]: Önceki sonuç
        """
        if self.can_undo():
            self.current_index -= 1
            return self.results[self.current_index]
        return None
        
    def redo(self) -> Optional[SegmentationResult]:
        """
        Bir sonraki sonuca gider.
        
        Returns:
            Optional[SegmentationResult]: Sonraki sonuç
        """
        if self.can_redo():
            self.current_index += 1
            return self.results[self.current_index]
        return None
        
    def get_current(self) -> Optional[SegmentationResult]:
        """
        Mevcut sonucu döndürür.
        
        Returns:
            Optional[SegmentationResult]: Mevcut sonuç
        """
        if 0 <= self.current_index < len(self.results):
            return self.results[self.current_index]
        return None
        
    def clear(self):
        """Geçmişi temizler."""
        self.results.clear()
        self.current_index = -1