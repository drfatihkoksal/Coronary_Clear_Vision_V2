"""
Tracking Data Models

İzleme (tracking) işlemleri için kullanılan veri yapılarını tanımlar.
Domain-driven design prensipleri uygulanmıştır.
İmmutable veri yapıları ile güvenli veri yönetimi sağlanır.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import numpy as np


class TrackingMethod(Enum):
    """
    İzleme yöntemleri.
    
    Farklı yöntemler farklı senaryolarda daha iyi performans gösterir.
    """
    TEMPLATE_MATCHING = "template_matching"      # Şablon eşleme
    OPTICAL_FLOW = "optical_flow"               # Optik akış
    KALMAN_FILTER = "kalman_filter"            # Kalman filtresi
    PARTICLE_FILTER = "particle_filter"         # Parçacık filtresi
    DEEP_LEARNING = "deep_learning"             # Derin öğrenme


class TrackingStatus(Enum):
    """
    İzleme durumu.
    
    Bir noktanın izleme durumunu belirtir.
    """
    ACTIVE = "active"          # Aktif olarak izleniyor
    LOST = "lost"             # İzleme kaybedildi
    OCCLUDED = "occluded"     # Geçici olarak görünmüyor
    MANUAL = "manual"         # Manuel olarak yerleştirildi
    PREDICTED = "predicted"    # Tahmin edildi
    VALIDATED = "validated"    # Doğrulandı


class MotionModel(Enum):
    """
    Hareket modeli.
    
    İzlenen nesnenin hareket karakteristiği.
    """
    STATIC = "static"              # Sabit
    LINEAR = "linear"              # Doğrusal hareket
    CIRCULAR = "circular"          # Dairesel hareket
    SINUSOIDAL = "sinusoidal"     # Sinüzoidal (kalp döngüsü)
    RANDOM_WALK = "random_walk"    # Rastgele yürüyüş


@dataclass(frozen=True)
class Point2D:
    """
    2D nokta.
    
    Görüntü koordinat sisteminde bir noktayı temsil eder.
    
    Attributes:
        x (float): X koordinatı (piksel)
        y (float): Y koordinatı (piksel)
    """
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
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
        
    def as_numpy(self) -> np.ndarray:
        """NumPy array olarak döndürür."""
        return np.array([self.x, self.y])
        
    def __add__(self, other: 'Point2D') -> 'Point2D':
        """İki noktayı toplar."""
        return Point2D(self.x + other.x, self.y + other.y)
        
    def __sub__(self, other: 'Point2D') -> 'Point2D':
        """İki nokta arasındaki farkı alır."""
        return Point2D(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class TrackingParameters:
    """
    İzleme parametreleri.
    
    İzleme algoritması için ayarlanabilir parametreler.
    
    Attributes:
        template_size (int): Şablon pencere boyutu
        search_radius (int): Arama yarıçapı
        confidence_threshold (float): Güven eşiği (0-1)
        max_motion (float): Maksimum hareket mesafesi
        motion_smoothing (float): Hareket yumuşatma faktörü (0-1)
        enable_prediction (bool): Tahmin etkin mi?
        adaptive_template (bool): Adaptif şablon güncelleme
        update_rate (float): Şablon güncelleme oranı (0-1)
    """
    template_size: int = 21
    search_radius: int = 30
    confidence_threshold: float = 0.7
    max_motion: float = 50.0
    motion_smoothing: float = 0.5
    enable_prediction: bool = True
    adaptive_template: bool = True
    update_rate: float = 0.1
    
    def __post_init__(self):
        """Parametre doğrulaması"""
        if self.template_size % 2 == 0:
            raise ValueError("Template size must be odd")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not 0 <= self.motion_smoothing <= 1:
            raise ValueError("Motion smoothing must be between 0 and 1")
        if not 0 <= self.update_rate <= 1:
            raise ValueError("Update rate must be between 0 and 1")


@dataclass
class TrackedPoint:
    """
    İzlenen nokta.
    
    Bir noktanın izleme geçmişi ve durumu.
    
    Attributes:
        id (str): Benzersiz tanımlayıcı
        name (str): İnsan okunabilir isim
        current_position (Point2D): Mevcut pozisyon
        status (TrackingStatus): İzleme durumu
        confidence (float): İzleme güveni (0-1)
        frame_number (int): Mevcut frame numarası
        history (List[TrackingFrame]): İzleme geçmişi
        metadata (Dict): Ek bilgiler
    """
    id: str
    name: str
    current_position: Point2D
    status: TrackingStatus = TrackingStatus.ACTIVE
    confidence: float = 1.0
    frame_number: int = 0
    history: List['TrackingFrame'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_frame(self, frame_data: 'TrackingFrame'):
        """
        Yeni frame verisi ekler.
        
        Args:
            frame_data: Frame verisi
        """
        self.history.append(frame_data)
        self.current_position = frame_data.position
        self.status = frame_data.status
        self.confidence = frame_data.confidence
        self.frame_number = frame_data.frame_number
        
    def get_trajectory(self) -> List[Point2D]:
        """
        Hareket yörüngesini döndürür.
        
        Returns:
            List[Point2D]: Pozisyon listesi
        """
        return [frame.position for frame in self.history]
        
    def get_velocity(self) -> Optional[Point2D]:
        """
        Anlık hızı hesaplar.
        
        Returns:
            Optional[Point2D]: Hız vektörü veya None
        """
        if len(self.history) < 2:
            return None
            
        prev = self.history[-2].position
        curr = self.history[-1].position
        
        return Point2D(curr.x - prev.x, curr.y - prev.y)
        
    def predict_next_position(self) -> Optional[Point2D]:
        """
        Sonraki pozisyonu tahmin eder.
        
        Returns:
            Optional[Point2D]: Tahmin edilen pozisyon
        """
        velocity = self.get_velocity()
        if velocity is None:
            return None
            
        return self.current_position + velocity


@dataclass(frozen=True)
class TrackingFrame:
    """
    Tek bir frame için izleme verisi.
    
    Attributes:
        frame_number (int): Frame numarası
        position (Point2D): Nokta pozisyonu
        confidence (float): İzleme güveni
        status (TrackingStatus): İzleme durumu
        timestamp (datetime): Zaman damgası
        template_match_score (float): Şablon eşleme skoru
        search_region (Tuple): Arama bölgesi (x, y, w, h)
    """
    frame_number: int
    position: Point2D
    confidence: float
    status: TrackingStatus
    timestamp: datetime = field(default_factory=datetime.now)
    template_match_score: float = 0.0
    search_region: Optional[Tuple[int, int, int, int]] = None


@dataclass
class TrackingRequest:
    """
    İzleme isteği.
    
    Attributes:
        image (np.ndarray): Mevcut frame
        points (List[TrackedPoint]): İzlenecek noktalar
        method (TrackingMethod): İzleme yöntemi
        parameters (TrackingParameters): İzleme parametreleri
        previous_image (Optional[np.ndarray]): Önceki frame
        motion_model (MotionModel): Hareket modeli
    """
    image: np.ndarray
    points: List[TrackedPoint]
    method: TrackingMethod = TrackingMethod.TEMPLATE_MATCHING
    parameters: TrackingParameters = field(default_factory=TrackingParameters)
    previous_image: Optional[np.ndarray] = None
    motion_model: MotionModel = MotionModel.LINEAR
    
    def __post_init__(self):
        """İstek doğrulaması"""
        if self.image is None or self.image.size == 0:
            raise ValueError("Image cannot be empty")
        if not self.points:
            raise ValueError("At least one point required for tracking")


@dataclass
class TrackingResult:
    """
    İzleme sonucu.
    
    Attributes:
        success (bool): Başarılı mı?
        tracked_points (List[TrackedPoint]): Güncellenmiş noktalar
        failed_points (List[str]): Başarısız nokta ID'leri
        processing_time_ms (float): İşlem süresi
        frame_number (int): İşlenen frame numarası
        error_message (Optional[str]): Hata mesajı
        visualization (Optional[np.ndarray]): Görselleştirme
    """
    success: bool
    tracked_points: List[TrackedPoint] = field(default_factory=list)
    failed_points: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    frame_number: int = 0
    error_message: Optional[str] = None
    visualization: Optional[np.ndarray] = None
    
    @property
    def success_rate(self) -> float:
        """
        Başarı oranını hesaplar.
        
        Returns:
            float: Başarı oranı (0-1)
        """
        total = len(self.tracked_points) + len(self.failed_points)
        if total == 0:
            return 0.0
        return len(self.tracked_points) / total
        
    def get_tracked_point_by_id(self, point_id: str) -> Optional[TrackedPoint]:
        """
        ID'ye göre izlenen noktayı döndürür.
        
        Args:
            point_id: Nokta ID'si
            
        Returns:
            Optional[TrackedPoint]: Nokta veya None
        """
        for point in self.tracked_points:
            if point.id == point_id:
                return point
        return None


@dataclass
class TrackingSession:
    """
    İzleme oturumu.
    
    Bir video/görüntü dizisi için tüm izleme verisi.
    
    Attributes:
        session_id (str): Oturum ID'si
        start_frame (int): Başlangıç frame'i
        end_frame (int): Bitiş frame'i
        tracked_points (Dict[str, TrackedPoint]): İzlenen noktalar
        parameters (TrackingParameters): Kullanılan parametreler
        method (TrackingMethod): Kullanılan yöntem
        created_at (datetime): Oluşturulma zamanı
        metadata (Dict): Ek bilgiler
    """
    session_id: str
    start_frame: int
    end_frame: int
    tracked_points: Dict[str, TrackedPoint] = field(default_factory=dict)
    parameters: TrackingParameters = field(default_factory=TrackingParameters)
    method: TrackingMethod = TrackingMethod.TEMPLATE_MATCHING
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_point(self, point: TrackedPoint):
        """Nokta ekler."""
        self.tracked_points[point.id] = point
        
    def remove_point(self, point_id: str):
        """Nokta kaldırır."""
        if point_id in self.tracked_points:
            del self.tracked_points[point_id]
            
    def get_point(self, point_id: str) -> Optional[TrackedPoint]:
        """ID'ye göre nokta döndürür."""
        return self.tracked_points.get(point_id)
        
    def get_all_points(self) -> List[TrackedPoint]:
        """Tüm noktaları döndürür."""
        return list(self.tracked_points.values())
        
    def get_frame_range(self) -> Tuple[int, int]:
        """Frame aralığını döndürür."""
        return (self.start_frame, self.end_frame)
        
    def is_active(self) -> bool:
        """Oturum aktif mi?"""
        return bool(self.tracked_points)