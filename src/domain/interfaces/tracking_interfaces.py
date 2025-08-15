"""
Tracking Interface Definitions

İzleme (tracking) işlemleri için kullanılan arayüzleri tanımlar.
Dependency Inversion Principle (DIP) uygulaması için abstraksiyonlar sağlar.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Tuple, Dict, Any, Callable, runtime_checkable
import numpy as np

from src.domain.models.tracking_models import (
    TrackingResult,
    TrackingRequest,
    TrackedPoint,
    TrackingSession,
    TrackingParameters,
    Point2D,
)


@runtime_checkable
class ITracker(Protocol):
    """
    Standardized tracking algorithm protocol.

    All tracking implementations must follow this interface.
    Supports dependency injection and runtime type checking.
    """

    def initialize(self, image: np.ndarray, point: Point2D, template_size: int) -> bool:
        """
        İzleyiciyi başlatır.

        Args:
            image: Başlangıç görüntüsü
            point: İzlenecek nokta
            template_size: Şablon boyutu

        Returns:
            bool: Başarılı mı?
        """
        ...

    def track(
        self, image: np.ndarray, search_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[bool, Point2D, float]:
        """
        Noktayı yeni frame'de izler.

        Args:
            image: Yeni frame
            search_region: Arama bölgesi (x, y, width, height)

        Returns:
            Tuple: (başarılı, yeni_pozisyon, güven_skoru)
        """
        ...

    def update_template(self, image: np.ndarray, position: Point2D, update_rate: float):
        """
        Şablonu günceller.

        Args:
            image: Güncel görüntü
            position: Güncel pozisyon
            update_rate: Güncelleme oranı (0-1)
        """
        ...

    @property
    def method_name(self) -> str:
        """İzleme yöntemi adı"""
        ...


class IMotionPredictor(Protocol):
    """
    Hareket tahmini için protokol.

    Gelecek pozisyonları tahmin eden algoritmalar.
    """

    def predict(self, history: List[Point2D], steps_ahead: int = 1) -> List[Point2D]:
        """
        Gelecek pozisyonları tahmin eder.

        Args:
            history: Geçmiş pozisyonlar
            steps_ahead: Kaç adım ileri tahmin

        Returns:
            List[Point2D]: Tahmin edilen pozisyonlar
        """
        ...

    def update_model(self, actual_position: Point2D, predicted_position: Point2D):
        """
        Tahmin modelini günceller.

        Args:
            actual_position: Gerçek pozisyon
            predicted_position: Tahmin edilen pozisyon
        """
        ...


class ITrackingService(ABC):
    """
    İzleme servisi için abstract base class.

    Ana izleme iş mantığını tanımlar.
    """

    @abstractmethod
    def track_frame(self, request: TrackingRequest) -> TrackingResult:
        """
        Tek frame için izleme yapar.

        Args:
            request: İzleme isteği

        Returns:
            TrackingResult: İzleme sonucu
        """

    @abstractmethod
    def track_sequence(
        self,
        images: List[np.ndarray],
        initial_points: List[TrackedPoint],
        parameters: TrackingParameters,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TrackingSession:
        """
        Görüntü dizisi üzerinde izleme yapar.

        Args:
            images: Görüntü dizisi
            initial_points: Başlangıç noktaları
            parameters: İzleme parametreleri
            progress_callback: İlerleme bildirimi

        Returns:
            TrackingSession: İzleme oturumu
        """

    @abstractmethod
    def add_tracker(self, point_id: str, tracker: ITracker):
        """
        Yeni izleyici ekler.

        Args:
            point_id: Nokta ID'si
            tracker: İzleyici instance
        """

    @abstractmethod
    def remove_tracker(self, point_id: str):
        """
        İzleyiciyi kaldırır.

        Args:
            point_id: Nokta ID'si
        """

    @abstractmethod
    def reset_trackers(self):
        """Tüm izleyicileri sıfırlar."""


class ITrackingSessionManager(Protocol):
    """
    İzleme oturumu yönetimi için protokol.
    """

    def create_session(self, start_frame: int, parameters: TrackingParameters) -> TrackingSession:
        """
        Yeni oturum oluşturur.

        Args:
            start_frame: Başlangıç frame'i
            parameters: İzleme parametreleri

        Returns:
            TrackingSession: Yeni oturum
        """
        ...

    def get_session(self, session_id: str) -> Optional[TrackingSession]:
        """
        Oturumu döndürür.

        Args:
            session_id: Oturum ID'si

        Returns:
            Optional[TrackingSession]: Oturum veya None
        """
        ...

    def save_session(self, session: TrackingSession) -> bool:
        """
        Oturumu kaydeder.

        Args:
            session: Kaydedilecek oturum

        Returns:
            bool: Başarılı mı?
        """
        ...

    def load_session(self, file_path: str) -> Optional[TrackingSession]:
        """
        Oturumu dosyadan yükler.

        Args:
            file_path: Dosya yolu

        Returns:
            Optional[TrackingSession]: Yüklenen oturum
        """
        ...

    def get_active_sessions(self) -> List[TrackingSession]:
        """
        Aktif oturumları döndürür.

        Returns:
            List[TrackingSession]: Aktif oturumlar
        """
        ...


class ITrackingValidator(Protocol):
    """
    İzleme doğrulama için protokol.
    """

    def validate_tracking(
        self, tracked_point: TrackedPoint, image: np.ndarray
    ) -> Tuple[bool, float]:
        """
        İzleme sonucunu doğrular.

        Args:
            tracked_point: İzlenen nokta
            image: Mevcut görüntü

        Returns:
            Tuple: (geçerli_mi, güven_skoru)
        """
        ...

    def validate_trajectory(self, trajectory: List[Point2D], max_acceleration: float) -> bool:
        """
        Hareket yörüngesini doğrular.

        Args:
            trajectory: Hareket yörüngesi
            max_acceleration: Maksimum ivme

        Returns:
            bool: Geçerli mi?
        """
        ...


class ITrackingVisualization(Protocol):
    """
    İzleme görselleştirme için protokol.
    """

    def visualize_tracking(
        self,
        image: np.ndarray,
        tracked_points: List[TrackedPoint],
        show_trajectory: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        İzleme sonuçlarını görselleştirir.

        Args:
            image: Görüntü
            tracked_points: İzlenen noktalar
            show_trajectory: Yörünge göster
            show_confidence: Güven göster

        Returns:
            np.ndarray: Görselleştirilmiş görüntü
        """
        ...

    def draw_search_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (255, 255, 0),
    ) -> np.ndarray:
        """
        Arama bölgesini çizer.

        Args:
            image: Görüntü
            region: Bölge (x, y, w, h)
            color: Renk

        Returns:
            np.ndarray: İşaretlenmiş görüntü
        """
        ...

    def create_tracking_overlay(
        self, tracked_points: List[TrackedPoint], image_size: Tuple[int, int], alpha: float = 0.5
    ) -> np.ndarray:
        """
        İzleme overlay'i oluşturur.

        Args:
            tracked_points: İzlenen noktalar
            image_size: Görüntü boyutu (h, w)
            alpha: Saydamlık

        Returns:
            np.ndarray: Overlay görüntüsü
        """
        ...


class ITrackerFactory(Protocol):
    """
    İzleyici factory için protokol.
    """

    def create_tracker(self, method: str, parameters: Dict[str, Any]) -> ITracker:
        """
        İzleyici oluşturur.

        Args:
            method: İzleme yöntemi
            parameters: Yöntem parametreleri

        Returns:
            ITracker: İzleyici instance
        """
        ...

    def get_available_methods(self) -> List[str]:
        """
        Mevcut izleme yöntemlerini döndürür.

        Returns:
            List[str]: Yöntem listesi
        """
        ...
