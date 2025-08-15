"""
Tracking Service Implementation

İzleme (tracking) işlemlerini yöneten ana servis.
Clean Architecture ve SOLID prensipleri uygulanmıştır.
"""

from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import logging
from datetime import datetime
import uuid

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from src.domain.models.tracking_models import (
    TrackingRequest,
    TrackingResult,
    TrackedPoint,
    TrackingFrame,
    TrackingSession,
    TrackingParameters,
    Point2D,
    TrackingStatus,
    TrackingMethod,
)
from src.domain.interfaces.tracking_interfaces import (
    ITracker,
    IMotionPredictor,
    ITrackingService,
    ITrackingValidator,
)

logger = logging.getLogger(__name__)


class TrackingService(QObject, ITrackingService):
    """
    İzleme servisi ana implementasyonu.

    Bu servis:
    - Birden fazla nokta izleme
    - Farklı izleme algoritmaları
    - Hareket tahmini
    - İzleme doğrulama
    - Performans optimizasyonu

    Signals:
        tracking_progress: İzleme ilerlemesi
        tracking_completed: İzleme tamamlandı
        tracking_failed: İzleme başarısız
        point_lost: Nokta kaybedildi
        point_recovered: Nokta tekrar bulundu
    """

    # Signals
    tracking_progress = pyqtSignal(int, int)  # current, total
    tracking_completed = pyqtSignal(TrackingSession)
    tracking_failed = pyqtSignal(str)  # error message
    point_lost = pyqtSignal(str)  # point_id
    point_recovered = pyqtSignal(str)  # point_id

    def __init__(
        self,
        tracker_factory: Optional[Callable[[str, Dict], ITracker]] = None,
        motion_predictor: Optional[IMotionPredictor] = None,
        validator: Optional[ITrackingValidator] = None,
        fps: float = 30.0,
    ):
        """
        TrackingService constructor.

        Args:
            tracker_factory: İzleyici oluşturma fonksiyonu
            motion_predictor: Hareket tahmin edici
            validator: İzleme doğrulayıcı
            fps: Video frame rate
        """
        super().__init__()

        self._tracker_factory = tracker_factory or self._default_tracker_factory
        self._motion_predictor = motion_predictor
        self._validator = validator
        self.fps = fps

        # İzleyici havuzu - her nokta için birden fazla izleyici (hybrid tracking)
        self._trackers: Dict[str, Dict[str, ITracker]] = {}  # {point_id: {method: tracker}}

        # Aktif oturum
        self._current_session: Optional[TrackingSession] = None

        # İşlem thread'i
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional["TrackingWorker"] = None

        # FPS'e göre varsayılan parametreler
        self._setup_fps_parameters()

        logger.info(f"TrackingService initialized with FPS={fps}")

    def _setup_fps_parameters(self):
        """FPS'e göre varsayılan parametreleri ayarla."""
        if self.fps < 20:
            # 15 FPS için optimize edilmiş parametreler
            self.default_params = {
                "search_radius": 60,
                "template_size": 25,
                "confidence_threshold": 0.4,
                "max_motion": 80,
                "update_rate": 0.05,
                "enable_prediction": True,
                "use_hybrid": True,  # Hybrid tracking aktif
            }
        else:
            # Normal FPS için parametreler
            self.default_params = {
                "search_radius": 30,
                "template_size": 21,
                "confidence_threshold": 0.5,
                "max_motion": 40,
                "update_rate": 0.1,
                "enable_prediction": True,
                "use_hybrid": False,
            }

    def track_frame(self, request: TrackingRequest) -> TrackingResult:
        """
        Tek frame için izleme yapar.

        Args:
            request: İzleme isteği

        Returns:
            TrackingResult: İzleme sonucu
        """
        try:
            # Başlangıç zamanı
            start_time = datetime.now()

            # Sonuç verilerini topla
            tracked_points = []
            failed_points = []

            # Her nokta için izleme yap
            for point in request.points:
                try:
                    # İzleyici al veya oluştur
                    tracker = self._get_or_create_tracker(
                        point.id, request.method, request.parameters
                    )

                    # İlk frame ise başlat
                    if point.frame_number == 0 or point.status == TrackingStatus.MANUAL:
                        success = tracker.initialize(
                            request.image, point.current_position, request.parameters.template_size
                        )
                        if not success:
                            logger.warning(f"Failed to initialize tracker for point {point.id}")
                            failed_points.append(point.id)
                            continue

                    # Hareket tahmini yap
                    predicted_position = None
                    if self._motion_predictor and request.parameters.enable_prediction:
                        trajectory = point.get_trajectory()
                        if len(trajectory) >= 3:
                            predicted_position = self._motion_predictor.predict(trajectory, 1)[0]

                    # Arama bölgesi belirle
                    search_region = self._calculate_search_region(
                        point.current_position,
                        predicted_position,
                        request.parameters.search_radius,
                        request.image.shape,
                    )

                    # İzle
                    success, new_position, confidence = tracker.track(request.image, search_region)
                    
                    # Hybrid tracking - düşük güven durumunda alternatif tracker kullan
                    if self.default_params.get("use_hybrid", False) and confidence < 0.6:
                        # Optical flow ile dene
                        optical_tracker = self._get_or_create_tracker(
                            point.id, TrackingMethod.OPTICAL_FLOW, request.parameters
                        )
                        optical_success, optical_pos, optical_conf = optical_tracker.track(
                            request.image, search_region
                        )
                        
                        if optical_success and optical_conf > confidence:
                            success = optical_success
                            new_position = optical_pos
                            confidence = optical_conf
                            logger.debug(f"Switched to optical flow for point {point.id}")

                    if success and confidence >= request.parameters.confidence_threshold:
                        # Hareket kontrolü
                        motion_distance = point.current_position.distance_to(new_position)
                        if motion_distance <= request.parameters.max_motion:
                            # Doğrulama yap
                            if self._validator:
                                is_valid, adjusted_confidence = self._validator.validate_tracking(
                                    TrackedPoint(
                                        id=point.id,
                                        name=point.name,
                                        current_position=new_position,
                                        confidence=confidence,
                                    ),
                                    request.image,
                                )
                                if not is_valid:
                                    logger.warning(
                                        f"Tracking validation failed for point {point.id}"
                                    )
                                    failed_points.append(point.id)
                                    continue
                                confidence = adjusted_confidence

                            # Başarılı izleme
                            frame_data = TrackingFrame(
                                frame_number=point.frame_number + 1,
                                position=new_position,
                                confidence=confidence,
                                status=TrackingStatus.ACTIVE,
                                search_region=search_region,
                            )

                            point.add_frame(frame_data)
                            tracked_points.append(point)

                            # Şablon güncelle
                            if request.parameters.adaptive_template:
                                tracker.update_template(
                                    request.image, new_position, request.parameters.update_rate
                                )

                            # Hareket tahmin modelini güncelle
                            if self._motion_predictor and predicted_position:
                                self._motion_predictor.update_model(
                                    new_position, predicted_position
                                )
                        else:
                            logger.warning(
                                f"Motion limit exceeded for point {point.id}: "
                                f"{motion_distance:.1f} > {request.parameters.max_motion}"
                            )
                            failed_points.append(point.id)
                    else:
                        logger.info(
                            f"Tracking failed for point {point.id}: confidence={confidence:.2f}"
                        )
                        failed_points.append(point.id)

                except Exception as e:
                    logger.error(f"Error tracking point {point.id}: {str(e)}")
                    failed_points.append(point.id)

            # İşlem süresi
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Sonuç oluştur
            result = TrackingResult(
                success=len(tracked_points) > 0,
                tracked_points=tracked_points,
                failed_points=failed_points,
                processing_time_ms=processing_time,
                frame_number=request.points[0].frame_number + 1 if request.points else 0,
            )

            return result

        except Exception as e:
            logger.error(f"Track frame error: {str(e)}")
            return TrackingResult(success=False, error_message=str(e))

    def track_sequence(
        self,
        images: List[np.ndarray],
        initial_points: List[TrackedPoint],
        parameters: TrackingParameters,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TrackingSession:
        """
        Görüntü dizisi üzerinde izleme yapar.

        Bu metod arkaplan thread'inde çalışır.

        Args:
            images: Görüntü dizisi
            initial_points: Başlangıç noktaları
            parameters: İzleme parametreleri
            progress_callback: İlerleme bildirimi

        Returns:
            TrackingSession: İzleme oturumu
        """
        # Yeni oturum oluştur
        session = TrackingSession(
            session_id=str(uuid.uuid4()),
            start_frame=0,
            end_frame=len(images) - 1,
            parameters=parameters,
            method=TrackingMethod.TEMPLATE_MATCHING,  # Default
        )

        # Başlangıç noktalarını ekle
        for point in initial_points:
            session.add_point(point)

        self._current_session = session

        # Worker thread başlat
        self._start_tracking_worker(images, initial_points, parameters, progress_callback)

        return session

    def add_tracker(self, point_id: str, tracker: ITracker, method: Optional[TrackingMethod] = None):
        """
        Yeni izleyici ekler.

        Args:
            point_id: Nokta ID'si
            tracker: İzleyici instance
            method: İzleme yöntemi
        """
        if point_id not in self._trackers:
            self._trackers[point_id] = {}
        
        method_key = method.value if method else "default"
        self._trackers[point_id][method_key] = tracker
        logger.debug(f"Added tracker for point {point_id}: {tracker.method_name}")

    def remove_tracker(self, point_id: str, method: Optional[TrackingMethod] = None):
        """
        İzleyiciyi kaldırır.

        Args:
            point_id: Nokta ID'si
            method: Kaldırılacak spesifik method (None ise tümü)
        """
        if point_id in self._trackers:
            if method:
                method_key = method.value
                if method_key in self._trackers[point_id]:
                    del self._trackers[point_id][method_key]
            else:
                del self._trackers[point_id]
            logger.debug(f"Removed tracker for point {point_id}")

    def reset_trackers(self):
        """Tüm izleyicileri sıfırlar."""
        self._trackers.clear()
        logger.info("All trackers reset")

    def _get_or_create_tracker(
        self, point_id: str, method: TrackingMethod, parameters: TrackingParameters
    ) -> ITracker:
        """
        İzleyici al veya oluştur.

        Args:
            point_id: Nokta ID'si
            method: İzleme yöntemi
            parameters: Parametreler

        Returns:
            ITracker: İzleyici instance
        """
        if point_id not in self._trackers:
            self._trackers[point_id] = {}
        
        method_key = method.value
        if method_key not in self._trackers[point_id]:
            # FPS ve diğer parametrelerle tracker oluştur
            tracker_params = {
                "template_size": parameters.template_size,
                "search_radius": parameters.search_radius,
                "fps": self.fps,
            }
            
            # FPS'e göre ek parametreler ekle
            if self.fps < 20:
                tracker_params.update({
                    "use_pyramid": True,
                    "pyramid_levels": 3,
                })
            
            tracker = self._tracker_factory(method.value, tracker_params)
            self._trackers[point_id][method_key] = tracker

        return self._trackers[point_id][method_key]

    def _default_tracker_factory(self, method: str, params: Dict) -> ITracker:
        """
        Varsayılan izleyici factory.

        Args:
            method: İzleme yöntemi
            params: Parametreler

        Returns:
            ITracker: İzleyici instance
        """
        from src.services.tracking.trackers import TemplateMatchingTracker, OpticalFlowTracker

        if method == TrackingMethod.TEMPLATE_MATCHING.value or method == "template_matching":
            return TemplateMatchingTracker(**params)
        elif method == TrackingMethod.OPTICAL_FLOW.value or method == "optical_flow":
            return OpticalFlowTracker(**params)
        else:
            # Varsayılan olarak template matching kullan
            logger.warning(f"Unknown tracking method: {method}, using template matching")
            return TemplateMatchingTracker(**params)

    def _calculate_search_region(
        self,
        current_position: Point2D,
        predicted_position: Optional[Point2D],
        search_radius: int,
        image_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Arama bölgesi hesapla.

        Args:
            current_position: Mevcut pozisyon
            predicted_position: Tahmin edilen pozisyon
            search_radius: Arama yarıçapı
            image_shape: Görüntü boyutu

        Returns:
            Tuple: (x, y, width, height)
        """
        # Merkez nokta
        if predicted_position:
            center_x = predicted_position.x
            center_y = predicted_position.y
        else:
            center_x = current_position.x
            center_y = current_position.y

        # Bölge sınırları
        x = max(0, int(center_x - search_radius))
        y = max(0, int(center_y - search_radius))
        x2 = min(image_shape[1], int(center_x + search_radius))
        y2 = min(image_shape[0], int(center_y + search_radius))

        width = x2 - x
        height = y2 - y

        return (x, y, width, height)

    def _start_tracking_worker(
        self,
        images: List[np.ndarray],
        initial_points: List[TrackedPoint],
        parameters: TrackingParameters,
        progress_callback: Optional[Callable[[int, int], None]],
    ):
        """
        Arkaplan izleme worker'ı başlat.

        Args:
            images: Görüntü dizisi
            initial_points: Başlangıç noktaları
            parameters: Parametreler
            progress_callback: İlerleme callback'i
        """
        # Mevcut worker'ı durdur
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait()

        # Yeni worker oluştur
        self._worker_thread = QThread()
        self._worker = TrackingWorker(self, images, initial_points, parameters)

        # Worker'ı thread'e taşı
        self._worker.moveToThread(self._worker_thread)

        # Sinyalleri bağla
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        if progress_callback:
            self._worker.progress.connect(progress_callback)

        # Başlat
        self._worker_thread.start()

    def _on_worker_progress(self, current: int, total: int):
        """Worker ilerleme bildirimi."""
        self.tracking_progress.emit(current, total)

    def _on_worker_completed(self, session: TrackingSession):
        """Worker tamamlandı."""
        self._current_session = session
        self.tracking_completed.emit(session)

    def _on_worker_failed(self, error: str):
        """Worker başarısız."""
        self.tracking_failed.emit(error)


class TrackingWorker(QObject):
    """
    Arkaplan izleme worker'ı.

    Uzun süren izleme işlemlerini arkaplan thread'inde çalıştırır.
    """

    # Signals
    progress = pyqtSignal(int, int)  # current, total
    completed = pyqtSignal(TrackingSession)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        service: TrackingService,
        images: List[np.ndarray],
        initial_points: List[TrackedPoint],
        parameters: TrackingParameters,
    ):
        """
        Worker constructor.

        Args:
            service: Ana tracking servisi
            images: Görüntü dizisi
            initial_points: Başlangıç noktaları
            parameters: İzleme parametreleri
        """
        super().__init__()
        self._service = service
        self._images = images
        self._initial_points = initial_points
        self._parameters = parameters
        self._is_cancelled = False

    def run(self):
        """Ana işlem döngüsü."""
        try:
            # Oturum oluştur
            session = TrackingSession(
                session_id=str(uuid.uuid4()),
                start_frame=0,
                end_frame=len(self._images) - 1,
                parameters=self._parameters,
                method=TrackingMethod.TEMPLATE_MATCHING,
            )

            # İlk noktaları ekle
            current_points = []
            for point in self._initial_points:
                session.add_point(point)
                current_points.append(point)

            # Her frame için izleme yap
            total_frames = len(self._images)

            for i, image in enumerate(self._images):
                if self._is_cancelled:
                    break

                # İlerleme bildirimi
                self.progress.emit(i + 1, total_frames)

                # İlk frame'i atla (başlangıç noktaları zaten var)
                if i == 0:
                    continue

                # İzleme isteği oluştur
                request = TrackingRequest(
                    image=image,
                    points=current_points,
                    method=TrackingMethod.TEMPLATE_MATCHING,
                    parameters=self._parameters,
                    previous_image=self._images[i - 1] if i > 0 else None,
                )

                # İzle
                result = self._service.track_frame(request)

                if result.success:
                    # Başarılı noktaları güncelle
                    current_points = result.tracked_points

                    # Oturuma ekle
                    for point in current_points:
                        session_point = session.get_point(point.id)
                        if session_point:
                            # Frame verisini ekle
                            session_point.history.append(point.history[-1])
                else:
                    logger.warning(f"Frame {i} tracking failed: {result.error_message}")
                    # Başarısız noktaları işaretle
                    for point_id in result.failed_points:
                        for point in current_points:
                            if point.id == point_id:
                                point.status = TrackingStatus.LOST

            # Tamamlandı
            self.completed.emit(session)

        except Exception as e:
            logger.error(f"Tracking worker error: {str(e)}")
            self.failed.emit(str(e))
        finally:
            self.finished.emit()

    def cancel(self):
        """İşlemi iptal et."""
        self._is_cancelled = True
