"""
QCA Analysis Service

QCA analizinin business logic'ini yöneten servis.
UI'dan bağımsız olarak analiz işlemlerini gerçekleştirir.
Clean architecture ve SOLID prensipleri uygulanmıştır.
"""

from typing import List, Optional, Dict, Any, Callable
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future

from src.domain.models.qca_models import (
    QCAAnalysisResult,
    QCAAnalysisRequest,
    QCASequentialResult,
    VesselMeasurement,
    StenosisData,
    AnalysisStatus,
    CalibrationData,
)
from src.domain.interfaces.qca_interfaces import (
    IQCAAnalysisService,
    IDiameterCalculator,
    IEdgeDetector,
    IStenosisDetector,
)

logger = logging.getLogger(__name__)


class QCAAnalysisService(IQCAAnalysisService):
    """
    QCA analiz servisi implementasyonu.

    Bu servis QCA analizinin tüm business logic'ini yönetir.
    UI'dan bağımsız olarak çalışır ve test edilebilir.
    """

    def __init__(
        self,
        diameter_calculator: IDiameterCalculator,
        edge_detector: IEdgeDetector,
        stenosis_detector: IStenosisDetector,
    ):
        """
        Servisi dependency injection ile başlat.

        Args:
            diameter_calculator: Çap hesaplama algoritması
            edge_detector: Kenar tespit algoritması
            stenosis_detector: Stenoz tespit algoritması
        """
        self._diameter_calculator = diameter_calculator
        self._edge_detector = edge_detector
        self._stenosis_detector = stenosis_detector
        self._executor = ThreadPoolExecutor(max_workers=4)

    def analyze_vessel(self, request: QCAAnalysisRequest) -> QCAAnalysisResult:
        """
        Tek frame için damar analizi yap.

        Args:
            request: Analiz parametreleri

        Returns:
            QCAAnalysisResult: Analiz sonucu
        """
        start_time = time.time()
        result = QCAAnalysisResult(
            frame_number=request.frame_number,
            calibration=request.calibration,
            status=AnalysisStatus.IN_PROGRESS,
        )

        try:
            # 1. Segmentasyon sonucunu al
            segmentation = request.segmentation_result
            if not self._validate_segmentation(segmentation):
                raise ValueError("Invalid segmentation result")

            # 2. Damar merkez hattını çıkar
            centerline = self._extract_centerline(segmentation)
            result.centerline = centerline.tolist()

            # 3. Kenar tespiti yap
            edges = self._detect_edges(
                segmentation.image, segmentation.contour, request.analysis_options
            )

            # 4. Çap ölçümlerini hesapla
            measurements = self._calculate_diameters(edges, centerline, request.calibration)
            result.measurements = measurements

            # 5. Stenoz analizi (opsiyonel)
            if request.analysis_options.get("detect_stenosis", True):
                stenosis = self._detect_stenosis(measurements)
                result.stenosis_data = stenosis

            # 6. Damar tipini belirle
            result.vessel_type = self._identify_vessel_type(centerline)

            # Başarılı tamamlandı
            result.status = AnalysisStatus.COMPLETED
            result.analysis_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"QCA analysis completed for frame {request.frame_number} "
                f"in {result.analysis_time_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(f"QCA analysis failed: {e}", exc_info=True)
            result.status = AnalysisStatus.FAILED
            result.error_message = str(e)

        return result

    def analyze_sequential(
        self,
        requests: List[QCAAnalysisRequest],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> QCASequentialResult:
        """
        Ardışık frame'ler için paralel analiz yap.

        Args:
            requests: Analiz istekleri listesi
            progress_callback: İlerleme bildirimi için callback (current, total)

        Returns:
            QCASequentialResult: Ardışık analiz sonuçları
        """
        logger.info(f"Starting sequential QCA analysis for {len(requests)} frames")

        sequential_result = QCASequentialResult()
        futures: Dict[Future, int] = {}

        # Paralel analiz başlat
        for request in requests:
            future = self._executor.submit(self.analyze_vessel, request)
            futures[future] = request.frame_number

        # Sonuçları topla
        completed = 0
        for future in futures:
            frame_number = futures[future]
            try:
                result = future.result(timeout=30)  # 30 saniye timeout
                sequential_result.results_by_frame[frame_number] = result
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(requests))

            except Exception as e:
                logger.error(f"Frame {frame_number} analysis failed: {e}")
                # Başarısız analiz için boş sonuç ekle
                sequential_result.results_by_frame[frame_number] = QCAAnalysisResult(
                    frame_number=frame_number, status=AnalysisStatus.FAILED, error_message=str(e)
                )

        # Özet istatistikleri hesapla
        self._calculate_summary_statistics(sequential_result)

        # Zamansal analizi yap
        self._perform_temporal_analysis(sequential_result)

        logger.info(
            f"Sequential analysis completed. "
            f"Success rate: {sequential_result.successful_count}/{sequential_result.frame_count}"
        )

        return sequential_result

    def set_diameter_calculator(self, calculator: IDiameterCalculator):
        """Çap hesaplama algoritmasını değiştir"""
        self._diameter_calculator = calculator
        logger.info(f"Diameter calculator changed to: {calculator.method_name}")

    def set_edge_detector(self, detector: IEdgeDetector):
        """Kenar tespit algoritmasını değiştir"""
        self._edge_detector = detector
        logger.info("Edge detector updated")

    def set_stenosis_detector(self, detector: IStenosisDetector):
        """Stenoz tespit algoritmasını değiştir"""
        self._stenosis_detector = detector
        logger.info("Stenosis detector updated")

    # Private helper methods

    def _validate_segmentation(self, segmentation: Any) -> bool:
        """
        Segmentasyon sonucunu doğrula.

        Args:
            segmentation: Doğrulanacak segmentasyon

        Returns:
            bool: Geçerli mi?
        """
        # Gerçek implementasyonda detaylı doğrulama yapılacak
        return (
            hasattr(segmentation, "image")
            and hasattr(segmentation, "contour")
            and segmentation.image is not None
            and segmentation.contour is not None
        )

    def _extract_centerline(self, segmentation: Any) -> np.ndarray:
        """
        Segmentasyondan damar merkez hattını çıkar.

        Args:
            segmentation: Segmentasyon sonucu

        Returns:
            np.ndarray: Merkez hattı noktaları
        """
        # Basitleştirilmiş implementasyon
        # Gerçek implementasyonda skeletonization veya
        # medial axis transform kullanılacak
        contour = np.array(segmentation.contour)

        # Konturu smooth et
        if len(contour) > 10:
            from scipy.interpolate import UnivariateSpline

            t = np.linspace(0, 1, len(contour))
            x_spline = UnivariateSpline(t, contour[:, 0], s=5)
            y_spline = UnivariateSpline(t, contour[:, 1], s=5)

            t_smooth = np.linspace(0, 1, 100)
            centerline = np.column_stack([x_spline(t_smooth), y_spline(t_smooth)])
        else:
            centerline = contour

        return centerline

    def _detect_edges(
        self, image: np.ndarray, initial_contour: np.ndarray, options: Dict[str, Any]
    ) -> np.ndarray:
        """
        Damar kenarlarını tespit et.

        Args:
            image: Görüntü
            initial_contour: Başlangıç konturu
            options: Analiz seçenekleri

        Returns:
            np.ndarray: Kenar noktaları
        """
        return self._edge_detector.detect_edges(image, initial_contour, **options)

    def _calculate_diameters(
        self, edges: np.ndarray, centerline: np.ndarray, calibration: CalibrationData
    ) -> List[VesselMeasurement]:
        """
        Damar çaplarını hesapla.

        Args:
            edges: Kenar noktaları
            centerline: Merkez hattı
            calibration: Kalibrasyon

        Returns:
            List[VesselMeasurement]: Çap ölçümleri
        """
        return self._diameter_calculator.calculate_diameters(edges, centerline, calibration)

    def _detect_stenosis(self, measurements: List[VesselMeasurement]) -> Optional[StenosisData]:
        """
        Stenoz tespit et.

        Args:
            measurements: Çap ölçümleri

        Returns:
            Optional[StenosisData]: Stenoz bilgileri
        """
        if not measurements:
            return None

        return self._stenosis_detector.detect_stenosis(measurements)

    def _identify_vessel_type(self, centerline: np.ndarray) -> "VesselType":
        """
        Damar tipini belirle.

        Basitleştirilmiş implementasyon - gerçekte görüntü
        özelliklerine ve konuma göre belirlenir.

        Args:
            centerline: Damar merkez hattı

        Returns:
            VesselType: Tahmin edilen damar tipi
        """
        from src.domain.models.qca_models import VesselType

        # Şimdilik UNKNOWN döndür
        # Gerçek implementasyonda ML modeli veya
        # rule-based sistem kullanılacak
        return VesselType.UNKNOWN

    def _calculate_summary_statistics(self, result: QCASequentialResult):
        """
        Özet istatistikleri hesapla.

        Args:
            result: Ardışık analiz sonucu
        """
        successful_results = [r for r in result.results_by_frame.values() if r.is_successful]

        if not successful_results:
            return

        # Ortalama çap istatistikleri
        all_diameters = []
        for r in successful_results:
            if r.mean_diameter_mm:
                all_diameters.append(r.mean_diameter_mm)

        if all_diameters:
            result.summary_statistics.update(
                {
                    "mean_diameter_overall": np.mean(all_diameters),
                    "std_diameter_overall": np.std(all_diameters),
                    "min_diameter_overall": np.min(all_diameters),
                    "max_diameter_overall": np.max(all_diameters),
                    "diameter_variation_percent": (np.max(all_diameters) - np.min(all_diameters))
                    / np.mean(all_diameters)
                    * 100,
                }
            )

        # Stenoz istatistikleri
        stenosis_count = sum(1 for r in successful_results if r.has_stenosis)
        result.summary_statistics["stenosis_detection_rate"] = stenosis_count / len(
            successful_results
        )

        # Başarı oranı
        result.summary_statistics["success_rate"] = len(successful_results) / result.frame_count

    def _perform_temporal_analysis(self, result: QCASequentialResult):
        """
        Zamansal analiz yap.

        Args:
            result: Ardışık analiz sonucu
        """
        frames, diameters = result.get_diameter_curve()

        if len(frames) < 3:
            return

        # Çap değişim hızı
        diameter_changes = np.diff(diameters)
        frame_intervals = np.diff(frames)

        change_rates = diameter_changes / frame_intervals

        result.temporal_analysis.update(
            {
                "max_diameter_change_rate": np.max(np.abs(change_rates)),
                "mean_diameter_change_rate": np.mean(np.abs(change_rates)),
                "diameter_stability": 1.0 - (np.std(diameters) / np.mean(diameters)),
                "pulsatility_index": (np.max(diameters) - np.min(diameters)) / np.mean(diameters),
            }
        )

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
