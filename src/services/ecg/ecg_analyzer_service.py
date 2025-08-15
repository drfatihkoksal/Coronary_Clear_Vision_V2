"""
ECG Analyzer Service

Ana ECG analiz servisi.
Tüm ECG işleme servislerini koordine eder.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Optional
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from src.domain.models.ecg_models import (
    ECGSignal,
    ECGProcessingRequest,
    ECGAnalysisResult,
    RPeak,
    RRInterval,
    CardiacPhaseAnalysis,
    ECGQualityMetrics,
    HRVMetrics,
)
from src.domain.interfaces.ecg_interfaces import (
    IECGAnalyzer,
    IECGProcessor,
    IRPeakDetector,
    ICardiacPhaseDetector,
    IECGQualityAssessor,
    IHRVCalculator,
)

logger = logging.getLogger(__name__)


class ECGAnalyzerService(IECGAnalyzer):
    """
    Ana ECG analiz servisi.

    Bu servis:
    - ECG sinyal işleme
    - R-peak tespiti
    - Kardiyak faz tespiti
    - Kalite değerlendirmesi
    - HRV hesaplama işlemlerini koordine eder

    Dependency Injection kullanarak loose coupling sağlar.
    """

    def __init__(
        self,
        processor: IECGProcessor,
        rpeak_detector: IRPeakDetector,
        phase_detector: ICardiacPhaseDetector,
        quality_assessor: IECGQualityAssessor,
        hrv_calculator: IHRVCalculator,
    ):
        """
        ECGAnalyzerService constructor.

        Args:
            processor: ECG işleme servisi
            rpeak_detector: R-peak tespit servisi
            phase_detector: Kardiyak faz tespit servisi
            quality_assessor: Kalite değerlendirme servisi
            hrv_calculator: HRV hesaplama servisi
        """
        self._processor = processor
        self._rpeak_detector = rpeak_detector
        self._phase_detector = phase_detector
        self._quality_assessor = quality_assessor
        self._hrv_calculator = hrv_calculator

        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info("ECGAnalyzerService initialized with all dependencies")

    def analyze(self, request: ECGProcessingRequest) -> ECGAnalysisResult:
        """
        ECG analizi yap.

        İşlem sırası:
        1. Sinyal işleme (filtreleme)
        2. Kalite değerlendirmesi (paralel)
        3. R-peak tespiti
        4. RR interval hesaplama
        5. HRV hesaplama (paralel)
        6. Kardiyak faz tespiti (paralel)

        Args:
            request: İşleme isteği

        Returns:
            ECGAnalysisResult: Analiz sonuçları
        """
        start_time = time.time()

        try:
            # 1. Sinyal işleme
            if request.apply_filters:
                logger.info("Processing ECG signal...")
                processed_signal = self._processor.process_signal(request.signal)
            else:
                processed_signal = request.signal

            # 2. Kalite değerlendirmesi (asenkron başlat)
            quality_future: Optional[Future] = None
            if request.assess_quality:
                logger.info("Starting quality assessment...")
                quality_future = self._executor.submit(
                    self._quality_assessor.assess_quality, processed_signal
                )

            # 3. R-peak tespiti
            r_peaks: List[RPeak] = []
            if request.detect_r_peaks:
                logger.info("Detecting R-peaks...")
                r_peaks = self._rpeak_detector.detect_r_peaks(processed_signal)
                logger.info(f"Found {len(r_peaks)} R-peaks")

            # 4. RR interval hesaplama
            rr_intervals: List[RRInterval] = []
            if r_peaks and len(r_peaks) > 1:
                rr_intervals = self.extract_rr_intervals(r_peaks)

            # 5. HRV hesaplama (asenkron başlat)
            hrv_future: Optional[Future] = None
            if request.calculate_hrv and rr_intervals:
                logger.info("Starting HRV calculation...")
                hrv_future = self._executor.submit(self._hrv_calculator.calculate_hrv, rr_intervals)

            # 6. Kardiyak faz tespiti (asenkron başlat)
            phase_future: Optional[Future] = None
            if request.detect_phases and r_peaks:
                logger.info("Starting cardiac phase detection...")
                phase_future = self._executor.submit(
                    self._phase_detector.detect_phases, processed_signal, r_peaks
                )

            # Asenkron işlemlerin sonuçlarını topla

            # Kalite sonucu
            quality_metrics: Optional[ECGQualityMetrics] = None
            if quality_future:
                try:
                    quality_metrics = quality_future.result(timeout=5.0)
                    logger.info(
                        f"Quality assessment completed: {quality_metrics.overall_quality.value}"
                    )
                except Exception as e:
                    logger.error(f"Quality assessment failed: {str(e)}")

            # HRV sonucu
            hrv_metrics: Optional[HRVMetrics] = None
            if hrv_future:
                try:
                    hrv_metrics = hrv_future.result(timeout=5.0)
                    logger.info(f"HRV calculation completed: Mean RR = {hrv_metrics.mean_rr:.1f}ms")
                except Exception as e:
                    logger.error(f"HRV calculation failed: {str(e)}")

            # Kardiyak faz sonucu
            cardiac_phases: Optional[CardiacPhaseAnalysis] = None
            if phase_future:
                try:
                    cardiac_phases = phase_future.result(timeout=5.0)
                    logger.info(f"Phase detection completed: {len(cardiac_phases.cycles)} cycles")
                except Exception as e:
                    logger.error(f"Phase detection failed: {str(e)}")

            # İşleme süresi
            processing_time_ms = (time.time() - start_time) * 1000

            # Sonuç oluştur
            result = ECGAnalysisResult(
                signal=processed_signal,
                r_peaks=r_peaks,
                rr_intervals=rr_intervals,
                cardiac_phases=cardiac_phases,
                quality_metrics=quality_metrics,
                hrv_metrics=hrv_metrics,
                processing_time_ms=processing_time_ms,
            )

            logger.info(f"ECG analysis completed in {processing_time_ms:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"Error in ECG analysis: {str(e)}")
            # Hata durumunda kısmi sonuç döndür
            processing_time_ms = (time.time() - start_time) * 1000
            return ECGAnalysisResult(signal=request.signal, processing_time_ms=processing_time_ms)

    def extract_rr_intervals(self, r_peaks: List[RPeak]) -> List[RRInterval]:
        """
        R-peak'lerden RR intervalleri çıkar.

        Args:
            r_peaks: R-peak listesi

        Returns:
            List[RRInterval]: RR interval listesi
        """
        rr_intervals = []

        for i in range(1, len(r_peaks)):
            prev_peak = r_peaks[i - 1]
            current_peak = r_peaks[i]

            # RR interval süresi (ms)
            duration_ms = (current_peak.time - prev_peak.time) * 1000

            # Anlık kalp hızı (BPM)
            if duration_ms > 0:
                heart_rate = 60000 / duration_ms
            else:
                continue  # Geçersiz interval

            # RRInterval oluştur
            rr_interval = RRInterval(
                start_peak=prev_peak,
                end_peak=current_peak,
                duration_ms=duration_ms,
                heart_rate=heart_rate,
            )

            rr_intervals.append(rr_interval)

        return rr_intervals

    def analyze_segment(
        self, signal: ECGSignal, start_time: float, end_time: float
    ) -> ECGAnalysisResult:
        """
        Belirli bir zaman segmentini analiz et.

        Args:
            signal: ECG sinyali
            start_time: Başlangıç zamanı (saniye)
            end_time: Bitiş zamanı (saniye)

        Returns:
            ECGAnalysisResult: Segment analiz sonuçları
        """
        try:
            fs = signal.metadata.sampling_rate

            # İndeksleri hesapla
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)

            # Sınırları kontrol et
            start_idx = max(0, start_idx)
            end_idx = min(len(signal.data), end_idx)

            if start_idx >= end_idx:
                logger.error("Invalid segment boundaries")
                return ECGAnalysisResult(signal=signal)

            # Segment verisini al
            segment_data = signal.data[start_idx:end_idx]

            # Yeni ECGSignal oluştur
            from src.domain.models.ecg_models import ECGMetadata

            segment_metadata = ECGMetadata(
                source=signal.metadata.source,
                sampling_rate=signal.metadata.sampling_rate,
                duration=(end_idx - start_idx) / fs,
                num_samples=len(segment_data),
                vendor=signal.metadata.vendor,
                modality=signal.metadata.modality,
                study_date=signal.metadata.study_date,
                additional_info=signal.metadata.additional_info,
            )

            segment_signal = ECGSignal(
                data=segment_data, metadata=segment_metadata, timestamp_offset=start_time
            )

            # Segment için analiz isteği oluştur
            segment_request = ECGProcessingRequest(
                signal=segment_signal,
                detect_r_peaks=True,
                detect_phases=True,
                calculate_hrv=True,
                assess_quality=True,
                apply_filters=True,
            )

            # Analiz yap
            return self.analyze(segment_request)

        except Exception as e:
            logger.error(f"Error analyzing segment: {str(e)}")
            return ECGAnalysisResult(signal=signal)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
