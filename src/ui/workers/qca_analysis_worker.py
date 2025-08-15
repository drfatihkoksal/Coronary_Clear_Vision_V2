"""
QCA Analysis Worker

QCA analizini arkaplan thread'inde gerçekleştiren worker.
UI'yı dondurmadan ağır hesaplamaları yapar.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from typing import Dict, List, Optional, Any
import logging

from src.domain.models.qca_models import QCAAnalysisResult, QCAAnalysisRequest, QCASequentialResult
from src.services.qca_analysis_service import QCAAnalysisService

logger = logging.getLogger(__name__)


class QCAAnalysisWorker(QThread):
    """
    Tekli QCA analizi için worker thread.

    Signals:
        started: Analiz başladı
        progress: İlerleme güncellemesi (percentage, message)
        finished: Analiz tamamlandı (result)
        error: Hata oluştu (error_message)
    """

    # Signals
    started = pyqtSignal()
    progress = pyqtSignal(int, str)  # percentage, message
    finished = pyqtSignal(QCAAnalysisResult)
    error = pyqtSignal(str)

    def __init__(
        self,
        service: QCAAnalysisService,
        request: QCAAnalysisRequest,
        parent: Optional[QObject] = None,
    ):
        """
        Worker'ı başlatır.

        Args:
            service: QCA analiz servisi
            request: Analiz isteği
            parent: Parent QObject
        """
        super().__init__(parent)
        self._service = service
        self._request = request
        self._is_cancelled = False

    def run(self):
        """
        Thread'de çalışacak ana metod.

        QThread.run() metodunu override eder.
        """
        try:
            # Başlangıç sinyali
            self.started.emit()
            self.progress.emit(0, "Initializing QCA analysis...")

            if self._is_cancelled:
                return

            # İlerleme bildirimleri
            self.progress.emit(20, "Extracting vessel centerline...")

            if self._is_cancelled:
                return

            self.progress.emit(40, "Detecting vessel edges...")

            if self._is_cancelled:
                return

            self.progress.emit(60, "Calculating diameters...")

            if self._is_cancelled:
                return

            self.progress.emit(80, "Analyzing stenosis...")

            # Gerçek analizi yap
            result = self._service.analyze_vessel(self._request)

            if self._is_cancelled:
                return

            # Tamamlandı
            self.progress.emit(100, "Analysis complete")
            self.finished.emit(result)

        except Exception as e:
            logger.error(f"QCA analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def cancel(self):
        """Analizi iptal eder."""
        self._is_cancelled = True
        logger.info("QCA analysis cancelled by user")


class QCASequentialWorker(QThread):
    """
    Ardışık QCA analizi için worker thread.

    Birden fazla frame'i paralel olarak analiz eder.

    Signals:
        started: Analiz başladı
        frame_started: Frame analizi başladı (frame_number)
        frame_completed: Frame analizi tamamlandı (frame_number, result)
        frame_failed: Frame analizi başarısız (frame_number, error)
        progress: Genel ilerleme (current, total, message)
        finished: Tüm analiz tamamlandı (sequential_result)
        error: Kritik hata (error_message)
    """

    # Signals
    started = pyqtSignal()
    frame_started = pyqtSignal(int)  # frame_number
    frame_completed = pyqtSignal(int, QCAAnalysisResult)  # frame_number, result
    frame_failed = pyqtSignal(int, str)  # frame_number, error
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(QCASequentialResult)
    error = pyqtSignal(str)

    def __init__(
        self,
        service: QCAAnalysisService,
        requests: List[QCAAnalysisRequest],
        parent: Optional[QObject] = None,
    ):
        """
        Sequential worker'ı başlatır.

        Args:
            service: QCA analiz servisi
            requests: Analiz istekleri listesi
            parent: Parent QObject
        """
        super().__init__(parent)
        self._service = service
        self._requests = requests
        self._is_cancelled = False

    def run(self):
        """
        Thread'de çalışacak ana metod.

        Paralel analiz yapar ve sonuçları toplar.
        """
        try:
            # Başlangıç
            self.started.emit()
            total_frames = len(self._requests)

            logger.info(f"Starting sequential QCA analysis for {total_frames} frames")

            # Progress callback
            def progress_callback(current: int, total: int):
                if not self._is_cancelled:
                    message = f"Processing frame {current} of {total}"
                    self.progress.emit(current, total, message)

            # Servisi kullanarak analiz yap
            result = self._service.analyze_sequential(
                self._requests, progress_callback=progress_callback
            )

            if self._is_cancelled:
                logger.info("Sequential analysis cancelled")
                return

            # Başarılı frame'leri bildir
            for frame_num, frame_result in result.results_by_frame.items():
                if frame_result.is_successful:
                    self.frame_completed.emit(frame_num, frame_result)
                else:
                    self.frame_failed.emit(frame_num, frame_result.error_message or "Unknown error")

            # Tamamlandı
            self.finished.emit(result)
            logger.info(
                f"Sequential analysis completed. "
                f"Success rate: {result.successful_count}/{result.frame_count}"
            )

        except Exception as e:
            logger.error(f"Sequential QCA analysis failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def cancel(self):
        """Analizi iptal eder."""
        self._is_cancelled = True
        logger.info("Sequential QCA analysis cancelled by user")


class QCARWSIntegrationWorker(QThread):
    """
    QCA ve RWS analizlerini entegre eden worker.

    QCA sonuçlarını kullanarak RWS analizi yapar.

    Signals:
        progress: İlerleme (stage, percentage, message)
        qca_completed: QCA tamamlandı
        rws_completed: RWS tamamlandı (rws_result)
        finished: Tüm analizler tamamlandı
        error: Hata (error_message)
    """

    # Signals
    progress = pyqtSignal(str, int, str)  # stage, percentage, message
    qca_completed = pyqtSignal()
    rws_completed = pyqtSignal(dict)  # rws_result
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        qca_service: QCAAnalysisService,
        rws_analyzer: Any,  # RWS analyzer instance
        qca_requests: List[QCAAnalysisRequest],
        cardiac_phase_info: Dict,
        parent: Optional[QObject] = None,
    ):
        """
        Integration worker'ı başlatır.

        Args:
            qca_service: QCA analiz servisi
            rws_analyzer: RWS analyzer instance
            qca_requests: QCA analiz istekleri
            cardiac_phase_info: Kardiyak faz bilgileri
            parent: Parent QObject
        """
        super().__init__(parent)
        self._qca_service = qca_service
        self._rws_analyzer = rws_analyzer
        self._qca_requests = qca_requests
        self._cardiac_phase_info = cardiac_phase_info
        self._is_cancelled = False

    def run(self):
        """Entegre analizi çalıştırır."""
        try:
            # QCA analizi
            self.progress.emit("QCA", 0, "Starting QCA analysis...")

            def qca_progress(current: int, total: int):
                if not self._is_cancelled:
                    percentage = int(current / total * 50)  # İlk %50 QCA için
                    self.progress.emit("QCA", percentage, f"QCA: Frame {current}/{total}")

            # QCA analizini yap
            qca_result = self._qca_service.analyze_sequential(
                self._qca_requests, progress_callback=qca_progress
            )

            if self._is_cancelled:
                return

            self.qca_completed.emit()

            # QCA sonuçlarını RWS formatına dönüştür
            self.progress.emit("RWS", 50, "Preparing RWS analysis...")

            qca_results_by_frame = {}
            for frame_num, result in qca_result.results_by_frame.items():
                if result.is_successful:
                    # RWS için gerekli formatı oluştur
                    qca_results_by_frame[frame_num] = {
                        "measurements": result.measurements,
                        "centerline": result.centerline,
                        "mean_diameter": result.mean_diameter_mm,
                    }

            if not qca_results_by_frame:
                raise ValueError("No successful QCA results for RWS analysis")

            # RWS analizi
            self.progress.emit("RWS", 75, "Calculating radial wall strain...")

            rws_result = self._rws_analyzer.calculate_rws(
                qca_results_by_frame, self._cardiac_phase_info
            )

            if self._is_cancelled:
                return

            self.progress.emit("RWS", 100, "RWS analysis complete")
            self.rws_completed.emit(rws_result)

            # Tamamlandı
            self.finished.emit()

        except Exception as e:
            logger.error(f"QCA-RWS integration failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def cancel(self):
        """Analizi iptal eder."""
        self._is_cancelled = True
        logger.info("QCA-RWS integration cancelled by user")
