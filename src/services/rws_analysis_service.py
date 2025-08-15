"""
RWS Analysis Service

RWS analizinin business logic'ini yöneten servis.
UI'dan bağımsız olarak analiz işlemlerini gerçekleştirir.
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from typing import Dict, Optional
import logging
from src.models.rws_models import (
    RWSAnalysisResult,
    RWSAnalysisProgress,
    RWSAnalysisRequest,
    RiskLevel,
)


logger = logging.getLogger(__name__)


class RWSAnalysisWorker(QThread):
    """
    RWS analizini arkaplan thread'inde gerçekleştiren worker.

    Thread-safe analiz için QThread kullanır.
    UI'yı dondurmadan ağır hesaplamaları yapar.

    Signals:
        progress_updated: İlerleme güncellemesi sinyali
        analysis_completed: Analiz tamamlandı sinyali
        analysis_failed: Analiz başarısız sinyali
    """

    # Signals
    progress_updated = pyqtSignal(RWSAnalysisProgress)
    analysis_completed = pyqtSignal(RWSAnalysisResult)
    analysis_failed = pyqtSignal(str)  # error message

    def __init__(self, qca_analyzer, request: RWSAnalysisRequest):
        """
        Worker'ı başlatır.

        Args:
            qca_analyzer: QCA analiz nesnesi
            request: Analiz isteği parametreleri
        """
        super().__init__()
        self.qca_analyzer = qca_analyzer
        self.request = request
        self._is_cancelled = False

    def run(self):
        """
        Thread'de çalışacak ana analiz metodu.

        QThread.run() metodunu override eder.
        """
        try:
            # İlerleme bildirimi - Başlangıç
            self._emit_progress("Initializing RWS analysis...", 10)

            if self._is_cancelled:
                return

            # İlerleme bildirimi - Analiz
            self._emit_progress("Calculating radial wall strain...", 50)

            # Gerçek analizi yap
            raw_result = self.qca_analyzer.calculate_rws(
                self.request.qca_results_by_frame, self.request.cardiac_phase_info
            )

            if self._is_cancelled:
                return

            # İlerleme bildirimi - Sonuçları işleme
            self._emit_progress("Processing results...", 80)

            # Ham sonuçları model'e dönüştür
            result = self._convert_to_model(raw_result)

            # İlerleme bildirimi - Tamamlandı
            self._emit_progress("Analysis complete", 100, is_complete=True)

            # Sonucu yayınla
            self.analysis_completed.emit(result)

        except Exception as e:
            logger.error(f"RWS analysis failed: {e}", exc_info=True)
            self.analysis_failed.emit(str(e))

    def cancel(self):
        """Analizi iptal eder."""
        self._is_cancelled = True
        self._emit_progress("Analysis cancelled", 0, is_cancelled=True)

    def _emit_progress(
        self, status: str, percentage: int, is_complete: bool = False, is_cancelled: bool = False
    ):
        """
        İlerleme durumu yayınlar.

        Args:
            status: Durum mesajı
            percentage: İlerleme yüzdesi
            is_complete: Tamamlandı mı
            is_cancelled: İptal edildi mi
        """
        progress = RWSAnalysisProgress(
            status=status, percentage=percentage, is_complete=is_complete, is_cancelled=is_cancelled
        )
        self.progress_updated.emit(progress)

    def _convert_to_model(self, raw_result: Dict) -> RWSAnalysisResult:
        """
        Ham analiz sonuçlarını model nesnesine dönüştürür.

        Args:
            raw_result: QCA analyzer'dan gelen ham sonuç

        Returns:
            RWSAnalysisResult: Yapılandırılmış sonuç modeli
        """
        # Başarı durumunu kontrol et
        success = raw_result.get("success", False)

        if not success:
            return RWSAnalysisResult(
                success=False, error=raw_result.get("error", "Unknown error occurred")
            )

        # Değerleri çıkar ve varsayılanları kullan
        rws_max = raw_result.get("rws_max", 0.0)

        # Risk seviyesini hesapla
        if rws_max < 10.0:
            risk_level = RiskLevel.LOW
        elif rws_max < 14.25:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.HIGH

        # Klinik yorumu oluştur
        interpretation = self._generate_interpretation(rws_max, risk_level)

        # Model nesnesini oluştur
        return RWSAnalysisResult(
            success=True,
            rws_max=rws_max,
            rws_stenosis=raw_result.get("rws_stenosis", 0.0),
            rws_max_location=raw_result.get("rws_max_location", -1),
            diameter_change_mm=raw_result.get("diameter_change_mm", 0.0),
            reference_diameter_mm=raw_result.get("reference_diameter_mm", 0.0),
            end_diastole_frame=raw_result.get("end_diastole_frame", -1),
            end_systole_frame=raw_result.get("end_systole_frame", -1),
            risk_level=risk_level,
            interpretation=interpretation,
            rws_values=raw_result.get("rws_values"),
            metadata=raw_result.get("metadata", {}),
        )

    def _generate_interpretation(self, rws_max: float, risk_level: RiskLevel) -> str:
        """
        RWS değerine göre klinik yorum oluşturur.

        Args:
            rws_max: Maksimum RWS değeri
            risk_level: Hesaplanan risk seviyesi

        Returns:
            str: Klinik yorum metni
        """
        interpretations = {
            RiskLevel.LOW: (
                f"Maximum RWS of {rws_max}% indicates low risk. "
                "The vessel wall deformation is within normal limits. "
                "No significant plaque vulnerability detected."
            ),
            RiskLevel.MODERATE: (
                f"Maximum RWS of {rws_max}% indicates moderate risk. "
                "Some vessel wall deformation is present. "
                "Consider closer monitoring and risk factor modification."
            ),
            RiskLevel.HIGH: (
                f"Maximum RWS of {rws_max}% indicates HIGH RISK. "
                "Significant vessel wall deformation detected. "
                "High probability of vulnerable plaque. "
                "Immediate clinical evaluation recommended."
            ),
            RiskLevel.UNKNOWN: (
                "Unable to determine risk level. " "Please verify the analysis parameters."
            ),
        }

        return interpretations.get(risk_level, interpretations[RiskLevel.UNKNOWN])


class RWSAnalysisService(QObject):
    """
    RWS analiz servisinin ana sınıfı.

    UI ile business logic arasında köprü görevi görür.
    Analiz lifecycle'ını yönetir.

    Signals:
        progress_updated: İlerleme güncellemesi
        analysis_completed: Analiz tamamlandı
        analysis_failed: Analiz başarısız
    """

    # Forwarded signals
    progress_updated = pyqtSignal(RWSAnalysisProgress)
    analysis_completed = pyqtSignal(RWSAnalysisResult)
    analysis_failed = pyqtSignal(str)

    def __init__(self, parent=None):
        """
        Servisi başlatır.

        Args:
            parent: Parent QObject (isteğe bağlı)
        """
        super().__init__(parent)
        self._current_worker: Optional[RWSAnalysisWorker] = None

    def start_analysis(self, qca_analyzer, request: RWSAnalysisRequest):
        """
        Yeni bir RWS analizi başlatır.

        Args:
            qca_analyzer: QCA analiz nesnesi
            request: Analiz parametreleri
        """
        # Mevcut analizi durdur
        self.stop_analysis()

        # Yeni worker oluştur
        self._current_worker = RWSAnalysisWorker(qca_analyzer, request)

        # Signal bağlantılarını yap
        self._connect_worker_signals()

        # Analizi başlat
        self._current_worker.start()

    def stop_analysis(self):
        """Mevcut analizi durdurur."""
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.cancel()
            self._current_worker.wait(5000)  # Max 5 saniye bekle
            self._current_worker = None

    def is_running(self) -> bool:
        """
        Analiz çalışıyor mu kontrol eder.

        Returns:
            bool: Çalışıyorsa True
        """
        return self._current_worker is not None and self._current_worker.isRunning()

    def _connect_worker_signals(self):
        """Worker sinyallerini servise bağlar."""
        if self._current_worker:
            # Sinyalleri forward et
            self._current_worker.progress_updated.connect(self.progress_updated.emit)
            self._current_worker.analysis_completed.connect(self._on_analysis_completed)
            self._current_worker.analysis_failed.connect(self.analysis_failed.emit)

    def _on_analysis_completed(self, result: RWSAnalysisResult):
        """
        Analiz tamamlandığında çağrılır.

        Args:
            result: Analiz sonucu
        """
        # Worker'ı temizle
        self._current_worker = None

        # Sonucu forward et
        self.analysis_completed.emit(result)
