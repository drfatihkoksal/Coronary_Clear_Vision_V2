"""
ECG Domain Interfaces

ECG işleme için domain arayüzleri.
Clean Architecture ve SOLID prensiplerine uygun olarak tasarlanmıştır.
"""

from typing import Protocol, Optional, List, Dict, Any, Tuple
import numpy as np
from abc import abstractmethod

from src.domain.models.ecg_models import (
    ECGSignal, ECGMetadata, ECGSource,
    RPeak, RRInterval, CardiacPhase, CardiacCycle,
    CardiacPhaseAnalysis, PhaseStatistics,
    ECGQualityMetrics, HRVMetrics,
    ECGAnalysisResult, ECGProcessingRequest,
    ECGSyncInfo, SignalQuality
)


class IECGExtractor(Protocol):
    """
    ECG veri çıkarma arayüzü.
    
    DICOM dosyalarından ECG verisi çıkarır.
    """
    
    @abstractmethod
    def extract_from_dicom(self, dicom_dataset: Any) -> Optional[ECGSignal]:
        """
        DICOM dataset'inden ECG verisi çıkar.
        
        Args:
            dicom_dataset: DICOM dataset objesi
            
        Returns:
            Optional[ECGSignal]: Çıkarılan ECG sinyali veya None
        """
        ...
    
    @abstractmethod
    def get_supported_sources(self) -> List[ECGSource]:
        """
        Desteklenen ECG kaynaklarını döndür.
        
        Returns:
            List[ECGSource]: Desteklenen kaynaklar
        """
        ...


class IECGProcessor(Protocol):
    """
    ECG sinyal işleme arayüzü.
    
    Ham ECG sinyalini işler ve filtreler.
    """
    
    @abstractmethod
    def process_signal(self, signal: ECGSignal) -> ECGSignal:
        """
        ECG sinyalini işle.
        
        Args:
            signal: Ham ECG sinyali
            
        Returns:
            ECGSignal: İşlenmiş sinyal
        """
        ...
    
    @abstractmethod
    def apply_bandpass_filter(self, signal: np.ndarray, 
                            low_freq: float, high_freq: float,
                            sampling_rate: float) -> np.ndarray:
        """
        Bandpass filtre uygula.
        
        Args:
            signal: Sinyal verisi
            low_freq: Alt frekans (Hz)
            high_freq: Üst frekans (Hz)
            sampling_rate: Örnekleme hızı (Hz)
            
        Returns:
            np.ndarray: Filtrelenmiş sinyal
        """
        ...
    
    @abstractmethod
    def remove_baseline_wander(self, signal: np.ndarray,
                             sampling_rate: float) -> np.ndarray:
        """
        Baseline kaymasını kaldır.
        
        Args:
            signal: Sinyal verisi
            sampling_rate: Örnekleme hızı (Hz)
            
        Returns:
            np.ndarray: Düzeltilmiş sinyal
        """
        ...
    
    @abstractmethod
    def remove_powerline_interference(self, signal: np.ndarray,
                                    sampling_rate: float,
                                    powerline_freq: float = 50.0) -> np.ndarray:
        """
        Güç hattı girişimini kaldır.
        
        Args:
            signal: Sinyal verisi
            sampling_rate: Örnekleme hızı (Hz)
            powerline_freq: Güç hattı frekansı (Hz)
            
        Returns:
            np.ndarray: Temizlenmiş sinyal
        """
        ...


class IRPeakDetector(Protocol):
    """
    R-peak tespit arayüzü.
    
    ECG sinyalinden R-peak'leri tespit eder.
    """
    
    @abstractmethod
    def detect_r_peaks(self, signal: ECGSignal) -> List[RPeak]:
        """
        R-peak'leri tespit et.
        
        Args:
            signal: ECG sinyali
            
        Returns:
            List[RPeak]: Tespit edilen R-peak'ler
        """
        ...
    
    @abstractmethod
    def refine_peaks(self, signal: ECGSignal, 
                    initial_peaks: List[int]) -> List[RPeak]:
        """
        İlk peak tahminlerini iyileştir.
        
        Args:
            signal: ECG sinyali
            initial_peaks: İlk peak indeksleri
            
        Returns:
            List[RPeak]: İyileştirilmiş peak'ler
        """
        ...
    
    @abstractmethod
    def calculate_confidence(self, signal: ECGSignal,
                           peaks: List[RPeak]) -> List[float]:
        """
        Peak güvenilirliklerini hesapla.
        
        Args:
            signal: ECG sinyali
            peaks: R-peak listesi
            
        Returns:
            List[float]: Güvenilirlik değerleri
        """
        ...


class ICardiacPhaseDetector(Protocol):
    """
    Kardiyak faz tespit arayüzü.
    
    R-peak'lere göre kardiyak fazları tespit eder.
    """
    
    @abstractmethod
    def detect_phases(self, signal: ECGSignal,
                     r_peaks: List[RPeak]) -> CardiacPhaseAnalysis:
        """
        Kardiyak fazları tespit et.
        
        Args:
            signal: ECG sinyali
            r_peaks: R-peak listesi
            
        Returns:
            CardiacPhaseAnalysis: Faz analiz sonuçları
        """
        ...
    
    @abstractmethod
    def get_phase_at_time(self, phase_analysis: CardiacPhaseAnalysis,
                         time_s: float) -> Optional[CardiacPhase]:
        """
        Belirli bir zamandaki fazı döndür.
        
        Args:
            phase_analysis: Faz analizi
            time_s: Zaman (saniye)
            
        Returns:
            Optional[CardiacPhase]: O andaki faz
        """
        ...
    
    @abstractmethod
    def calculate_phase_statistics(self, 
                                 cycles: List[CardiacCycle]) -> PhaseStatistics:
        """
        Faz istatistiklerini hesapla.
        
        Args:
            cycles: Kardiyak döngüler
            
        Returns:
            PhaseStatistics: İstatistikler
        """
        ...


class IECGQualityAssessor(Protocol):
    """
    ECG kalite değerlendirme arayüzü.
    
    ECG sinyal kalitesini değerlendirir.
    """
    
    @abstractmethod
    def assess_quality(self, signal: ECGSignal) -> ECGQualityMetrics:
        """
        Sinyal kalitesini değerlendir.
        
        Args:
            signal: ECG sinyali
            
        Returns:
            ECGQualityMetrics: Kalite metrikleri
        """
        ...
    
    @abstractmethod
    def calculate_snr(self, signal: np.ndarray) -> float:
        """
        Sinyal-gürültü oranını hesapla.
        
        Args:
            signal: Sinyal verisi
            
        Returns:
            float: SNR (dB)
        """
        ...
    
    @abstractmethod
    def detect_artifacts(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Artefaktları tespit et.
        
        Args:
            signal: Sinyal verisi
            
        Returns:
            Dict[str, float]: Artefakt seviyeleri
        """
        ...


class IHRVCalculator(Protocol):
    """
    Kalp hızı değişkenliği (HRV) hesaplama arayüzü.
    """
    
    @abstractmethod
    def calculate_hrv(self, rr_intervals: List[RRInterval]) -> HRVMetrics:
        """
        HRV metriklerini hesapla.
        
        Args:
            rr_intervals: RR interval listesi
            
        Returns:
            HRVMetrics: HRV metrikleri
        """
        ...
    
    @abstractmethod
    def calculate_time_domain(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """
        Zaman alanı HRV metriklerini hesapla.
        
        Args:
            rr_intervals_ms: RR intervaller (ms)
            
        Returns:
            Dict[str, float]: Zaman alanı metrikleri
        """
        ...


class IECGAnalyzer(Protocol):
    """
    ECG analiz servisi arayüzü.
    
    Tüm ECG analiz işlemlerini koordine eder.
    """
    
    @abstractmethod
    def analyze(self, request: ECGProcessingRequest) -> ECGAnalysisResult:
        """
        ECG analizi yap.
        
        Args:
            request: İşleme isteği
            
        Returns:
            ECGAnalysisResult: Analiz sonuçları
        """
        ...
    
    @abstractmethod
    def extract_rr_intervals(self, r_peaks: List[RPeak]) -> List[RRInterval]:
        """
        R-peak'lerden RR intervalleri çıkar.
        
        Args:
            r_peaks: R-peak listesi
            
        Returns:
            List[RRInterval]: RR interval listesi
        """
        ...


class IECGSynchronizer(Protocol):
    """
    ECG-Video senkronizasyon arayüzü.
    """
    
    @abstractmethod
    def synchronize(self, ecg_signal: ECGSignal,
                   video_duration: float,
                   frame_rate: float) -> ECGSyncInfo:
        """
        ECG ve video senkronizasyonu.
        
        Args:
            ecg_signal: ECG sinyali
            video_duration: Video süresi (saniye)
            frame_rate: Frame hızı (fps)
            
        Returns:
            ECGSyncInfo: Senkronizasyon bilgileri
        """
        ...
    
    @abstractmethod
    def map_frame_to_time(self, frame_index: int,
                         frame_rate: float) -> float:
        """
        Frame indeksini zamana çevir.
        
        Args:
            frame_index: Frame indeksi
            frame_rate: Frame hızı (fps)
            
        Returns:
            float: Zaman (saniye)
        """
        ...
    
    @abstractmethod
    def map_time_to_frame(self, time_s: float,
                         frame_rate: float) -> int:
        """
        Zamanı frame indeksine çevir.
        
        Args:
            time_s: Zaman (saniye)
            frame_rate: Frame hızı (fps)
            
        Returns:
            int: Frame indeksi
        """
        ...


class IECGVisualizer(Protocol):
    """
    ECG görselleştirme arayüzü.
    """
    
    @abstractmethod
    def create_signal_plot(self, signal: ECGSignal,
                          r_peaks: Optional[List[RPeak]] = None,
                          phases: Optional[CardiacPhaseAnalysis] = None) -> Any:
        """
        ECG sinyal grafiği oluştur.
        
        Args:
            signal: ECG sinyali
            r_peaks: R-peak'ler (opsiyonel)
            phases: Kardiyak fazlar (opsiyonel)
            
        Returns:
            Any: Grafik objesi
        """
        ...
    
    @abstractmethod
    def create_heart_rate_plot(self, rr_intervals: List[RRInterval]) -> Any:
        """
        Kalp hızı grafiği oluştur.
        
        Args:
            rr_intervals: RR interval listesi
            
        Returns:
            Any: Grafik objesi
        """
        ...


class IECGExporter(Protocol):
    """
    ECG veri dışa aktarma arayüzü.
    """
    
    @abstractmethod
    def export_to_csv(self, analysis_result: ECGAnalysisResult,
                     output_path: str) -> bool:
        """
        Analiz sonuçlarını CSV'ye aktar.
        
        Args:
            analysis_result: Analiz sonuçları
            output_path: Çıktı dosya yolu
            
        Returns:
            bool: Başarılı mı?
        """
        ...
    
    @abstractmethod
    def export_to_json(self, analysis_result: ECGAnalysisResult,
                      output_path: str) -> bool:
        """
        Analiz sonuçlarını JSON'a aktar.
        
        Args:
            analysis_result: Analiz sonuçları
            output_path: Çıktı dosya yolu
            
        Returns:
            bool: Başarılı mı?
        """
        ...