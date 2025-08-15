"""
ECG Domain Models

ECG verilerini temsil eden domain modelleri.
Clean Architecture prensiplerine uygun olarak tasarlanmıştır.
Tüm modeller immutable (frozen=True) olarak tanımlanmıştır.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np
from datetime import datetime


class ECGSource(Enum):
    """ECG veri kaynağı tipleri."""
    WAVEFORM_SEQUENCE = "WaveformSequence"
    LEGACY_CURVE = "LegacyCurve"
    SIEMENS_PRIVATE = "SiemensPrivate"
    GE_PRIVATE = "GEPrivate"
    UNKNOWN = "Unknown"


class SignalQuality(Enum):
    """ECG sinyal kalitesi seviyeleri."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class CardiacPhase(Enum):
    """Kardiyak faz tipleri."""
    D1 = "d1"  # Mid-diastole
    D2 = "d2"  # End-diastole
    S1 = "s1"  # Early-systole
    S2 = "s2"  # End-systole


class PhaseTransition(Enum):
    """Kardiyak faz geçişleri."""
    END_DIASTOLE = "End-diastole"      # D2→S1
    EARLY_SYSTOLE = "Early-systole"     # S1→S2
    END_SYSTOLE = "End-systole"         # S2→D1
    MID_DIASTOLE = "Mid-diastole"       # D1→D2


@dataclass(frozen=True)
class ECGMetadata:
    """
    ECG metadata bilgileri.
    
    Attributes:
        source: Veri kaynağı
        sampling_rate: Örnekleme hızı (Hz)
        duration: Süre (saniye)
        num_samples: Örnek sayısı
        vendor: Cihaz üreticisi
        modality: Modalite
        study_date: Çalışma tarihi
        additional_info: Ek bilgiler
    """
    source: ECGSource
    sampling_rate: float
    duration: float
    num_samples: int
    vendor: Optional[str] = None
    modality: Optional[str] = None
    study_date: Optional[datetime] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validasyon."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.duration < 0:
            raise ValueError("Duration cannot be negative")
        if self.num_samples < 0:
            raise ValueError("Number of samples cannot be negative")


@dataclass(frozen=True)
class ECGSignal:
    """
    ECG sinyal verisi.
    
    Attributes:
        data: Sinyal verisi (numpy array)
        metadata: Metadata bilgileri
        timestamp_offset: Zaman offset'i (saniye)
    """
    data: np.ndarray
    metadata: ECGMetadata
    timestamp_offset: float = 0.0
    
    def __post_init__(self):
        """Validasyon ve dondurma."""
        # Numpy array'i read-only yap
        object.__setattr__(self, 'data', self.data.copy())
        self.data.flags.writeable = False
        
        # Boyut kontrolü
        if len(self.data) != self.metadata.num_samples:
            raise ValueError(f"Data length ({len(self.data)}) doesn't match num_samples ({self.metadata.num_samples})")
    
    @property
    def time_array(self) -> np.ndarray:
        """Zaman dizisini döndür."""
        return np.arange(self.metadata.num_samples) / self.metadata.sampling_rate + self.timestamp_offset


@dataclass(frozen=True)
class RPeak:
    """
    R-peak bilgisi.
    
    Attributes:
        index: Sinyal içindeki indeks
        time: Zaman (saniye)
        amplitude: Genlik değeri
        confidence: Güvenilirlik (0-1)
    """
    index: int
    time: float
    amplitude: float
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validasyon."""
        if self.index < 0:
            raise ValueError("Peak index cannot be negative")
        if self.time < 0:
            raise ValueError("Peak time cannot be negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class RRInterval:
    """
    RR interval bilgisi.
    
    Attributes:
        start_peak: Başlangıç R-peak
        end_peak: Bitiş R-peak
        duration_ms: Süre (milisaniye)
        heart_rate: Kalp hızı (BPM)
    """
    start_peak: RPeak
    end_peak: RPeak
    duration_ms: float
    heart_rate: float
    
    def __post_init__(self):
        """Validasyon."""
        if self.duration_ms <= 0:
            raise ValueError("RR interval duration must be positive")
        if self.heart_rate <= 0:
            raise ValueError("Heart rate must be positive")


@dataclass(frozen=True)
class CardiacCyclePhase:
    """
    Tek bir kardiyak döngüdeki faz bilgisi.
    
    Attributes:
        phase: Faz tipi
        index: Sinyal içindeki indeks
        time: Zaman (saniye)
        phase_name: Faz adı
    """
    phase: CardiacPhase
    index: int
    time: float
    phase_name: str


@dataclass(frozen=True)
class CardiacCycle:
    """
    Tek bir kardiyak döngü.
    
    Attributes:
        cycle_number: Döngü numarası
        r_peak: R-peak bilgisi
        phases: Faz bilgileri
        rr_interval_ms: RR interval (ms)
        heart_rate: Kalp hızı (BPM)
    """
    cycle_number: int
    r_peak: RPeak
    phases: Dict[CardiacPhase, CardiacCyclePhase]
    rr_interval_ms: float
    heart_rate: float
    
    def get_phase(self, phase: CardiacPhase) -> Optional[CardiacCyclePhase]:
        """Belirli bir fazı döndür."""
        return self.phases.get(phase)
    
    def get_phase_duration(self, from_phase: CardiacPhase, to_phase: CardiacPhase) -> Optional[float]:
        """İki faz arasındaki süreyi hesapla (ms)."""
        from_p = self.phases.get(from_phase)
        to_p = self.phases.get(to_phase)
        
        if from_p and to_p:
            return (to_p.time - from_p.time) * 1000
        return None


@dataclass(frozen=True)
class PhaseStatistics:
    """
    Kardiyak faz istatistikleri.
    
    Attributes:
        d2_offset_ms_mean: D2 offset ortalaması (ms)
        d2_offset_ms_std: D2 offset standart sapması
        s2_offset_ms_mean: S2 offset ortalaması (ms)
        s2_offset_ms_std: S2 offset standart sapması
        d1_offset_ms_mean: D1 offset ortalaması (ms)
        d1_offset_ms_std: D1 offset standart sapması
        systole_duration_ms_mean: Sistol süresi ortalaması
        systole_duration_ms_std: Sistol süresi standart sapması
        diastole_duration_ms_mean: Diyastol süresi ortalaması
        diastole_duration_ms_std: Diyastol süresi standart sapması
    """
    d2_offset_ms_mean: float
    d2_offset_ms_std: float
    s2_offset_ms_mean: float
    s2_offset_ms_std: float
    d1_offset_ms_mean: float
    d1_offset_ms_std: float
    systole_duration_ms_mean: float
    systole_duration_ms_std: float
    diastole_duration_ms_mean: float
    diastole_duration_ms_std: float


@dataclass(frozen=True)
class CardiacPhaseAnalysis:
    """
    Kardiyak faz analizi sonuçları.
    
    Attributes:
        cycles: Kardiyak döngüler
        statistics: Faz istatistikleri
        sampling_rate: Örnekleme hızı
    """
    cycles: List[CardiacCycle] = field(default_factory=list)
    statistics: Optional[PhaseStatistics] = None
    sampling_rate: float = 1000.0
    
    def get_phase_at_time(self, time_s: float) -> Optional[CardiacPhase]:
        """Belirli bir zamandaki fazı döndür."""
        for cycle in self.cycles:
            # Döngü içindeki fazları kontrol et
            sorted_phases = sorted(
                cycle.phases.items(),
                key=lambda x: x[1].time
            )
            
            for i in range(len(sorted_phases) - 1):
                current_phase, current_info = sorted_phases[i]
                next_phase, next_info = sorted_phases[i + 1]
                
                if current_info.time <= time_s < next_info.time:
                    return current_phase
        
        return None


@dataclass(frozen=True)
class ECGQualityMetrics:
    """
    ECG sinyal kalitesi metrikleri.
    
    Attributes:
        snr_db: Sinyal-gürültü oranı (dB)
        baseline_wander: Baseline kayması seviyesi
        powerline_interference: Güç hattı girişimi seviyesi
        motion_artifacts: Hareket artefaktları seviyesi
        clipping_ratio: Kırpılma oranı
        overall_quality: Genel kalite
        quality_score: Kalite skoru (0-1)
        messages: Kalite mesajları
    """
    snr_db: float
    baseline_wander: float
    powerline_interference: float
    motion_artifacts: float
    clipping_ratio: float
    overall_quality: SignalQuality
    quality_score: float
    messages: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validasyon."""
        if not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be between 0 and 1")


@dataclass(frozen=True)
class HRVMetrics:
    """
    Kalp hızı değişkenliği (HRV) metrikleri.
    
    Attributes:
        mean_rr: Ortalama RR interval (ms)
        std_rr: RR interval standart sapması (ms)
        rmssd: Ardışık RR farkları RMS'i (ms)
        pnn50: 50ms'den büyük RR farkları yüzdesi
        mean_hr: Ortalama kalp hızı (BPM)
        std_hr: Kalp hızı standart sapması (BPM)
    """
    mean_rr: float
    std_rr: float
    rmssd: float
    pnn50: float
    mean_hr: float
    std_hr: float


@dataclass(frozen=True)
class ECGAnalysisResult:
    """
    ECG analiz sonuçları.
    
    Attributes:
        signal: İşlenmiş ECG sinyali
        r_peaks: R-peak listesi
        rr_intervals: RR interval listesi
        cardiac_phases: Kardiyak faz analizi
        quality_metrics: Kalite metrikleri
        hrv_metrics: HRV metrikleri
        processing_time_ms: İşleme süresi (ms)
    """
    signal: ECGSignal
    r_peaks: List[RPeak] = field(default_factory=list)
    rr_intervals: List[RRInterval] = field(default_factory=list)
    cardiac_phases: Optional[CardiacPhaseAnalysis] = None
    quality_metrics: Optional[ECGQualityMetrics] = None
    hrv_metrics: Optional[HRVMetrics] = None
    processing_time_ms: float = 0.0


@dataclass(frozen=True)
class ECGProcessingRequest:
    """
    ECG işleme isteği.
    
    Attributes:
        signal: Ham ECG sinyali
        detect_r_peaks: R-peak tespiti yapılsın mı?
        detect_phases: Kardiyak faz tespiti yapılsın mı?
        calculate_hrv: HRV hesaplansın mı?
        assess_quality: Kalite değerlendirmesi yapılsın mı?
        apply_filters: Filtreler uygulansın mı?
    """
    signal: ECGSignal
    detect_r_peaks: bool = True
    detect_phases: bool = True
    calculate_hrv: bool = True
    assess_quality: bool = True
    apply_filters: bool = True


@dataclass(frozen=True)
class ECGSyncInfo:
    """
    ECG senkronizasyon bilgileri.
    
    Attributes:
        video_duration: Video süresi (saniye)
        ecg_duration: ECG süresi (saniye)
        time_offset: Zaman offset'i (saniye)
        sync_quality: Senkronizasyon kalitesi
        frame_to_time_mapping: Frame-zaman eşleşmesi
    """
    video_duration: float
    ecg_duration: float
    time_offset: float
    sync_quality: SignalQuality
    frame_to_time_mapping: Optional[Dict[int, float]] = None
    
    @property
    def duration_difference_ms(self) -> float:
        """Süre farkını milisaniye olarak döndür."""
        return abs(self.video_duration - self.ecg_duration) * 1000