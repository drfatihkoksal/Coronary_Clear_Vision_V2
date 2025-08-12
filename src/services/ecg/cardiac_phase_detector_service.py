"""
Cardiac Phase Detector Service

Kardiyak faz tespit servisi.
R-peak'lere göre kardiyak fazları (D1, D2, S1, S2) tespit eder.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Dict, Optional
import numpy as np
import logging
from scipy import signal as scipy_signal

from src.domain.models.ecg_models import (
    ECGSignal, RPeak, CardiacPhase, CardiacCycle,
    CardiacCyclePhase, CardiacPhaseAnalysis,
    PhaseStatistics, RRInterval
)
from src.domain.interfaces.ecg_interfaces import ICardiacPhaseDetector

logger = logging.getLogger(__name__)


class CardiacPhaseDetectorService(ICardiacPhaseDetector):
    """
    Kardiyak faz tespit servisi.
    
    Kardiyak döngü fazları:
    - D2: End-diastole (R dalgasından hemen önce)
    - S1: Early-systole (R dalgası zirvesi)
    - S2: End-systole (T dalgası sonu)
    - D1: Mid-diastole (Diyastolik dönemin ortası)
    
    Faz geçişleri:
    - D2→S1: End-diastole fazı
    - S1→S2: Early-systole fazı
    - S2→D1: End-systole fazı
    - D1→D2: Mid-diastole fazı
    """
    
    def __init__(self):
        """CardiacPhaseDetectorService constructor."""
        self._base_d2_offset_ms = 50  # D2'nin R'den önceki offset'i
        self._base_systole_fraction = 0.35  # Sistol süresi oranı
        logger.info("CardiacPhaseDetectorService initialized")
    
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
        try:
            if len(r_peaks) < 2:
                logger.warning("Insufficient R-peaks for phase detection")
                return CardiacPhaseAnalysis(cycles=[], sampling_rate=signal.metadata.sampling_rate)
            
            fs = signal.metadata.sampling_rate
            cycles = []
            
            # Her kardiyak döngü için fazları tespit et
            for i in range(len(r_peaks)):
                # RR interval hesapla
                if i < len(r_peaks) - 1:
                    # Normal durum: sonraki R-peak var
                    current_r = r_peaks[i]
                    next_r = r_peaks[i + 1]
                    rr_interval_ms = (next_r.time - current_r.time) * 1000
                else:
                    # Son R-peak: önceki RR'lardan tahmin et
                    if i > 0:
                        # Önceki RR intervallerinin ortalaması
                        prev_intervals = []
                        for j in range(max(0, i-3), i):
                            interval = (r_peaks[j+1].time - r_peaks[j].time) * 1000
                            prev_intervals.append(interval)
                        rr_interval_ms = np.mean(prev_intervals) if prev_intervals else 800
                    else:
                        # Tek R-peak var, varsayılan değer kullan
                        rr_interval_ms = 800
                
                # Kalp hızı hesapla
                heart_rate = 60000 / rr_interval_ms if rr_interval_ms > 0 else 75
                
                # Bu döngü için fazları tespit et
                cycle_phases = self._detect_cycle_phases(
                    signal=signal,
                    r_peak=r_peaks[i],
                    rr_interval_ms=rr_interval_ms,
                    heart_rate=heart_rate,
                    next_r_index=r_peaks[i+1].index if i < len(r_peaks)-1 else None
                )
                
                # Kardiyak döngü oluştur
                cycle = CardiacCycle(
                    cycle_number=i + 1,
                    r_peak=r_peaks[i],
                    phases=cycle_phases,
                    rr_interval_ms=rr_interval_ms,
                    heart_rate=heart_rate
                )
                
                cycles.append(cycle)
            
            # İstatistikleri hesapla
            statistics = self.calculate_phase_statistics(cycles)
            
            # Analiz sonucunu oluştur
            analysis = CardiacPhaseAnalysis(
                cycles=cycles,
                statistics=statistics,
                sampling_rate=fs
            )
            
            logger.info(f"Detected phases for {len(cycles)} cardiac cycles")
            return analysis
            
        except Exception as e:
            logger.error(f"Error detecting cardiac phases: {str(e)}")
            return CardiacPhaseAnalysis(cycles=[], sampling_rate=signal.metadata.sampling_rate)
    
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
        try:
            # Tüm faz olaylarını topla ve sırala
            phase_events = []
            
            for cycle in phase_analysis.cycles:
                for phase, phase_info in cycle.phases.items():
                    phase_events.append((phase_info.time, phase))
            
            # Zamana göre sırala
            phase_events.sort(key=lambda x: x[0])
            
            # Hangi aralıkta olduğunu bul
            for i in range(len(phase_events) - 1):
                current_time, current_phase = phase_events[i]
                next_time, next_phase = phase_events[i + 1]
                
                if current_time <= time_s < next_time:
                    # Bu aralıktaki faz
                    return current_phase
            
            # Son fazdan sonra mı?
            if phase_events and time_s >= phase_events[-1][0]:
                return phase_events[-1][1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting phase at time: {str(e)}")
            return None
    
    def calculate_phase_statistics(self, 
                                 cycles: List[CardiacCycle]) -> PhaseStatistics:
        """
        Faz istatistiklerini hesapla.
        
        Args:
            cycles: Kardiyak döngüler
            
        Returns:
            PhaseStatistics: İstatistikler
        """
        try:
            if not cycles:
                # Boş istatistikler
                return PhaseStatistics(
                    d2_offset_ms_mean=0, d2_offset_ms_std=0,
                    s2_offset_ms_mean=0, s2_offset_ms_std=0,
                    d1_offset_ms_mean=0, d1_offset_ms_std=0,
                    systole_duration_ms_mean=0, systole_duration_ms_std=0,
                    diastole_duration_ms_mean=0, diastole_duration_ms_std=0
                )
            
            # İstatistik listeleri
            d2_offsets = []
            s2_offsets = []
            d1_offsets = []
            systole_durations = []
            diastole_durations = []
            
            for i, cycle in enumerate(cycles):
                r_peak_time = cycle.r_peak.time
                
                # D2 offset (R'den önce)
                d2_phase = cycle.get_phase(CardiacPhase.D2)
                if d2_phase:
                    d2_offset = (r_peak_time - d2_phase.time) * 1000
                    d2_offsets.append(d2_offset)
                
                # S2 offset (R'den sonra)
                s2_phase = cycle.get_phase(CardiacPhase.S2)
                if s2_phase:
                    s2_offset = (s2_phase.time - r_peak_time) * 1000
                    s2_offsets.append(s2_offset)
                
                # D1 offset (R'den sonra)
                d1_phase = cycle.get_phase(CardiacPhase.D1)
                if d1_phase:
                    d1_offset = (d1_phase.time - r_peak_time) * 1000
                    d1_offsets.append(d1_offset)
                
                # Sistol süresi (S1'den S2'ye)
                s1_phase = cycle.get_phase(CardiacPhase.S1)
                if s1_phase and s2_phase:
                    systole_duration = (s2_phase.time - s1_phase.time) * 1000
                    systole_durations.append(systole_duration)
                
                # Diyastol süresi (S2'den sonraki D2'ye)
                if s2_phase and i < len(cycles) - 1:
                    next_cycle = cycles[i + 1]
                    next_d2 = next_cycle.get_phase(CardiacPhase.D2)
                    if next_d2:
                        diastole_duration = (next_d2.time - s2_phase.time) * 1000
                        diastole_durations.append(diastole_duration)
            
            # İstatistikleri hesapla
            statistics = PhaseStatistics(
                d2_offset_ms_mean=np.mean(d2_offsets) if d2_offsets else 0,
                d2_offset_ms_std=np.std(d2_offsets) if d2_offsets else 0,
                s2_offset_ms_mean=np.mean(s2_offsets) if s2_offsets else 0,
                s2_offset_ms_std=np.std(s2_offsets) if s2_offsets else 0,
                d1_offset_ms_mean=np.mean(d1_offsets) if d1_offsets else 0,
                d1_offset_ms_std=np.std(d1_offsets) if d1_offsets else 0,
                systole_duration_ms_mean=np.mean(systole_durations) if systole_durations else 0,
                systole_duration_ms_std=np.std(systole_durations) if systole_durations else 0,
                diastole_duration_ms_mean=np.mean(diastole_durations) if diastole_durations else 0,
                diastole_duration_ms_std=np.std(diastole_durations) if diastole_durations else 0
            )
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating phase statistics: {str(e)}")
            # Hata durumunda boş istatistikler döndür
            return PhaseStatistics(
                d2_offset_ms_mean=0, d2_offset_ms_std=0,
                s2_offset_ms_mean=0, s2_offset_ms_std=0,
                d1_offset_ms_mean=0, d1_offset_ms_std=0,
                systole_duration_ms_mean=0, systole_duration_ms_std=0,
                diastole_duration_ms_mean=0, diastole_duration_ms_std=0
            )
    
    def _detect_cycle_phases(self, signal: ECGSignal, r_peak: RPeak,
                           rr_interval_ms: float, heart_rate: float,
                           next_r_index: Optional[int]) -> Dict[CardiacPhase, CardiacCyclePhase]:
        """Tek bir kardiyak döngü için fazları tespit et."""
        phases = {}
        fs = signal.metadata.sampling_rate
        
        # S1: Early-systole (R-peak kendisi)
        phases[CardiacPhase.S1] = CardiacCyclePhase(
            phase=CardiacPhase.S1,
            index=r_peak.index,
            time=r_peak.time,
            phase_name="Early-systole"
        )
        
        # D2: End-diastole (R'den önce)
        d2_offset_ms = self._calculate_d2_offset(heart_rate)
        d2_offset_samples = int(d2_offset_ms * fs / 1000)
        d2_index = max(0, r_peak.index - d2_offset_samples)
        
        phases[CardiacPhase.D2] = CardiacCyclePhase(
            phase=CardiacPhase.D2,
            index=d2_index,
            time=d2_index / fs,
            phase_name="End-diastole"
        )
        
        # S2: End-systole (T dalgası sonu)
        s2_index = self._detect_t_wave_end(
            signal.data, r_peak.index,
            int(rr_interval_ms * fs / 1000),
            fs
        )
        
        # S2 bulunamadıysa tahmin et
        if s2_index is None:
            systole_fraction = self._calculate_systole_fraction(heart_rate)
            s2_offset_samples = int(systole_fraction * rr_interval_ms * fs / 1000)
            s2_index = r_peak.index + s2_offset_samples
        
        # S2'nin sonraki R'yi geçmediğinden emin ol
        if next_r_index is not None:
            max_s2_index = next_r_index - int(0.1 * rr_interval_ms * fs / 1000)
            s2_index = min(s2_index, max_s2_index)
        
        phases[CardiacPhase.S2] = CardiacCyclePhase(
            phase=CardiacPhase.S2,
            index=s2_index,
            time=s2_index / fs,
            phase_name="End-systole"
        )
        
        # D1: Mid-diastole
        # S2 ile sonraki D2 arasının ortası
        if next_r_index is not None:
            next_d2_index = max(0, next_r_index - d2_offset_samples)
        else:
            # Son döngü için tahmin
            next_d2_index = s2_index + int(0.5 * rr_interval_ms * fs / 1000)
        
        d1_index = s2_index + (next_d2_index - s2_index) // 2
        
        phases[CardiacPhase.D1] = CardiacCyclePhase(
            phase=CardiacPhase.D1,
            index=d1_index,
            time=d1_index / fs,
            phase_name="Mid-diastole"
        )
        
        return phases
    
    def _calculate_d2_offset(self, heart_rate: float) -> float:
        """Kalp hızına göre D2 offset'ini hesapla (ms)."""
        # Kalp hızı arttıkça offset azalır
        # Normal HR (60-80): 40-60ms
        # Yüksek HR (>100): 30-40ms
        # Düşük HR (<60): 50-70ms
        
        hr_factor = 70 / heart_rate  # 70 BPM'e normalize et
        d2_offset = self._base_d2_offset_ms * hr_factor
        
        # Sınırlar içinde tut
        return np.clip(d2_offset, 30, 70)
    
    def _calculate_systole_fraction(self, heart_rate: float) -> float:
        """Kalp hızına göre sistol süresinin RR'ye oranını hesapla."""
        # Kalp hızı arttıkça sistol oranı artar
        # Normal HR: %35-40
        # Yüksek HR: %40-45
        # Düşük HR: %30-35
        
        hr_adjustment = 0.10 * (heart_rate - 70) / 70
        systole_fraction = self._base_systole_fraction + hr_adjustment
        
        # Sınırlar içinde tut
        return np.clip(systole_fraction, 0.30, 0.45)
    
    def _detect_t_wave_end(self, signal: np.ndarray, r_peak_index: int,
                          rr_interval_samples: int, fs: float) -> Optional[int]:
        """T dalgası sonunu tespit et."""
        try:
            # Arama penceresi (R'den 200-400ms sonra)
            hr = 60000 / ((rr_interval_samples / fs) * 1000)
            hr_factor = 70 / hr
            
            start_ms = 200 * hr_factor
            end_ms = 400 * hr_factor
            
            # Sınırlar
            start_ms = np.clip(start_ms, 150, 250)
            end_ms = np.clip(end_ms, 350, 450)
            
            search_start = r_peak_index + int(start_ms * fs / 1000)
            search_end = min(
                r_peak_index + int(end_ms * fs / 1000),
                r_peak_index + int(0.5 * rr_interval_samples)
            )
            
            if search_start >= len(signal) or search_end >= len(signal):
                return None
            
            # Arama penceresini al
            window = signal[search_start:search_end]
            if len(window) < 10:
                return None
            
            # Sinyal yumuşatma
            window_smooth = scipy_signal.savgol_filter(
                window,
                window_length=min(11, len(window) // 2 * 2 + 1),
                polyorder=3
            )
            
            # Türev hesapla
            derivative = np.gradient(window_smooth)
            
            # Sıfır geçişlerini bul
            zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
            
            if len(zero_crossings) > 0:
                # Son önemli sıfır geçişi
                t_end_idx = zero_crossings[-1]
                return search_start + t_end_idx
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting T-wave end: {str(e)}")
            return None