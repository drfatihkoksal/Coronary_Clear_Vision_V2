"""
R-Peak Detector Service

ECG sinyalinden R-peak tespiti servisi.
Modified Pan-Tompkins algoritması kullanır.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Dict, Tuple
import numpy as np
import logging
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d

from src.domain.models.ecg_models import ECGSignal, RPeak
from src.domain.interfaces.ecg_interfaces import IRPeakDetector

logger = logging.getLogger(__name__)


class RPeakDetectorService(IRPeakDetector):
    """
    R-peak tespit servisi.
    
    Modified Pan-Tompkins algoritması kullanarak güvenilir
    R-peak tespiti yapar. Dinamik eşikleme ve post-processing
    ile yüksek doğruluk sağlar.
    """
    
    def __init__(self):
        """RPeakDetectorService constructor."""
        self._min_rr_interval_ms = 300  # 200 BPM maksimum
        self._max_rr_interval_ms = 2000  # 30 BPM minimum
        logger.info("RPeakDetectorService initialized")
    
    def detect_r_peaks(self, signal: ECGSignal) -> List[RPeak]:
        """
        R-peak'leri tespit et.
        
        İşlem adımları:
        1. QRS enhancement
        2. Türev hesaplama
        3. Kare alma
        4. Moving average
        5. Dinamik eşikleme
        6. Peak refinement
        7. Post-processing
        
        Args:
            signal: ECG sinyali
            
        Returns:
            List[RPeak]: Tespit edilen R-peak'ler
        """
        try:
            data = signal.data
            fs = signal.metadata.sampling_rate
            
            # 1. QRS kompleksini vurgula (5-15 Hz bandpass)
            qrs_enhanced = self._enhance_qrs(data, fs)
            
            # 2. Türev al (eğimleri vurgula)
            derivative = np.gradient(qrs_enhanced)
            
            # 3. Kare al (pozitif yap ve vurgula)
            squared = derivative ** 2
            
            # 4. Moving window integration (150ms pencere)
            window_size = int(0.15 * fs)
            integrated = uniform_filter1d(squared, size=window_size)
            
            # 5. Dinamik eşikleme ile peak'leri bul
            peak_indices = self._find_peaks_dynamic(integrated, fs)
            
            # 6. Peak konumlarını iyileştir (orijinal sinyalde)
            refined_indices = self._refine_peak_locations(data, peak_indices, fs)
            
            # 7. Post-processing (false positive'leri kaldır)
            final_indices = self._post_process_peaks(data, refined_indices, fs)
            
            # 8. RPeak objelerine çevir
            r_peaks = self._create_rpeak_objects(data, final_indices, fs)
            
            # 9. Güvenilirlik hesapla
            confidences = self.calculate_confidence(signal, r_peaks)
            for i, conf in enumerate(confidences):
                # RPeak immutable olduğu için yeni obje oluştur
                peak = r_peaks[i]
                r_peaks[i] = RPeak(
                    index=peak.index,
                    time=peak.time,
                    amplitude=peak.amplitude,
                    confidence=conf
                )
            
            logger.info(f"Detected {len(r_peaks)} R-peaks")
            return r_peaks
            
        except Exception as e:
            logger.error(f"Error detecting R-peaks: {str(e)}")
            return []
    
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
        try:
            data = signal.data
            fs = signal.metadata.sampling_rate
            
            # Peak konumlarını iyileştir
            refined_indices = self._refine_peak_locations(data, initial_peaks, fs)
            
            # RPeak objelerine çevir
            r_peaks = self._create_rpeak_objects(data, refined_indices, fs)
            
            return r_peaks
            
        except Exception as e:
            logger.error(f"Error refining peaks: {str(e)}")
            return []
    
    def calculate_confidence(self, signal: ECGSignal,
                           peaks: List[RPeak]) -> List[float]:
        """
        Peak güvenilirliklerini hesapla.
        
        Güvenilirlik faktörleri:
        - Peak genliği tutarlılığı
        - RR interval düzenliliği
        - Morfoloji benzerliği
        - Sinyal kalitesi
        
        Args:
            signal: ECG sinyali
            peaks: R-peak listesi
            
        Returns:
            List[float]: Güvenilirlik değerleri (0-1)
        """
        try:
            if len(peaks) < 2:
                return [0.5] * len(peaks)
            
            confidences = []
            data = signal.data
            
            # Peak genliklerini al
            amplitudes = [p.amplitude for p in peaks]
            mean_amplitude = np.mean(amplitudes)
            std_amplitude = np.std(amplitudes)
            
            # RR intervalleri hesapla
            rr_intervals = []
            for i in range(1, len(peaks)):
                rr_ms = (peaks[i].time - peaks[i-1].time) * 1000
                rr_intervals.append(rr_ms)
            
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            cv_rr = std_rr / mean_rr if mean_rr > 0 else 1.0
            
            # Her peak için güvenilirlik hesapla
            for i, peak in enumerate(peaks):
                confidence = 1.0
                
                # 1. Genlik tutarlılığı
                if std_amplitude > 0:
                    amplitude_z = abs(peak.amplitude - mean_amplitude) / std_amplitude
                    amplitude_conf = np.exp(-amplitude_z / 2)  # Gaussian benzeri
                    confidence *= amplitude_conf
                
                # 2. RR interval düzenliliği
                if i > 0 and i < len(peaks) - 1:
                    prev_rr = (peak.time - peaks[i-1].time) * 1000
                    next_rr = (peaks[i+1].time - peak.time) * 1000
                    
                    # Önceki ve sonraki RR benzer mi?
                    rr_diff = abs(prev_rr - next_rr) / mean_rr
                    rr_conf = np.exp(-rr_diff * 2)
                    confidence *= rr_conf
                
                # 3. Morfoloji kontrolü (basit)
                # Peak civarında sinyal kalitesi
                window = int(0.05 * signal.metadata.sampling_rate)  # 50ms
                start = max(0, peak.index - window)
                end = min(len(data), peak.index + window)
                
                if start < end:
                    local_signal = data[start:end]
                    local_snr = self._estimate_local_snr(local_signal)
                    snr_conf = min(local_snr / 20, 1.0)  # 20dB'de maksimum
                    confidence *= snr_conf
                
                # Güvenilirliği sınırla
                confidence = max(0.1, min(confidence, 1.0))
                confidences.append(confidence)
            
            return confidences
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return [0.5] * len(peaks)
    
    def _enhance_qrs(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """QRS kompleksini vurgula."""
        try:
            # QRS frekans aralığı için bandpass filtre (5-15 Hz)
            nyquist = fs / 2
            low = 5 / nyquist
            high = min(15 / nyquist, 0.99)
            
            if low < high:
                sos = scipy_signal.butter(2, [low, high], btype='band', output='sos')
                filtered = scipy_signal.sosfiltfilt(sos, signal)
                return filtered
            else:
                return signal
                
        except Exception as e:
            logger.error(f"Error enhancing QRS: {str(e)}")
            return signal
    
    def _find_peaks_dynamic(self, integrated: np.ndarray, fs: float) -> List[int]:
        """Dinamik eşikleme ile peak'leri bul."""
        try:
            # İlk yüksek eşik ile başla
            threshold = 0.6 * np.max(integrated)
            min_distance = int(self._min_rr_interval_ms * fs / 1000)
            
            # Peak'leri bul
            peaks, properties = scipy_signal.find_peaks(
                integrated,
                height=threshold,
                distance=min_distance
            )
            
            # Çok az peak varsa eşiği düşür
            if len(peaks) < 5:
                threshold = 0.4 * np.max(integrated)
                peaks, properties = scipy_signal.find_peaks(
                    integrated,
                    height=threshold,
                    distance=min_distance
                )
            
            # Hala çok az varsa daha da düşür
            if len(peaks) < 3:
                threshold = 0.3 * np.max(integrated)
                peaks, properties = scipy_signal.find_peaks(
                    integrated,
                    height=threshold,
                    distance=min_distance
                )
            
            return peaks.tolist()
            
        except Exception as e:
            logger.error(f"Error finding peaks: {str(e)}")
            return []
    
    def _refine_peak_locations(self, original: np.ndarray, 
                              peak_indices: List[int], fs: float) -> List[int]:
        """Peak konumlarını orijinal sinyalde iyileştir."""
        try:
            refined_peaks = []
            search_window = int(0.05 * fs)  # 50ms pencere
            
            for peak in peak_indices:
                start = max(0, peak - search_window)
                end = min(len(original), peak + search_window)
                
                if start < end:
                    # Pencere içinde maksimum genliği bul
                    window = original[start:end]
                    
                    # Hem pozitif hem negatif peak'leri kontrol et
                    max_idx = np.argmax(np.abs(window))
                    refined_peaks.append(start + max_idx)
                else:
                    refined_peaks.append(peak)
            
            return refined_peaks
            
        except Exception as e:
            logger.error(f"Error refining peak locations: {str(e)}")
            return peak_indices
    
    def _post_process_peaks(self, signal: np.ndarray, 
                           peaks: List[int], fs: float) -> List[int]:
        """Post-processing ile false positive'leri kaldır."""
        try:
            if len(peaks) < 2:
                return peaks
            
            # Sırala
            peaks = sorted(peaks)
            
            # Çok yakın peak'leri birleştir
            min_distance = int(self._min_rr_interval_ms * fs / 1000)
            final_peaks = [peaks[0]]
            
            for peak in peaks[1:]:
                if peak - final_peaks[-1] >= min_distance:
                    final_peaks.append(peak)
                else:
                    # Hangisinin genliği daha yüksek?
                    if abs(signal[peak]) > abs(signal[final_peaks[-1]]):
                        final_peaks[-1] = peak
            
            # Genlik bazlı outlier'ları kaldır
            if len(final_peaks) > 3:
                amplitudes = [abs(signal[p]) for p in final_peaks]
                median_amp = np.median(amplitudes)
                
                # Median'ın %30'undan düşük genlikli peak'leri kaldır
                valid_peaks = []
                for i, peak in enumerate(final_peaks):
                    if amplitudes[i] >= 0.3 * median_amp:
                        valid_peaks.append(peak)
                
                if len(valid_peaks) >= 3:
                    final_peaks = valid_peaks
            
            return final_peaks
            
        except Exception as e:
            logger.error(f"Error in post-processing peaks: {str(e)}")
            return peaks
    
    def _create_rpeak_objects(self, signal: np.ndarray, 
                             indices: List[int], fs: float) -> List[RPeak]:
        """İndekslerden RPeak objeleri oluştur."""
        r_peaks = []
        
        for idx in indices:
            if 0 <= idx < len(signal):
                r_peaks.append(RPeak(
                    index=idx,
                    time=idx / fs,
                    amplitude=float(signal[idx]),
                    confidence=1.0  # Başlangıç güvenilirliği
                ))
        
        return r_peaks
    
    def _estimate_local_snr(self, signal_window: np.ndarray) -> float:
        """Yerel sinyal-gürültü oranını tahmin et."""
        try:
            if len(signal_window) < 3:
                return 10.0  # Varsayılan
            
            # Basit SNR tahmini
            signal_power = np.std(signal_window)
            
            # Gürültüyü türevden tahmin et
            noise_estimate = np.std(np.diff(signal_window)) / np.sqrt(2)
            
            if noise_estimate > 0:
                snr = 20 * np.log10(signal_power / noise_estimate)
                return max(0, snr)
            else:
                return 20.0  # Gürültü çok düşük
                
        except:
            return 10.0  # Varsayılan