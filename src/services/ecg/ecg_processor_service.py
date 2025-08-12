"""
ECG Processor Service

ECG sinyal işleme servisi.
Filtreleme, baseline düzeltme ve gürültü temizleme işlemleri.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import Dict, Any
import numpy as np
import logging
from scipy import signal
from scipy.ndimage import median_filter

from src.domain.models.ecg_models import ECGSignal, ECGMetadata
from src.domain.interfaces.ecg_interfaces import IECGProcessor

logger = logging.getLogger(__name__)


class ECGProcessorService(IECGProcessor):
    """
    ECG sinyal işleme servisi.
    
    Bu servis:
    - Bandpass filtreleme
    - Baseline wander kaldırma
    - Güç hattı girişimi temizleme
    - Gürültü azaltma işlemleri yapar
    """
    
    def __init__(self):
        """ECGProcessorService constructor."""
        self._default_bandpass = (0.5, 40.0)  # Hz
        self._default_powerline_freqs = [50.0, 60.0]  # Hz
        logger.info("ECGProcessorService initialized")
    
    def process_signal(self, signal: ECGSignal) -> ECGSignal:
        """
        ECG sinyalini işle.
        
        İşlem sırası:
        1. DC offset kaldırma
        2. Baseline wander kaldırma
        3. Bandpass filtreleme
        4. Güç hattı girişimi temizleme (opsiyonel)
        
        Args:
            signal: Ham ECG sinyali
            
        Returns:
            ECGSignal: İşlenmiş sinyal
        """
        try:
            # Veriyi kopyala
            data = signal.data.copy()
            sampling_rate = signal.metadata.sampling_rate
            
            # 1. DC offset'i kaldır
            data = data - np.mean(data)
            logger.debug("DC offset removed")
            
            # 2. Baseline wander'ı kaldır
            data = self.remove_baseline_wander(data, sampling_rate)
            logger.debug("Baseline wander removed")
            
            # 3. Bandpass filtre uygula
            data = self.apply_bandpass_filter(
                data, 
                self._default_bandpass[0],
                self._default_bandpass[1],
                sampling_rate
            )
            logger.debug(f"Bandpass filter applied: {self._default_bandpass} Hz")
            
            # 4. Güç hattı girişimini tespit ve temizle
            powerline_level = self._detect_powerline_interference(data, sampling_rate)
            if powerline_level > 0.1:  # Eşik değeri
                logger.info(f"Powerline interference detected (level: {powerline_level:.2f})")
                # Hem 50Hz hem 60Hz'i temizle
                for freq in self._default_powerline_freqs:
                    data = self.remove_powerline_interference(data, sampling_rate, freq)
                logger.debug("Powerline interference removed")
            
            # Yeni ECGSignal oluştur
            processed_signal = ECGSignal(
                data=data,
                metadata=signal.metadata,
                timestamp_offset=signal.timestamp_offset
            )
            
            logger.info("ECG signal processing completed")
            return processed_signal
            
        except Exception as e:
            logger.error(f"Error processing ECG signal: {str(e)}")
            # Hata durumunda orijinal sinyali döndür
            return signal
    
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
        try:
            nyquist = sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Frekansların geçerli aralıkta olduğunu kontrol et
            if low >= high or low <= 0 or high >= 1:
                logger.warning(f"Invalid filter frequencies: {low_freq}-{high_freq} Hz")
                return signal
            
            # Butterworth filtre tasarla
            # SOS (Second-Order Sections) formatı daha stabil
            sos = signal.butter(
                N=2,  # Filtre derecesi
                Wn=[low, high],
                btype='band',
                output='sos'
            )
            
            # Filtre uygula (forward-backward için filtfilt)
            filtered = signal.sosfiltfilt(sos, signal)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error applying bandpass filter: {str(e)}")
            return signal
    
    def remove_baseline_wander(self, signal: np.ndarray,
                             sampling_rate: float) -> np.ndarray:
        """
        Baseline kaymasını kaldır.
        
        Median filtre kullanarak yavaş değişen baseline'ı tahmin eder
        ve sinyalden çıkarır.
        
        Args:
            signal: Sinyal verisi
            sampling_rate: Örnekleme hızı (Hz)
            
        Returns:
            np.ndarray: Düzeltilmiş sinyal
        """
        try:
            # Median filtre pencere boyutu (200ms)
            window_size = int(0.2 * sampling_rate)
            
            # Tek sayı olması gerekiyor
            if window_size % 2 == 0:
                window_size += 1
            
            # Minimum pencere boyutu
            window_size = max(window_size, 3)
            
            # Baseline'ı tahmin et
            baseline = median_filter(signal, size=window_size, mode='reflect')
            
            # Baseline'ı çıkar
            corrected = signal - baseline
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error removing baseline wander: {str(e)}")
            return signal
    
    def remove_powerline_interference(self, signal: np.ndarray,
                                    sampling_rate: float,
                                    powerline_freq: float = 50.0) -> np.ndarray:
        """
        Güç hattı girişimini kaldır.
        
        Notch filtre kullanarak belirli frekanstaki girişimi temizler.
        
        Args:
            signal: Sinyal verisi
            sampling_rate: Örnekleme hızı (Hz)
            powerline_freq: Güç hattı frekansı (Hz)
            
        Returns:
            np.ndarray: Temizlenmiş sinyal
        """
        try:
            nyquist = sampling_rate / 2
            
            # Frekansın geçerli aralıkta olduğunu kontrol et
            if powerline_freq >= nyquist:
                logger.warning(f"Powerline frequency {powerline_freq}Hz is above Nyquist frequency")
                return signal
            
            # Notch filtre parametreleri
            notch_freq = powerline_freq / nyquist
            quality_factor = 30.0  # Q faktörü (dar bant için yüksek)
            
            # Notch filtre tasarla
            b, a = signal.iirnotch(notch_freq, quality_factor)
            
            # Filtre uygula
            filtered = signal.filtfilt(b, a, signal)
            
            # Harmonikleri de temizle (2x, 3x frekanslar)
            for harmonic in [2, 3]:
                harmonic_freq = harmonic * powerline_freq
                if harmonic_freq < nyquist:
                    notch_freq = harmonic_freq / nyquist
                    b, a = signal.iirnotch(notch_freq, quality_factor)
                    filtered = signal.filtfilt(b, a, filtered)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error removing powerline interference: {str(e)}")
            return signal
    
    def _detect_powerline_interference(self, signal: np.ndarray,
                                     sampling_rate: float) -> float:
        """
        Güç hattı girişim seviyesini tespit et.
        
        Args:
            signal: Sinyal verisi
            sampling_rate: Örnekleme hızı (Hz)
            
        Returns:
            float: Girişim seviyesi (0-1)
        """
        try:
            # FFT hesapla
            freqs, psd = signal.periodogram(signal, sampling_rate)
            
            # Güç hattı frekanslarında güç hesapla
            total_power = np.sum(psd)
            powerline_power = 0
            
            for freq in self._default_powerline_freqs:
                # Frekans civarında ±1Hz pencere
                mask = (freqs >= freq - 1) & (freqs <= freq + 1)
                if np.any(mask):
                    powerline_power += np.sum(psd[mask])
            
            # Girişim seviyesi
            if total_power > 0:
                interference_level = powerline_power / total_power
            else:
                interference_level = 0
            
            return min(interference_level, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting powerline interference: {str(e)}")
            return 0.0
    
    def apply_smoothing(self, signal: np.ndarray, 
                       window_size: int = 5) -> np.ndarray:
        """
        Sinyal yumuşatma (smoothing) uygula.
        
        Args:
            signal: Sinyal verisi
            window_size: Pencere boyutu
            
        Returns:
            np.ndarray: Yumuşatılmış sinyal
        """
        try:
            if window_size < 3:
                return signal
            
            # Savitzky-Golay filtre kullan
            # Polinom derecesi pencere boyutundan küçük olmalı
            polyorder = min(3, window_size - 1)
            
            smoothed = signal.savgol_filter(
                signal,
                window_length=window_size,
                polyorder=polyorder
            )
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {str(e)}")
            return signal
    
    def enhance_qrs_complex(self, signal: np.ndarray,
                           sampling_rate: float) -> np.ndarray:
        """
        QRS kompleksini vurgula.
        
        R-peak tespiti için QRS kompleksini güçlendirir.
        
        Args:
            signal: Sinyal verisi
            sampling_rate: Örnekleme hızı (Hz)
            
        Returns:
            np.ndarray: QRS vurgulanmış sinyal
        """
        try:
            # QRS frekans aralığı (5-15 Hz)
            filtered = self.apply_bandpass_filter(signal, 5.0, 15.0, sampling_rate)
            
            # Türev al (değişimleri vurgula)
            derivative = np.gradient(filtered)
            
            # Kare al (pozitif yap ve vurgula)
            squared = derivative ** 2
            
            return squared
            
        except Exception as e:
            logger.error(f"Error enhancing QRS complex: {str(e)}")
            return signal
    
    def normalize_amplitude(self, signal: np.ndarray,
                          target_range: tuple = (-1.0, 1.0)) -> np.ndarray:
        """
        Sinyal genliğini normalize et.
        
        Args:
            signal: Sinyal verisi
            target_range: Hedef aralık (min, max)
            
        Returns:
            np.ndarray: Normalize edilmiş sinyal
        """
        try:
            # Min-max normalizasyon
            min_val = np.min(signal)
            max_val = np.max(signal)
            
            if max_val > min_val:
                # Normalize et
                normalized = (signal - min_val) / (max_val - min_val)
                
                # Hedef aralığa ölçekle
                target_min, target_max = target_range
                scaled = normalized * (target_max - target_min) + target_min
                
                return scaled
            else:
                # Sabit sinyal
                return np.full_like(signal, target_range[0])
                
        except Exception as e:
            logger.error(f"Error normalizing amplitude: {str(e)}")
            return signal