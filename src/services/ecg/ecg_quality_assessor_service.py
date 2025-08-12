"""
ECG Quality Assessor Service

ECG sinyal kalitesi değerlendirme servisi.
Sinyal-gürültü oranı, artefakt tespiti ve genel kalite değerlendirmesi.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import Dict, List
import numpy as np
import logging
from scipy import signal as scipy_signal

from src.domain.models.ecg_models import (
    ECGSignal, ECGQualityMetrics, SignalQuality
)
from src.domain.interfaces.ecg_interfaces import IECGQualityAssessor

logger = logging.getLogger(__name__)


class ECGQualityAssessorService(IECGQualityAssessor):
    """
    ECG kalite değerlendirme servisi.
    
    Bu servis:
    - Sinyal-gürültü oranı (SNR) hesaplar
    - Baseline wander seviyesini ölçer
    - Güç hattı girişimini tespit eder
    - Hareket artefaktlarını tespit eder
    - Sinyal kırpılmasını kontrol eder
    - Genel kalite skoru hesaplar
    """
    
    def __init__(self):
        """ECGQualityAssessorService constructor."""
        # Kalite eşik değerleri
        self._snr_thresholds = {
            'excellent': 20.0,  # dB
            'good': 15.0,
            'fair': 10.0,
            'poor': 5.0
        }
        
        self._artifact_thresholds = {
            'baseline_wander': 0.1,
            'powerline': 0.05,
            'motion': 0.2,
            'clipping': 0.01
        }
        
        logger.info("ECGQualityAssessorService initialized")
    
    def assess_quality(self, signal: ECGSignal) -> ECGQualityMetrics:
        """
        Sinyal kalitesini değerlendir.
        
        Args:
            signal: ECG sinyali
            
        Returns:
            ECGQualityMetrics: Kalite metrikleri
        """
        try:
            data = signal.data
            fs = signal.metadata.sampling_rate
            
            # SNR hesapla
            snr_db = self.calculate_snr(data)
            
            # Artefaktları tespit et
            artifacts = self.detect_artifacts(data)
            
            # Baseline wander seviyesi
            baseline_wander = self._assess_baseline_wander(data, fs)
            
            # Güç hattı girişimi
            powerline_interference = self._assess_powerline_interference(data, fs)
            
            # Hareket artefaktları
            motion_artifacts = artifacts.get('motion', 0.0)
            
            # Kırpılma oranı
            clipping_ratio = self._calculate_clipping_ratio(data)
            
            # Genel kalite değerlendirmesi
            overall_quality = self._determine_overall_quality(
                snr_db, baseline_wander, powerline_interference,
                motion_artifacts, clipping_ratio
            )
            
            # Kalite skoru (0-1)
            quality_score = self._calculate_quality_score(
                snr_db, baseline_wander, powerline_interference,
                motion_artifacts, clipping_ratio
            )
            
            # Kalite mesajları
            messages = self._generate_quality_messages(
                snr_db, baseline_wander, powerline_interference,
                motion_artifacts, clipping_ratio
            )
            
            # Metrikleri oluştur
            metrics = ECGQualityMetrics(
                snr_db=snr_db,
                baseline_wander=baseline_wander,
                powerline_interference=powerline_interference,
                motion_artifacts=motion_artifacts,
                clipping_ratio=clipping_ratio,
                overall_quality=overall_quality,
                quality_score=quality_score,
                messages=messages
            )
            
            logger.info(f"Quality assessment completed: {overall_quality.value} (score: {quality_score:.2f})")
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing quality: {str(e)}")
            # Hata durumunda varsayılan metrikler
            return ECGQualityMetrics(
                snr_db=0.0,
                baseline_wander=1.0,
                powerline_interference=1.0,
                motion_artifacts=1.0,
                clipping_ratio=1.0,
                overall_quality=SignalQuality.UNUSABLE,
                quality_score=0.0,
                messages=["Quality assessment failed"]
            )
    
    def calculate_snr(self, signal: np.ndarray) -> float:
        """
        Sinyal-gürültü oranını hesapla.
        
        Args:
            signal: Sinyal verisi
            
        Returns:
            float: SNR (dB)
        """
        try:
            if len(signal) < 10:
                return 0.0
            
            # Sinyal gücü
            signal_power = np.std(signal)
            
            # Gürültü tahmini (yüksek frekanslı bileşenlerden)
            # Türev yöntemi kullan
            noise_estimate = np.std(np.diff(signal)) / np.sqrt(2)
            
            # Alternatif: Median Absolute Deviation (MAD) yöntemi
            # median_signal = np.median(signal)
            # mad = np.median(np.abs(signal - median_signal))
            # noise_estimate = 1.4826 * mad  # Gaussian dağılım için
            
            # SNR hesapla
            if noise_estimate > 0:
                snr_db = 20 * np.log10(signal_power / noise_estimate)
                return max(0, snr_db)  # Negatif SNR'yi 0 yap
            else:
                return 30.0  # Çok düşük gürültü
                
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            return 0.0
    
    def detect_artifacts(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Artefaktları tespit et.
        
        Args:
            signal: Sinyal verisi
            
        Returns:
            Dict[str, float]: Artefakt seviyeleri (0-1)
        """
        try:
            artifacts = {}
            
            # Hareket artefaktları
            # Büyük ve ani genlik değişimleri
            diff_signal = np.diff(signal)
            motion_level = np.std(diff_signal) / (np.std(signal) + 1e-10)
            artifacts['motion'] = min(motion_level / 2, 1.0)  # Normalize
            
            # Ani spike'lar
            median_diff = np.median(np.abs(diff_signal))
            spike_threshold = 10 * median_diff
            spike_count = np.sum(np.abs(diff_signal) > spike_threshold)
            spike_ratio = spike_count / len(diff_signal)
            artifacts['spikes'] = min(spike_ratio * 10, 1.0)
            
            # Düz bölgeler (sinyal kaybı)
            flat_threshold = np.std(signal) * 0.01
            flat_regions = []
            window_size = 50
            
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i+window_size]
                if np.std(window) < flat_threshold:
                    flat_regions.append(i)
            
            flat_ratio = len(flat_regions) * window_size / len(signal)
            artifacts['flat_regions'] = min(flat_ratio * 2, 1.0)
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error detecting artifacts: {str(e)}")
            return {}
    
    def _assess_baseline_wander(self, signal: np.ndarray, fs: float) -> float:
        """Baseline wander seviyesini değerlendir."""
        try:
            # Düşük frekanslı bileşenleri çıkar (< 0.5 Hz)
            nyquist = fs / 2
            low_cutoff = 0.5 / nyquist
            
            if low_cutoff < 1:
                # Highpass filtre ile baseline'ı kaldır
                sos = scipy_signal.butter(2, low_cutoff, btype='high', output='sos')
                filtered = scipy_signal.sosfiltfilt(sos, signal)
                
                # Kaldırılan baseline miktarı
                baseline = signal - filtered
                baseline_power = np.std(baseline)
                signal_power = np.std(signal)
                
                # Baseline/sinyal oranı
                if signal_power > 0:
                    baseline_ratio = baseline_power / signal_power
                    return min(baseline_ratio, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error assessing baseline wander: {str(e)}")
            return 0.5
    
    def _assess_powerline_interference(self, signal: np.ndarray, fs: float) -> float:
        """Güç hattı girişim seviyesini değerlendir."""
        try:
            # FFT ile frekans analizi
            freqs, psd = scipy_signal.periodogram(signal, fs)
            
            # Toplam güç
            total_power = np.sum(psd)
            if total_power == 0:
                return 0.0
            
            # 50Hz ve 60Hz civarındaki güç
            powerline_power = 0
            powerline_freqs = [50.0, 60.0]
            
            for pl_freq in powerline_freqs:
                # ±2Hz pencere
                mask = (freqs >= pl_freq - 2) & (freqs <= pl_freq + 2)
                if np.any(mask):
                    powerline_power += np.sum(psd[mask])
            
            # Güç hattı/toplam güç oranı
            interference_ratio = powerline_power / total_power
            
            return min(interference_ratio * 5, 1.0)  # Amplify ve normalize
            
        except Exception as e:
            logger.error(f"Error assessing powerline interference: {str(e)}")
            return 0.5
    
    def _calculate_clipping_ratio(self, signal: np.ndarray) -> float:
        """Sinyal kırpılma oranını hesapla."""
        try:
            # Maksimum ve minimum değerler
            max_val = np.max(signal)
            min_val = np.min(signal)
            signal_range = max_val - min_val
            
            if signal_range == 0:
                return 1.0
            
            # Kırpılma eşiği (%95)
            clip_threshold = 0.95
            
            # Üst ve alt sınırlara yakın değerler
            upper_threshold = min_val + signal_range * clip_threshold
            lower_threshold = min_val + signal_range * (1 - clip_threshold)
            
            # Kırpılmış örnek sayısı
            clipped_samples = np.sum((signal >= upper_threshold) | (signal <= lower_threshold))
            
            # Kırpılma oranı
            clipping_ratio = clipped_samples / len(signal)
            
            return clipping_ratio
            
        except Exception as e:
            logger.error(f"Error calculating clipping ratio: {str(e)}")
            return 0.0
    
    def _determine_overall_quality(self, snr_db: float, baseline_wander: float,
                                 powerline_interference: float, motion_artifacts: float,
                                 clipping_ratio: float) -> SignalQuality:
        """Genel sinyal kalitesini belirle."""
        # Kötü faktörleri kontrol et
        if clipping_ratio > 0.05:
            return SignalQuality.UNUSABLE
        
        if snr_db < self._snr_thresholds['poor']:
            return SignalQuality.UNUSABLE
        
        # Artefakt skorları
        artifact_score = (
            baseline_wander * 0.2 +
            powerline_interference * 0.3 +
            motion_artifacts * 0.4 +
            clipping_ratio * 0.1
        )
        
        # SNR ve artefakt skoruna göre kalite belirle
        if snr_db >= self._snr_thresholds['excellent'] and artifact_score < 0.1:
            return SignalQuality.EXCELLENT
        elif snr_db >= self._snr_thresholds['good'] and artifact_score < 0.2:
            return SignalQuality.GOOD
        elif snr_db >= self._snr_thresholds['fair'] and artifact_score < 0.4:
            return SignalQuality.FAIR
        elif snr_db >= self._snr_thresholds['poor'] and artifact_score < 0.6:
            return SignalQuality.POOR
        else:
            return SignalQuality.UNUSABLE
    
    def _calculate_quality_score(self, snr_db: float, baseline_wander: float,
                               powerline_interference: float, motion_artifacts: float,
                               clipping_ratio: float) -> float:
        """Sayısal kalite skoru hesapla (0-1)."""
        # SNR skoru (0-1)
        snr_score = min(snr_db / 30, 1.0)  # 30dB'de maksimum
        
        # Artefakt skorları (ters çevir, 1=iyi, 0=kötü)
        baseline_score = 1.0 - baseline_wander
        powerline_score = 1.0 - powerline_interference
        motion_score = 1.0 - motion_artifacts
        clipping_score = 1.0 - clipping_ratio
        
        # Ağırlıklı ortalama
        weights = {
            'snr': 0.4,
            'baseline': 0.15,
            'powerline': 0.15,
            'motion': 0.2,
            'clipping': 0.1
        }
        
        quality_score = (
            snr_score * weights['snr'] +
            baseline_score * weights['baseline'] +
            powerline_score * weights['powerline'] +
            motion_score * weights['motion'] +
            clipping_score * weights['clipping']
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_quality_messages(self, snr_db: float, baseline_wander: float,
                                 powerline_interference: float, motion_artifacts: float,
                                 clipping_ratio: float) -> List[str]:
        """Kalite mesajları oluştur."""
        messages = []
        
        # SNR mesajı
        if snr_db < self._snr_thresholds['poor']:
            messages.append(f"Çok düşük sinyal-gürültü oranı: {snr_db:.1f} dB")
        elif snr_db < self._snr_thresholds['fair']:
            messages.append(f"Düşük sinyal-gürültü oranı: {snr_db:.1f} dB")
        
        # Baseline wander
        if baseline_wander > self._artifact_thresholds['baseline_wander']:
            messages.append(f"Baseline kayması tespit edildi (seviye: {baseline_wander:.2f})")
        
        # Güç hattı girişimi
        if powerline_interference > self._artifact_thresholds['powerline']:
            messages.append(f"Güç hattı girişimi tespit edildi (seviye: {powerline_interference:.2f})")
        
        # Hareket artefaktları
        if motion_artifacts > self._artifact_thresholds['motion']:
            messages.append(f"Hareket artefaktları tespit edildi (seviye: {motion_artifacts:.2f})")
        
        # Kırpılma
        if clipping_ratio > self._artifact_thresholds['clipping']:
            messages.append(f"Sinyal kırpılması tespit edildi (%{clipping_ratio*100:.1f})")
        
        # İyi kalite mesajı
        if not messages:
            messages.append("ECG sinyali iyi kalitede")
        
        return messages