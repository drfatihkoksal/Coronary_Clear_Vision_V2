"""
HRV Calculator Service

Kalp hızı değişkenliği (HRV) hesaplama servisi.
Zaman alanı HRV metriklerini hesaplar.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Dict
import numpy as np
import logging

from src.domain.models.ecg_models import RRInterval, HRVMetrics
from src.domain.interfaces.ecg_interfaces import IHRVCalculator

logger = logging.getLogger(__name__)


class HRVCalculatorService(IHRVCalculator):
    """
    HRV hesaplama servisi.
    
    Bu servis zaman alanı HRV metriklerini hesaplar:
    - SDNN (Standard Deviation of NN intervals)
    - RMSSD (Root Mean Square of Successive Differences)
    - pNN50 (Percentage of successive NN intervals differing by more than 50ms)
    - Mean HR (Ortalama kalp hızı)
    - SD HR (Kalp hızı standart sapması)
    """
    
    def __init__(self):
        """HRVCalculatorService constructor."""
        self._min_rr_intervals = 5  # Minimum RR interval sayısı
        logger.info("HRVCalculatorService initialized")
    
    def calculate_hrv(self, rr_intervals: List[RRInterval]) -> HRVMetrics:
        """
        HRV metriklerini hesapla.
        
        Args:
            rr_intervals: RR interval listesi
            
        Returns:
            HRVMetrics: HRV metrikleri
        """
        try:
            # Yeterli veri var mı kontrol et
            if len(rr_intervals) < self._min_rr_intervals:
                logger.warning(f"Insufficient RR intervals for HRV calculation: {len(rr_intervals)}")
                return self._create_empty_metrics()
            
            # RR intervallerini milisaniye dizisine çevir
            rr_intervals_ms = np.array([rr.duration_ms for rr in rr_intervals])
            
            # Outlier'ları filtrele
            rr_intervals_ms = self._filter_outliers(rr_intervals_ms)
            
            if len(rr_intervals_ms) < self._min_rr_intervals:
                logger.warning("Too many outliers removed, insufficient data for HRV")
                return self._create_empty_metrics()
            
            # Zaman alanı metriklerini hesapla
            time_domain_metrics = self.calculate_time_domain(rr_intervals_ms)
            
            # Kalp hızı metriklerini hesapla
            heart_rates = 60000 / rr_intervals_ms  # BPM
            mean_hr = np.mean(heart_rates)
            std_hr = np.std(heart_rates)
            
            # HRVMetrics oluştur
            metrics = HRVMetrics(
                mean_rr=time_domain_metrics['mean_rr'],
                std_rr=time_domain_metrics['sdnn'],
                rmssd=time_domain_metrics['rmssd'],
                pnn50=time_domain_metrics['pnn50'],
                mean_hr=mean_hr,
                std_hr=std_hr
            )
            
            logger.info(f"HRV calculation completed - Mean RR: {metrics.mean_rr:.1f}ms, RMSSD: {metrics.rmssd:.1f}ms")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating HRV: {str(e)}")
            return self._create_empty_metrics()
    
    def calculate_time_domain(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """
        Zaman alanı HRV metriklerini hesapla.
        
        Args:
            rr_intervals_ms: RR intervaller (ms)
            
        Returns:
            Dict[str, float]: Zaman alanı metrikleri
        """
        try:
            metrics = {}
            
            # Mean RR
            metrics['mean_rr'] = float(np.mean(rr_intervals_ms))
            
            # SDNN (Standard Deviation of NN intervals)
            metrics['sdnn'] = float(np.std(rr_intervals_ms))
            
            # SDANN (Standard Deviation of Average NN intervals)
            # 5 dakikalık segmentler için hesaplanır, kısa kayıtlarda uygulanamaz
            if len(rr_intervals_ms) > 300:  # ~5 dakika için yeterli
                segment_size = 300  # ~5 dakika (60 BPM varsayımıyla)
                segment_means = []
                for i in range(0, len(rr_intervals_ms), segment_size):
                    segment = rr_intervals_ms[i:i+segment_size]
                    if len(segment) > 10:
                        segment_means.append(np.mean(segment))
                
                if len(segment_means) > 1:
                    metrics['sdann'] = float(np.std(segment_means))
                else:
                    metrics['sdann'] = 0.0
            else:
                metrics['sdann'] = 0.0
            
            # RMSSD (Root Mean Square of Successive Differences)
            if len(rr_intervals_ms) > 1:
                successive_diffs = np.diff(rr_intervals_ms)
                metrics['rmssd'] = float(np.sqrt(np.mean(successive_diffs ** 2)))
            else:
                metrics['rmssd'] = 0.0
            
            # SDSD (Standard Deviation of Successive Differences)
            if len(rr_intervals_ms) > 1:
                successive_diffs = np.diff(rr_intervals_ms)
                metrics['sdsd'] = float(np.std(successive_diffs))
            else:
                metrics['sdsd'] = 0.0
            
            # pNN50 (Percentage of successive NN intervals differing by more than 50ms)
            if len(rr_intervals_ms) > 1:
                successive_diffs = np.abs(np.diff(rr_intervals_ms))
                nn50_count = np.sum(successive_diffs > 50)
                metrics['pnn50'] = float(nn50_count / len(successive_diffs) * 100)
            else:
                metrics['pnn50'] = 0.0
            
            # pNN20 (Percentage of successive NN intervals differing by more than 20ms)
            if len(rr_intervals_ms) > 1:
                successive_diffs = np.abs(np.diff(rr_intervals_ms))
                nn20_count = np.sum(successive_diffs > 20)
                metrics['pnn20'] = float(nn20_count / len(successive_diffs) * 100)
            else:
                metrics['pnn20'] = 0.0
            
            # Triangular index
            # Histogram tabanlı metrik
            if len(rr_intervals_ms) > 20:
                # 7.8125ms bin genişliği (1/128 saniye)
                bin_width = 7.8125
                hist_range = (
                    np.floor(np.min(rr_intervals_ms) / bin_width) * bin_width,
                    np.ceil(np.max(rr_intervals_ms) / bin_width) * bin_width
                )
                n_bins = int((hist_range[1] - hist_range[0]) / bin_width)
                
                if n_bins > 0:
                    hist, _ = np.histogram(rr_intervals_ms, bins=n_bins, range=hist_range)
                    # Triangular index = toplam RR sayısı / maksimum histogram değeri
                    if np.max(hist) > 0:
                        metrics['triangular_index'] = float(len(rr_intervals_ms) / np.max(hist))
                    else:
                        metrics['triangular_index'] = 0.0
                else:
                    metrics['triangular_index'] = 0.0
            else:
                metrics['triangular_index'] = 0.0
            
            # TINN (Triangular Interpolation of NN interval histogram)
            # Daha karmaşık, şimdilik atla
            metrics['tinn'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating time domain metrics: {str(e)}")
            return {
                'mean_rr': 0.0,
                'sdnn': 0.0,
                'sdann': 0.0,
                'rmssd': 0.0,
                'sdsd': 0.0,
                'pnn50': 0.0,
                'pnn20': 0.0,
                'triangular_index': 0.0,
                'tinn': 0.0
            }
    
    def _filter_outliers(self, rr_intervals_ms: np.ndarray) -> np.ndarray:
        """
        RR interval outlier'larını filtrele.
        
        Fizyolojik olmayan RR intervallerini kaldırır:
        - Çok kısa (<300ms, >200 BPM)
        - Çok uzun (>2000ms, <30 BPM)
        - Ardışık RR'ler arasında %20'den fazla değişim
        
        Args:
            rr_intervals_ms: RR intervaller (ms)
            
        Returns:
            np.ndarray: Filtrelenmiş RR intervaller
        """
        try:
            if len(rr_intervals_ms) < 2:
                return rr_intervals_ms
            
            # Fizyolojik sınırlar
            min_rr = 300  # ms (200 BPM)
            max_rr = 2000  # ms (30 BPM)
            
            # Sınırlar içindeki değerleri al
            mask = (rr_intervals_ms >= min_rr) & (rr_intervals_ms <= max_rr)
            
            # Ardışık RR değişim filtresi
            if np.sum(mask) > 1:
                filtered = rr_intervals_ms[mask]
                
                # %20 değişim kuralı
                valid_indices = [0]  # İlk değer her zaman geçerli
                
                for i in range(1, len(filtered)):
                    rr_change = abs(filtered[i] - filtered[i-1]) / filtered[i-1]
                    if rr_change < 0.2:  # %20'den az değişim
                        valid_indices.append(i)
                    else:
                        # Bir sonraki RR'ye bak (spike kontrolü)
                        if i < len(filtered) - 1:
                            next_change = abs(filtered[i+1] - filtered[i-1]) / filtered[i-1]
                            if next_change < 0.2:
                                # Bu bir spike, atla
                                continue
                            else:
                                # Gerçek değişim
                                valid_indices.append(i)
                        else:
                            # Son değer, öncekiyle karşılaştır
                            if i > 1:
                                prev_change = abs(filtered[i] - filtered[i-2]) / filtered[i-2]
                                if prev_change < 0.2:
                                    valid_indices.append(i)
                
                return filtered[valid_indices]
            else:
                return rr_intervals_ms[mask]
                
        except Exception as e:
            logger.error(f"Error filtering outliers: {str(e)}")
            return rr_intervals_ms
    
    def _create_empty_metrics(self) -> HRVMetrics:
        """Boş HRV metrikleri oluştur."""
        return HRVMetrics(
            mean_rr=0.0,
            std_rr=0.0,
            rmssd=0.0,
            pnn50=0.0,
            mean_hr=0.0,
            std_hr=0.0
        )
    
    def calculate_hrv_indices(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """
        Ek HRV indeksleri hesapla.
        
        Args:
            rr_intervals_ms: RR intervaller (ms)
            
        Returns:
            Dict[str, float]: HRV indeksleri
        """
        try:
            indices = {}
            
            # Poincaré plot parametreleri
            if len(rr_intervals_ms) > 1:
                # SD1: Kısa dönem değişkenlik
                successive_diffs = np.diff(rr_intervals_ms)
                indices['sd1'] = float(np.sqrt(0.5 * np.var(successive_diffs)))
                
                # SD2: Uzun dönem değişkenlik
                rr_mean = np.mean(rr_intervals_ms)
                rr_centered = rr_intervals_ms - rr_mean
                indices['sd2'] = float(np.sqrt(2 * np.var(rr_centered) - 0.5 * np.var(successive_diffs)))
                
                # SD1/SD2 oranı
                if indices['sd2'] > 0:
                    indices['sd_ratio'] = indices['sd1'] / indices['sd2']
                else:
                    indices['sd_ratio'] = 0.0
            else:
                indices['sd1'] = 0.0
                indices['sd2'] = 0.0
                indices['sd_ratio'] = 0.0
            
            # Kalp hızı aralığı
            if len(rr_intervals_ms) > 0:
                heart_rates = 60000 / rr_intervals_ms
                indices['hr_min'] = float(np.min(heart_rates))
                indices['hr_max'] = float(np.max(heart_rates))
                indices['hr_range'] = indices['hr_max'] - indices['hr_min']
            else:
                indices['hr_min'] = 0.0
                indices['hr_max'] = 0.0
                indices['hr_range'] = 0.0
            
            return indices
            
        except Exception as e:
            logger.error(f"Error calculating HRV indices: {str(e)}")
            return {
                'sd1': 0.0,
                'sd2': 0.0,
                'sd_ratio': 0.0,
                'hr_min': 0.0,
                'hr_max': 0.0,
                'hr_range': 0.0
            }