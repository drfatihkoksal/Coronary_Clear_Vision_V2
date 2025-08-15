"""
ECG Synchronizer Service

ECG-Video senkronizasyon servisi.
ECG ve video verilerini zamansal olarak eşleştirir.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import Dict
import numpy as np
import logging

from src.domain.models.ecg_models import ECGSignal, ECGSyncInfo, SignalQuality
from src.domain.interfaces.ecg_interfaces import IECGSynchronizer

logger = logging.getLogger(__name__)


class ECGSynchronizerService(IECGSynchronizer):
    """
    ECG-Video senkronizasyon servisi.

    Bu servis:
    - ECG ve video sürelerini karşılaştırır
    - Zaman offset'lerini hesaplar
    - Frame-zaman eşleşmesi yapar
    - Senkronizasyon kalitesini değerlendirir
    """

    def __init__(self):
        """ECGSynchronizerService constructor."""
        self._tolerance_ms = 1.0  # Senkronizasyon toleransı (ms)
        logger.info("ECGSynchronizerService initialized")

    def synchronize(
        self, ecg_signal: ECGSignal, video_duration: float, frame_rate: float
    ) -> ECGSyncInfo:
        """
        ECG ve video senkronizasyonu.

        Args:
            ecg_signal: ECG sinyali
            video_duration: Video süresi (saniye)
            frame_rate: Frame hızı (fps)

        Returns:
            ECGSyncInfo: Senkronizasyon bilgileri
        """
        try:
            # ECG süresi
            ecg_duration = ecg_signal.metadata.duration

            # Süre farkı
            duration_diff_ms = abs(video_duration - ecg_duration) * 1000

            # Senkronizasyon kalitesini değerlendir
            sync_quality = self._evaluate_sync_quality(duration_diff_ms)

            # Zaman offset'i hesapla
            # Basit durum: ECG ve video aynı anda başlar
            # Gelişmiş durum: Cross-correlation ile offset bulunabilir
            time_offset = 0.0

            # Eğer süreler farklıysa, daha kısa olanı merkeze al
            if ecg_duration < video_duration:
                # ECG daha kısa, video'nun ortasına yerleştir
                time_offset = (video_duration - ecg_duration) / 2
            elif video_duration < ecg_duration:
                # Video daha kısa, ECG'nin başından itibaren kullan
                time_offset = 0.0

            # Frame-zaman eşleşmesi oluştur
            frame_to_time_mapping = self._create_frame_time_mapping(
                video_duration, frame_rate, time_offset
            )

            # Senkronizasyon bilgilerini oluştur
            sync_info = ECGSyncInfo(
                video_duration=video_duration,
                ecg_duration=ecg_duration,
                time_offset=time_offset,
                sync_quality=sync_quality,
                frame_to_time_mapping=frame_to_time_mapping,
            )

            logger.info(
                f"Synchronization completed - "
                f"Video: {video_duration:.2f}s, ECG: {ecg_duration:.2f}s, "
                f"Offset: {time_offset:.2f}s, Quality: {sync_quality.value}"
            )

            return sync_info

        except Exception as e:
            logger.error(f"Error in synchronization: {str(e)}")
            # Hata durumunda varsayılan senkronizasyon
            return ECGSyncInfo(
                video_duration=video_duration,
                ecg_duration=ecg_signal.metadata.duration,
                time_offset=0.0,
                sync_quality=SignalQuality.POOR,
                frame_to_time_mapping=None,
            )

    def map_frame_to_time(self, frame_index: int, frame_rate: float) -> float:
        """
        Frame indeksini zamana çevir.

        Args:
            frame_index: Frame indeksi
            frame_rate: Frame hızı (fps)

        Returns:
            float: Zaman (saniye)
        """
        if frame_rate <= 0:
            logger.error("Invalid frame rate")
            return 0.0

        return frame_index / frame_rate

    def map_time_to_frame(self, time_s: float, frame_rate: float) -> int:
        """
        Zamanı frame indeksine çevir.

        Args:
            time_s: Zaman (saniye)
            frame_rate: Frame hızı (fps)

        Returns:
            int: Frame indeksi
        """
        if frame_rate <= 0:
            logger.error("Invalid frame rate")
            return 0

        # En yakın frame'i bul
        frame_index = int(round(time_s * frame_rate))

        # Negatif olmamalı
        return max(0, frame_index)

    def _evaluate_sync_quality(self, duration_diff_ms: float) -> SignalQuality:
        """
        Senkronizasyon kalitesini değerlendir.

        Args:
            duration_diff_ms: Süre farkı (ms)

        Returns:
            SignalQuality: Senkronizasyon kalitesi
        """
        if duration_diff_ms < self._tolerance_ms:
            return SignalQuality.EXCELLENT
        elif duration_diff_ms < self._tolerance_ms * 5:  # 5ms
            return SignalQuality.GOOD
        elif duration_diff_ms < self._tolerance_ms * 10:  # 10ms
            return SignalQuality.FAIR
        elif duration_diff_ms < self._tolerance_ms * 50:  # 50ms
            return SignalQuality.POOR
        else:
            return SignalQuality.UNUSABLE

    def _create_frame_time_mapping(
        self, video_duration: float, frame_rate: float, time_offset: float
    ) -> Dict[int, float]:
        """
        Frame-zaman eşleşme tablosu oluştur.

        Args:
            video_duration: Video süresi (saniye)
            frame_rate: Frame hızı (fps)
            time_offset: Zaman offset'i (saniye)

        Returns:
            Dict[int, float]: Frame indeksi -> ECG zamanı eşleşmesi
        """
        try:
            total_frames = int(video_duration * frame_rate)

            # Her 10 frame'de bir eşleşme kaydet (bellek optimizasyonu)
            step = max(1, total_frames // 1000)  # Maksimum 1000 kayıt

            mapping = {}
            for frame_idx in range(0, total_frames, step):
                video_time = self.map_frame_to_time(frame_idx, frame_rate)
                ecg_time = video_time - time_offset
                mapping[frame_idx] = ecg_time

            # Son frame'i de ekle
            if total_frames - 1 not in mapping:
                video_time = self.map_frame_to_time(total_frames - 1, frame_rate)
                ecg_time = video_time - time_offset
                mapping[total_frames - 1] = ecg_time

            return mapping

        except Exception as e:
            logger.error(f"Error creating frame-time mapping: {str(e)}")
            return {}

    def find_ecg_time_for_frame(
        self, frame_index: int, sync_info: ECGSyncInfo, frame_rate: float
    ) -> float:
        """
        Belirli bir frame için ECG zamanını bul.

        Args:
            frame_index: Frame indeksi
            sync_info: Senkronizasyon bilgileri
            frame_rate: Frame hızı (fps)

        Returns:
            float: ECG zamanı (saniye)
        """
        # Önce mapping tablosuna bak
        if sync_info.frame_to_time_mapping and frame_index in sync_info.frame_to_time_mapping:
            return sync_info.frame_to_time_mapping[frame_index]

        # Mapping'de yoksa hesapla
        video_time = self.map_frame_to_time(frame_index, frame_rate)
        ecg_time = video_time - sync_info.time_offset

        # ECG sınırları içinde olduğundan emin ol
        ecg_time = max(0, min(ecg_time, sync_info.ecg_duration))

        return ecg_time

    def find_frame_for_ecg_time(
        self, ecg_time: float, sync_info: ECGSyncInfo, frame_rate: float
    ) -> int:
        """
        Belirli bir ECG zamanı için frame bul.

        Args:
            ecg_time: ECG zamanı (saniye)
            sync_info: Senkronizasyon bilgileri
            frame_rate: Frame hızı (fps)

        Returns:
            int: Frame indeksi
        """
        # Video zamanına çevir
        video_time = ecg_time + sync_info.time_offset

        # Frame indeksine çevir
        frame_index = self.map_time_to_frame(video_time, frame_rate)

        # Video sınırları içinde olduğundan emin ol
        max_frame = int(sync_info.video_duration * frame_rate) - 1
        frame_index = max(0, min(frame_index, max_frame))

        return frame_index

    def calculate_sync_drift(self, ecg_signal: ECGSignal, video_timestamps: np.ndarray) -> float:
        """
        Senkronizasyon kaymasını hesapla.

        Zamanla oluşan senkronizasyon kaymasını tespit eder.

        Args:
            ecg_signal: ECG sinyali
            video_timestamps: Video frame zaman damgaları

        Returns:
            float: Ortalama kayma (ms)
        """
        try:
            if len(video_timestamps) < 2:
                return 0.0

            # Video frame aralıklarını hesapla
            video_intervals = np.diff(video_timestamps)
            mean_video_interval = np.mean(video_intervals)

            # ECG örnekleme aralığı
            1.0 / ecg_signal.metadata.sampling_rate

            # Beklenen video frame sayısı
            expected_frames = ecg_signal.metadata.duration / mean_video_interval
            actual_frames = len(video_timestamps)

            # Kayma hesapla
            drift_frames = abs(expected_frames - actual_frames)
            drift_ms = drift_frames * mean_video_interval * 1000

            return drift_ms

        except Exception as e:
            logger.error(f"Error calculating sync drift: {str(e)}")
            return 0.0
