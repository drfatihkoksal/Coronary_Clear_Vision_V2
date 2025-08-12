"""
ECG Extractor Service

DICOM dosyalarından ECG verisi çıkarma servisi.
Farklı üretici formatlarını destekler.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import logging
from datetime import datetime
from pydicom.tag import Tag

from src.domain.models.ecg_models import (
    ECGSignal, ECGMetadata, ECGSource
)
from src.domain.interfaces.ecg_interfaces import IECGExtractor

logger = logging.getLogger(__name__)


class ECGExtractorService(IECGExtractor):
    """
    ECG veri çıkarma servisi.
    
    Bu servis:
    - Modern waveform sequence formatını destekler
    - Legacy curve data formatını destekler
    - Siemens private tag'lerini destekler
    - GE private tag'lerini destekler
    """
    
    # DICOM tag tanımları
    CURVE_DATA_TAG = Tag(0x5000, 0x3000)  # Legacy curve data
    WAVEFORM_SEQUENCE_TAG = Tag(0x5400, 0x0100)  # Modern waveform sequence
    SIEMENS_ECG_TAG = Tag(0x0019, 0x1010)  # Siemens private
    GE_ECG_TAG = Tag(0x0009, 0x1010)  # GE private
    
    def __init__(self):
        """ECGExtractorService constructor."""
        self._supported_sources = [
            ECGSource.WAVEFORM_SEQUENCE,
            ECGSource.LEGACY_CURVE,
            ECGSource.SIEMENS_PRIVATE,
            ECGSource.GE_PRIVATE
        ]
        logger.info("ECGExtractorService initialized")
    
    def extract_from_dicom(self, dicom_dataset: Any) -> Optional[ECGSignal]:
        """
        DICOM dataset'inden ECG verisi çıkar.
        
        Args:
            dicom_dataset: DICOM dataset objesi
            
        Returns:
            Optional[ECGSignal]: Çıkarılan ECG sinyali veya None
        """
        # Modern waveform sequence'ı dene
        signal = self._extract_waveform_sequence(dicom_dataset)
        if signal:
            logger.info("ECG extracted from WaveformSequence")
            return signal
        
        # Legacy curve data'yı dene
        signal = self._extract_legacy_curve_data(dicom_dataset)
        if signal:
            logger.info("ECG extracted from Legacy Curve Data")
            return signal
        
        # Siemens private tag'leri dene
        signal = self._extract_siemens_private(dicom_dataset)
        if signal:
            logger.info("ECG extracted from Siemens Private Tags")
            return signal
        
        # GE private tag'leri dene
        signal = self._extract_ge_private(dicom_dataset)
        if signal:
            logger.info("ECG extracted from GE Private Tags")
            return signal
        
        logger.warning("No ECG data found in DICOM")
        return None
    
    def get_supported_sources(self) -> List[ECGSource]:
        """
        Desteklenen ECG kaynaklarını döndür.
        
        Returns:
            List[ECGSource]: Desteklenen kaynaklar
        """
        return self._supported_sources.copy()
    
    def _extract_waveform_sequence(self, ds: Any) -> Optional[ECGSignal]:
        """Modern waveform sequence'dan ECG çıkar."""
        try:
            if not hasattr(ds, 'WaveformSequence') or len(ds.WaveformSequence) == 0:
                return None
            
            waveform = ds.WaveformSequence[0]
            
            # Örnekleme hızı
            sampling_rate = float(waveform.SamplingFrequency) if hasattr(waveform, 'SamplingFrequency') else 1000.0
            
            # Örnek sayısı
            num_samples = int(waveform.NumberOfWaveformSamples) if hasattr(waveform, 'NumberOfWaveformSamples') else 0
            
            if num_samples == 0 or not hasattr(waveform, 'WaveformData'):
                return None
            
            # Veri tipini belirle
            raw_data = waveform.WaveformData
            bits = int(waveform.WaveformBitsAllocated) if hasattr(waveform, 'WaveformBitsAllocated') else 16
            
            # Veriyi numpy array'e çevir
            if bits == 16:
                data = np.frombuffer(raw_data, dtype=np.int16)
            elif bits == 8:
                data = np.frombuffer(raw_data, dtype=np.int8)
            else:
                data = np.frombuffer(raw_data, dtype=np.int16)
            
            # Float'a çevir ve normalize et
            data = data.astype(np.float32)
            data = self._normalize_waveform_data(data, waveform)
            
            # Metadata oluştur
            metadata = self._create_metadata(
                source=ECGSource.WAVEFORM_SEQUENCE,
                sampling_rate=sampling_rate,
                num_samples=len(data),
                dataset=ds
            )
            
            return ECGSignal(data=data, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error extracting waveform sequence: {str(e)}")
            return None
    
    def _extract_legacy_curve_data(self, ds: Any) -> Optional[ECGSignal]:
        """Legacy curve data'dan ECG çıkar."""
        try:
            # Curve data gruplarını kontrol et (0x5000-0x501E)
            for group in range(0x5000, 0x5020, 0x0002):
                curve_data_tag = Tag(group, 0x3000)
                curve_dimensions_tag = Tag(group, 0x0005)
                curve_samples_tag = Tag(group, 0x0010)
                
                if curve_data_tag not in ds:
                    continue
                
                # ECG verisi mi kontrol et
                if not self._is_ecg_curve(ds, group):
                    continue
                
                # Boyutlar ve örnek sayısı
                dimensions = ds.get(curve_dimensions_tag, 1)
                num_points = ds.get(curve_samples_tag, 0)
                
                if num_points == 0:
                    continue
                
                # Ham veriyi al
                raw_data = ds[curve_data_tag].value
                
                # Veri tipini belirle
                curve_data_repr_tag = Tag(group, 0x0103)
                if curve_data_repr_tag in ds:
                    data_repr = ds[curve_data_repr_tag].value
                    if data_repr == 0:  # unsigned
                        raw_values = np.frombuffer(raw_data, dtype=np.uint16)
                    else:  # signed
                        raw_values = np.frombuffer(raw_data, dtype=np.int16)
                else:
                    # Varsayılan olarak unsigned 16-bit
                    raw_values = np.frombuffer(raw_data, dtype=np.uint16)
                
                # ECG verisini çıkar
                if dimensions == 2:
                    # 2D curve - sadece ECG değerlerini al
                    data = raw_values[:num_points].astype(np.float32)
                else:
                    # 1D curve
                    data = raw_values.astype(np.float32)
                
                # Siemens verisi için özel işleme
                data = self._process_siemens_curve_data(data)
                
                # Örnekleme hızını hesapla
                sampling_rate = self._calculate_sampling_rate(ds, len(data))
                
                # Metadata oluştur
                metadata = self._create_metadata(
                    source=ECGSource.LEGACY_CURVE,
                    sampling_rate=sampling_rate,
                    num_samples=len(data),
                    dataset=ds,
                    additional_info={'curve_group': f'{group:04X}'}
                )
                
                return ECGSignal(data=data, metadata=metadata)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting legacy curve data: {str(e)}")
            return None
    
    def _extract_siemens_private(self, ds: Any) -> Optional[ECGSignal]:
        """Siemens private tag'lerden ECG çıkar."""
        try:
            if self.SIEMENS_ECG_TAG not in ds:
                return None
            
            raw_data = ds[self.SIEMENS_ECG_TAG].value
            
            # Header'ı atla (ilk 4 byte)
            if len(raw_data) <= 4:
                return None
            
            # ECG verisini çıkar
            data = np.frombuffer(raw_data[4:], dtype=np.int16)
            data = data.astype(np.float32)
            
            # Normalize et
            data = self._normalize_siemens_data(data)
            
            # Metadata oluştur
            metadata = self._create_metadata(
                source=ECGSource.SIEMENS_PRIVATE,
                sampling_rate=1000.0,  # Siemens genelde 1000Hz
                num_samples=len(data),
                dataset=ds,
                vendor="Siemens"
            )
            
            return ECGSignal(data=data, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error extracting Siemens private data: {str(e)}")
            return None
    
    def _extract_ge_private(self, ds: Any) -> Optional[ECGSignal]:
        """GE private tag'lerden ECG çıkar."""
        try:
            if self.GE_ECG_TAG not in ds:
                return None
            
            raw_data = ds[self.GE_ECG_TAG].value
            
            # GE formatı değişken olabilir
            data_array = np.frombuffer(raw_data, dtype=np.int16)
            
            if len(data_array) < 100:
                return None
            
            data = data_array.astype(np.float32)
            
            # Normalize et
            data = self._normalize_ge_data(data)
            
            # Metadata oluştur
            metadata = self._create_metadata(
                source=ECGSource.GE_PRIVATE,
                sampling_rate=500.0,  # GE genelde 500Hz veya 1000Hz
                num_samples=len(data),
                dataset=ds,
                vendor="GE"
            )
            
            return ECGSignal(data=data, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error extracting GE private data: {str(e)}")
            return None
    
    def _is_ecg_curve(self, ds: Any, group: int) -> bool:
        """Curve'ün ECG verisi olup olmadığını kontrol et."""
        curve_label_tag = Tag(group, 0x0040)
        curve_type_tag = Tag(group, 0x0020)
        
        # Type kontrolü
        if curve_type_tag in ds:
            curve_type = str(ds[curve_type_tag].value).upper()
            if 'ECG' in curve_type or 'EKG' in curve_type:
                return True
        
        # Label kontrolü
        if curve_label_tag in ds:
            label = str(ds[curve_label_tag].value).lower()
            if 'ecg' in label or 'ekg' in label:
                return True
        
        # Koroner anjiyografi için varsayılan olarak ECG kabul et
        return True
    
    def _calculate_sampling_rate(self, ds: Any, num_samples: int) -> float:
        """Video süresine göre örnekleme hızını hesapla."""
        try:
            # Frame sayısı ve FPS'den video süresini hesapla
            fps = float(ds.get('CineRate', 15))
            num_frames = int(ds.get('NumberOfFrames', 1))
            video_duration = num_frames / fps
            
            # Örnekleme hızı = örnek sayısı / süre
            sampling_rate = num_samples / video_duration
            
            # Makul aralıkta olduğunu kontrol et
            if 100 <= sampling_rate <= 5000:
                return sampling_rate
            else:
                return 1000.0  # Varsayılan
                
        except:
            return 1000.0  # Varsayılan
    
    def _normalize_waveform_data(self, data: np.ndarray, waveform_item: Any) -> np.ndarray:
        """Waveform verisini normalize et."""
        # Kanal hassasiyeti varsa kullan
        if hasattr(waveform_item, 'ChannelSensitivity'):
            sensitivity = float(waveform_item.ChannelSensitivity)
            data = data * sensitivity
        elif hasattr(waveform_item, 'ChannelSensitivityCorrectionFactor'):
            factor = float(waveform_item.ChannelSensitivityCorrectionFactor)
            data = data * factor
        else:
            # Varsayılan normalizasyon
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = (data / max_val) * 5.0  # ±5mV aralığına ölçekle
        
        return data
    
    def _process_siemens_curve_data(self, data: np.ndarray) -> np.ndarray:
        """Siemens curve verisini işle."""
        # Siemens ECG özellikleri:
        # - 12-bit ADC (0-4095 aralığı)
        # - Baseline 2048 civarında
        # - Unsigned 16-bit olarak saklanır
        
        # DC offset'i kaldır
        baseline = 2048.0  # Tipik 12-bit ADC merkezi
        data = data - baseline
        
        # Milivolta çevir
        # Tipik ECG kazancı: 1 mV = ~200 ADC birimi
        adc_per_mv = 200.0
        data = data / adc_per_mv
        
        # Hafif smoothing uygula
        if len(data) > 10:
            from scipy.ndimage import uniform_filter1d
            data = uniform_filter1d(data, size=3)
        
        return data
    
    def _normalize_siemens_data(self, data: np.ndarray) -> np.ndarray:
        """Siemens private verisini normalize et."""
        # DC offset'i kaldır
        data = data - np.mean(data)
        
        # Milivolta ölçekle
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 5.0
        
        return data
    
    def _normalize_ge_data(self, data: np.ndarray) -> np.ndarray:
        """GE private verisini normalize et."""
        # DC offset'i kaldır
        data = data - np.mean(data)
        
        # Milivolta ölçekle
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val) * 5.0
        
        return data
    
    def _create_metadata(self, source: ECGSource, sampling_rate: float,
                        num_samples: int, dataset: Any,
                        vendor: Optional[str] = None,
                        additional_info: Optional[Dict[str, Any]] = None) -> ECGMetadata:
        """ECG metadata oluştur."""
        # Süreyi hesapla
        duration = num_samples / sampling_rate
        
        # Vendor'ı belirle
        if not vendor and hasattr(dataset, 'Manufacturer'):
            vendor = str(dataset.Manufacturer)
        
        # Modalite
        modality = str(dataset.Modality) if hasattr(dataset, 'Modality') else None
        
        # Çalışma tarihi
        study_date = None
        if hasattr(dataset, 'StudyDate'):
            try:
                study_date = datetime.strptime(str(dataset.StudyDate), '%Y%m%d')
            except:
                pass
        
        # Ek bilgiler
        info = additional_info or {}
        if hasattr(dataset, 'PatientID'):
            info['patient_id'] = str(dataset.PatientID)
        if hasattr(dataset, 'StudyInstanceUID'):
            info['study_uid'] = str(dataset.StudyInstanceUID)
        
        return ECGMetadata(
            source=source,
            sampling_rate=sampling_rate,
            duration=duration,
            num_samples=num_samples,
            vendor=vendor,
            modality=modality,
            study_date=study_date,
            additional_info=info
        )