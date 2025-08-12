"""
Calibration Interface Definitions

Kalibrasyon işlemleri için kullanılan arayüzleri tanımlar.
Dependency Inversion Principle (DIP) uygulaması için abstraksiyonlar sağlar.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Tuple
import numpy as np

from src.domain.models.calibration_models import (
    CalibrationResult, CalibrationRequest, CatheterMeasurement,
    CalibrationValidation, CalibrationPoint, CatheterSize
)


class ICatheterSegmentation(Protocol):
    """
    Kateter segmentasyonu için protokol.
    
    Farklı segmentasyon yöntemleri bu protokolü implement eder.
    """
    
    def segment_catheter(self, 
                        image: np.ndarray,
                        roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Kateter segmentasyonu yapar.
        
        Args:
            image: Giriş görüntüsü
            roi: İlgi alanı (x, y, width, height)
            
        Returns:
            np.ndarray: Segmentasyon maskesi
        """
        ...
        
    @property
    def method_name(self) -> str:
        """Segmentasyon yöntemi adı"""
        ...


class ICatheterMeasurement(Protocol):
    """
    Kateter genişliği ölçümü için protokol.
    
    Farklı ölçüm algoritmaları bu protokolü implement eder.
    """
    
    def measure_width(self,
                     segmentation_mask: np.ndarray,
                     contour: np.ndarray) -> CatheterMeasurement:
        """
        Kateter genişliğini ölçer.
        
        Args:
            segmentation_mask: Segmentasyon maskesi
            contour: Kateter konturu
            
        Returns:
            CatheterMeasurement: Ölçüm sonucu
        """
        ...


class ICalibrationService(ABC):
    """
    Kalibrasyon servisi için abstract base class.
    
    Ana kalibrasyon iş mantığını tanımlar.
    """
    
    @abstractmethod
    def calibrate(self, request: CalibrationRequest) -> CalibrationResult:
        """
        Kalibrasyon işlemini gerçekleştirir.
        
        Args:
            request: Kalibrasyon parametreleri
            
        Returns:
            CalibrationResult: Kalibrasyon sonucu
        """
        pass
    
    @abstractmethod
    def validate_calibration(self, result: CalibrationResult) -> CalibrationValidation:
        """
        Kalibrasyon sonucunu doğrular.
        
        Args:
            result: Doğrulanacak kalibrasyon sonucu
            
        Returns:
            CalibrationValidation: Doğrulama sonucu
        """
        pass
    
    @abstractmethod
    def calculate_manual_calibration(self,
                                   points: List[CalibrationPoint],
                                   known_distance_mm: float) -> CalibrationResult:
        """
        Manuel nokta seçimi ile kalibrasyon yapar.
        
        Args:
            points: Seçilen noktalar
            known_distance_mm: Bilinen mesafe (mm)
            
        Returns:
            CalibrationResult: Kalibrasyon sonucu
        """
        pass
    
    @abstractmethod
    def calculate_catheter_calibration(self,
                                     measurement: CatheterMeasurement,
                                     catheter_size: CatheterSize) -> CalibrationResult:
        """
        Kateter ölçümünden kalibrasyon hesaplar.
        
        Args:
            measurement: Kateter ölçüm sonucu
            catheter_size: Kateter boyutu
            
        Returns:
            CalibrationResult: Kalibrasyon sonucu
        """
        pass


class ICalibrationPersistence(Protocol):
    """
    Kalibrasyon verilerinin saklanması için protokol.
    """
    
    def save_calibration(self, 
                        calibration: CalibrationResult,
                        study_id: str) -> bool:
        """
        Kalibrasyonu saklar.
        
        Args:
            calibration: Saklancak kalibrasyon
            study_id: Çalışma ID'si
            
        Returns:
            bool: Başarılı mı?
        """
        ...
        
    def load_calibration(self, study_id: str) -> Optional[CalibrationResult]:
        """
        Kalibrasyonu yükler.
        
        Args:
            study_id: Çalışma ID'si
            
        Returns:
            Optional[CalibrationResult]: Varsa kalibrasyon
        """
        ...
        
    def get_recent_calibrations(self, limit: int = 10) -> List[CalibrationResult]:
        """
        Son kalibrasyonları getirir.
        
        Args:
            limit: Maksimum sayı
            
        Returns:
            List[CalibrationResult]: Kalibrasyon listesi
        """
        ...


class ICalibrationVisualization(Protocol):
    """
    Kalibrasyon görselleştirmesi için protokol.
    """
    
    def visualize_calibration(self,
                            image: np.ndarray,
                            calibration: CalibrationResult) -> np.ndarray:
        """
        Kalibrasyon sonucunu görselleştirir.
        
        Args:
            image: Orijinal görüntü
            calibration: Kalibrasyon sonucu
            
        Returns:
            np.ndarray: İşaretlenmiş görüntü
        """
        ...
        
    def draw_measurement_line(self,
                            image: np.ndarray,
                            start_point: Tuple[float, float],
                            end_point: Tuple[float, float],
                            length_mm: float) -> np.ndarray:
        """
        Ölçüm çizgisi çizer.
        
        Args:
            image: Görüntü
            start_point: Başlangıç noktası
            end_point: Bitiş noktası
            length_mm: Uzunluk (mm)
            
        Returns:
            np.ndarray: İşaretlenmiş görüntü
        """
        ...
        
    def draw_catheter_overlay(self,
                            image: np.ndarray,
                            measurement: CatheterMeasurement) -> np.ndarray:
        """
        Kateter overlay'i çizer.
        
        Args:
            image: Görüntü
            measurement: Kateter ölçümü
            
        Returns:
            np.ndarray: İşaretlenmiş görüntü
        """
        ...


class IDICOMCalibrationExtractor(Protocol):
    """
    DICOM metadata'dan kalibrasyon bilgisi çıkarma protokolü.
    """
    
    def extract_pixel_spacing(self, dicom_dataset) -> Optional[Tuple[float, float]]:
        """
        DICOM'dan pixel spacing bilgisini çıkarır.
        
        Args:
            dicom_dataset: DICOM dataset
            
        Returns:
            Optional[Tuple[float, float]]: (row_spacing, col_spacing) mm/pixel
        """
        ...
        
    def extract_imager_pixel_spacing(self, dicom_dataset) -> Optional[Tuple[float, float]]:
        """
        Imager pixel spacing bilgisini çıkarır.
        
        Args:
            dicom_dataset: DICOM dataset
            
        Returns:
            Optional[Tuple[float, float]]: (row_spacing, col_spacing) mm/pixel
        """
        ...
        
    def calculate_calibration_from_dicom(self, dicom_dataset) -> Optional[CalibrationResult]:
        """
        DICOM metadata'dan kalibrasyon hesaplar.
        
        Args:
            dicom_dataset: DICOM dataset
            
        Returns:
            Optional[CalibrationResult]: Hesaplanan kalibrasyon
        """
        ...