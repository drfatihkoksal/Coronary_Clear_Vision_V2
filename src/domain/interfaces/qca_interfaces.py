"""
QCA Interface Definitions

Bu modül QCA analizi için kullanılan arayüzleri (interfaces/protocols) tanımlar.
Dependency Inversion Principle (DIP) uygulaması için abstraksiyonlar sağlar.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Tuple
import numpy as np

from src.domain.models.qca_models import (
    QCAAnalysisResult, QCAAnalysisRequest, CalibrationData,
    VesselMeasurement, StenosisData
)


class IDiameterCalculator(Protocol):
    """
    Damar çapı hesaplama algoritmaları için protokol.
    
    Farklı çap hesaplama yöntemleri bu protokolü implement eder.
    """
    
    def calculate_diameters(self, 
                          edge_points: np.ndarray,
                          centerline: np.ndarray,
                          calibration: CalibrationData) -> List[VesselMeasurement]:
        """
        Damar çaplarını hesapla.
        
        Args:
            edge_points: Damar kenar noktaları
            centerline: Damar merkez hattı
            calibration: Kalibrasyon bilgisi
            
        Returns:
            List[VesselMeasurement]: Çap ölçümleri
        """
        ...
        
    @property
    def method_name(self) -> str:
        """Algoritma adı"""
        ...


class IEdgeDetector(Protocol):
    """
    Damar kenarı tespit algoritmaları için protokol.
    
    Gradient, Canny, vb. edge detection metodları için.
    """
    
    def detect_edges(self, 
                    image: np.ndarray,
                    initial_contour: np.ndarray,
                    **kwargs) -> np.ndarray:
        """
        Damar kenarlarını tespit et.
        
        Args:
            image: Giriş görüntüsü
            initial_contour: Başlangıç konturu
            **kwargs: Algoritma spesifik parametreler
            
        Returns:
            np.ndarray: Tespit edilen kenar noktaları
        """
        ...


class IStenosisDetector(Protocol):
    """
    Stenoz tespit algoritmaları için protokol.
    
    Darlık analizi yapan algoritmalar için.
    """
    
    def detect_stenosis(self,
                       measurements: List[VesselMeasurement],
                       threshold_percent: float = 50.0) -> Optional[StenosisData]:
        """
        Stenoz tespit et ve analiz et.
        
        Args:
            measurements: Damar çap ölçümleri
            threshold_percent: Stenoz eşik değeri
            
        Returns:
            Optional[StenosisData]: Stenoz bilgileri veya None
        """
        ...


class ICalibrationService(ABC):
    """
    Kalibrasyon servisi için abstract base class.
    
    Farklı kalibrasyon yöntemlerini destekler.
    """
    
    @abstractmethod
    def calibrate_manual(self, 
                        points: List[Tuple[float, float]], 
                        known_length_mm: float) -> CalibrationData:
        """
        Manuel kalibrasyon yap.
        
        Args:
            points: Kalibrasyon noktaları (piksel)
            known_length_mm: Bilinen uzunluk (mm)
            
        Returns:
            CalibrationData: Kalibrasyon sonucu
        """
        pass
    
    @abstractmethod
    def calibrate_catheter(self,
                          catheter_pixels: float,
                          catheter_size_fr: int) -> CalibrationData:
        """
        Kateter bazlı kalibrasyon yap.
        
        Args:
            catheter_pixels: Kateter genişliği (piksel)
            catheter_size_fr: Kateter boyutu (French)
            
        Returns:
            CalibrationData: Kalibrasyon sonucu
        """
        pass
    
    @abstractmethod
    def validate_calibration(self, calibration: CalibrationData) -> bool:
        """
        Kalibrasyonu doğrula.
        
        Args:
            calibration: Doğrulanacak kalibrasyon
            
        Returns:
            bool: Geçerli mi?
        """
        pass


class IQCAAnalysisService(ABC):
    """
    QCA analiz servisi için abstract base class.
    
    Ana QCA analiz mantığını tanımlar.
    """
    
    @abstractmethod
    def analyze_vessel(self, request: QCAAnalysisRequest) -> QCAAnalysisResult:
        """
        Tek frame için damar analizi yap.
        
        Args:
            request: Analiz parametreleri
            
        Returns:
            QCAAnalysisResult: Analiz sonucu
        """
        pass
    
    @abstractmethod
    def analyze_sequential(self, 
                          requests: List[QCAAnalysisRequest],
                          progress_callback=None) -> dict:
        """
        Ardışık frame'ler için analiz yap.
        
        Args:
            requests: Analiz istekleri listesi
            progress_callback: İlerleme bildirimi için callback
            
        Returns:
            dict: Ardışık analiz sonuçları
        """
        pass
    
    @abstractmethod
    def set_diameter_calculator(self, calculator: IDiameterCalculator):
        """Çap hesaplama algoritmasını ayarla"""
        pass
    
    @abstractmethod
    def set_edge_detector(self, detector: IEdgeDetector):
        """Kenar tespit algoritmasını ayarla"""
        pass
    
    @abstractmethod
    def set_stenosis_detector(self, detector: IStenosisDetector):
        """Stenoz tespit algoritmasını ayarla"""
        pass


class IQCAExportService(Protocol):
    """
    QCA sonuçlarını dışa aktarma servisi için protokol.
    """
    
    def export_to_excel(self, 
                       results: QCAAnalysisResult,
                       file_path: str) -> bool:
        """Excel'e aktar"""
        ...
        
    def export_to_csv(self,
                     results: QCAAnalysisResult,
                     file_path: str) -> bool:
        """CSV'ye aktar"""
        ...
        
    def export_to_pdf(self,
                     results: QCAAnalysisResult,
                     file_path: str,
                     include_images: bool = True) -> bool:
        """PDF raporu oluştur"""
        ...


class IQCAVisualizationService(Protocol):
    """
    QCA görselleştirme servisi için protokol.
    """
    
    def create_diameter_graph(self,
                            measurements: List[VesselMeasurement]) -> Any:
        """Çap grafiği oluştur"""
        ...
        
    def create_vessel_overlay(self,
                            image: np.ndarray,
                            centerline: np.ndarray,
                            edges: np.ndarray) -> np.ndarray:
        """Damar overlay'i oluştur"""
        ...
        
    def create_stenosis_visualization(self,
                                    image: np.ndarray,
                                    stenosis: StenosisData) -> np.ndarray:
        """Stenoz görselleştirmesi"""
        ...