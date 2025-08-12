"""
Segmentation Interface Definitions

Segmentasyon işlemleri için kullanılan arayüzleri tanımlar.
Dependency Inversion Principle (DIP) uygulaması için abstraksiyonlar sağlar.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Tuple, Dict, Any
import numpy as np

from src.domain.models.segmentation_models import (
    SegmentationResult, SegmentationRequest, VesselFeatures,
    SegmentationValidation, UserPoint
)


class ISegmentationEngine(Protocol):
    """
    Segmentasyon motoru için protokol.
    
    Farklı segmentasyon algoritmaları bu protokolü implement eder.
    """
    
    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        """
        Görüntü segmentasyonu yapar.
        
        Args:
            request: Segmentasyon parametreleri
            
        Returns:
            SegmentationResult: Segmentasyon sonucu
        """
        ...
        
    @property
    def method_name(self) -> str:
        """Segmentasyon yöntemi adı"""
        ...
        
    @property
    def requires_user_input(self) -> bool:
        """Kullanıcı girişi gerekli mi?"""
        ...


class IFeatureExtractor(Protocol):
    """
    Damar özellik çıkarımı için protokol.
    
    Segmentasyon maskesinden anlamlı özellikler çıkarır.
    """
    
    def extract_features(self, mask: np.ndarray) -> VesselFeatures:
        """
        Damar özelliklerini çıkarır.
        
        Args:
            mask: Segmentasyon maskesi
            
        Returns:
            VesselFeatures: Çıkarılan özellikler
        """
        ...
        
    def extract_centerline(self, mask: np.ndarray) -> np.ndarray:
        """
        Damar merkez hattını çıkarır.
        
        Args:
            mask: Segmentasyon maskesi
            
        Returns:
            np.ndarray: Merkez hattı noktaları
        """
        ...
        
    def extract_boundaries(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Damar sınırlarını çıkarır.
        
        Args:
            mask: Segmentasyon maskesi
            
        Returns:
            List[np.ndarray]: Sınır konturları
        """
        ...


class IPostProcessor(Protocol):
    """
    Segmentasyon sonrası işleme için protokol.
    
    Maskeyi iyileştirme ve temizleme işlemleri.
    """
    
    def process(self, 
               mask: np.ndarray,
               options: Dict[str, Any]) -> np.ndarray:
        """
        Maskeyi işler ve iyileştirir.
        
        Args:
            mask: Ham segmentasyon maskesi
            options: İşleme seçenekleri
            
        Returns:
            np.ndarray: İşlenmiş maske
        """
        ...
        
    def remove_small_components(self,
                              mask: np.ndarray,
                              min_size: int) -> np.ndarray:
        """
        Küçük bileşenleri kaldırır.
        
        Args:
            mask: Segmentasyon maskesi
            min_size: Minimum bileşen boyutu
            
        Returns:
            np.ndarray: Temizlenmiş maske
        """
        ...
        
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Maskedeki delikleri doldurur.
        
        Args:
            mask: Segmentasyon maskesi
            
        Returns:
            np.ndarray: Doldurulmuş maske
        """
        ...
        
    def smooth_boundaries(self,
                        mask: np.ndarray,
                        iterations: int) -> np.ndarray:
        """
        Sınırları yumuşatır.
        
        Args:
            mask: Segmentasyon maskesi
            iterations: Yumuşatma iterasyonu
            
        Returns:
            np.ndarray: Yumuşatılmış maske
        """
        ...


class ISegmentationService(ABC):
    """
    Segmentasyon servisi için abstract base class.
    
    Ana segmentasyon iş mantığını tanımlar.
    """
    
    @abstractmethod
    def segment_vessel(self, request: SegmentationRequest) -> SegmentationResult:
        """
        Damar segmentasyonu yapar.
        
        Args:
            request: Segmentasyon isteği
            
        Returns:
            SegmentationResult: Segmentasyon sonucu
        """
        pass
        
    @abstractmethod
    def refine_segmentation(self,
                          current_mask: np.ndarray,
                          user_corrections: List[UserPoint],
                          options: Dict[str, Any]) -> SegmentationResult:
        """
        Kullanıcı düzeltmeleriyle segmentasyonu iyileştirir.
        
        Args:
            current_mask: Mevcut segmentasyon
            user_corrections: Kullanıcı düzeltme noktaları
            options: İyileştirme seçenekleri
            
        Returns:
            SegmentationResult: İyileştirilmiş sonuç
        """
        pass
        
    @abstractmethod
    def validate_segmentation(self, result: SegmentationResult) -> SegmentationValidation:
        """
        Segmentasyon sonucunu doğrular.
        
        Args:
            result: Doğrulanacak sonuç
            
        Returns:
            SegmentationValidation: Doğrulama sonucu
        """
        pass
        
    @abstractmethod
    def set_segmentation_engine(self, engine: ISegmentationEngine):
        """Segmentasyon motorunu ayarlar"""
        pass
        
    @abstractmethod
    def set_feature_extractor(self, extractor: IFeatureExtractor):
        """Özellik çıkarıcıyı ayarlar"""
        pass
        
    @abstractmethod
    def set_post_processor(self, processor: IPostProcessor):
        """Post processor'ı ayarlar"""
        pass


class IModelManager(Protocol):
    """
    AI model yönetimi için protokol.
    
    Model indirme, yükleme ve güncelleme işlemleri.
    """
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Model mevcut mu kontrol eder.
        
        Args:
            model_name: Model adı
            
        Returns:
            bool: Mevcut mu?
        """
        ...
        
    def download_model(self,
                      model_name: str,
                      progress_callback: Optional[callable] = None) -> bool:
        """
        Modeli indirir.
        
        Args:
            model_name: Model adı
            progress_callback: İlerleme bildirimi
            
        Returns:
            bool: Başarılı mı?
        """
        ...
        
    def load_model(self, model_name: str) -> Any:
        """
        Modeli yükler.
        
        Args:
            model_name: Model adı
            
        Returns:
            Any: Yüklenen model
        """
        ...
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Model bilgilerini döndürür.
        
        Args:
            model_name: Model adı
            
        Returns:
            Dict: Model bilgileri
        """
        ...


class ISegmentationVisualization(Protocol):
    """
    Segmentasyon görselleştirme için protokol.
    """
    
    def visualize_segmentation(self,
                             image: np.ndarray,
                             mask: np.ndarray,
                             features: Optional[VesselFeatures] = None) -> np.ndarray:
        """
        Segmentasyonu görselleştirir.
        
        Args:
            image: Orijinal görüntü
            mask: Segmentasyon maskesi
            features: Damar özellikleri
            
        Returns:
            np.ndarray: Görselleştirilmiş görüntü
        """
        ...
        
    def create_overlay(self,
                      image: np.ndarray,
                      mask: np.ndarray,
                      alpha: float = 0.5,
                      color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Maskeyi görüntü üzerine overlay yapar.
        
        Args:
            image: Orijinal görüntü
            mask: Segmentasyon maskesi
            alpha: Saydamlık
            color: Overlay rengi
            
        Returns:
            np.ndarray: Overlay'li görüntü
        """
        ...
        
    def draw_centerline(self,
                       image: np.ndarray,
                       centerline: np.ndarray,
                       color: Tuple[int, int, int] = (255, 0, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        Merkez hattını çizer.
        
        Args:
            image: Görüntü
            centerline: Merkez hattı noktaları
            color: Çizgi rengi
            thickness: Çizgi kalınlığı
            
        Returns:
            np.ndarray: İşaretlenmiş görüntü
        """
        ...


class ISegmentationPersistence(Protocol):
    """
    Segmentasyon verilerinin saklanması için protokol.
    """
    
    def save_segmentation(self,
                        result: SegmentationResult,
                        file_path: str,
                        format: str = "npy") -> bool:
        """
        Segmentasyonu dosyaya kaydeder.
        
        Args:
            result: Segmentasyon sonucu
            file_path: Dosya yolu
            format: Dosya formatı
            
        Returns:
            bool: Başarılı mı?
        """
        ...
        
    def load_segmentation(self, file_path: str) -> Optional[SegmentationResult]:
        """
        Segmentasyonu dosyadan yükler.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            Optional[SegmentationResult]: Yüklenen sonuç
        """
        ...
        
    def export_mask(self,
                   mask: np.ndarray,
                   file_path: str,
                   format: str = "png") -> bool:
        """
        Maskeyi görüntü olarak dışa aktarır.
        
        Args:
            mask: Segmentasyon maskesi
            file_path: Dosya yolu
            format: Görüntü formatı
            
        Returns:
            bool: Başarılı mı?
        """
        ...