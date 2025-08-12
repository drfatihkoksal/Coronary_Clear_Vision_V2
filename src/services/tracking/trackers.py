"""
Tracking Algorithm Implementations

Farklı izleme algoritmaları implementasyonları.
Her algoritma ITracker protokolünü implement eder.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
import logging
from abc import ABC, abstractmethod

from src.domain.models.tracking_models import Point2D
from src.domain.interfaces.tracking_interfaces import ITracker

logger = logging.getLogger(__name__)


class BaseTracker(ABC):
    """
    İzleyici base class.
    
    Ortak fonksiyonalite ve yardımcı metodlar.
    """
    
    def __init__(self, template_size: int = 21, search_radius: int = 30):
        """
        BaseTracker constructor.
        
        Args:
            template_size: Şablon pencere boyutu (tek sayı)
            search_radius: Arama yarıçapı
        """
        # Tek sayı olmalı
        if template_size % 2 == 0:
            template_size += 1
            
        self.template_size = template_size
        self.search_radius = search_radius
        self.template = None
        self.last_position = None
        self.confidence_history = []
        
    @property
    @abstractmethod
    def method_name(self) -> str:
        """İzleme yöntemi adı"""
        pass
    
    def _extract_template(self, 
                         image: np.ndarray, 
                         center: Point2D,
                         size: int) -> Optional[np.ndarray]:
        """
        Görüntüden şablon çıkar.
        
        Args:
            image: Kaynak görüntü
            center: Merkez nokta
            size: Şablon boyutu
            
        Returns:
            Optional[np.ndarray]: Şablon veya None
        """
        half_size = size // 2
        
        # Sınırları kontrol et
        x1 = int(center.x - half_size)
        y1 = int(center.y - half_size)
        x2 = int(center.x + half_size + 1)
        y2 = int(center.y + half_size + 1)
        
        # Görüntü sınırları içinde mi?
        if (x1 < 0 or y1 < 0 or 
            x2 > image.shape[1] or y2 > image.shape[0]):
            logger.warning(f"Template extraction out of bounds: ({x1},{y1}) to ({x2},{y2})")
            return None
        
        # Şablonu çıkar
        template = image[y1:y2, x1:x2].copy()
        
        return template
    
    def _get_search_region(self,
                          image: np.ndarray,
                          center: Point2D,
                          region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, int, int]:
        """
        Arama bölgesini al.
        
        Args:
            image: Kaynak görüntü
            center: Merkez nokta
            region: Özel arama bölgesi (x, y, w, h)
            
        Returns:
            Tuple: (bölge_görüntüsü, offset_x, offset_y)
        """
        if region:
            x, y, w, h = region
            search_region = image[y:y+h, x:x+w]
            return search_region, x, y
        else:
            # Varsayılan arama bölgesi
            x1 = max(0, int(center.x - self.search_radius))
            y1 = max(0, int(center.y - self.search_radius))
            x2 = min(image.shape[1], int(center.x + self.search_radius))
            y2 = min(image.shape[0], int(center.y + self.search_radius))
            
            search_region = image[y1:y2, x1:x2]
            return search_region, x1, y1
    
    def _refine_position(self,
                        image: np.ndarray,
                        position: Point2D,
                        window_size: int = 5) -> Point2D:
        """
        Sub-piksel hassasiyetinde pozisyon iyileştirme.
        
        Args:
            image: Görüntü (genellikle korelasyon haritası)
            position: İlk pozisyon
            window_size: İyileştirme pencere boyutu
            
        Returns:
            Point2D: İyileştirilmiş pozisyon
        """
        x, y = int(position.x), int(position.y)
        half_win = window_size // 2
        
        # Pencere sınırlarını kontrol et
        x1 = max(half_win, x - half_win)
        y1 = max(half_win, y - half_win)
        x2 = min(image.shape[1] - half_win, x + half_win)
        y2 = min(image.shape[0] - half_win, y + half_win)
        
        # Pencere içinde ağırlık merkezi hesapla
        window = image[y1:y2+1, x1:x2+1].astype(np.float32)
        
        # Ağırlıkları normalize et
        window = window - window.min()
        if window.max() > 0:
            window = window / window.max()
        
        # Koordinat grid'leri
        yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
        
        # Ağırlıklı ortalama
        total_weight = np.sum(window)
        if total_weight > 0:
            refined_x = np.sum(xx * window) / total_weight
            refined_y = np.sum(yy * window) / total_weight
            return Point2D(refined_x, refined_y)
        else:
            return position


class TemplateMatchingTracker(BaseTracker, ITracker):
    """
    Şablon eşleme tabanlı izleyici.
    
    OpenCV template matching kullanarak izleme yapar.
    Hızlı ve güvenilir, ancak rotasyon/ölçek değişimlerine duyarlı.
    """
    
    def __init__(self, 
                 template_size: int = 21,
                 search_radius: int = 30,
                 matching_method: int = cv2.TM_CCOEFF_NORMED):
        """
        TemplateMatchingTracker constructor.
        
        Args:
            template_size: Şablon boyutu
            search_radius: Arama yarıçapı
            matching_method: OpenCV eşleme yöntemi
        """
        super().__init__(template_size, search_radius)
        self.matching_method = matching_method
        self.min_correlation = 0.5  # Minimum kabul edilebilir korelasyon
        
    @property
    def method_name(self) -> str:
        """İzleme yöntemi adı"""
        return "Template Matching"
    
    def initialize(self, 
                  image: np.ndarray, 
                  point: Point2D,
                  template_size: int) -> bool:
        """
        İzleyiciyi başlat.
        
        Args:
            image: Başlangıç görüntüsü
            point: İzlenecek nokta
            template_size: Şablon boyutu
            
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Şablonu çıkar
            self.template_size = template_size
            self.template = self._extract_template(gray, point, template_size)
            
            if self.template is None:
                logger.error("Failed to extract template")
                return False
            
            self.last_position = point
            self.confidence_history = [1.0]  # İlk güven %100
            
            logger.debug(f"Template initialized at ({point.x:.1f}, {point.y:.1f}), "
                        f"size: {self.template.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Template initialization error: {str(e)}")
            return False
    
    def track(self, 
             image: np.ndarray,
             search_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, Point2D, float]:
        """
        Noktayı yeni frame'de izle.
        
        Args:
            image: Yeni frame
            search_region: Arama bölgesi (x, y, width, height)
            
        Returns:
            Tuple: (başarılı, yeni_pozisyon, güven_skoru)
        """
        try:
            if self.template is None or self.last_position is None:
                logger.error("Tracker not initialized")
                return False, Point2D(0, 0), 0.0
            
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Arama bölgesini al
            search_img, offset_x, offset_y = self._get_search_region(
                gray, self.last_position, search_region
            )
            
            # Arama bölgesi çok küçükse
            if (search_img.shape[0] < self.template.shape[0] or 
                search_img.shape[1] < self.template.shape[1]):
                logger.warning("Search region too small for template")
                return False, self.last_position, 0.0
            
            # Şablon eşleme
            result = cv2.matchTemplate(
                search_img, 
                self.template, 
                self.matching_method
            )
            
            # En iyi eşleşmeyi bul
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Yönteme göre en iyi konumu seç
            if self.matching_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                best_loc = min_loc
                confidence = 1.0 - min_val  # Düşük değer daha iyi
            else:
                best_loc = max_loc
                confidence = max_val  # Yüksek değer daha iyi
            
            # Minimum korelasyon kontrolü
            if confidence < self.min_correlation:
                logger.debug(f"Low correlation: {confidence:.3f} < {self.min_correlation}")
                return False, self.last_position, confidence
            
            # Global koordinatlara çevir
            new_x = best_loc[0] + offset_x + self.template.shape[1] // 2
            new_y = best_loc[1] + offset_y + self.template.shape[0] // 2
            
            new_position = Point2D(float(new_x), float(new_y))
            
            # Sub-piksel iyileştirme
            if confidence > 0.8:  # Yüksek güven durumunda
                new_position = self._refine_position(result, Point2D(best_loc[0], best_loc[1]))
                new_position = Point2D(
                    new_position.x + offset_x + self.template.shape[1] // 2,
                    new_position.y + offset_y + self.template.shape[0] // 2
                )
            
            # Güncelle
            self.last_position = new_position
            self.confidence_history.append(confidence)
            
            # Güven geçmişini sınırla
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)
            
            return True, new_position, confidence
            
        except Exception as e:
            logger.error(f"Template tracking error: {str(e)}")
            return False, self.last_position, 0.0
    
    def update_template(self, 
                       image: np.ndarray,
                       position: Point2D,
                       update_rate: float):
        """
        Şablonu güncelle.
        
        Args:
            image: Güncel görüntü
            position: Güncel pozisyon
            update_rate: Güncelleme oranı (0-1)
        """
        try:
            if update_rate <= 0:
                return
            
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Yeni şablonu çıkar
            new_template = self._extract_template(gray, position, self.template_size)
            
            if new_template is None:
                logger.warning("Failed to extract new template for update")
                return
            
            # Ağırlıklı ortalama ile güncelle
            if self.template is not None:
                self.template = cv2.addWeighted(
                    self.template, 1.0 - update_rate,
                    new_template, update_rate,
                    0
                )
            else:
                self.template = new_template
                
        except Exception as e:
            logger.error(f"Template update error: {str(e)}")


class OpticalFlowTracker(BaseTracker, ITracker):
    """
    Optik akış tabanlı izleyici.
    
    Lucas-Kanade optik akış kullanarak izleme yapar.
    Küçük hareketler için çok hassas.
    """
    
    def __init__(self,
                 template_size: int = 21,
                 search_radius: int = 30,
                 pyramid_levels: int = 3):
        """
        OpticalFlowTracker constructor.
        
        Args:
            template_size: Şablon boyutu
            search_radius: Arama yarıçapı
            pyramid_levels: Piramit seviyeleri
        """
        super().__init__(template_size, search_radius)
        self.pyramid_levels = pyramid_levels
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=pyramid_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_pts = None
        
    @property
    def method_name(self) -> str:
        """İzleme yöntemi adı"""
        return "Optical Flow"
    
    def initialize(self, 
                  image: np.ndarray, 
                  point: Point2D,
                  template_size: int) -> bool:
        """
        İzleyiciyi başlat.
        
        Args:
            image: Başlangıç görüntüsü
            point: İzlenecek nokta
            template_size: Şablon boyutu
            
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                self.prev_gray = image.copy()
            
            # İzlenecek noktayı ayarla
            self.prev_pts = np.array([[point.x, point.y]], dtype=np.float32)
            self.last_position = point
            self.template_size = template_size
            
            # Template region'da özellik noktaları bul
            mask = np.zeros(self.prev_gray.shape, dtype=np.uint8)
            half_size = template_size // 2
            cv2.rectangle(
                mask,
                (int(point.x - half_size), int(point.y - half_size)),
                (int(point.x + half_size), int(point.y + half_size)),
                255, -1
            )
            
            # Ek özellik noktaları
            p0 = cv2.goodFeaturesToTrack(
                self.prev_gray,
                mask=mask,
                **self.feature_params
            )
            
            if p0 is not None and len(p0) > 0:
                # Ana noktayı ve yakın özellikleri birleştir
                self.prev_pts = np.vstack([self.prev_pts, p0.reshape(-1, 2)])
            
            logger.debug(f"Optical flow initialized with {len(self.prev_pts)} points")
            
            return True
            
        except Exception as e:
            logger.error(f"Optical flow initialization error: {str(e)}")
            return False
    
    def track(self, 
             image: np.ndarray,
             search_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, Point2D, float]:
        """
        Noktayı yeni frame'de izle.
        
        Args:
            image: Yeni frame
            search_region: Arama bölgesi (x, y, width, height)
            
        Returns:
            Tuple: (başarılı, yeni_pozisyon, güven_skoru)
        """
        try:
            if self.prev_gray is None or self.prev_pts is None:
                logger.error("Optical flow tracker not initialized")
                return False, Point2D(0, 0), 0.0
            
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Optik akış hesapla
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_pts,
                None,
                **self.lk_params
            )
            
            # Başarılı noktaları filtrele
            if next_pts is not None and status is not None:
                good_new = next_pts[status == 1]
                good_old = self.prev_pts[status == 1]
                
                if len(good_new) > 0:
                    # Ana nokta (ilk nokta) takibi
                    if status[0] == 1:
                        new_position = Point2D(next_pts[0][0], next_pts[0][1])
                        
                        # Güven hesapla
                        # 1. Error değerine göre
                        point_error = error[0][0] if error is not None else 10.0
                        error_confidence = np.exp(-point_error / 10.0)
                        
                        # 2. Hareket tutarlılığına göre
                        if len(good_new) > 1:
                            # Diğer noktaların ortalama hareketi
                            mean_motion = np.mean(good_new - good_old, axis=0)
                            point_motion = next_pts[0] - self.prev_pts[0]
                            
                            # Hareket tutarlılığı
                            motion_diff = np.linalg.norm(point_motion - mean_motion)
                            motion_confidence = np.exp(-motion_diff / 5.0)
                        else:
                            motion_confidence = 0.8
                        
                        # Toplam güven
                        confidence = error_confidence * motion_confidence
                        
                        # Güncelle
                        self.prev_gray = gray
                        self.prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
                        self.last_position = new_position
                        
                        return True, new_position, confidence
                    else:
                        # Ana nokta kaybedildi
                        logger.warning("Main tracking point lost in optical flow")
                        
                        # En yakın başarılı noktayı kullan
                        if len(good_new) > 0:
                            distances = np.linalg.norm(
                                good_old - self.prev_pts[0].reshape(1, -1), 
                                axis=1
                            )
                            nearest_idx = np.argmin(distances)
                            new_position = Point2D(
                                good_new[nearest_idx][0], 
                                good_new[nearest_idx][1]
                            )
                            
                            # Düşük güven
                            confidence = 0.5
                            
                            # Güncelle
                            self.prev_gray = gray
                            self.prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
                            self.last_position = new_position
                            
                            return True, new_position, confidence
            
            logger.warning("Optical flow tracking failed")
            return False, self.last_position, 0.0
            
        except Exception as e:
            logger.error(f"Optical flow tracking error: {str(e)}")
            return False, self.last_position, 0.0
    
    def update_template(self, 
                       image: np.ndarray,
                       position: Point2D,
                       update_rate: float):
        """
        Şablonu güncelle (Optik akış için özellik noktalarını güncelle).
        
        Args:
            image: Güncel görüntü
            position: Güncel pozisyon
            update_rate: Güncelleme oranı (0-1)
        """
        try:
            if update_rate <= 0:
                return
            
            # Yeni özellik noktaları bul
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Mevcut nokta etrafında maske
            mask = np.zeros(gray.shape, dtype=np.uint8)
            half_size = self.template_size // 2
            cv2.rectangle(
                mask,
                (int(position.x - half_size), int(position.y - half_size)),
                (int(position.x + half_size), int(position.y + half_size)),
                255, -1
            )
            
            # Yeni özellikler
            new_features = cv2.goodFeaturesToTrack(
                gray,
                mask=mask,
                **self.feature_params
            )
            
            if new_features is not None and len(new_features) > 0:
                # Ana noktayı koru, yeni özellikleri ekle
                main_point = np.array([[position.x, position.y]], dtype=np.float32)
                
                # Eski ve yeni özellikleri birleştir
                if update_rate < 1.0:
                    # Kısmi güncelleme - bazı eski noktaları koru
                    n_keep = int(len(self.prev_pts) * (1 - update_rate))
                    if n_keep > 0:
                        old_features = self.prev_pts[1:n_keep+1]
                        self.prev_pts = np.vstack([
                            main_point,
                            old_features,
                            new_features.reshape(-1, 2)
                        ])
                    else:
                        self.prev_pts = np.vstack([
                            main_point,
                            new_features.reshape(-1, 2)
                        ])
                else:
                    # Tam güncelleme
                    self.prev_pts = np.vstack([
                        main_point,
                        new_features.reshape(-1, 2)
                    ])
                
                # Maksimum nokta sayısını sınırla
                if len(self.prev_pts) > 50:
                    # En yakın noktaları koru
                    distances = np.linalg.norm(
                        self.prev_pts[1:] - main_point,
                        axis=1
                    )
                    keep_indices = np.argsort(distances)[:49]
                    self.prev_pts = np.vstack([
                        main_point,
                        self.prev_pts[1:][keep_indices]
                    ])
                
        except Exception as e:
            logger.error(f"Feature update error: {str(e)}")


def create_tracker(method: str, params: Dict[str, Any]) -> ITracker:
    """
    İzleyici factory fonksiyonu.
    
    Args:
        method: İzleme yöntemi
        params: Parametreler
        
    Returns:
        ITracker: İzleyici instance
    """
    if method == "template_matching":
        return TemplateMatchingTracker(**params)
    elif method == "optical_flow":
        return OpticalFlowTracker(**params)
    else:
        raise ValueError(f"Unknown tracking method: {method}")