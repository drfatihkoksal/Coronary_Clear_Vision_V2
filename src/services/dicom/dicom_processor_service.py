"""
DICOM Processor Service

DICOM görüntü işleme servisi.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import cv2
import logging
from scipy import ndimage

from src.domain.models.dicom_models import DicomSeries
from src.domain.interfaces.dicom_interfaces import IDicomPixelProcessor, IDicomFrameExtractor

logger = logging.getLogger(__name__)


class DicomProcessorService(IDicomPixelProcessor, IDicomFrameExtractor):
    """
    DICOM görüntü işleme servisi.

    Bu servis:
    - Piksel verisi dönüşümleri
    - Pencere/seviye uygulamaları
    - Kontrast iyileştirme
    - Frame çıkarma ve işleme
    """

    def __init__(self):
        """DicomProcessorService constructor."""
        logger.info("DicomProcessorService initialized")

    # IDicomPixelProcessor implementasyonu

    def normalize_pixels(
        self, pixel_array: np.ndarray, output_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Piksel değerlerini normalize et.

        Args:
            pixel_array: Piksel verisi
            output_range: Çıktı aralığı

        Returns:
            np.ndarray: Normalize edilmiş veri
        """
        try:
            # Min-max normalizasyon
            min_val = np.min(pixel_array)
            max_val = np.max(pixel_array)

            if max_val > min_val:
                normalized = (pixel_array - min_val) / (max_val - min_val)

                # Çıktı aralığına ölçekle
                out_min, out_max = output_range
                scaled = normalized * (out_max - out_min) + out_min

                return scaled
            else:
                # Sabit değer
                return np.full_like(pixel_array, output_range[0], dtype=np.float32)

        except Exception as e:
            logger.error(f"Error normalizing pixels: {str(e)}")
            return pixel_array.astype(np.float32)

    def apply_lut(self, pixel_array: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Look-up table uygula.

        Args:
            pixel_array: Piksel verisi
            lut: Look-up table

        Returns:
            np.ndarray: Dönüştürülmüş veri
        """
        try:
            # LUT boyutu kontrolü
            if len(lut) < 256:
                logger.warning(f"LUT size {len(lut)} is less than 256")
                return pixel_array

            # Değerleri 0-255 aralığına normalize et
            normalized = self.normalize_pixels(pixel_array, (0, 255))
            indices = normalized.astype(np.uint8)

            # LUT uygula
            transformed = lut[indices]

            return transformed

        except Exception as e:
            logger.error(f"Error applying LUT: {str(e)}")
            return pixel_array

    def convert_to_hounsfield(
        self, pixel_array: np.ndarray, slope: float, intercept: float
    ) -> np.ndarray:
        """
        Hounsfield birimlerine çevir (CT için).

        HU = pixel_value * slope + intercept

        Args:
            pixel_array: Piksel verisi
            slope: Rescale slope
            intercept: Rescale intercept

        Returns:
            np.ndarray: HU değerleri
        """
        try:
            # Hounsfield dönüşümü
            hu_values = pixel_array.astype(np.float32) * slope + intercept

            # CT HU aralığı genellikle [-1024, 3071]
            # Ancak sınırlama yapmıyoruz, ham değerleri döndürüyoruz

            return hu_values

        except Exception as e:
            logger.error(f"Error converting to Hounsfield: {str(e)}")
            return pixel_array.astype(np.float32)

    def enhance_contrast(self, pixel_array: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Kontrast iyileştirme.

        Args:
            pixel_array: Piksel verisi
            method: İyileştirme yöntemi
                - "clahe": Contrast Limited Adaptive Histogram Equalization
                - "histogram": Histogram eşitleme
                - "gamma": Gamma düzeltme
                - "sigmoid": Sigmoid kontrast

        Returns:
            np.ndarray: İyileştirilmiş veri
        """
        try:
            # 8-bit'e normalize et
            normalized = self.normalize_pixels(pixel_array, (0, 255))
            uint8_image = normalized.astype(np.uint8)

            if method == "clahe":
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(uint8_image)

            elif method == "histogram":
                # Histogram eşitleme
                enhanced = cv2.equalizeHist(uint8_image)

            elif method == "gamma":
                # Gamma düzeltme (gamma=0.5 için daha parlak)
                gamma = 0.5
                inv_gamma = 1.0 / gamma
                table = np.array(
                    [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
                ).astype("uint8")
                enhanced = cv2.LUT(uint8_image, table)

            elif method == "sigmoid":
                # Sigmoid kontrast
                # Normalize to [-1, 1]
                norm = (normalized / 255.0) * 2 - 1
                # Sigmoid function
                enhanced_norm = 1 / (1 + np.exp(-5 * norm))  # 5 is gain
                # Back to [0, 255]
                enhanced = (enhanced_norm * 255).astype(np.uint8)

            else:
                logger.warning(f"Unknown enhancement method: {method}")
                enhanced = uint8_image

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing contrast: {str(e)}")
            return pixel_array

    # IDicomFrameExtractor implementasyonu

    def extract_frame(
        self, series: DicomSeries, frame_index: int, apply_window_level: bool = True
    ) -> Optional[np.ndarray]:
        """
        Frame çıkar.

        Args:
            series: DICOM serisi
            frame_index: Frame indeksi
            apply_window_level: Pencere/seviye uygulansın mı?

        Returns:
            Optional[np.ndarray]: Frame verisi
        """
        try:
            # Frame'i al
            frame = series.get_frame(frame_index)
            if not frame:
                return None

            # Kopyala (orijinali değiştirmemek için)
            pixel_data = frame.pixel_array.copy()

            # Pencere/seviye uygula
            if apply_window_level:
                pixel_data = series.window_level.apply(pixel_data)

            return pixel_data

        except Exception as e:
            logger.error(f"Error extracting frame {frame_index}: {str(e)}")
            return None

    def extract_all_frames(
        self, series: DicomSeries, apply_window_level: bool = True
    ) -> List[np.ndarray]:
        """
        Tüm frame'leri çıkar.

        Args:
            series: DICOM serisi
            apply_window_level: Pencere/seviye uygulansın mı?

        Returns:
            List[np.ndarray]: Frame listesi
        """
        frames = []

        for i in range(series.num_frames):
            frame = self.extract_frame(series, i, apply_window_level)
            if frame is not None:
                frames.append(frame)

        return frames

    def extract_time_range(
        self,
        series: DicomSeries,
        start_time: float,
        end_time: float,
        apply_window_level: bool = True,
    ) -> List[np.ndarray]:
        """
        Zaman aralığındaki frame'leri çıkar.

        Args:
            series: DICOM serisi
            start_time: Başlangıç zamanı (saniye)
            end_time: Bitiş zamanı (saniye)
            apply_window_level: Pencere/seviye uygulansın mı?

        Returns:
            List[np.ndarray]: Frame listesi
        """
        frames = []

        # Frame indekslerini hesapla
        start_frame = int(start_time * series.frame_rate)
        end_frame = int(end_time * series.frame_rate)

        # Sınırları kontrol et
        start_frame = max(0, start_frame)
        end_frame = min(series.num_frames - 1, end_frame)

        # Frame'leri çıkar
        for i in range(start_frame, end_frame + 1):
            frame = self.extract_frame(series, i, apply_window_level)
            if frame is not None:
                frames.append(frame)

        return frames

    # Ek işleme metodları

    def apply_gaussian_blur(self, pixel_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Gaussian bulanıklaştırma uygula.

        Args:
            pixel_array: Piksel verisi
            sigma: Standart sapma

        Returns:
            np.ndarray: Bulanıklaştırılmış veri
        """
        try:
            blurred = ndimage.gaussian_filter(pixel_array, sigma=sigma)
            return blurred
        except Exception as e:
            logger.error(f"Error applying Gaussian blur: {str(e)}")
            return pixel_array

    def apply_median_filter(self, pixel_array: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Median filtre uygula (gürültü azaltma).

        Args:
            pixel_array: Piksel verisi
            size: Filtre boyutu

        Returns:
            np.ndarray: Filtrelenmiş veri
        """
        try:
            filtered = ndimage.median_filter(pixel_array, size=size)
            return filtered
        except Exception as e:
            logger.error(f"Error applying median filter: {str(e)}")
            return pixel_array

    def detect_edges(self, pixel_array: np.ndarray, method: str = "canny", **kwargs) -> np.ndarray:
        """
        Kenar tespiti.

        Args:
            pixel_array: Piksel verisi
            method: Tespit yöntemi ("canny", "sobel", "laplacian")
            **kwargs: Yöntem parametreleri

        Returns:
            np.ndarray: Kenar haritası
        """
        try:
            # 8-bit'e normalize et
            normalized = self.normalize_pixels(pixel_array, (0, 255))
            uint8_image = normalized.astype(np.uint8)

            if method == "canny":
                # Canny kenar tespiti
                low_threshold = kwargs.get("low_threshold", 50)
                high_threshold = kwargs.get("high_threshold", 150)
                edges = cv2.Canny(uint8_image, low_threshold, high_threshold)

            elif method == "sobel":
                # Sobel gradyan
                grad_x = cv2.Sobel(uint8_image, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(uint8_image, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(grad_x**2 + grad_y**2)
                edges = self.normalize_pixels(edges, (0, 255)).astype(np.uint8)

            elif method == "laplacian":
                # Laplacian
                edges = cv2.Laplacian(uint8_image, cv2.CV_64F)
                edges = np.absolute(edges)
                edges = self.normalize_pixels(edges, (0, 255)).astype(np.uint8)

            else:
                logger.warning(f"Unknown edge detection method: {method}")
                edges = uint8_image

            return edges

        except Exception as e:
            logger.error(f"Error detecting edges: {str(e)}")
            return pixel_array

    def resize_frame(
        self, pixel_array: np.ndarray, target_size: Tuple[int, int], preserve_aspect: bool = True
    ) -> np.ndarray:
        """
        Frame'i yeniden boyutlandır.

        Args:
            pixel_array: Piksel verisi
            target_size: Hedef boyut (width, height)
            preserve_aspect: En-boy oranını koru

        Returns:
            np.ndarray: Yeniden boyutlandırılmış veri
        """
        try:
            current_height, current_width = pixel_array.shape[:2]
            target_width, target_height = target_size

            if preserve_aspect:
                # En-boy oranını koru
                aspect = current_width / current_height

                if target_width / target_height > aspect:
                    # Yüksekliğe göre ölçekle
                    new_height = target_height
                    new_width = int(target_height * aspect)
                else:
                    # Genişliğe göre ölçekle
                    new_width = target_width
                    new_height = int(target_width / aspect)

                # Yeniden boyutlandır
                resized = cv2.resize(
                    pixel_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )

                # Padding ekle
                pad_top = (target_height - new_height) // 2
                pad_bottom = target_height - new_height - pad_top
                pad_left = (target_width - new_width) // 2
                pad_right = target_width - new_width - pad_left

                padded = np.pad(
                    resized,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )

                return padded

            else:
                # Direkt yeniden boyutlandır
                resized = cv2.resize(pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                return resized

        except Exception as e:
            logger.error(f"Error resizing frame: {str(e)}")
            return pixel_array

    def calculate_pixel_statistics(self, pixel_array: np.ndarray) -> Dict[str, float]:
        """
        Piksel istatistiklerini hesapla.

        Args:
            pixel_array: Piksel verisi

        Returns:
            Dict[str, float]: İstatistikler
        """
        try:
            return {
                "min": float(np.min(pixel_array)),
                "max": float(np.max(pixel_array)),
                "mean": float(np.mean(pixel_array)),
                "std": float(np.std(pixel_array)),
                "median": float(np.median(pixel_array)),
                "percentile_5": float(np.percentile(pixel_array, 5)),
                "percentile_95": float(np.percentile(pixel_array, 95)),
            }
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}

    def create_thumbnail(self, pixel_array: np.ndarray, max_size: int = 256) -> np.ndarray:
        """
        Küçük resim oluştur.

        Args:
            pixel_array: Piksel verisi
            max_size: Maksimum boyut

        Returns:
            np.ndarray: Küçük resim
        """
        height, width = pixel_array.shape[:2]

        # Ölçek faktörü
        scale = min(max_size / width, max_size / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return self.resize_frame(pixel_array, (new_width, new_height), True)
        else:
            return pixel_array.copy()
