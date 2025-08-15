"""
Catheter Measurement Strategies

Kateter genişliği ölçümü için farklı strateji implementasyonları.
Strategy pattern kullanarak farklı ölçüm yöntemlerini destekler.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging

from src.domain.models.calibration_models import CatheterMeasurement, MeasurementMethod
from src.domain.interfaces.calibration_interfaces import ICatheterMeasurement

logger = logging.getLogger(__name__)


class DistanceTransformMeasurement(ICatheterMeasurement):
    """
    Distance transform kullanarak kateter genişliği ölçümü.

    En güvenilir yöntemlerden biri. Kateter eksenine dik
    en geniş noktayı bulur.
    """

    def measure_width(
        self, segmentation_mask: np.ndarray, contour: np.ndarray
    ) -> CatheterMeasurement:
        """
        Distance transform ile genişlik ölçer.

        Args:
            segmentation_mask: Kateter segmentasyon maskesi
            contour: Kateter konturu

        Returns:
            CatheterMeasurement: Ölçüm sonucu
        """
        logger.debug("Measuring catheter width using distance transform")

        # Distance transform hesapla
        dist_transform = cv2.distanceTransform(
            segmentation_mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        # Maksimum değeri bul (merkez nokta)
        max_loc = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
        max_dist = dist_transform[max_loc]

        # Genişlik = 2 * maksimum mesafe
        width_pixels = 2 * max_dist

        # Ölçüm çizgisini hesapla
        measurement_line = self._calculate_measurement_line(segmentation_mask, max_loc, max_dist)

        # Güven skorunu hesapla
        confidence = self._calculate_confidence(dist_transform, max_dist)

        logger.info(f"Distance transform width: {width_pixels:.1f}px, confidence: {confidence:.2f}")

        return CatheterMeasurement(
            width_pixels=width_pixels,
            method=MeasurementMethod.DISTANCE_TRANSFORM,
            confidence=confidence,
            contour_points=contour.reshape(-1, 2).tolist(),
            measurement_line=measurement_line,
        )

    def _calculate_measurement_line(
        self, mask: np.ndarray, center: Tuple[int, int], radius: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Ölçüm çizgisini hesaplar.

        Args:
            mask: Segmentasyon maskesi
            center: Merkez nokta
            radius: Yarıçap

        Returns:
            Tuple: (başlangıç, bitiş) noktaları
        """
        # Gradyan hesapla (kenar yönünü bulmak için)
        gy, gx = np.gradient(mask.astype(float))

        # Merkez noktadaki gradyan
        cy, cx = center
        grad_y = gy[cy, cx]
        grad_x = gx[cy, cx]

        # Gradyana dik yön (genişlik yönü)
        if abs(grad_x) > 0.001 or abs(grad_y) > 0.001:
            norm = np.sqrt(grad_x**2 + grad_y**2)
            perp_x = -grad_y / norm
            perp_y = grad_x / norm
        else:
            # Gradyan yoksa yatay çizgi
            perp_x, perp_y = 1, 0

        # Başlangıç ve bitiş noktaları
        start = (cx - radius * perp_x, cy - radius * perp_y)
        end = (cx + radius * perp_x, cy + radius * perp_y)

        return (start, end)

    def _calculate_confidence(self, dist_transform: np.ndarray, max_dist: float) -> float:
        """
        Ölçüm güvenilirliğini hesaplar.

        Args:
            dist_transform: Distance transform
            max_dist: Maksimum mesafe

        Returns:
            float: Güven skoru (0-1)
        """
        # Maksimum değere yakın nokta sayısı
        threshold = max_dist * 0.9
        high_value_pixels = np.sum(dist_transform > threshold)

        # Toplam kateter piksel sayısı
        total_pixels = np.sum(dist_transform > 0)

        if total_pixels == 0:
            return 0.0

        # Yüksek değerli piksellerin oranı güveni belirler
        ratio = high_value_pixels / total_pixels

        # Çok dar veya çok geniş oranlar düşük güven
        if ratio < 0.01:  # Çok az merkez piksel
            confidence = 0.5
        elif ratio > 0.3:  # Çok fazla merkez piksel (blob)
            confidence = 0.7
        else:
            confidence = 0.9

        return confidence


class MinAreaRectMeasurement(ICatheterMeasurement):
    """
    Minimum alan dikdörtgeni kullanarak genişlik ölçümü.

    Düz kateterler için etkili, eğri kateterler için
    daha az güvenilir.
    """

    def measure_width(
        self, segmentation_mask: np.ndarray, contour: np.ndarray
    ) -> CatheterMeasurement:
        """
        Minimum alan dikdörtgeni ile genişlik ölçer.

        Args:
            segmentation_mask: Kateter segmentasyon maskesi
            contour: Kateter konturu

        Returns:
            CatheterMeasurement: Ölçüm sonucu
        """
        logger.debug("Measuring catheter width using minimum area rectangle")

        # Minimum alan dikdörtgenini bul
        rect = cv2.minAreaRect(contour)
        center, (width, height), angle = rect

        # Kısa kenar genişliktir
        width_pixels = min(width, height)
        length_pixels = max(width, height)

        # Uzunluk/genişlik oranı güveni etkiler
        aspect_ratio = length_pixels / width_pixels if width_pixels > 0 else 0

        # Ölçüm çizgisi (dikdörtgenin kısa kenarı ortasından)
        measurement_line = self._calculate_rect_measurement_line(rect)

        # Güven skoru
        confidence = self._calculate_confidence(aspect_ratio, segmentation_mask, rect)

        logger.info(f"Min area rect width: {width_pixels:.1f}px, confidence: {confidence:.2f}")

        return CatheterMeasurement(
            width_pixels=width_pixels,
            method=MeasurementMethod.MIN_AREA_RECT,
            confidence=confidence,
            contour_points=contour.reshape(-1, 2).tolist(),
            measurement_line=measurement_line,
        )

    def _calculate_rect_measurement_line(
        self, rect: Tuple
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Dikdörtgen için ölçüm çizgisini hesaplar.

        Args:
            rect: OpenCV minAreaRect sonucu

        Returns:
            Tuple: (başlangıç, bitiş) noktaları
        """
        center, (width, height), angle = rect

        # Kısa kenar genişliktir
        if width < height:
            # Genişlik yönünde çizgi
            angle_rad = np.radians(angle)
        else:
            # 90 derece döndür
            angle_rad = np.radians(angle + 90)

        # Çizgi yönü
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Genişlik/2 uzaklıkta başlangıç ve bitiş
        half_width = min(width, height) / 2

        start = (center[0] - half_width * dx, center[1] - half_width * dy)
        end = (center[0] + half_width * dx, center[1] + half_width * dy)

        return (start, end)

    def _calculate_confidence(self, aspect_ratio: float, mask: np.ndarray, rect: Tuple) -> float:
        """
        Min area rect ölçümü için güven hesaplar.

        Args:
            aspect_ratio: Uzunluk/genişlik oranı
            mask: Segmentasyon maskesi
            rect: Dikdörtgen

        Returns:
            float: Güven skoru (0-1)
        """
        # Aspect ratio güveni
        if aspect_ratio < 3:  # Çok kısa/kalın
            ar_confidence = 0.5
        elif aspect_ratio > 20:  # Çok uzun/ince
            ar_confidence = 0.7
        else:
            ar_confidence = 0.9

        # Dikdörtgen doluluk oranı
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Dikdörtgen maskesi
        rect_mask = np.zeros_like(mask)
        cv2.fillPoly(rect_mask, [box], 1)

        # Doluluk oranı
        intersection = np.logical_and(mask > 0, rect_mask > 0)
        fill_ratio = np.sum(intersection) / np.sum(rect_mask) if np.sum(rect_mask) > 0 else 0

        # Toplam güven
        confidence = ar_confidence * fill_ratio

        return confidence


class SkeletonBasedMeasurement(ICatheterMeasurement):
    """
    İskelet (skeleton) analizi kullanarak genişlik ölçümü.

    Karmaşık şekilli kateterler için daha doğru sonuç verir.
    """

    def measure_width(
        self, segmentation_mask: np.ndarray, contour: np.ndarray
    ) -> CatheterMeasurement:
        """
        İskelet analizi ile genişlik ölçer.

        Args:
            segmentation_mask: Kateter segmentasyon maskesi
            contour: Kateter konturu

        Returns:
            CatheterMeasurement: Ölçüm sonucu
        """
        logger.debug("Measuring catheter width using skeleton analysis")

        # İskeleti çıkar
        skeleton = self._extract_skeleton(segmentation_mask)

        # İskelet noktalarını bul
        skeleton_points = np.column_stack(np.where(skeleton > 0))

        if len(skeleton_points) < 5:
            # Yetersiz iskelet noktası, fallback
            return self._fallback_measurement(segmentation_mask, contour)

        # Her iskelet noktası için genişlik ölç
        widths = []
        measurement_lines = []

        for point in skeleton_points[::5]:  # Her 5 noktada bir ölç
            width, line = self._measure_width_at_point(segmentation_mask, point, skeleton)
            if width > 0:
                widths.append(width)
                measurement_lines.append(line)

        if not widths:
            return self._fallback_measurement(segmentation_mask, contour)

        # Medyan genişlik (outlier'lara karşı dayanıklı)
        width_pixels = np.median(widths)

        # En güvenilir ölçüm çizgisi (medyana en yakın)
        best_idx = np.argmin(np.abs(np.array(widths) - width_pixels))
        measurement_line = measurement_lines[best_idx]

        # Güven skoru
        confidence = self._calculate_confidence(widths)

        logger.info(f"Skeleton-based width: {width_pixels:.1f}px, confidence: {confidence:.2f}")

        return CatheterMeasurement(
            width_pixels=width_pixels,
            method=MeasurementMethod.SKELETON_BASED,
            confidence=confidence,
            contour_points=contour.reshape(-1, 2).tolist(),
            measurement_line=measurement_line,
        )

    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Morfoljik iskelet çıkarır.

        Args:
            mask: Binary maske

        Returns:
            np.ndarray: İskelet görüntüsü
        """
        from skimage.morphology import skeletonize

        # Skeletonize
        skeleton = skeletonize(mask > 0)

        return skeleton.astype(np.uint8) * 255

    def _measure_width_at_point(
        self, mask: np.ndarray, point: np.ndarray, skeleton: np.ndarray
    ) -> Tuple[float, Optional[Tuple]]:
        """
        Belirli bir iskelet noktasında genişlik ölçer.

        Args:
            mask: Segmentasyon maskesi
            point: İskelet noktası
            skeleton: İskelet görüntüsü

        Returns:
            Tuple: (genişlik, ölçüm çizgisi)
        """
        y, x = point

        # İskelet yönünü bul
        direction = self._get_skeleton_direction(skeleton, point)

        # Yöne dik doğrultuda tara
        perp_dir = np.array([-direction[1], direction[0]])

        # Her iki yönde kenarları bul
        edge1 = self._find_edge(mask, (x, y), perp_dir)
        edge2 = self._find_edge(mask, (x, y), -perp_dir)

        if edge1 is None or edge2 is None:
            return 0, None

        # Genişlik
        width = np.linalg.norm(edge1 - edge2)

        # Ölçüm çizgisi
        measurement_line = (tuple(edge1), tuple(edge2))

        return width, measurement_line

    def _get_skeleton_direction(self, skeleton: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        İskelet noktasındaki yerel yönü hesaplar.

        Args:
            skeleton: İskelet görüntüsü
            point: İskelet noktası

        Returns:
            np.ndarray: Birim yön vektörü
        """
        y, x = point

        # Yerel pencere
        window_size = 5
        y1 = max(0, y - window_size)
        y2 = min(skeleton.shape[0], y + window_size + 1)
        x1 = max(0, x - window_size)
        x2 = min(skeleton.shape[1], x + window_size + 1)

        # Penceredeki iskelet noktaları
        window = skeleton[y1:y2, x1:x2]
        local_points = np.column_stack(np.where(window > 0))

        if len(local_points) < 2:
            return np.array([1, 0])  # Varsayılan yatay

        # PCA ile ana yönü bul
        local_points = local_points - np.mean(local_points, axis=0)
        cov = np.cov(local_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # En büyük özdeğere ait özvektör
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]

        # Y,X'ten X,Y'ye dönüştür ve normalize et
        direction = np.array([main_direction[1], main_direction[0]])
        direction = direction / np.linalg.norm(direction)

        return direction

    def _find_edge(
        self, mask: np.ndarray, start: Tuple[float, float], direction: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Belirli yönde kenar bulur.

        Args:
            mask: Segmentasyon maskesi
            start: Başlangıç noktası
            direction: Arama yönü

        Returns:
            Optional[np.ndarray]: Kenar noktası
        """
        max_dist = 100  # Maksimum arama mesafesi

        for dist in range(1, max_dist):
            point = start + direction * dist
            x, y = int(point[0]), int(point[1])

            # Sınır kontrolü
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # Maskenin dışına çıktık mı?
            if mask[y, x] == 0:
                # Bir önceki nokta kenar
                return start + direction * (dist - 1)

        return None

    def _calculate_confidence(self, widths: List[float]) -> float:
        """
        Genişlik ölçümlerinin tutarlılığına göre güven hesaplar.

        Args:
            widths: Genişlik ölçümleri

        Returns:
            float: Güven skoru (0-1)
        """
        if len(widths) < 3:
            return 0.5

        # Varyasyon katsayısı (CV)
        mean_width = np.mean(widths)
        std_width = np.std(widths)

        if mean_width == 0:
            return 0.0

        cv = std_width / mean_width

        # CV düşükse güven yüksek
        if cv < 0.1:  # %10'dan az varyasyon
            confidence = 0.95
        elif cv < 0.2:  # %20'den az varyasyon
            confidence = 0.85
        elif cv < 0.3:  # %30'dan az varyasyon
            confidence = 0.7
        else:
            confidence = 0.5

        return confidence

    def _fallback_measurement(self, mask: np.ndarray, contour: np.ndarray) -> CatheterMeasurement:
        """
        İskelet analizi başarısız olursa fallback ölçüm.

        Args:
            mask: Segmentasyon maskesi
            contour: Kontur

        Returns:
            CatheterMeasurement: Basit ölçüm sonucu
        """
        # Basit alan/uzunluk yaklaşımı
        area = np.sum(mask > 0)
        perimeter = cv2.arcLength(contour, True)

        # Yaklaşık genişlik
        width_pixels = 2 * area / perimeter if perimeter > 0 else 0

        return CatheterMeasurement(
            width_pixels=width_pixels,
            method=MeasurementMethod.SKELETON_BASED,
            confidence=0.3,  # Düşük güven
            contour_points=contour.reshape(-1, 2).tolist(),
        )
