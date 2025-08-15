"""
QCA Tracked Points Integration

Bu modül QCA analizi ile tracked points sistemi arasındaki entegrasyonu sağlar.
Tracking servisinden alınan noktaları QCA analizinde kullanmak için gerekli
helper fonksiyonları içerir.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from ..domain.models.tracking_models import TrackedPoint, TrackingSession
from .qca_analysis import QCAAnalysis
from .minimum_cost_path import MinimumCostPathGenerator

logger = logging.getLogger(__name__)


class QCATrackedIntegration:
    """
    QCA analizi ile tracked points entegrasyonu
    """

    def __init__(self):
        """Initialize integration"""
        self.qca_analyzer = QCAAnalysis()
        self.cost_path_generator = MinimumCostPathGenerator()

    def analyze_with_tracked_points(
        self,
        segmentation_result: Dict,
        tracked_points: List[TrackedPoint],
        calibration_factor: Optional[float] = None,
        original_image: Optional[np.ndarray] = None,
        smooth_factor: float = 1.0,
        use_vessel_guidance: bool = True,
    ) -> Dict:
        """
        Tracked points kullanarak QCA analizi yap

        Args:
            segmentation_result: AngioPy segmentasyon sonucu
            tracked_points: İzlenen noktalar listesi
            calibration_factor: Kalibrasyon faktörü (mm/pixel)
            original_image: Orijinal görüntü
            smooth_factor: Smoothing faktörü
            use_vessel_guidance: Vessel mask'ı kullanarak guidance

        Returns:
            QCA analiz sonuçları
        """
        try:
            logger.info(f"Starting QCA analysis with {len(tracked_points)} tracked points")

            # Kalibrasyon ayarla
            if calibration_factor is not None:
                self.qca_analyzer.calibration_factor = calibration_factor
                logger.info(f"Set calibration factor: {calibration_factor:.5f} mm/pixel")

            # Tracked points'leri (x, y) koordinatlarına dönüştür
            point_coordinates = []
            for point in tracked_points:
                x, y = point.current_position.x, point.current_position.y
                point_coordinates.append((x, y))

            logger.info(f"Extracted coordinates from {len(point_coordinates)} tracked points")

            # QCA analizini tracked points ile yap
            result = self.qca_analyzer.analyze_from_angiopy(
                segmentation_result=segmentation_result,
                original_image=original_image,
                tracked_points=point_coordinates,
                use_tracked_centerline=True,
            )

            # Sonuçlara tracking bilgilerini ekle
            if result.get("success"):
                result["tracking_info"] = {
                    "num_tracked_points": len(tracked_points),
                    "tracking_confidence": self._calculate_tracking_confidence(tracked_points),
                    "point_ids": [point.id for point in tracked_points],
                    "smooth_factor": smooth_factor,
                    "vessel_guidance_used": use_vessel_guidance,
                }

                logger.info("QCA analysis with tracked points completed successfully")
            else:
                logger.error(f"QCA analysis failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"QCA tracked integration failed: {e}")
            return {"success": False, "error": str(e)}

    def analyze_with_tracking_session(
        self,
        segmentation_result: Dict,
        tracking_session: TrackingSession,
        frame_number: int,
        calibration_factor: Optional[float] = None,
        original_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Tracking session'dan belirli frame için QCA analizi yap

        Args:
            segmentation_result: AngioPy segmentasyon sonucu
            tracking_session: Tracking oturumu
            frame_number: Analiz edilecek frame numarası
            calibration_factor: Kalibrasyon faktörü
            original_image: Orijinal görüntü

        Returns:
            QCA analiz sonuçları
        """
        try:
            # Session'dan tracked points'leri al
            tracked_points = []
            for point in tracking_session.get_all_points():
                # Belirtilen frame'deki pozisyonu bul
                frame_position = self._get_point_position_at_frame(point, frame_number)
                if frame_position:
                    tracked_points.append(point)

            if not tracked_points:
                return {
                    "success": False,
                    "error": f"No tracked points found for frame {frame_number}",
                }

            logger.info(f"Found {len(tracked_points)} tracked points for frame {frame_number}")

            # QCA analizini yap
            return self.analyze_with_tracked_points(
                segmentation_result=segmentation_result,
                tracked_points=tracked_points,
                calibration_factor=calibration_factor,
                original_image=original_image,
            )

        except Exception as e:
            logger.error(f"Session-based QCA analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def validate_tracked_points_for_qca(self, tracked_points: List[TrackedPoint]) -> Dict[str, Any]:
        """
        Tracked points'lerin QCA analizi için uygunluğunu kontrol et

        Args:
            tracked_points: İzlenen noktalar

        Returns:
            Validation sonuçları
        """
        validation = {"valid": True, "warnings": [], "errors": [], "recommendations": []}

        # Minimum nokta sayısı kontrolü
        if len(tracked_points) < 2:
            validation["valid"] = False
            validation["errors"].append("At least 2 tracked points required for QCA analysis")
            return validation

        # Nokta güven skorları kontrolü
        low_confidence_points = []
        for point in tracked_points:
            if point.confidence < 0.7:
                low_confidence_points.append(point.id)

        if low_confidence_points:
            validation["warnings"].append(
                f"Low confidence points detected: {low_confidence_points}"
            )

        # Nokta dağılımı kontrolü
        coordinates = [(p.current_position.x, p.current_position.y) for p in tracked_points]
        total_distance = self._calculate_path_length(coordinates)

        if total_distance < 50:  # pixels
            validation["warnings"].append(
                f"Short vessel segment detected ({total_distance:.1f} pixels). "
                "Consider adding more points for better analysis."
            )

        # Nokta sıralama kontrolü
        if not self._check_point_ordering(coordinates):
            validation["recommendations"].append(
                "Points may not be in optimal order. Consider reordering for better results."
            )

        return validation

    def create_qca_centerline_visualization(
        self, mask: np.ndarray, tracked_points: List[TrackedPoint], show_cost_map: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        QCA centerline görselleştirmesi oluştur

        Args:
            mask: Vessel mask
            tracked_points: Tracked points
            show_cost_map: Cost map'i göster

        Returns:
            Görselleştirme verileri
        """
        try:
            # Koordinatları çıkar
            coordinates = [(p.current_position.x, p.current_position.y) for p in tracked_points]

            # Centerline oluştur
            centerline = self.cost_path_generator.generate_centerline_from_tracked_points(
                mask=mask, tracked_points=coordinates, smooth_factor=1.0, use_vessel_guidance=True
            )

            # Görselleştirme oluştur
            visualization = {}

            # Ana görselleştirme
            vis_image = self._create_centerline_overlay(mask, centerline, coordinates)
            visualization["centerline_overlay"] = vis_image

            # Cost map görselleştirmesi
            if show_cost_map:
                cost_map_vis = self.cost_path_generator.visualize_cost_map()
                if cost_map_vis is not None:
                    visualization["cost_map"] = cost_map_vis

            # Kalite metrikleri
            quality_metrics = self.cost_path_generator.get_path_quality_metrics(centerline)
            visualization["quality_metrics"] = quality_metrics

            return visualization

        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return {}

    def _calculate_tracking_confidence(self, tracked_points: List[TrackedPoint]) -> float:
        """Tracked points'lerin ortalama güven skorunu hesapla"""
        if not tracked_points:
            return 0.0

        confidences = [point.confidence for point in tracked_points]
        return sum(confidences) / len(confidences)

    def _get_point_position_at_frame(
        self, point: TrackedPoint, frame_number: int
    ) -> Optional[Tuple[float, float]]:
        """Belirli frame'deki nokta pozisyonunu al"""
        for frame_data in point.history:
            if frame_data.frame_number == frame_number:
                pos = frame_data.position
                return (pos.x, pos.y)
        return None

    def _calculate_path_length(self, coordinates: List[Tuple[float, float]]) -> float:
        """Nokta dizisi boyunca toplam mesafe hesapla"""
        if len(coordinates) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(coordinates)):
            p1 = np.array(coordinates[i - 1])
            p2 = np.array(coordinates[i])
            total_length += np.linalg.norm(p2 - p1)

        return total_length

    def _check_point_ordering(self, coordinates: List[Tuple[float, float]]) -> bool:
        """Noktaların mantıklı sıralamada olup olmadığını kontrol et"""
        if len(coordinates) < 3:
            return True

        # Basit kontrol: ardışık noktalar arasındaki mesafe çok büyük mü?
        distances = []
        for i in range(1, len(coordinates)):
            p1 = np.array(coordinates[i - 1])
            p2 = np.array(coordinates[i])
            distances.append(np.linalg.norm(p2 - p1))

        mean_distance = np.mean(distances)
        max_distance = np.max(distances)

        # Eğer maksimum mesafe ortalamadan çok büyükse sıralama problemi olabilir
        return max_distance < mean_distance * 3.0

    def _create_centerline_overlay(
        self, mask: np.ndarray, centerline: np.ndarray, tracked_points: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Centerline ve tracked points overlay'i oluştur"""
        import cv2

        # Mask'i 3 kanallı görüntüye dönüştür
        if len(mask.shape) == 2:
            vis_image = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            vis_image = mask.copy()

        # Centerline çiz (mavi)
        if len(centerline) > 1:
            points = centerline.astype(np.int32)
            for i in range(len(points) - 1):
                cv2.line(
                    vis_image,
                    (points[i][1], points[i][0]),  # (x, y) format
                    (points[i + 1][1], points[i + 1][0]),
                    (255, 0, 0),
                    2,
                )  # Mavi

        # Tracked points çiz (yeşil)
        for i, (x, y) in enumerate(tracked_points):
            cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Yeşil
            cv2.putText(
                vis_image,
                str(i + 1),
                (int(x + 8), int(y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return vis_image


# Convenience functions
def analyze_qca_with_tracked_points(
    segmentation_result: Dict,
    tracked_points: List[TrackedPoint],
    calibration_factor: Optional[float] = None,
    original_image: Optional[np.ndarray] = None,
) -> Dict:
    """
    Convenience function for QCA analysis with tracked points

    Args:
        segmentation_result: AngioPy segmentation result
        tracked_points: List of tracked points
        calibration_factor: Calibration factor (mm/pixel)
        original_image: Original image

    Returns:
        QCA analysis results
    """
    integration = QCATrackedIntegration()
    return integration.analyze_with_tracked_points(
        segmentation_result=segmentation_result,
        tracked_points=tracked_points,
        calibration_factor=calibration_factor,
        original_image=original_image,
    )


def validate_tracking_for_qca(tracked_points: List[TrackedPoint]) -> Dict[str, Any]:
    """
    Convenience function for validating tracked points for QCA

    Args:
        tracked_points: List of tracked points

    Returns:
        Validation results
    """
    integration = QCATrackedIntegration()
    return integration.validate_tracked_points_for_qca(tracked_points)
