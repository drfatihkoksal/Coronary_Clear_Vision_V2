"""
Segmentation Service

Segmentasyon işlemlerinin business logic'ini yöneten servis.
UI'dan bağımsız olarak segmentasyon operasyonlarını gerçekleştirir.
"""

import logging
import time
from typing import List, Dict, Any
import numpy as np

from src.domain.models.segmentation_models import (
    SegmentationResult,
    SegmentationRequest,
    SegmentationValidation,
    VesselFeatures,
    UserPoint,
    SegmentationQuality,
    SegmentationMethod,
)
from src.domain.interfaces.segmentation_interfaces import (
    ISegmentationService,
    ISegmentationEngine,
    IFeatureExtractor,
    IPostProcessor,
)

logger = logging.getLogger(__name__)


class SegmentationService(ISegmentationService):
    """
    Segmentasyon servisi implementasyonu.

    Bu servis tüm segmentasyon business logic'ini yönetir.
    UI'dan bağımsız olarak çalışır ve test edilebilir.
    """

    # Kalite eşikleri
    MIN_VESSEL_PIXELS = 100  # Minimum damar piksel sayısı
    MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum güven eşiği
    MAX_HOLE_RATIO = 0.1  # Maksimum delik oranı

    def __init__(
        self,
        segmentation_engine: ISegmentationEngine,
        feature_extractor: IFeatureExtractor,
        post_processor: IPostProcessor,
    ):
        """
        Servisi dependency injection ile başlat.

        Args:
            segmentation_engine: Segmentasyon motoru
            feature_extractor: Özellik çıkarıcı
            post_processor: Post processor
        """
        self._engine = segmentation_engine
        self._feature_extractor = feature_extractor
        self._post_processor = post_processor

    def segment_vessel(self, request: SegmentationRequest) -> SegmentationResult:
        """
        Damar segmentasyonu yapar.

        Args:
            request: Segmentasyon isteği

        Returns:
            SegmentationResult: Segmentasyon sonucu
        """
        logger.info(f"Starting vessel segmentation with method: {request.method.value}")
        start_time = time.time()

        try:
            # Giriş doğrulama
            self._validate_request(request)

            # Segmentasyon
            logger.debug("Performing segmentation...")
            result = self._engine.segment(request)

            if not result.success:
                return result

            # Post-processing
            if request.options.get("post_process", True):
                logger.debug("Applying post-processing...")
                result.mask = self._apply_post_processing(result.mask, request.options)

            # Özellik çıkarımı
            if request.options.get("extract_features", True):
                logger.debug("Extracting vessel features...")
                result.features = self._extract_features(result.mask)

            # Kalite değerlendirmesi
            result.quality = self._evaluate_quality(result)

            # İşlem süresini güncelle
            result.processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Segmentation completed in {result.processing_time_ms:.2f}ms "
                f"with quality: {result.quality.value}"
            )

            return result

        except Exception as e:
            logger.error(f"Segmentation failed: {e}", exc_info=True)
            return SegmentationResult(
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def refine_segmentation(
        self, current_mask: np.ndarray, user_corrections: List[UserPoint], options: Dict[str, Any]
    ) -> SegmentationResult:
        """
        Kullanıcı düzeltmeleriyle segmentasyonu iyileştirir.

        Args:
            current_mask: Mevcut segmentasyon
            user_corrections: Kullanıcı düzeltme noktaları
            options: İyileştirme seçenekleri

        Returns:
            SegmentationResult: İyileştirilmiş sonuç
        """
        logger.info(f"Refining segmentation with {len(user_corrections)} corrections")

        try:
            # Düzeltme tipine göre işlem
            correction_type = options.get("correction_type", "add")

            if correction_type == "add":
                # Ekleme düzeltmesi
                refined_mask = self._add_to_mask(current_mask, user_corrections, options)
            elif correction_type == "remove":
                # Çıkarma düzeltmesi
                refined_mask = self._remove_from_mask(current_mask, user_corrections, options)
            else:
                # Yeniden segmentasyon
                # Düzeltme noktalarını yeni başlangıç noktaları olarak kullan
                request = SegmentationRequest(
                    image=options.get("original_image"),
                    method=SegmentationMethod.SEMI_AUTOMATIC,
                    user_points=user_corrections,
                    previous_mask=current_mask,
                    options=options,
                )
                return self.segment_vessel(request)

            # Post-processing
            if options.get("post_process", True):
                refined_mask = self._apply_post_processing(refined_mask, options)

            # Sonuç oluştur
            result = SegmentationResult(
                success=True,
                mask=refined_mask,
                confidence=0.9,  # Kullanıcı düzeltmesi yüksek güven
                method=SegmentationMethod.SEMI_AUTOMATIC,
            )

            # Özellik çıkarımı
            if options.get("extract_features", True):
                result.features = self._extract_features(refined_mask)

            return result

        except Exception as e:
            logger.error(f"Refinement failed: {e}", exc_info=True)
            return SegmentationResult(success=False, error_message=str(e))

    def validate_segmentation(self, result: SegmentationResult) -> SegmentationValidation:
        """
        Segmentasyon sonucunu doğrular.

        Args:
            result: Doğrulanacak sonuç

        Returns:
            SegmentationValidation: Doğrulama sonucu
        """
        validation = SegmentationValidation()

        if not result.success:
            validation.add_issue("Segmentation was not successful")
            return validation

        if result.mask is None:
            validation.add_issue("No segmentation mask found")
            return validation

        # Piksel sayısı kontrolü
        vessel_pixels = result.vessel_pixels
        validation.set_metric("vessel_pixels", vessel_pixels)

        if vessel_pixels < self.MIN_VESSEL_PIXELS:
            validation.add_issue(f"Too few vessel pixels detected: {vessel_pixels}")
            validation.add_suggestion("Try selecting more points or adjusting threshold")

        # Güven kontrolü
        validation.set_metric("confidence", result.confidence)

        if result.confidence < self.MIN_CONFIDENCE_THRESHOLD:
            validation.add_suggestion("Low confidence segmentation. Consider manual refinement")

        # Delik kontrolü
        hole_ratio = self._calculate_hole_ratio(result.mask)
        validation.set_metric("hole_ratio", hole_ratio)

        if hole_ratio > self.MAX_HOLE_RATIO:
            validation.add_issue(f"Too many holes in segmentation: {hole_ratio:.1%}")
            validation.add_suggestion("Enable hole filling in post-processing")

        # Özellik kontrolü
        if result.has_features:
            features = result.features

            # Merkez hattı kontrolü
            if len(features.centerline) < 10:
                validation.add_issue("Centerline too short")

            # Kıvrımlılık kontrolü
            if features.tortuosity > 2.0:
                validation.add_suggestion(
                    "High vessel tortuosity detected. Verify segmentation accuracy"
                )

        # Kalite skoru
        validation.set_metric("quality_score", result.get_quality_score())

        return validation

    def set_segmentation_engine(self, engine: ISegmentationEngine):
        """Segmentasyon motorunu değiştir"""
        self._engine = engine
        logger.info(f"Segmentation engine changed to: {engine.method_name}")

    def set_feature_extractor(self, extractor: IFeatureExtractor):
        """Özellik çıkarıcıyı değiştir"""
        self._feature_extractor = extractor
        logger.info("Feature extractor updated")

    def set_post_processor(self, processor: IPostProcessor):
        """Post processor'ı değiştir"""
        self._post_processor = processor
        logger.info("Post processor updated")

    # Private helper methods

    def _validate_request(self, request: SegmentationRequest):
        """
        Segmentasyon isteğini doğrular.

        Args:
            request: Doğrulanacak istek

        Raises:
            ValueError: Geçersiz parametre
        """
        if request.image is None or request.image.size == 0:
            raise ValueError("Image cannot be empty")

        # Kullanıcı noktası gerekliliği kontrolü
        if self._engine.requires_user_input and not request.user_points:
            raise ValueError(f"{self._engine.method_name} requires user input points")

    def _apply_post_processing(self, mask: np.ndarray, options: Dict[str, Any]) -> np.ndarray:
        """
        Post-processing uygular.

        Args:
            mask: Ham maske
            options: İşleme seçenekleri

        Returns:
            np.ndarray: İşlenmiş maske
        """
        # Post processor'ı kullan
        processed_mask = self._post_processor.process(mask, options)

        # Ek işlemler
        min_size = options.get("min_vessel_size", 50)
        if min_size > 0:
            processed_mask = self._post_processor.remove_small_components(processed_mask, min_size)

        if options.get("fill_holes", True):
            processed_mask = self._post_processor.fill_holes(processed_mask)

        smoothing_iterations = options.get("smoothing_iterations", 2)
        if smoothing_iterations > 0:
            processed_mask = self._post_processor.smooth_boundaries(
                processed_mask, smoothing_iterations
            )

        return processed_mask

    def _extract_features(self, mask: np.ndarray) -> VesselFeatures:
        """
        Damar özelliklerini çıkarır.

        Args:
            mask: Segmentasyon maskesi

        Returns:
            VesselFeatures: Çıkarılan özellikler
        """
        return self._feature_extractor.extract_features(mask)

    def _evaluate_quality(self, result: SegmentationResult) -> SegmentationQuality:
        """
        Segmentasyon kalitesini değerlendirir.

        Args:
            result: Segmentasyon sonucu

        Returns:
            SegmentationQuality: Kalite seviyesi
        """
        # Güven skoruna göre
        if result.confidence >= 0.9:
            base_quality = SegmentationQuality.EXCELLENT
        elif result.confidence >= 0.7:
            base_quality = SegmentationQuality.GOOD
        elif result.confidence >= 0.5:
            base_quality = SegmentationQuality.FAIR
        else:
            base_quality = SegmentationQuality.POOR

        # Ek kontroller
        if result.vessel_pixels < self.MIN_VESSEL_PIXELS:
            # Çok az piksel varsa kaliteyi düşür
            quality_levels = list(SegmentationQuality)
            current_index = quality_levels.index(base_quality)
            if current_index < len(quality_levels) - 1:
                base_quality = quality_levels[current_index + 1]

        return base_quality

    def _calculate_hole_ratio(self, mask: np.ndarray) -> float:
        """
        Maskedeki delik oranını hesaplar.

        Args:
            mask: Segmentasyon maskesi

        Returns:
            float: Delik oranı (0-1)
        """

        # Doldurulmuş versiyon
        filled = self._post_processor.fill_holes(mask)

        # Delik pikselleri
        holes = filled.astype(int) - mask.astype(int)
        hole_pixels = np.sum(holes > 0)

        # Toplam damar pikseli
        vessel_pixels = np.sum(mask > 0)

        if vessel_pixels == 0:
            return 0.0

        return hole_pixels / vessel_pixels

    def _add_to_mask(
        self, mask: np.ndarray, points: List[UserPoint], options: Dict[str, Any]
    ) -> np.ndarray:
        """
        Maskeye ekleme yapar.

        Args:
            mask: Mevcut maske
            points: Ekleme noktaları
            options: Ekleme seçenekleri

        Returns:
            np.ndarray: Güncellenmiş maske
        """
        import cv2

        # Kopyala
        refined_mask = mask.copy()

        # Fırça boyutu
        brush_size = options.get("brush_size", 10)

        # Her nokta için ekleme yap
        for point in points:
            cv2.circle(refined_mask, (int(point.x), int(point.y)), brush_size, 255, -1)

        return refined_mask

    def _remove_from_mask(
        self, mask: np.ndarray, points: List[UserPoint], options: Dict[str, Any]
    ) -> np.ndarray:
        """
        Maskeden çıkarma yapar.

        Args:
            mask: Mevcut maske
            points: Çıkarma noktaları
            options: Çıkarma seçenekleri

        Returns:
            np.ndarray: Güncellenmiş maske
        """
        import cv2

        # Kopyala
        refined_mask = mask.copy()

        # Fırça boyutu
        brush_size = options.get("brush_size", 10)

        # Her nokta için çıkarma yap
        for point in points:
            cv2.circle(refined_mask, (int(point.x), int(point.y)), brush_size, 0, -1)

        return refined_mask
