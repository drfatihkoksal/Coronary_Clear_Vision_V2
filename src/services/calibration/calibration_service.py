"""
Calibration Service

Kalibrasyon işlemlerinin business logic'ini yöneten servis.
UI'dan bağımsız olarak kalibrasyon hesaplamalarını gerçekleştirir.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime

from src.domain.models.calibration_models import (
    CalibrationResult, CalibrationRequest, CalibrationMethod,
    CatheterMeasurement, CalibrationValidation, CalibrationPoint,
    CatheterSize
)
from src.domain.interfaces.calibration_interfaces import (
    ICalibrationService, ICatheterSegmentation, ICatheterMeasurement
)

logger = logging.getLogger(__name__)


class CalibrationService(ICalibrationService):
    """
    Kalibrasyon servisi implementasyonu.
    
    Bu servis tüm kalibrasyon business logic'ini yönetir.
    UI'dan bağımsız olarak çalışır ve test edilebilir.
    """
    
    # Kalibrasyon limitleri
    MIN_CALIBRATION_FACTOR = 0.01  # 0.01 px/mm (çok büyük piksel)
    MAX_CALIBRATION_FACTOR = 100.0  # 100 px/mm (çok küçük piksel)
    MIN_CATHETER_WIDTH_PIXELS = 15  # Minimum kateter genişliği
    
    def __init__(self,
                 catheter_segmentation: ICatheterSegmentation,
                 catheter_measurement: ICatheterMeasurement):
        """
        Servisi dependency injection ile başlat.
        
        Args:
            catheter_segmentation: Kateter segmentasyon servisi
            catheter_measurement: Kateter ölçüm servisi
        """
        self._segmentation = catheter_segmentation
        self._measurement = catheter_measurement
        
    def calibrate(self, request: CalibrationRequest) -> CalibrationResult:
        """
        Kalibrasyon işlemini gerçekleştirir.
        
        Args:
            request: Kalibrasyon parametreleri
            
        Returns:
            CalibrationResult: Kalibrasyon sonucu
        """
        # Input validation
        if request is None:
            logger.error("Calibration request is None")
            return CalibrationResult(
                success=False,
                error_message="Invalid calibration request"
            )
        
        if request.method is None:
            logger.error("Calibration method not specified")
            return CalibrationResult(
                success=False,
                error_message="Calibration method must be specified"
            )
        
        logger.info(f"Starting calibration with method: {request.method.value}")
        
        try:
            # Validate image if required
            if request.method == CalibrationMethod.CATHETER:
                if request.image is None or request.image.size == 0:
                    return CalibrationResult(
                        success=False,
                        error_message="Image is required for catheter calibration"
                    )
            
            # Execute calibration based on method
            if request.method == CalibrationMethod.MANUAL:
                result = self._calibrate_manual(request)
            elif request.method == CalibrationMethod.CATHETER:
                result = self._calibrate_catheter(request)
            else:
                return CalibrationResult(
                    success=False,
                    error_message=f"Unsupported calibration method: {request.method.value}"
                )
            
            # Validate result before returning
            if result and result.success:
                validation = self.validate_calibration(result)
                if not validation.is_valid:
                    result.success = False
                    result.error_message = "; ".join(validation.issues)
            
            return result
                
        except MemoryError as e:
            logger.error(f"Out of memory during calibration: {e}")
            return CalibrationResult(
                success=False,
                error_message="Insufficient memory for calibration. Try with a smaller image."
            )
        except ValueError as e:
            logger.error(f"Invalid value during calibration: {e}")
            return CalibrationResult(
                success=False,
                error_message=f"Invalid calibration parameters: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Calibration failed: {e}", exc_info=True)
            return CalibrationResult(
                success=False,
                error_message=f"Calibration failed: {str(e)}"
            )
            
    def validate_calibration(self, result: CalibrationResult) -> CalibrationValidation:
        """
        Kalibrasyon sonucunu doğrular.
        
        Args:
            result: Doğrulanacak kalibrasyon sonucu
            
        Returns:
            CalibrationValidation: Doğrulama sonucu
        """
        validation = CalibrationValidation(is_valid=True)
        
        # Temel kontroller
        if not result.success:
            validation.add_issue("Calibration was not successful")
            return validation
            
        # Faktör aralığı kontrolü
        if result.factor <= self.MIN_CALIBRATION_FACTOR:
            validation.add_issue(
                f"Calibration factor too small: {result.factor:.4f} px/mm"
            )
            validation.add_recommendation(
                "Check if the correct units are being used"
            )
            
        if result.factor >= self.MAX_CALIBRATION_FACTOR:
            validation.add_issue(
                f"Calibration factor too large: {result.factor:.4f} px/mm"
            )
            validation.add_recommendation(
                "Verify the catheter size or measurement"
            )
            
        # Güven kontrolü
        if result.confidence < 0.8:
            validation.add_recommendation(
                "Low confidence calibration. Consider recalibrating with better image quality"
            )
            validation.confidence_score = result.confidence
            
        # Kateter kalibrasyonu için ek kontroller
        if result.method == CalibrationMethod.CATHETER and result.measurement:
            measurement = result.measurement
            
            if measurement.width_pixels < self.MIN_CATHETER_WIDTH_PIXELS:
                validation.add_issue(
                    f"Catheter width too small: {measurement.width_pixels:.1f} pixels"
                )
                validation.add_recommendation(
                    "Ensure the catheter is clearly visible and properly segmented"
                )
                
        # Genel güven skoru hesapla
        if validation.is_valid:
            validation.confidence_score = result.confidence
            
        return validation
        
    def calculate_manual_calibration(self,
                                   points: List[CalibrationPoint],
                                   known_distance_mm: float) -> CalibrationResult:
        """
        Manuel nokta seçimi ile kalibrasyon yapar.
        
        Args:
            points: Seçilen noktalar (en az 2)
            known_distance_mm: Bilinen mesafe (mm)
            
        Returns:
            CalibrationResult: Kalibrasyon sonucu
        """
        # Validate inputs
        if not points or len(points) < 2:
            return CalibrationResult(
                success=False,
                error_message="At least 2 points required for manual calibration"
            )
        
        if known_distance_mm <= 0:
            return CalibrationResult(
                success=False,
                error_message=f"Invalid known distance: {known_distance_mm}mm (must be > 0)"
            )
        
        if known_distance_mm > 50:  # Typical max for coronary reference
            logger.warning(f"Large reference distance: {known_distance_mm}mm - verify this is correct")
            
        # İlk iki nokta arasındaki mesafeyi hesapla
        try:
            point1, point2 = points[0], points[1]
            distance_pixels = point1.distance_to(point2)
        except (AttributeError, TypeError) as e:
            logger.error(f"Error calculating distance between points: {e}")
            return CalibrationResult(
                success=False,
                error_message="Invalid calibration points provided"
            )
        
        # Minimum distance check
        MIN_PIXEL_DISTANCE = 10.0  # At least 10 pixels for accuracy
        if distance_pixels < MIN_PIXEL_DISTANCE:
            return CalibrationResult(
                success=False,
                error_message=f"Points are too close together ({distance_pixels:.1f} pixels). "
                            f"Minimum distance is {MIN_PIXEL_DISTANCE} pixels."
            )
            
        # Kalibrasyon faktörünü hesapla
        try:
            factor = distance_pixels / known_distance_mm
        except ZeroDivisionError:
            return CalibrationResult(
                success=False,
                error_message="Division by zero in calibration calculation"
            )
        
        # Güven skorunu hesapla
        confidence = min(point1.confidence, point2.confidence)
        
        # Ek noktalar varsa, tutarlılığı kontrol et
        if len(points) > 2:
            # TODO: Çoklu nokta tutarlılık kontrolü
            pass
            
        logger.info(f"Manual calibration: {factor:.4f} px/mm, confidence: {confidence:.2f}")
        
        return CalibrationResult(
            success=True,
            factor=factor,
            method=CalibrationMethod.MANUAL,
            confidence=confidence,
            metadata={
                'distance_pixels': distance_pixels,
                'distance_mm': known_distance_mm,
                'points': [(p.x, p.y) for p in points]
            }
        )
        
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
        # Validate inputs
        if measurement is None:
            return CalibrationResult(
                success=False,
                error_message="No catheter measurement provided"
            )
        
        if catheter_size is None:
            return CalibrationResult(
                success=False,
                error_message="No catheter size specified"
            )
        
        # Validate measurement values
        if not hasattr(measurement, 'width_pixels') or measurement.width_pixels <= 0:
            return CalibrationResult(
                success=False,
                error_message=f"Invalid catheter width measurement"
            )
        
        # Validate catheter size
        if not hasattr(catheter_size, 'diameter_mm') or catheter_size.diameter_mm <= 0:
            return CalibrationResult(
                success=False,
                error_message=f"Invalid catheter size specification"
            )
        
        # Kalibrasyon faktörünü hesapla
        try:
            factor = measurement.width_pixels / catheter_size.diameter_mm
        except (ZeroDivisionError, TypeError) as e:
            logger.error(f"Error calculating calibration factor: {e}")
            return CalibrationResult(
                success=False,
                error_message="Failed to calculate calibration factor"
            )
        
        # Validate factor is in reasonable range
        if factor < self.MIN_CALIBRATION_FACTOR or factor > self.MAX_CALIBRATION_FACTOR:
            return CalibrationResult(
                success=False,
                error_message=f"Calibration factor {factor:.4f} px/mm is outside valid range "
                            f"({self.MIN_CALIBRATION_FACTOR}-{self.MAX_CALIBRATION_FACTOR})"
            )
        
        logger.info(
            f"Catheter calibration: {factor:.4f} px/mm "
            f"(width: {measurement.width_pixels:.1f}px, "
            f"catheter: {catheter_size})"
        )
        
        return CalibrationResult(
            success=True,
            factor=factor,
            method=CalibrationMethod.CATHETER,
            confidence=measurement.confidence,
            measurement=measurement,
            metadata={
                'catheter_size': str(catheter_size),
                'measurement_method': measurement.method.value
            }
        )
        
    def _calibrate_manual(self, request: CalibrationRequest) -> CalibrationResult:
        """
        Manuel kalibrasyon işlemi.
        
        Args:
            request: Kalibrasyon isteği
            
        Returns:
            CalibrationResult: Sonuç
        """
        if not request.manual_points:
            return CalibrationResult(
                success=False,
                error_message="No manual points provided"
            )
            
        # Varsayılan olarak ilk iki nokta arasını 10mm kabul et
        # TODO: Kullanıcıdan mesafe bilgisi al
        known_distance_mm = request.options.get('known_distance_mm', 10.0)
        
        return self.calculate_manual_calibration(
            request.manual_points,
            known_distance_mm
        )
        
    def _calibrate_catheter(self, request: CalibrationRequest) -> CalibrationResult:
        """
        Kateter bazlı kalibrasyon işlemi.
        
        Args:
            request: Kalibrasyon isteği
            
        Returns:
            CalibrationResult: Sonuç
        """
        if not request.catheter_size:
            return CalibrationResult(
                success=False,
                error_message="No catheter size specified"
            )
            
        try:
            # Kateter segmentasyonu
            logger.info("Performing catheter segmentation...")
            segmentation_mask = self._segmentation.segment_catheter(
                request.image,
                request.roi
            )
            
            # Kontur çıkarımı
            contours = self._extract_contours(segmentation_mask)
            if not contours:
                return CalibrationResult(
                    success=False,
                    error_message="No catheter contour found in segmentation"
                )
                
            # En büyük konturu seç (muhtemelen kateter)
            largest_contour = max(contours, key=lambda c: len(c))
            
            # Genişlik ölçümü
            logger.info("Measuring catheter width...")
            measurement = self._measurement.measure_width(
                segmentation_mask,
                largest_contour
            )
            
            # Minimum genişlik kontrolü
            if measurement.width_pixels < self.MIN_CATHETER_WIDTH_PIXELS:
                return CalibrationResult(
                    success=False,
                    error_message=f"Catheter width too small: {measurement.width_pixels:.1f} pixels",
                    measurement=measurement
                )
                
            # Kalibrasyon hesaplama
            result = self.calculate_catheter_calibration(
                measurement,
                request.catheter_size
            )
            
            # Görselleştirme ekle
            if request.options.get('create_visualization', True):
                result.visualization = self._create_visualization(
                    request.image,
                    segmentation_mask,
                    measurement
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Catheter calibration failed: {e}", exc_info=True)
            return CalibrationResult(
                success=False,
                error_message=f"Catheter calibration error: {str(e)}"
            )
            
    def _extract_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Maskeden konturları çıkarır.
        
        Args:
            mask: Binary maske
            
        Returns:
            List[np.ndarray]: Kontur listesi
        """
        import cv2
        
        # Maske uint8 olmalı
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
            
        # Konturları bul
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
        
    def _create_visualization(self,
                            image: np.ndarray,
                            mask: np.ndarray,
                            measurement: CatheterMeasurement) -> np.ndarray:
        """
        Kalibrasyon görselleştirmesi oluşturur.
        
        Args:
            image: Orijinal görüntü
            mask: Segmentasyon maskesi
            measurement: Ölçüm sonucu
            
        Returns:
            np.ndarray: Görselleştirme
        """
        import cv2
        
        # Görüntüyü renkli yap
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
            
        # Maskeyi overlay olarak ekle
        overlay = np.zeros_like(vis)
        overlay[mask > 0] = [0, 255, 0]  # Yeşil
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Ölçüm çizgisini çiz
        if measurement.measurement_line:
            start, end = measurement.measurement_line
            cv2.line(vis, 
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    (255, 0, 0), 2)  # Mavi
                    
            # Genişlik bilgisini yaz
            mid_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            cv2.putText(vis,
                       f"{measurement.width_pixels:.1f}px",
                       (int(mid_point[0] + 10), int(mid_point[1])),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 0), 1)
                       
        return vis