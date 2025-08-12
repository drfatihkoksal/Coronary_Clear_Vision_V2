"""
Calibration Results Display Component

Kalibrasyon sonuçlarını gösteren UI bileşeni.
Single Responsibility: Sadece sonuçları görselleştirir.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QGroupBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
import numpy as np
from typing import Optional

from src.domain.models.calibration_models import (
    CalibrationResult, CalibrationValidation, CalibrationMethod
)


class CalibrationResultsDisplay(QWidget):
    """
    Kalibrasyon sonuçlarını gösteren widget.
    
    Kalibrasyon faktörü, güven skoru ve görselleştirmeyi sunar.
    Business logic içermez, sadece veri sunumu yapar.
    
    Signals:
        recalibrate_requested: Yeniden kalibrasyon istendi
        details_requested: Detaylı bilgi istendi
    """
    
    # Signals
    recalibrate_requested = pyqtSignal()
    details_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Results display widget'ını başlatır.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_result: Optional[CalibrationResult] = None
        self._validation: Optional[CalibrationValidation] = None
        self._init_ui()
        
    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Ana sonuç grubu
        self.results_group = QGroupBox("Calibration Results")
        results_layout = QVBoxLayout()
        
        # Durum göstergesi
        self.status_widget = self._create_status_widget()
        results_layout.addWidget(self.status_widget)
        
        # Metrikler
        metrics_layout = self._create_metrics_layout()
        results_layout.addLayout(metrics_layout)
        
        # Görselleştirme
        self.visualization_label = QLabel()
        self.visualization_label.setObjectName("calibrationVisualization")
        self.visualization_label.setMinimumHeight(200)
        self.visualization_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_label.setStyleSheet("""
            QLabel#calibrationVisualization {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
        """)
        results_layout.addWidget(self.visualization_label)
        
        # Doğrulama mesajları
        self.validation_widget = self._create_validation_widget()
        results_layout.addWidget(self.validation_widget)
        
        self.results_group.setLayout(results_layout)
        layout.addWidget(self.results_group)
        
        self.setLayout(layout)
        
        # Başlangıçta gizle
        self.hide()
        
    def _create_status_widget(self) -> QWidget:
        """
        Durum gösterge widget'ı oluşturur.
        
        Returns:
            QWidget: Durum widget'ı
        """
        widget = QFrame()
        widget.setObjectName("calibrationStatus")
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Durum ikonu
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(24, 24)
        layout.addWidget(self.status_icon)
        
        # Durum metni
        self.status_label = QLabel("No calibration")
        self.status_label.setObjectName("calibrationStatusText")
        self.status_label.setStyleSheet("""
            QLabel#calibrationStatusText {
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Yeniden kalibrasyon butonu
        self.recalibrate_label = QLabel('<a href="#">Recalibrate</a>')
        self.recalibrate_label.setObjectName("recalibrateLink")
        self.recalibrate_label.linkActivated.connect(
            lambda: self.recalibrate_requested.emit()
        )
        layout.addWidget(self.recalibrate_label)
        
        widget.setLayout(layout)
        self._update_status_style(False)
        
        return widget
        
    def _create_metrics_layout(self) -> QVBoxLayout:
        """
        Kalibrasyon metriklerini gösteren layout oluşturur.
        
        Returns:
            QVBoxLayout: Metrik layout'u
        """
        layout = QVBoxLayout()
        
        # Kalibrasyon faktörü
        factor_layout = QHBoxLayout()
        factor_layout.addWidget(self._create_metric_label("Calibration Factor:"))
        self.factor_value_label = self._create_value_label("-- px/mm")
        factor_layout.addWidget(self.factor_value_label)
        factor_layout.addStretch()
        layout.addLayout(factor_layout)
        
        # Piksel boyutu
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(self._create_metric_label("Pixel Size:"))
        self.pixel_size_label = self._create_value_label("-- mm/px")
        pixel_layout.addWidget(self.pixel_size_label)
        pixel_layout.addStretch()
        layout.addLayout(pixel_layout)
        
        # Güven skoru
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(self._create_metric_label("Confidence:"))
        self.confidence_label = self._create_value_label("--%")
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addStretch()
        layout.addLayout(confidence_layout)
        
        # Yöntem
        method_layout = QHBoxLayout()
        method_layout.addWidget(self._create_metric_label("Method:"))
        self.method_label = self._create_value_label("--")
        method_layout.addWidget(self.method_label)
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        return layout
        
    def _create_validation_widget(self) -> QWidget:
        """
        Doğrulama mesajları widget'ı oluşturur.
        
        Returns:
            QWidget: Doğrulama widget'ı
        """
        widget = QFrame()
        widget.setObjectName("calibrationValidation")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Sorunlar
        self.issues_label = QLabel()
        self.issues_label.setObjectName("calibrationIssues")
        self.issues_label.setWordWrap(True)
        self.issues_label.hide()
        layout.addWidget(self.issues_label)
        
        # Öneriler
        self.recommendations_label = QLabel()
        self.recommendations_label.setObjectName("calibrationRecommendations")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.hide()
        layout.addWidget(self.recommendations_label)
        
        widget.setLayout(layout)
        widget.hide()
        
        return widget
        
    def _create_metric_label(self, text: str) -> QLabel:
        """Metrik etiketi oluşturur."""
        label = QLabel(text)
        label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                min-width: 120px;
            }
        """)
        return label
        
    def _create_value_label(self, text: str) -> QLabel:
        """Değer etiketi oluşturur."""
        label = QLabel(text)
        label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #333;
                min-width: 100px;
            }
        """)
        return label
        
    def display_result(self, result: CalibrationResult):
        """
        Kalibrasyon sonucunu gösterir.
        
        Args:
            result: Kalibrasyon sonucu
        """
        self._current_result = result
        self.show()
        
        # Durum güncelle
        self._update_status(result)
        
        # Metrikleri güncelle
        self._update_metrics(result)
        
        # Görselleştirmeyi güncelle
        self._update_visualization(result)
        
        # Doğrulama widget'ını gizle (validation yoksa)
        self.validation_widget.hide()
        
    def display_validation(self, validation: CalibrationValidation):
        """
        Kalibrasyon doğrulama sonucunu gösterir.
        
        Args:
            validation: Doğrulama sonucu
        """
        self._validation = validation
        
        if not validation.issues and not validation.recommendations:
            self.validation_widget.hide()
            return
            
        self.validation_widget.show()
        
        # Sorunları göster
        if validation.issues:
            issues_text = "<b>Issues:</b><ul>"
            for issue in validation.issues:
                issues_text += f"<li>{issue}</li>"
            issues_text += "</ul>"
            
            self.issues_label.setText(issues_text)
            self.issues_label.setStyleSheet("""
                QLabel#calibrationIssues {
                    color: #D32F2F;
                    font-size: 11px;
                    padding: 5px;
                    background-color: #FFEBEE;
                    border-radius: 3px;
                }
            """)
            self.issues_label.show()
        else:
            self.issues_label.hide()
            
        # Önerileri göster
        if validation.recommendations:
            rec_text = "<b>Recommendations:</b><ul>"
            for rec in validation.recommendations:
                rec_text += f"<li>{rec}</li>"
            rec_text += "</ul>"
            
            self.recommendations_label.setText(rec_text)
            self.recommendations_label.setStyleSheet("""
                QLabel#calibrationRecommendations {
                    color: #F57C00;
                    font-size: 11px;
                    padding: 5px;
                    background-color: #FFF3E0;
                    border-radius: 3px;
                }
            """)
            self.recommendations_label.show()
        else:
            self.recommendations_label.hide()
            
    def _update_status(self, result: CalibrationResult):
        """Durum göstergesini günceller."""
        if result.success:
            self.status_label.setText("Calibration Successful")
            self._update_status_style(True)
            self._set_status_icon("✓", QColor(76, 175, 80))
        else:
            self.status_label.setText("Calibration Failed")
            self._update_status_style(False)
            self._set_status_icon("✗", QColor(244, 67, 54))
            
    def _update_status_style(self, success: bool):
        """Durum widget stilini günceller."""
        if success:
            self.status_widget.setStyleSheet("""
                QFrame#calibrationStatus {
                    background-color: #E8F5E9;
                    border: 1px solid #4CAF50;
                    border-radius: 4px;
                }
            """)
        else:
            self.status_widget.setStyleSheet("""
                QFrame#calibrationStatus {
                    background-color: #FFEBEE;
                    border: 1px solid #F44336;
                    border-radius: 4px;
                }
            """)
            
    def _set_status_icon(self, text: str, color: QColor):
        """Durum ikonunu ayarlar."""
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Daire arka plan
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 20, 20)
        
        # Metin
        painter.setPen(Qt.GlobalColor.white)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        
        painter.end()
        
        self.status_icon.setPixmap(pixmap)
        
    def _update_metrics(self, result: CalibrationResult):
        """Kalibrasyon metriklerini günceller."""
        if result.success:
            # Faktör
            self.factor_value_label.setText(f"{result.factor:.3f} px/mm")
            
            # Piksel boyutu
            self.pixel_size_label.setText(f"{result.mm_per_pixel:.4f} mm/px")
            
            # Güven
            confidence_percent = result.confidence * 100
            self.confidence_label.setText(f"{confidence_percent:.0f}%")
            
            # Güven rengini ayarla
            if result.confidence >= 0.8:
                confidence_color = "#4CAF50"  # Yeşil
            elif result.confidence >= 0.6:
                confidence_color = "#FF9800"  # Turuncu
            else:
                confidence_color = "#F44336"  # Kırmızı
                
            self.confidence_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 13px;
                    font-weight: bold;
                    color: {confidence_color};
                    min-width: 100px;
                }}
            """)
            
            # Yöntem
            method_names = {
                CalibrationMethod.MANUAL: "Manual",
                CalibrationMethod.CATHETER: "Catheter",
                CalibrationMethod.DICOM_METADATA: "DICOM",
                CalibrationMethod.AUTO_DETECT: "Auto"
            }
            self.method_label.setText(method_names.get(result.method, "Unknown"))
            
        else:
            # Başarısız durumda varsayılan değerler
            self.factor_value_label.setText("-- px/mm")
            self.pixel_size_label.setText("-- mm/px")
            self.confidence_label.setText("--%")
            self.method_label.setText("--")
            
    def _update_visualization(self, result: CalibrationResult):
        """Görselleştirmeyi günceller."""
        if result.visualization is not None:
            # NumPy array'i QPixmap'e dönüştür
            height, width = result.visualization.shape[:2]
            
            if len(result.visualization.shape) == 3:
                # Renkli görüntü
                bytes_per_line = 3 * width
                q_image = QPixmap.fromImage(
                    result.visualization.data,
                    width, height,
                    bytes_per_line,
                    QPixmap.Format.Format_RGB888
                )
            else:
                # Gri tonlama
                bytes_per_line = width
                q_image = QPixmap.fromImage(
                    result.visualization.data,
                    width, height,
                    bytes_per_line,
                    QPixmap.Format.Format_Grayscale8
                )
                
            # Ölçekle ve göster
            scaled_pixmap = q_image.scaled(
                self.visualization_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.visualization_label.setPixmap(scaled_pixmap)
            
        else:
            # Görselleştirme yoksa placeholder göster
            self.visualization_label.setText("No visualization available")
            self.visualization_label.setPixmap(QPixmap())
            
    def clear_results(self):
        """Gösterilen sonuçları temizler."""
        self._current_result = None
        self._validation = None
        
        # Metrikleri sıfırla
        self.status_label.setText("No calibration")
        self._update_status_style(False)
        self.status_icon.clear()
        
        self.factor_value_label.setText("-- px/mm")
        self.pixel_size_label.setText("-- mm/px")
        self.confidence_label.setText("--%")
        self.method_label.setText("--")
        
        # Görselleştirmeyi temizle
        self.visualization_label.clear()
        self.visualization_label.setText("No calibration data")
        
        # Doğrulama mesajlarını gizle
        self.validation_widget.hide()
        
        # Widget'ı gizle
        self.hide()
        
    def get_current_result(self) -> Optional[CalibrationResult]:
        """
        Mevcut kalibrasyon sonucunu döndürür.
        
        Returns:
            Optional[CalibrationResult]: Mevcut sonuç veya None
        """
        return self._current_result