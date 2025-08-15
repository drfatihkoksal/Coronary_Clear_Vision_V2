"""
Segmentation Results Display Component

Segmentasyon sonuçlarını gösteren UI bileşeni.
Single Responsibility: Sadece sonuçları görselleştirir.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import numpy as np
from typing import Optional

from src.domain.models.segmentation_models import SegmentationResult, SegmentationQuality


class SegmentationResultsDisplay(QWidget):
    """
    Segmentasyon sonuçlarını gösteren widget.

    Maske önizleme, özellikler ve kalite metriklerini sunar.
    Business logic içermez, sadece veri sunumu yapar.

    Signals:
        feature_clicked: Özellik tıklandığında (feature_name)
        export_requested: Export istendi
    """

    # Signals
    feature_clicked = pyqtSignal(str)
    export_requested = pyqtSignal()

    # Kalite renkleri
    QUALITY_COLORS = {
        SegmentationQuality.EXCELLENT: "#4CAF50",  # Yeşil
        SegmentationQuality.GOOD: "#8BC34A",  # Açık yeşil
        SegmentationQuality.FAIR: "#FF9800",  # Turuncu
        SegmentationQuality.POOR: "#F44336",  # Kırmızı
    }

    def __init__(self, parent=None):
        """
        Results display widget'ını başlatır.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_result: Optional[SegmentationResult] = None
        self._init_ui()

    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Başlık ve durum
        header_layout = self._create_header_layout()
        layout.addLayout(header_layout)

        # Ana içerik
        content_layout = QHBoxLayout()

        # Sol panel - Maske önizleme
        preview_widget = self._create_preview_widget()
        content_layout.addWidget(preview_widget, 1)

        # Sağ panel - Özellikler
        features_widget = self._create_features_widget()
        content_layout.addWidget(features_widget, 1)

        layout.addLayout(content_layout)

        # Alt panel - Metrikler
        metrics_widget = self._create_metrics_widget()
        layout.addWidget(metrics_widget)

        self.setLayout(layout)

    def _create_header_layout(self) -> QHBoxLayout:
        """
        Başlık ve durum göstergesini oluşturur.

        Returns:
            QHBoxLayout: Başlık layout'u
        """
        layout = QHBoxLayout()

        # Başlık
        self.title_label = QLabel("Segmentation Results")
        self.title_label.setObjectName("segResultsTitle")
        self.title_label.setStyleSheet(
            """
            QLabel#segResultsTitle {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
        """
        )
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Kalite göstergesi
        self.quality_indicator = self._create_quality_indicator()
        layout.addWidget(self.quality_indicator)

        return layout

    def _create_quality_indicator(self) -> QFrame:
        """
        Kalite gösterge widget'ı oluşturur.

        Returns:
            QFrame: Kalite göstergesi
        """
        frame = QFrame()
        frame.setObjectName("qualityIndicator")
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)

        self.quality_label = QLabel("Quality: --")
        self.quality_label.setObjectName("qualityLabel")
        layout.addWidget(self.quality_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setObjectName("confidenceBar")
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%p%")
        self.confidence_bar.setMaximumWidth(100)
        layout.addWidget(self.confidence_bar)

        frame.setLayout(layout)
        self._update_quality_style(None)

        return frame

    def _create_preview_widget(self) -> QGroupBox:
        """
        Maske önizleme widget'ı oluşturur.

        Returns:
            QGroupBox: Önizleme grubu
        """
        group = QGroupBox("Segmentation Preview")
        layout = QVBoxLayout()

        # Görüntü etiketi
        self.preview_label = QLabel()
        self.preview_label.setObjectName("segPreview")
        self.preview_label.setMinimumSize(200, 200)
        self.preview_label.setScaledContents(False)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            """
            QLabel#segPreview {
                border: 2px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
        """
        )
        layout.addWidget(self.preview_label)

        # Piksel sayısı
        self.pixel_count_label = QLabel("Vessel pixels: --")
        self.pixel_count_label.setObjectName("pixelCountLabel")
        self.pixel_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.pixel_count_label)

        group.setLayout(layout)
        return group

    def _create_features_widget(self) -> QGroupBox:
        """
        Damar özellikleri widget'ı oluşturur.

        Returns:
            QGroupBox: Özellikler grubu
        """
        group = QGroupBox("Vessel Features")
        layout = QVBoxLayout()

        # Özellikler tablosu
        self.features_table = QTableWidget()
        self.features_table.setObjectName("featuresTable")
        self.features_table.setColumnCount(2)
        self.features_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.features_table.horizontalHeader().setStretchLastSection(True)
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setMaximumHeight(250)

        # Tıklama eventi
        self.features_table.itemClicked.connect(self._on_feature_clicked)

        layout.addWidget(self.features_table)

        group.setLayout(layout)
        return group

    def _create_metrics_widget(self) -> QGroupBox:
        """
        Performans metrikleri widget'ı oluşturur.

        Returns:
            QGroupBox: Metrikler grubu
        """
        group = QGroupBox("Performance Metrics")
        layout = QHBoxLayout()

        # İşlem süresi
        self.time_label = self._create_metric_label("Processing Time:", "-- ms")
        layout.addWidget(self.time_label)

        # Yöntem
        self.method_label = self._create_metric_label("Method:", "--")
        layout.addWidget(self.method_label)

        # Durum
        self.status_label = self._create_metric_label("Status:", "--")
        layout.addWidget(self.status_label)

        layout.addStretch()

        group.setLayout(layout)
        return group

    def _create_metric_label(self, title: str, value: str) -> QWidget:
        """
        Metrik etiketi oluşturur.

        Args:
            title: Metrik başlığı
            value: Varsayılan değer

        Returns:
            QWidget: Metrik widget'ı
        """
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 10, 0)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName(f"{title.lower().replace(' ', '_').replace(':', '')}_value")
        value_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(value_label)

        widget.setLayout(layout)
        return widget

    def display_result(self, result: SegmentationResult):
        """
        Segmentasyon sonucunu gösterir.

        Args:
            result: Segmentasyon sonucu
        """
        self._current_result = result

        if not result.success:
            self._display_error(result)
            return

        # Kalite göstergesini güncelle
        self._update_quality_display(result)

        # Önizlemeyi güncelle
        self._update_preview(result)

        # Özellikleri güncelle
        self._update_features(result)

        # Metrikleri güncelle
        self._update_metrics(result)

    def _display_error(self, result: SegmentationResult):
        """
        Hata durumunu gösterir.

        Args:
            result: Başarısız sonuç
        """
        self.quality_label.setText("Quality: Failed")
        self.confidence_bar.setValue(0)
        self._update_quality_style(None)

        self.preview_label.setText(f"Segmentation Failed\n\n{result.error_message}")
        self.preview_label.setPixmap(QPixmap())

        self.features_table.setRowCount(0)

        # Durum etiketi
        status_value = self.status_label.findChild(QLabel, "status_value")
        if status_value:
            status_value.setText("Failed")
            status_value.setStyleSheet("font-weight: bold; font-size: 11px; color: #F44336;")

    def _update_quality_display(self, result: SegmentationResult):
        """
        Kalite göstergesini günceller.

        Args:
            result: Segmentasyon sonucu
        """
        self.quality_label.setText(f"Quality: {result.quality.value.upper()}")
        self.confidence_bar.setValue(int(result.confidence * 100))
        self._update_quality_style(result.quality)

    def _update_quality_style(self, quality: Optional[SegmentationQuality]):
        """
        Kalite göstergesi stilini günceller.

        Args:
            quality: Kalite seviyesi
        """
        if quality:
            color = self.QUALITY_COLORS.get(quality, "#999")
            self.quality_indicator.setStyleSheet(
                f"""
                QFrame#qualityIndicator {{
                    background-color: {color}20;
                    border: 1px solid {color};
                    border-radius: 4px;
                }}
                QLabel#qualityLabel {{
                    color: {color};
                    font-weight: bold;
                }}
                QProgressBar#confidenceBar::chunk {{
                    background-color: {color};
                }}
            """
            )
        else:
            self.quality_indicator.setStyleSheet(
                """
                QFrame#qualityIndicator {
                    background-color: #f5f5f5;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
            """
            )

    def _update_preview(self, result: SegmentationResult):
        """
        Maske önizlemesini günceller.

        Args:
            result: Segmentasyon sonucu
        """
        if result.mask is None:
            self.preview_label.setText("No mask available")
            return

        # Görselleştirme varsa kullan
        if result.visualization is not None:
            preview_image = result.visualization
        else:
            # Maskeyi görselleştir
            preview_image = self._create_mask_visualization(result.mask)

        # QPixmap'e dönüştür
        height, width = preview_image.shape[:2]

        if len(preview_image.shape) == 3:
            # Renkli
            bytes_per_line = 3 * width
            q_image = QImage(
                preview_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            # Gri tonlama
            bytes_per_line = width
            q_image = QImage(
                preview_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8
            )

        pixmap = QPixmap.fromImage(q_image)

        # Ölçekle
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.preview_label.setPixmap(scaled_pixmap)

        # Piksel sayısı
        vessel_pixels = result.vessel_pixels
        total_pixels = result.mask.size
        percentage = (vessel_pixels / total_pixels * 100) if total_pixels > 0 else 0

        self.pixel_count_label.setText(f"Vessel pixels: {vessel_pixels:,} ({percentage:.1f}%)")

    def _create_mask_visualization(self, mask: np.ndarray) -> np.ndarray:
        """
        Maske görselleştirmesi oluşturur.

        Args:
            mask: Segmentasyon maskesi

        Returns:
            np.ndarray: RGB görselleştirme
        """
        # Maskeyi RGB'ye çevir
        if len(mask.shape) == 2:
            # Yeşil kanal olarak göster
            vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            vis[:, :, 1] = mask  # Yeşil kanal
        else:
            vis = mask

        return vis

    def _update_features(self, result: SegmentationResult):
        """
        Damar özelliklerini günceller.

        Args:
            result: Segmentasyon sonucu
        """
        self.features_table.setRowCount(0)

        if not result.has_features or result.features is None:
            return

        features = result.features

        # Özellik listesi
        feature_data = [
            ("Mean Diameter", f"{features.mean_diameter:.2f} px"),
            ("Total Length", f"{features.length:.1f} px"),
            ("Vessel Area", f"{features.area:.0f} px²"),
            ("Tortuosity", f"{features.tortuosity:.3f}"),
            ("Main Orientation", f"{features.get_main_orientation():.1f}°"),
            ("Branch Points", str(len(features.branch_points))),
            ("End Points", str(len(features.end_points))),
            ("Centerline Points", str(len(features.centerline))),
        ]

        # Tabloya ekle
        for feature_name, value in feature_data:
            row = self.features_table.rowCount()
            self.features_table.insertRow(row)

            name_item = QTableWidgetItem(feature_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.features_table.setItem(row, 0, name_item)

            value_item = QTableWidgetItem(value)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.features_table.setItem(row, 1, value_item)

    def _update_metrics(self, result: SegmentationResult):
        """
        Performans metriklerini günceller.

        Args:
            result: Segmentasyon sonucu
        """
        # İşlem süresi
        time_value = self.time_label.findChild(QLabel, "processing_time_value")
        if time_value:
            time_value.setText(f"{result.processing_time_ms:.1f} ms")

        # Yöntem
        method_value = self.method_label.findChild(QLabel, "method_value")
        if method_value:
            method_value.setText(result.method.value.replace("_", " ").title())

        # Durum
        status_value = self.status_label.findChild(QLabel, "status_value")
        if status_value:
            status_value.setText("Success")
            status_value.setStyleSheet("font-weight: bold; font-size: 11px; color: #4CAF50;")

    def _on_feature_clicked(self, item: QTableWidgetItem):
        """
        Özellik tıklandığında çağrılır.

        Args:
            item: Tıklanan tablo öğesi
        """
        if item.column() == 0:  # Özellik adı sütunu
            feature_name = item.text()
            self.feature_clicked.emit(feature_name)

    def clear_results(self):
        """Gösterilen sonuçları temizler."""
        self._current_result = None

        # Kalite göstergesini sıfırla
        self.quality_label.setText("Quality: --")
        self.confidence_bar.setValue(0)
        self._update_quality_style(None)

        # Önizlemeyi temizle
        self.preview_label.clear()
        self.preview_label.setText("No segmentation")
        self.pixel_count_label.setText("Vessel pixels: --")

        # Özellikleri temizle
        self.features_table.setRowCount(0)

        # Metrikleri sıfırla
        for widget in [self.time_label, self.method_label, self.status_label]:
            value_label = widget.findChild(QLabel)
            if value_label and value_label.objectName().endswith("_value"):
                value_label.setText("--")
                value_label.setStyleSheet("font-weight: bold; font-size: 11px;")

    def get_current_result(self) -> Optional[SegmentationResult]:
        """
        Mevcut segmentasyon sonucunu döndürür.

        Returns:
            Optional[SegmentationResult]: Mevcut sonuç veya None
        """
        return self._current_result
