"""
QCA Results Display Component

QCA analiz sonuçlarını gösteren UI bileşeni.
Single Responsibility: Sadece sonuçları görselleştirir, analiz yapmaz.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                            QTableWidget, QTableWidgetItem, QGroupBox,
                            QHeaderView, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from typing import Optional, List, Dict, Any

from src.domain.models.qca_models import (
    QCAAnalysisResult, StenosisGrade, VesselMeasurement
)


class QCAResultsDisplay(QWidget):
    """
    QCA analiz sonuçlarını gösteren widget.
    
    Tablo, metrikler ve stenoz bilgilerini görselleştirir.
    Business logic içermez, sadece veri sunumu yapar.
    
    Signals:
        measurement_selected: Ölçüm seçildiğinde (index)
    """
    
    # Signals
    measurement_selected = pyqtSignal(int)
    
    # Stenoz renkleri
    STENOSIS_COLORS = {
        StenosisGrade.MINIMAL: QColor(76, 175, 80),      # Yeşil
        StenosisGrade.MILD: QColor(255, 235, 59),        # Sarı
        StenosisGrade.MODERATE: QColor(255, 152, 0),     # Turuncu
        StenosisGrade.SEVERE: QColor(244, 67, 54),       # Kırmızı
        StenosisGrade.TOTAL_OCCLUSION: QColor(183, 28, 28)  # Koyu kırmızı
    }
    
    def __init__(self, parent=None):
        """
        Results display widget'ını başlatır.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_result: Optional[QCAAnalysisResult] = None
        self._init_ui()
        
    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget ile organize et
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("qcaResultsTabs")
        
        # Özet sekmesi
        self.summary_widget = self._create_summary_widget()
        self.tab_widget.addTab(self.summary_widget, "Summary")
        
        # Detaylı ölçümler sekmesi
        self.measurements_widget = self._create_measurements_widget()
        self.tab_widget.addTab(self.measurements_widget, "Measurements")
        
        # Stenoz analizi sekmesi
        self.stenosis_widget = self._create_stenosis_widget()
        self.tab_widget.addTab(self.stenosis_widget, "Stenosis Analysis")
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def _create_summary_widget(self) -> QWidget:
        """
        Özet bilgileri gösteren widget oluşturur.
        
        Returns:
            QWidget: Özet widget'ı
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Anahtar metrikler
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QVBoxLayout()
        
        # Metrik etiketleri
        self.mean_diameter_label = self._create_metric_label("Mean Diameter:")
        self.min_diameter_label = self._create_metric_label("Min Diameter:")
        self.max_diameter_label = self._create_metric_label("Max Diameter:")
        self.vessel_type_label = self._create_metric_label("Vessel Type:")
        self.analysis_time_label = self._create_metric_label("Analysis Time:")
        
        metrics_layout.addWidget(self.mean_diameter_label)
        metrics_layout.addWidget(self.min_diameter_label)
        metrics_layout.addWidget(self.max_diameter_label)
        metrics_layout.addWidget(self.vessel_type_label)
        metrics_layout.addWidget(self.analysis_time_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Kalibrasyon bilgisi
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()
        
        self.calib_factor_label = self._create_metric_label("Factor:")
        self.calib_method_label = self._create_metric_label("Method:")
        self.calib_confidence_label = self._create_metric_label("Confidence:")
        
        calib_layout.addWidget(self.calib_factor_label)
        calib_layout.addWidget(self.calib_method_label)
        calib_layout.addWidget(self.calib_confidence_label)
        
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def _create_measurements_widget(self) -> QWidget:
        """
        Detaylı ölçümler tablosunu içeren widget.
        
        Returns:
            QWidget: Ölçümler widget'ı
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Ölçümler tablosu
        self.measurements_table = QTableWidget()
        self.measurements_table.setObjectName("qcaMeasurementsTable")
        self.measurements_table.setColumnCount(4)
        self.measurements_table.setHorizontalHeaderLabels([
            "Position (mm)", "Diameter (mm)", "Area (mm²)", "Confidence"
        ])
        
        # Tablo özellikleri
        header = self.measurements_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.measurements_table.setAlternatingRowColors(True)
        self.measurements_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        
        # Seçim sinyali
        self.measurements_table.itemSelectionChanged.connect(
            self._on_measurement_selection_changed
        )
        
        layout.addWidget(self.measurements_table)
        
        # İstatistikler
        stats_layout = QHBoxLayout()
        self.total_measurements_label = QLabel("Total Measurements: 0")
        self.avg_confidence_label = QLabel("Average Confidence: --")
        stats_layout.addWidget(self.total_measurements_label)
        stats_layout.addWidget(self.avg_confidence_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        widget.setLayout(layout)
        return widget
        
    def _create_stenosis_widget(self) -> QWidget:
        """
        Stenoz analizi widget'ı oluşturur.
        
        Returns:
            QWidget: Stenoz widget'ı
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Stenoz varlığı
        self.stenosis_status_label = QLabel("No stenosis detected")
        self.stenosis_status_label.setObjectName("qcaStenosisStatus")
        self.stenosis_status_label.setStyleSheet("""
            QLabel#qcaStenosisStatus {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                background-color: #E8F5E9;
                color: #2E7D32;
            }
        """)
        layout.addWidget(self.stenosis_status_label)
        
        # Stenoz detayları
        self.stenosis_details_group = QGroupBox("Stenosis Details")
        details_layout = QVBoxLayout()
        
        self.stenosis_percent_label = self._create_metric_label("Diameter Stenosis:")
        self.stenosis_area_label = self._create_metric_label("Area Stenosis:")
        self.stenosis_location_label = self._create_metric_label("Location:")
        self.stenosis_length_label = self._create_metric_label("Length:")
        self.stenosis_grade_label = self._create_metric_label("Clinical Grade:")
        self.reference_diameter_label = self._create_metric_label("Reference Diameter:")
        self.mld_label = self._create_metric_label("Minimal Lumen Diameter:")
        
        details_layout.addWidget(self.stenosis_percent_label)
        details_layout.addWidget(self.stenosis_area_label)
        details_layout.addWidget(self.stenosis_location_label)
        details_layout.addWidget(self.stenosis_length_label)
        details_layout.addWidget(self.stenosis_grade_label)
        details_layout.addWidget(self.reference_diameter_label)
        details_layout.addWidget(self.mld_label)
        
        self.stenosis_details_group.setLayout(details_layout)
        self.stenosis_details_group.setVisible(False)
        layout.addWidget(self.stenosis_details_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def _create_metric_label(self, text: str) -> QLabel:
        """
        Metrik gösterimi için formatlanmış etiket oluşturur.
        
        Args:
            text: Etiket metni
            
        Returns:
            QLabel: Formatlanmış etiket
        """
        label = QLabel(f"{text} --")
        label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                padding: 5px;
                min-height: 25px;
            }
        """)
        return label
        
    def display_result(self, result: QCAAnalysisResult):
        """
        QCA analiz sonucunu gösterir.
        
        Args:
            result: Gösterilecek analiz sonucu
        """
        self._current_result = result
        
        if not result.is_successful:
            self._display_error(result)
            return
            
        # Özet metrikleri güncelle
        self._update_summary_metrics(result)
        
        # Ölçümleri güncelle
        self._update_measurements_table(result)
        
        # Stenoz bilgilerini güncelle
        self._update_stenosis_info(result)
        
    def _display_error(self, result: QCAAnalysisResult):
        """
        Hata durumunu gösterir.
        
        Args:
            result: Başarısız analiz sonucu
        """
        # Tabloları temizle
        self.measurements_table.setRowCount(0)
        
        # Hata mesajı göster
        error_msg = result.error_message or "Analysis failed"
        self.mean_diameter_label.setText(f"Error: {error_msg}")
        self.mean_diameter_label.setStyleSheet("""
            QLabel {
                color: #F44336;
                font-weight: bold;
            }
        """)
        
    def _update_summary_metrics(self, result: QCAAnalysisResult):
        """
        Özet metrikleri günceller.
        
        Args:
            result: Analiz sonucu
        """
        # Çap metrikleri
        if result.mean_diameter_mm is not None:
            self.mean_diameter_label.setText(f"Mean Diameter: {result.mean_diameter_mm:.3f} mm")
        if result.min_diameter_mm is not None:
            self.min_diameter_label.setText(f"Min Diameter: {result.min_diameter_mm:.3f} mm")
        if result.max_diameter_mm is not None:
            self.max_diameter_label.setText(f"Max Diameter: {result.max_diameter_mm:.3f} mm")
            
        # Damar tipi ve analiz süresi
        self.vessel_type_label.setText(f"Vessel Type: {result.vessel_type.value}")
        self.analysis_time_label.setText(f"Analysis Time: {result.analysis_time_ms:.1f} ms")
        
        # Kalibrasyon bilgisi
        if result.calibration:
            calib = result.calibration
            self.calib_factor_label.setText(f"Factor: {calib.factor:.4f} px/mm")
            self.calib_method_label.setText(f"Method: {calib.method}")
            self.calib_confidence_label.setText(f"Confidence: {calib.confidence:.1%}")
            
    def _update_measurements_table(self, result: QCAAnalysisResult):
        """
        Ölçümler tablosunu günceller.
        
        Args:
            result: Analiz sonucu
        """
        measurements = result.measurements
        self.measurements_table.setRowCount(len(measurements))
        
        for i, measurement in enumerate(measurements):
            # Position
            pos_item = QTableWidgetItem(f"{measurement.position_mm:.2f}")
            pos_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.measurements_table.setItem(i, 0, pos_item)
            
            # Diameter
            diam_item = QTableWidgetItem(f"{measurement.diameter_mm:.3f}")
            diam_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.measurements_table.setItem(i, 1, diam_item)
            
            # Area
            area_item = QTableWidgetItem(f"{measurement.area_mm2:.3f}")
            area_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.measurements_table.setItem(i, 2, area_item)
            
            # Confidence
            conf_item = QTableWidgetItem(f"{measurement.confidence:.1%}")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Düşük güven değerlerini renklendir
            if measurement.confidence < 0.8:
                conf_item.setBackground(QColor(255, 235, 59, 50))  # Sarı arka plan
                
            self.measurements_table.setItem(i, 3, conf_item)
            
        # İstatistikleri güncelle
        self.total_measurements_label.setText(f"Total Measurements: {len(measurements)}")
        
        if measurements:
            avg_conf = sum(m.confidence for m in measurements) / len(measurements)
            self.avg_confidence_label.setText(f"Average Confidence: {avg_conf:.1%}")
        else:
            self.avg_confidence_label.setText("Average Confidence: --")
            
    def _update_stenosis_info(self, result: QCAAnalysisResult):
        """
        Stenoz bilgilerini günceller.
        
        Args:
            result: Analiz sonucu
        """
        if not result.has_stenosis:
            self.stenosis_status_label.setText("No stenosis detected")
            self.stenosis_status_label.setStyleSheet("""
                QLabel#qcaStenosisStatus {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                    background-color: #E8F5E9;
                    color: #2E7D32;
                }
            """)
            self.stenosis_details_group.setVisible(False)
            return
            
        # Stenoz var
        stenosis = result.stenosis_data
        self.stenosis_status_label.setText("Stenosis Detected!")
        
        # Renk kodlaması
        color = self.STENOSIS_COLORS.get(stenosis.grade, QColor(128, 128, 128))
        self.stenosis_status_label.setStyleSheet(f"""
            QLabel#qcaStenosisStatus {{
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                background-color: {color.name()};
                color: white;
            }}
        """)
        
        # Detayları göster
        self.stenosis_details_group.setVisible(True)
        
        self.stenosis_percent_label.setText(
            f"Diameter Stenosis: {stenosis.percent_diameter:.1f}%"
        )
        self.stenosis_area_label.setText(
            f"Area Stenosis: {stenosis.percent_area:.1f}%"
        )
        self.stenosis_location_label.setText(
            f"Location: {stenosis.location_mm:.1f} mm"
        )
        self.stenosis_length_label.setText(
            f"Length: {stenosis.length_mm:.1f} mm"
        )
        self.stenosis_grade_label.setText(
            f"Clinical Grade: {stenosis.grade.value.upper()}"
        )
        self.reference_diameter_label.setText(
            f"Reference Diameter: {stenosis.reference_diameter_mm:.2f} mm"
        )
        self.mld_label.setText(
            f"Minimal Lumen Diameter: {stenosis.minimal_lumen_diameter_mm:.2f} mm"
        )
        
    def _on_measurement_selection_changed(self):
        """Ölçüm seçimi değiştiğinde çağrılır."""
        selected_rows = self.measurements_table.selectionModel().selectedRows()
        if selected_rows:
            index = selected_rows[0].row()
            self.measurement_selected.emit(index)
            
    def clear_results(self):
        """Gösterilen sonuçları temizler."""
        self._current_result = None
        
        # Metrikleri sıfırla
        for label in [self.mean_diameter_label, self.min_diameter_label,
                     self.max_diameter_label, self.vessel_type_label,
                     self.analysis_time_label, self.calib_factor_label,
                     self.calib_method_label, self.calib_confidence_label]:
            text = label.text().split(':')[0] + ": --"
            label.setText(text)
            
        # Tabloyu temizle
        self.measurements_table.setRowCount(0)
        self.total_measurements_label.setText("Total Measurements: 0")
        self.avg_confidence_label.setText("Average Confidence: --")
        
        # Stenoz bilgilerini temizle
        self.stenosis_status_label.setText("No stenosis detected")
        self.stenosis_details_group.setVisible(False)
        
    def get_current_result(self) -> Optional[QCAAnalysisResult]:
        """
        Mevcut gösterilen sonucu döndürür.
        
        Returns:
            Optional[QCAAnalysisResult]: Mevcut sonuç veya None
        """
        return self._current_result