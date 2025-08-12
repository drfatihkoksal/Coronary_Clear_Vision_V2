"""
QCA Control Panel Component

QCA analizi kontrol butonları ve seçenekleri için UI bileşeni.
Single Responsibility: Sadece kullanıcı kontrollerini yönetir.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, 
                            QLabel, QComboBox, QCheckBox, QGroupBox, QSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt
from typing import Dict, Any


class QCAControlPanel(QWidget):
    """
    QCA analizi için kontrol paneli bileşeni.
    
    Analiz başlatma, yöntem seçimi ve analiz seçeneklerini yönetir.
    Business logic içermez, sadece UI kontrollerini sağlar.
    
    Signals:
        analyze_requested: Analiz butonu tıklandığında
        sequential_analyze_requested: Ardışık analiz istendi
        options_changed: Analiz seçenekleri değişti
        export_requested: Export istendi
    """
    
    # Signals
    analyze_requested = pyqtSignal()
    sequential_analyze_requested = pyqtSignal()
    options_changed = pyqtSignal(dict)  # options dict
    export_requested = pyqtSignal(str)  # export format
    
    def __init__(self, parent=None):
        """
        Control panel'i başlatır.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Ana kontrol butonları
        layout.addLayout(self._create_main_controls())
        
        # Analiz seçenekleri
        layout.addWidget(self._create_analysis_options())
        
        # Export kontrolleri
        layout.addLayout(self._create_export_controls())
        
        self.setLayout(layout)
        
    def _create_main_controls(self) -> QHBoxLayout:
        """
        Ana kontrol butonlarını oluşturur.
        
        Returns:
            QHBoxLayout: Buton layout'u
        """
        layout = QHBoxLayout()
        
        # Tekli analiz butonu
        self.analyze_button = QPushButton("Analyze Frame")
        self.analyze_button.setObjectName("qcaAnalyzeButton")
        self.analyze_button.setToolTip("Analyze current frame")
        self._apply_primary_button_style(self.analyze_button)
        layout.addWidget(self.analyze_button)
        
        # Ardışık analiz butonu
        self.sequential_button = QPushButton("Sequential Analysis")
        self.sequential_button.setObjectName("qcaSequentialButton")
        self.sequential_button.setToolTip("Analyze multiple frames")
        self._apply_secondary_button_style(self.sequential_button)
        layout.addWidget(self.sequential_button)
        
        layout.addStretch()
        
        return layout
        
    def _create_analysis_options(self) -> QGroupBox:
        """
        Analiz seçenekleri grubunu oluşturur.
        
        Returns:
            QGroupBox: Seçenekler grubu
        """
        group = QGroupBox("Analysis Options")
        layout = QVBoxLayout()
        
        # Çap hesaplama yöntemi
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Diameter Method:"))
        
        self.diameter_method_combo = QComboBox()
        self.diameter_method_combo.addItems([
            "Ribbon Method",
            "Densitometric",
            "Edge Detection",
            "Advanced"
        ])
        self.diameter_method_combo.setObjectName("qcaDiameterMethod")
        method_layout.addWidget(self.diameter_method_combo)
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # Analiz seçenekleri checkboxes
        self.detect_stenosis_cb = QCheckBox("Detect Stenosis")
        self.detect_stenosis_cb.setChecked(True)
        self.detect_stenosis_cb.setObjectName("qcaDetectStenosis")
        layout.addWidget(self.detect_stenosis_cb)
        
        self.smooth_centerline_cb = QCheckBox("Smooth Centerline")
        self.smooth_centerline_cb.setChecked(True)
        self.smooth_centerline_cb.setObjectName("qcaSmoothCenterline")
        layout.addWidget(self.smooth_centerline_cb)
        
        self.auto_calibration_cb = QCheckBox("Auto Calibration")
        self.auto_calibration_cb.setObjectName("qcaAutoCalibration")
        layout.addWidget(self.auto_calibration_cb)
        
        # Frame aralığı (sequential için)
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame Interval:"))
        
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setRange(1, 10)
        self.frame_interval_spin.setValue(1)
        self.frame_interval_spin.setSuffix(" frames")
        self.frame_interval_spin.setObjectName("qcaFrameInterval")
        frame_layout.addWidget(self.frame_interval_spin)
        frame_layout.addStretch()
        layout.addLayout(frame_layout)
        
        group.setLayout(layout)
        return group
        
    def _create_export_controls(self) -> QHBoxLayout:
        """
        Export kontrollerini oluşturur.
        
        Returns:
            QHBoxLayout: Export layout'u
        """
        layout = QHBoxLayout()
        
        layout.addWidget(QLabel("Export:"))
        
        # Export butonları
        self.export_excel_btn = QPushButton("Excel")
        self.export_excel_btn.setObjectName("qcaExportExcel")
        self.export_excel_btn.setEnabled(False)
        layout.addWidget(self.export_excel_btn)
        
        self.export_csv_btn = QPushButton("CSV")
        self.export_csv_btn.setObjectName("qcaExportCSV")
        self.export_csv_btn.setEnabled(False)
        layout.addWidget(self.export_csv_btn)
        
        self.export_pdf_btn = QPushButton("PDF Report")
        self.export_pdf_btn.setObjectName("qcaExportPDF")
        self.export_pdf_btn.setEnabled(False)
        layout.addWidget(self.export_pdf_btn)
        
        layout.addStretch()
        
        return layout
        
    def _apply_primary_button_style(self, button: QPushButton):
        """Birincil buton stili uygula."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BBBBBB;
                color: #666666;
            }
        """)
        
    def _apply_secondary_button_style(self, button: QPushButton):
        """İkincil buton stili uygula."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #E65100;
            }
            QPushButton:disabled {
                background-color: #BBBBBB;
                color: #666666;
            }
        """)
        
    def _connect_signals(self):
        """Signal bağlantılarını yapar."""
        # Ana butonlar
        self.analyze_button.clicked.connect(self.analyze_requested.emit)
        self.sequential_button.clicked.connect(self.sequential_analyze_requested.emit)
        
        # Export butonları
        self.export_excel_btn.clicked.connect(lambda: self.export_requested.emit("excel"))
        self.export_csv_btn.clicked.connect(lambda: self.export_requested.emit("csv"))
        self.export_pdf_btn.clicked.connect(lambda: self.export_requested.emit("pdf"))
        
        # Seçenek değişimleri
        self.diameter_method_combo.currentTextChanged.connect(self._on_options_changed)
        self.detect_stenosis_cb.stateChanged.connect(self._on_options_changed)
        self.smooth_centerline_cb.stateChanged.connect(self._on_options_changed)
        self.auto_calibration_cb.stateChanged.connect(self._on_options_changed)
        self.frame_interval_spin.valueChanged.connect(self._on_options_changed)
        
    def _on_options_changed(self):
        """Seçenekler değiştiğinde çağrılır."""
        options = self.get_analysis_options()
        self.options_changed.emit(options)
        
    def get_analysis_options(self) -> Dict[str, Any]:
        """
        Mevcut analiz seçeneklerini döndürür.
        
        Returns:
            Dict[str, Any]: Analiz seçenekleri
        """
        return {
            'diameter_method': self.diameter_method_combo.currentText().lower().replace(' ', '_'),
            'detect_stenosis': self.detect_stenosis_cb.isChecked(),
            'smooth_centerline': self.smooth_centerline_cb.isChecked(),
            'auto_calibration': self.auto_calibration_cb.isChecked(),
            'frame_interval': self.frame_interval_spin.value()
        }
        
    def set_export_enabled(self, enabled: bool):
        """
        Export butonlarını etkinleştirir/devre dışı bırakır.
        
        Args:
            enabled: Etkin mi?
        """
        self.export_excel_btn.setEnabled(enabled)
        self.export_csv_btn.setEnabled(enabled)
        self.export_pdf_btn.setEnabled(enabled)
        
    def set_analysis_enabled(self, enabled: bool):
        """
        Analiz butonlarını etkinleştirir/devre dışı bırakır.
        
        Args:
            enabled: Etkin mi?
        """
        self.analyze_button.setEnabled(enabled)
        self.sequential_button.setEnabled(enabled)
        
    def set_sequential_mode(self, is_sequential: bool):
        """
        Ardışık analiz modunu ayarlar.
        
        Args:
            is_sequential: Ardışık mod mu?
        """
        self.frame_interval_spin.setEnabled(is_sequential)
        
        # Buton stillerini değiştir
        if is_sequential:
            self._apply_primary_button_style(self.sequential_button)
            self._apply_secondary_button_style(self.analyze_button)
        else:
            self._apply_primary_button_style(self.analyze_button)
            self._apply_secondary_button_style(self.sequential_button)