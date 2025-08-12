"""
ECG Control Panel Component

ECG kontrollerini içeren panel.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QCheckBox, QLabel, QSlider,
    QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
import logging

logger = logging.getLogger(__name__)


class ECGControlPanel(QWidget):
    """
    ECG kontrol paneli.
    
    Bu panel:
    - Görüntüleme seçenekleri
    - Analiz kontrolleri
    - Zoom kontrolleri
    - Dışa aktarma seçenekleri sağlar
    
    Signals:
        show_rpeaks_changed: R-peak gösterimi değişti
        show_phases_changed: Faz gösterimi değişti
        analysis_requested: Analiz istendi
        export_requested: Dışa aktarma istendi
        zoom_changed: Zoom seviyesi değişti
    """
    
    # Signals
    show_rpeaks_changed = pyqtSignal(bool)
    show_phases_changed = pyqtSignal(bool)
    analysis_requested = pyqtSignal()
    export_requested = pyqtSignal(str)  # format
    zoom_changed = pyqtSignal(float)  # zoom level
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        ECGControlPanel constructor.
        
        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        logger.info("ECGControlPanel initialized")
    
    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Görüntüleme seçenekleri
        self._create_display_options(layout)
        
        # Analiz kontrolleri
        self._create_analysis_controls(layout)
        
        # Zoom kontrolleri
        self._create_zoom_controls(layout)
        
        # Dışa aktarma
        self._create_export_controls(layout)
        
        # Spacer
        layout.addStretch()
    
    def _create_display_options(self, parent_layout: QVBoxLayout):
        """Görüntüleme seçenekleri oluştur."""
        group = QGroupBox("Görüntüleme Seçenekleri")
        layout = QVBoxLayout()
        
        # R-peak gösterimi
        self.show_rpeaks_checkbox = QCheckBox("R-peak'leri Göster")
        self.show_rpeaks_checkbox.setChecked(True)
        layout.addWidget(self.show_rpeaks_checkbox)
        
        # Kardiyak faz gösterimi
        self.show_phases_checkbox = QCheckBox("Kardiyak Fazları Göster")
        self.show_phases_checkbox.setChecked(False)
        layout.addWidget(self.show_phases_checkbox)
        
        # Grid gösterimi
        self.show_grid_checkbox = QCheckBox("Grid Göster")
        self.show_grid_checkbox.setChecked(True)
        layout.addWidget(self.show_grid_checkbox)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _create_analysis_controls(self, parent_layout: QVBoxLayout):
        """Analiz kontrolleri oluştur."""
        group = QGroupBox("Analiz")
        layout = QVBoxLayout()
        
        # Analiz butonu
        self.analyze_button = QPushButton("ECG Analizi Yap")
        self.analyze_button.setIcon(QIcon.fromTheme("system-run"))
        layout.addWidget(self.analyze_button)
        
        # Kalite göstergesi
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Sinyal Kalitesi:"))
        self.quality_label = QLabel("--")
        self.quality_label.setStyleSheet("font-weight: bold;")
        quality_layout.addWidget(self.quality_label)
        quality_layout.addStretch()
        layout.addLayout(quality_layout)
        
        # İstatistikler
        stats_layout = QVBoxLayout()
        
        # Kalp hızı
        hr_layout = QHBoxLayout()
        hr_layout.addWidget(QLabel("Kalp Hızı:"))
        self.hr_label = QLabel("-- BPM")
        hr_layout.addWidget(self.hr_label)
        hr_layout.addStretch()
        stats_layout.addLayout(hr_layout)
        
        # R-peak sayısı
        peak_layout = QHBoxLayout()
        peak_layout.addWidget(QLabel("R-peak Sayısı:"))
        self.peak_count_label = QLabel("--")
        peak_layout.addWidget(self.peak_count_label)
        peak_layout.addStretch()
        stats_layout.addLayout(peak_layout)
        
        # HRV
        hrv_layout = QHBoxLayout()
        hrv_layout.addWidget(QLabel("HRV (RMSSD):"))
        self.hrv_label = QLabel("-- ms")
        hrv_layout.addWidget(self.hrv_label)
        hrv_layout.addStretch()
        stats_layout.addLayout(hrv_layout)
        
        layout.addLayout(stats_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _create_zoom_controls(self, parent_layout: QVBoxLayout):
        """Zoom kontrolleri oluştur."""
        group = QGroupBox("Zoom")
        layout = QVBoxLayout()
        
        # Zoom slider
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)  # 0.1x
        self.zoom_slider.setMaximum(500)  # 5x
        self.zoom_slider.setValue(100)  # 1x
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("1.0x")
        self.zoom_label.setMinimumWidth(40)
        zoom_layout.addWidget(self.zoom_label)
        
        layout.addLayout(zoom_layout)
        
        # Hızlı zoom butonları
        button_layout = QHBoxLayout()
        
        self.fit_button = QPushButton("Sığdır")
        self.fit_button.setIcon(QIcon.fromTheme("zoom-fit-best"))
        button_layout.addWidget(self.fit_button)
        
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setMaximumWidth(30)
        button_layout.addWidget(self.zoom_in_button)
        
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setMaximumWidth(30)
        button_layout.addWidget(self.zoom_out_button)
        
        layout.addLayout(button_layout)
        
        # Zaman penceresi
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Pencere:"))
        
        self.window_spinbox = QSpinBox()
        self.window_spinbox.setMinimum(1)
        self.window_spinbox.setMaximum(60)
        self.window_spinbox.setValue(10)
        self.window_spinbox.setSuffix(" s")
        window_layout.addWidget(self.window_spinbox)
        
        layout.addLayout(window_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _create_export_controls(self, parent_layout: QVBoxLayout):
        """Dışa aktarma kontrolleri oluştur."""
        group = QGroupBox("Dışa Aktarma")
        layout = QVBoxLayout()
        
        # Format seçimi
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(['CSV', 'JSON', 'PDF', 'PNG'])
        format_layout.addWidget(self.export_format_combo)
        
        layout.addLayout(format_layout)
        
        # Dışa aktarma butonu
        self.export_button = QPushButton("Dışa Aktar")
        self.export_button.setIcon(QIcon.fromTheme("document-save"))
        layout.addWidget(self.export_button)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _connect_signals(self):
        """Sinyal bağlantılarını yap."""
        # Görüntüleme
        self.show_rpeaks_checkbox.toggled.connect(self.show_rpeaks_changed.emit)
        self.show_phases_checkbox.toggled.connect(self.show_phases_changed.emit)
        
        # Analiz
        self.analyze_button.clicked.connect(self.analysis_requested.emit)
        
        # Zoom
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.zoom_in_button.clicked.connect(self._on_zoom_in)
        self.zoom_out_button.clicked.connect(self._on_zoom_out)
        
        # Dışa aktarma
        self.export_button.clicked.connect(self._on_export_clicked)
    
    def _on_zoom_slider_changed(self, value: int):
        """Zoom slider değişti."""
        zoom_level = value / 100.0
        self.zoom_label.setText(f"{zoom_level:.1f}x")
        self.zoom_changed.emit(zoom_level)
    
    def _on_zoom_in(self):
        """Zoom in."""
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(min(current + 20, self.zoom_slider.maximum()))
    
    def _on_zoom_out(self):
        """Zoom out."""
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(max(current - 20, self.zoom_slider.minimum()))
    
    def _on_export_clicked(self):
        """Dışa aktarma tıklandı."""
        format_text = self.export_format_combo.currentText()
        self.export_requested.emit(format_text.lower())
    
    def update_quality_indicator(self, quality: str, color: str):
        """
        Kalite göstergesini güncelle.
        
        Args:
            quality: Kalite metni
            color: Renk kodu
        """
        self.quality_label.setText(quality)
        self.quality_label.setStyleSheet(f"font-weight: bold; color: {color};")
    
    def update_statistics(self, hr: Optional[float] = None,
                         peak_count: Optional[int] = None,
                         hrv_rmssd: Optional[float] = None):
        """
        İstatistikleri güncelle.
        
        Args:
            hr: Kalp hızı (BPM)
            peak_count: R-peak sayısı
            hrv_rmssd: HRV RMSSD değeri (ms)
        """
        if hr is not None:
            self.hr_label.setText(f"{hr:.0f} BPM")
        else:
            self.hr_label.setText("-- BPM")
        
        if peak_count is not None:
            self.peak_count_label.setText(str(peak_count))
        else:
            self.peak_count_label.setText("--")
        
        if hrv_rmssd is not None:
            self.hrv_label.setText(f"{hrv_rmssd:.1f} ms")
        else:
            self.hrv_label.setText("-- ms")
    
    def set_analysis_enabled(self, enabled: bool):
        """Analiz kontrollerini etkinleştir/devre dışı bırak."""
        self.analyze_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)