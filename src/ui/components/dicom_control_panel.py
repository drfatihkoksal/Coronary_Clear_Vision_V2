"""
DICOM Control Panel Component

DICOM kontrollerini içeren panel.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSlider, QSpinBox,
    QComboBox, QCheckBox, QToolButton, QMenu,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QAction
from pathlib import Path
import logging

from src.domain.models.dicom_models import (
    DicomStudy, DicomSeries, DicomWindowLevel,
    DicomProjectionInfo
)

logger = logging.getLogger(__name__)


class DicomControlPanel(QWidget):
    """
    DICOM kontrol paneli.
    
    Bu panel:
    - Dosya yükleme
    - Frame navigasyonu
    - Pencere/seviye ayarları
    - Projeksiyon seçimi
    - Oynatma kontrolleri
    
    Signals:
        file_requested: Dosya açma istendi
        frame_changed: Frame değişti (index)
        window_level_changed: Pencere/seviye değişti
        projection_changed: Projeksiyon değişti
        export_requested: Dışa aktarma istendi (format)
        playback_toggled: Oynatma durumu değişti (playing)
    """
    
    # Signals
    file_requested = pyqtSignal()
    frame_changed = pyqtSignal(int)
    window_level_changed = pyqtSignal(DicomWindowLevel)
    projection_changed = pyqtSignal(int)  # series index
    export_requested = pyqtSignal(str)  # format
    playback_toggled = pyqtSignal(bool)  # playing
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        DicomControlPanel constructor.
        
        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self._current_study: Optional[DicomStudy] = None
        self._current_series_index = 0
        self._is_playing = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_timer)
        
        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()
        
    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Dosya kontrolleri
        self._create_file_controls(layout)
        
        # Çalışma bilgileri
        self._create_study_info(layout)
        
        # Projeksiyon seçimi
        self._create_projection_controls(layout)
        
        # Frame navigasyonu
        self._create_frame_controls(layout)
        
        # Pencere/seviye kontrolleri
        self._create_window_level_controls(layout)
        
        # Oynatma kontrolleri
        self._create_playback_controls(layout)
        
        # Dışa aktarma
        self._create_export_controls(layout)
        
        # Spacer
        layout.addStretch()
        
    def _create_file_controls(self, parent_layout: QVBoxLayout):
        """Dosya kontrolleri oluştur."""
        group = QGroupBox("Dosya İşlemleri")
        layout = QVBoxLayout()
        
        # Dosya aç butonu
        self.open_file_btn = QPushButton("DICOM Aç")
        self.open_file_btn.setIcon(QIcon.fromTheme("document-open"))
        layout.addWidget(self.open_file_btn)
        
        # Son dosyalar
        self.recent_files_btn = QPushButton("Son Dosyalar")
        self.recent_files_btn.setIcon(QIcon.fromTheme("document-open-recent"))
        self.recent_menu = QMenu(self)
        self.recent_files_btn.setMenu(self.recent_menu)
        layout.addWidget(self.recent_files_btn)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def _create_study_info(self, parent_layout: QVBoxLayout):
        """Çalışma bilgileri oluştur."""
        group = QGroupBox("Çalışma Bilgileri")
        layout = QVBoxLayout()
        
        # Hasta
        self.patient_label = QLabel("Hasta: -")
        layout.addWidget(self.patient_label)
        
        # Tarih
        self.study_date_label = QLabel("Tarih: -")
        layout.addWidget(self.study_date_label)
        
        # Modalite
        self.modality_label = QLabel("Modalite: -")
        layout.addWidget(self.modality_label)
        
        # Seri sayısı
        self.series_count_label = QLabel("Seri: -")
        layout.addWidget(self.series_count_label)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def _create_projection_controls(self, parent_layout: QVBoxLayout):
        """Projeksiyon kontrolleri oluştur."""
        self.projection_group = QGroupBox("Projeksiyon")
        layout = QVBoxLayout()
        
        # Projeksiyon seçimi
        self.projection_combo = QComboBox()
        self.projection_combo.setToolTip("Görüntüleme açısını seçin")
        layout.addWidget(self.projection_combo)
        
        # Açı bilgisi
        self.angle_label = QLabel("Açı: -")
        layout.addWidget(self.angle_label)
        
        # Navigasyon butonları
        nav_layout = QHBoxLayout()
        
        self.prev_projection_btn = QToolButton()
        self.prev_projection_btn.setIcon(QIcon.fromTheme("go-previous"))
        self.prev_projection_btn.setToolTip("Önceki projeksiyon")
        nav_layout.addWidget(self.prev_projection_btn)
        
        nav_layout.addStretch()
        
        self.next_projection_btn = QToolButton()
        self.next_projection_btn.setIcon(QIcon.fromTheme("go-next"))
        self.next_projection_btn.setToolTip("Sonraki projeksiyon")
        nav_layout.addWidget(self.next_projection_btn)
        
        layout.addLayout(nav_layout)
        
        self.projection_group.setLayout(layout)
        parent_layout.addWidget(self.projection_group)
        
    def _create_frame_controls(self, parent_layout: QVBoxLayout):
        """Frame kontrolleri oluştur."""
        group = QGroupBox("Frame Navigasyonu")
        layout = QVBoxLayout()
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(10)
        layout.addWidget(self.frame_slider)
        
        # Frame bilgisi
        info_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame:")
        info_layout.addWidget(self.frame_label)
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
        info_layout.addWidget(self.frame_spinbox)
        
        self.total_frames_label = QLabel("/ 0")
        info_layout.addWidget(self.total_frames_label)
        
        info_layout.addStretch()
        
        self.time_label = QLabel("Zaman: 0.0s")
        info_layout.addWidget(self.time_label)
        
        layout.addLayout(info_layout)
        
        # Frame navigasyon butonları
        nav_layout = QHBoxLayout()
        
        self.first_frame_btn = QToolButton()
        self.first_frame_btn.setIcon(QIcon.fromTheme("go-first"))
        self.first_frame_btn.setToolTip("İlk frame")
        nav_layout.addWidget(self.first_frame_btn)
        
        self.prev_frame_btn = QToolButton()
        self.prev_frame_btn.setIcon(QIcon.fromTheme("go-previous"))
        self.prev_frame_btn.setToolTip("Önceki frame")
        nav_layout.addWidget(self.prev_frame_btn)
        
        nav_layout.addStretch()
        
        self.next_frame_btn = QToolButton()
        self.next_frame_btn.setIcon(QIcon.fromTheme("go-next"))
        self.next_frame_btn.setToolTip("Sonraki frame")
        nav_layout.addWidget(self.next_frame_btn)
        
        self.last_frame_btn = QToolButton()
        self.last_frame_btn.setIcon(QIcon.fromTheme("go-last"))
        self.last_frame_btn.setToolTip("Son frame")
        nav_layout.addWidget(self.last_frame_btn)
        
        layout.addLayout(nav_layout)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def _create_window_level_controls(self, parent_layout: QVBoxLayout):
        """Pencere/seviye kontrolleri oluştur."""
        group = QGroupBox("Pencere/Seviye")
        layout = QVBoxLayout()
        
        # Preset'ler
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['Varsayılan', 'Anjio', 'Kemik', 'Yumuşak Doku'])
        self.preset_combo.setToolTip("Hazır pencere/seviye ayarları")
        layout.addWidget(self.preset_combo)
        
        # Pencere merkezi
        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Merkez:"))
        
        self.window_center_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_center_slider.setMinimum(-1000)
        self.window_center_slider.setMaximum(1000)
        self.window_center_slider.setValue(128)
        center_layout.addWidget(self.window_center_slider)
        
        self.window_center_label = QLabel("128")
        self.window_center_label.setMinimumWidth(40)
        center_layout.addWidget(self.window_center_label)
        
        layout.addLayout(center_layout)
        
        # Pencere genişliği
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Genişlik:"))
        
        self.window_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_width_slider.setMinimum(1)
        self.window_width_slider.setMaximum(2000)
        self.window_width_slider.setValue(256)
        width_layout.addWidget(self.window_width_slider)
        
        self.window_width_label = QLabel("256")
        self.window_width_label.setMinimumWidth(40)
        width_layout.addWidget(self.window_width_label)
        
        layout.addLayout(width_layout)
        
        # Otomatik ayar
        self.auto_window_btn = QPushButton("Otomatik Ayarla")
        self.auto_window_btn.setIcon(QIcon.fromTheme("view-refresh"))
        layout.addWidget(self.auto_window_btn)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def _create_playback_controls(self, parent_layout: QVBoxLayout):
        """Oynatma kontrolleri oluştur."""
        group = QGroupBox("Oynatma")
        layout = QVBoxLayout()
        
        # Oynatma butonları
        button_layout = QHBoxLayout()
        
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_pause_btn.setText("Oynat")
        self.play_pause_btn.setCheckable(True)
        button_layout.addWidget(self.play_pause_btn)
        
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_btn.setText("Durdur")
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # Hız kontrolü
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Hız:"))
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(50)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(40)
        speed_layout.addWidget(self.speed_label)
        
        layout.addLayout(speed_layout)
        
        # Döngü
        self.loop_checkbox = QCheckBox("Döngü")
        self.loop_checkbox.setChecked(True)
        layout.addWidget(self.loop_checkbox)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def _create_export_controls(self, parent_layout: QVBoxLayout):
        """Dışa aktarma kontrolleri oluştur."""
        group = QGroupBox("Dışa Aktarma")
        layout = QVBoxLayout()
        
        # Frame dışa aktarma
        self.export_frame_btn = QPushButton("Frame'i Kaydet")
        self.export_frame_btn.setIcon(QIcon.fromTheme("document-save"))
        export_frame_menu = QMenu(self)
        export_frame_menu.addAction("PNG", lambda: self.export_requested.emit("png"))
        export_frame_menu.addAction("JPEG", lambda: self.export_requested.emit("jpg"))
        export_frame_menu.addAction("TIFF", lambda: self.export_requested.emit("tiff"))
        self.export_frame_btn.setMenu(export_frame_menu)
        layout.addWidget(self.export_frame_btn)
        
        # Video dışa aktarma
        self.export_video_btn = QPushButton("Video Olarak Kaydet")
        self.export_video_btn.setIcon(QIcon.fromTheme("video-x-generic"))
        export_video_menu = QMenu(self)
        export_video_menu.addAction("MP4", lambda: self.export_requested.emit("mp4"))
        export_video_menu.addAction("AVI", lambda: self.export_requested.emit("avi"))
        self.export_video_btn.setMenu(export_video_menu)
        layout.addWidget(self.export_video_btn)
        
        # Rapor
        self.export_report_btn = QPushButton("Rapor Oluştur")
        self.export_report_btn.setIcon(QIcon.fromTheme("x-office-document"))
        export_report_menu = QMenu(self)
        export_report_menu.addAction("HTML", lambda: self.export_requested.emit("html"))
        export_report_menu.addAction("PDF", lambda: self.export_requested.emit("pdf"))
        export_report_menu.addAction("JSON", lambda: self.export_requested.emit("json"))
        self.export_report_btn.setMenu(export_report_menu)
        layout.addWidget(self.export_report_btn)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
        
    def _connect_signals(self):
        """Sinyal bağlantılarını yap."""
        # Dosya
        self.open_file_btn.clicked.connect(self.file_requested.emit)
        
        # Projeksiyon
        self.projection_combo.currentIndexChanged.connect(self._on_projection_changed)
        self.prev_projection_btn.clicked.connect(self._on_prev_projection)
        self.next_projection_btn.clicked.connect(self._on_next_projection)
        
        # Frame
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        self.first_frame_btn.clicked.connect(self._on_first_frame)
        self.prev_frame_btn.clicked.connect(self._on_prev_frame)
        self.next_frame_btn.clicked.connect(self._on_next_frame)
        self.last_frame_btn.clicked.connect(self._on_last_frame)
        
        # Pencere/seviye
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.window_center_slider.valueChanged.connect(self._on_window_level_changed)
        self.window_width_slider.valueChanged.connect(self._on_window_level_changed)
        self.auto_window_btn.clicked.connect(self._on_auto_window)
        
        # Oynatma
        self.play_pause_btn.toggled.connect(self._on_play_pause_toggled)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        
    def set_study(self, study: DicomStudy):
        """
        DICOM çalışmasını ayarla.
        
        Args:
            study: DICOM çalışması
        """
        self._current_study = study
        self._current_series_index = 0
        
        # UI'ı güncelle
        self._update_study_info()
        self._update_projection_list()
        self._update_frame_controls()
        self._update_ui_state()
        
    def get_current_series(self) -> Optional[DicomSeries]:
        """
        Mevcut seriyi döndür.
        
        Returns:
            Optional[DicomSeries]: Mevcut seri
        """
        if self._current_study and 0 <= self._current_series_index < self._current_study.num_series:
            return self._current_study.series_list[self._current_series_index]
        return None
        
    def get_current_frame_index(self) -> int:
        """
        Mevcut frame indeksini döndür.
        
        Returns:
            int: Frame indeksi
        """
        return self.frame_slider.value()
        
    def get_window_level(self) -> DicomWindowLevel:
        """
        Mevcut pencere/seviye ayarlarını döndür.
        
        Returns:
            DicomWindowLevel: Pencere/seviye
        """
        return DicomWindowLevel(
            center=float(self.window_center_slider.value()),
            width=float(self.window_width_slider.value()),
            name="Custom"
        )
        
    def add_recent_file(self, file_path: Path):
        """
        Son dosyalara ekle.
        
        Args:
            file_path: Dosya yolu
        """
        action = QAction(file_path.name, self)
        action.setData(str(file_path))
        action.triggered.connect(lambda: self._on_recent_file_selected(file_path))
        
        # En üste ekle
        actions = self.recent_menu.actions()
        if actions:
            self.recent_menu.insertAction(actions[0], action)
        else:
            self.recent_menu.addAction(action)
            
        # Maksimum 10 dosya
        while len(self.recent_menu.actions()) > 10:
            self.recent_menu.removeAction(self.recent_menu.actions()[-1])
            
    def _update_study_info(self):
        """Çalışma bilgilerini güncelle."""
        if self._current_study:
            # Hasta
            patient_name = self._current_study.patient_info.patient_name
            if patient_name == "Anonymous":
                patient_name = "Anonim"
            self.patient_label.setText(f"Hasta: {patient_name}")
            
            # Tarih
            study_date = self._current_study.study_info.study_date
            if study_date:
                self.study_date_label.setText(f"Tarih: {study_date.strftime('%d.%m.%Y')}")
            else:
                self.study_date_label.setText("Tarih: -")
                
            # Modalite
            if self._current_study.series_list:
                modality = self._current_study.series_list[0].info.modality.value
                self.modality_label.setText(f"Modalite: {modality}")
            else:
                self.modality_label.setText("Modalite: -")
                
            # Seri sayısı
            self.series_count_label.setText(f"Seri: {self._current_study.num_series}")
        else:
            self.patient_label.setText("Hasta: -")
            self.study_date_label.setText("Tarih: -")
            self.modality_label.setText("Modalite: -")
            self.series_count_label.setText("Seri: -")
            
    def _update_projection_list(self):
        """Projeksiyon listesini güncelle."""
        self.projection_combo.clear()
        
        if self._current_study:
            for i, series in enumerate(self._current_study.series_list):
                # Projeksiyon açıklaması
                if series.projection_info:
                    desc = series.projection_info.angle_description
                else:
                    desc = series.info.series_description or f"Seri {i+1}"
                    
                self.projection_combo.addItem(desc)
                
        # Multi-projeksiyon kontrolü
        has_multiple = self.projection_combo.count() > 1
        self.projection_group.setVisible(has_multiple)
        
    def _update_frame_controls(self):
        """Frame kontrollerini güncelle."""
        series = self.get_current_series()
        
        if series:
            # Frame sayısı
            num_frames = series.num_frames
            self.frame_slider.setMaximum(num_frames - 1)
            self.frame_spinbox.setMaximum(num_frames - 1)
            self.total_frames_label.setText(f"/ {num_frames}")
            
            # Süre
            duration = series.duration
            self.time_label.setText(f"Süre: {duration:.1f}s")
            
            # Multi-frame kontrolü
            is_multi = series.is_multi_frame
            self.frame_slider.setEnabled(is_multi)
            self.frame_spinbox.setEnabled(is_multi)
            self.play_pause_btn.setEnabled(is_multi)
            
            # Pencere/seviye
            self._update_window_level_from_series(series)
        else:
            self.frame_slider.setMaximum(0)
            self.frame_spinbox.setMaximum(0)
            self.total_frames_label.setText("/ 0")
            self.time_label.setText("Süre: 0.0s")
            
    def _update_window_level_from_series(self, series: DicomSeries):
        """Seriden pencere/seviye ayarlarını güncelle."""
        wl = series.window_level
        
        # Slider aralıklarını ayarla
        center_min = int(wl.center - wl.width * 2)
        center_max = int(wl.center + wl.width * 2)
        self.window_center_slider.setRange(center_min, center_max)
        self.window_center_slider.setValue(int(wl.center))
        
        width_max = int(wl.width * 4)
        self.window_width_slider.setMaximum(width_max)
        self.window_width_slider.setValue(int(wl.width))
        
    def _update_ui_state(self):
        """UI durumunu güncelle."""
        has_study = self._current_study is not None
        
        # Projeksiyon
        self.projection_combo.setEnabled(has_study)
        self.prev_projection_btn.setEnabled(has_study and self.projection_combo.count() > 1)
        self.next_projection_btn.setEnabled(has_study and self.projection_combo.count() > 1)
        
        # Frame
        has_frames = has_study and self.get_current_series() is not None
        self.frame_slider.setEnabled(has_frames)
        self.frame_spinbox.setEnabled(has_frames)
        self.first_frame_btn.setEnabled(has_frames)
        self.prev_frame_btn.setEnabled(has_frames)
        self.next_frame_btn.setEnabled(has_frames)
        self.last_frame_btn.setEnabled(has_frames)
        
        # Pencere/seviye
        self.preset_combo.setEnabled(has_frames)
        self.window_center_slider.setEnabled(has_frames)
        self.window_width_slider.setEnabled(has_frames)
        self.auto_window_btn.setEnabled(has_frames)
        
        # Oynatma
        is_multi = has_frames and self.get_current_series().is_multi_frame if has_frames else False
        self.play_pause_btn.setEnabled(is_multi)
        self.stop_btn.setEnabled(is_multi)
        self.speed_slider.setEnabled(is_multi)
        self.loop_checkbox.setEnabled(is_multi)
        
        # Dışa aktarma
        self.export_frame_btn.setEnabled(has_frames)
        self.export_video_btn.setEnabled(is_multi)
        self.export_report_btn.setEnabled(has_study)
        
    def _on_projection_changed(self, index: int):
        """Projeksiyon değişti."""
        if index >= 0:
            self._current_series_index = index
            self._update_frame_controls()
            self.projection_changed.emit(index)
            
            # Açı bilgisini güncelle
            series = self.get_current_series()
            if series and series.projection_info:
                self.angle_label.setText(f"Açı: {series.projection_info.angle_description}")
            else:
                self.angle_label.setText("Açı: -")
                
    def _on_prev_projection(self):
        """Önceki projeksiyon."""
        current = self.projection_combo.currentIndex()
        if current > 0:
            self.projection_combo.setCurrentIndex(current - 1)
            
    def _on_next_projection(self):
        """Sonraki projeksiyon."""
        current = self.projection_combo.currentIndex()
        if current < self.projection_combo.count() - 1:
            self.projection_combo.setCurrentIndex(current + 1)
            
    def _on_frame_slider_changed(self, value: int):
        """Frame slider değişti."""
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(value)
        self.frame_spinbox.blockSignals(False)
        
        # Zaman bilgisini güncelle
        series = self.get_current_series()
        if series:
            time = value / series.frame_rate
            self.time_label.setText(f"Zaman: {time:.2f}s")
            
        self.frame_changed.emit(value)
        
    def _on_frame_spinbox_changed(self, value: int):
        """Frame spinbox değişti."""
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        
        self.frame_changed.emit(value)
        
    def _on_first_frame(self):
        """İlk frame."""
        self.frame_slider.setValue(0)
        
    def _on_prev_frame(self):
        """Önceki frame."""
        current = self.frame_slider.value()
        if current > 0:
            self.frame_slider.setValue(current - 1)
            
    def _on_next_frame(self):
        """Sonraki frame."""
        current = self.frame_slider.value()
        if current < self.frame_slider.maximum():
            self.frame_slider.setValue(current + 1)
            
    def _on_last_frame(self):
        """Son frame."""
        self.frame_slider.setValue(self.frame_slider.maximum())
        
    def _on_preset_changed(self, index: int):
        """Preset değişti."""
        presets = {
            0: (128, 256),    # Varsayılan
            1: (300, 600),    # Anjio
            2: (400, 1500),   # Kemik
            3: (40, 400)      # Yumuşak doku
        }
        
        if index in presets:
            center, width = presets[index]
            self.window_center_slider.setValue(center)
            self.window_width_slider.setValue(width)
            
    def _on_window_level_changed(self):
        """Pencere/seviye değişti."""
        center = self.window_center_slider.value()
        width = self.window_width_slider.value()
        
        self.window_center_label.setText(str(center))
        self.window_width_label.setText(str(width))
        
        # Custom preset
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(-1)
        self.preset_combo.blockSignals(False)
        
        wl = DicomWindowLevel(float(center), float(width), "Custom")
        self.window_level_changed.emit(wl)
        
    def _on_auto_window(self):
        """Otomatik pencere/seviye."""
        # Bu sinyal viewer tarafından işlenecek
        pass
        
    def _on_play_pause_toggled(self, checked: bool):
        """Oynat/duraklat."""
        self._is_playing = checked
        
        if checked:
            self.play_pause_btn.setText("Duraklat")
            self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-pause"))
            
            # Timer'ı başlat
            series = self.get_current_series()
            if series:
                interval = int(1000 / series.frame_rate * (100 / self.speed_slider.value()))
                self._playback_timer.setInterval(interval)
                self._playback_timer.start()
        else:
            self.play_pause_btn.setText("Oynat")
            self.play_pause_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            self._playback_timer.stop()
            
        self.playback_toggled.emit(checked)
        
    def _on_stop_clicked(self):
        """Durdur."""
        self.play_pause_btn.setChecked(False)
        self.frame_slider.setValue(0)
        
    def _on_speed_changed(self, value: int):
        """Hız değişti."""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.1f}x")
        
        # Timer intervalini güncelle
        if self._is_playing:
            series = self.get_current_series()
            if series:
                interval = int(1000 / series.frame_rate * (100 / value))
                self._playback_timer.setInterval(interval)
                
    def _on_playback_timer(self):
        """Oynatma timer'ı."""
        current = self.frame_slider.value()
        maximum = self.frame_slider.maximum()
        
        if current < maximum:
            self.frame_slider.setValue(current + 1)
        else:
            if self.loop_checkbox.isChecked():
                self.frame_slider.setValue(0)
            else:
                self.play_pause_btn.setChecked(False)
                
    def _on_recent_file_selected(self, file_path: Path):
        """Son dosya seçildi."""
        # Bu viewer tarafından işlenecek
        pass