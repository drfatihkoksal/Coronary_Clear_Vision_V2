"""
Segmentation Control Panel Component

Segmentasyon kontrolleri için UI bileşeni.
Single Responsibility: Sadece kullanıcı kontrollerini yönetir.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QComboBox, QCheckBox, QGroupBox,
                            QSlider, QSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt
from typing import Dict, Any

from src.domain.models.segmentation_models import SegmentationMethod


class SegmentationControlPanel(QWidget):
    """
    Segmentasyon kontrol paneli.
    
    Segmentasyon başlatma, yöntem seçimi ve parametre ayarlarını sağlar.
    Business logic içermez, sadece UI kontrollerini yönetir.
    
    Signals:
        segment_requested: Segmentasyon istendi
        refine_requested: İyileştirme istendi  
        clear_requested: Temizleme istendi
        undo_requested: Geri al istendi
        redo_requested: İleri al istendi
        method_changed: Yöntem değişti
        parameters_changed: Parametreler değişti
    """
    
    # Signals
    segment_requested = pyqtSignal()
    refine_requested = pyqtSignal(str)  # refine type
    clear_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    method_changed = pyqtSignal(SegmentationMethod)
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Control panel'i başlatır.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        self._current_method = SegmentationMethod.AI_ANGIOPY
        
    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Ana kontroller
        main_controls = self._create_main_controls()
        layout.addWidget(main_controls)
        
        # Yöntem seçimi
        method_group = self._create_method_selection()
        layout.addWidget(method_group)
        
        # Parametreler
        params_group = self._create_parameters_group()
        layout.addWidget(params_group)
        
        # İyileştirme kontrolleri
        refine_group = self._create_refinement_controls()
        layout.addWidget(refine_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def _create_main_controls(self) -> QGroupBox:
        """
        Ana kontrol butonlarını oluşturur.
        
        Returns:
            QGroupBox: Kontrol grubu
        """
        group = QGroupBox("Segmentation Controls")
        layout = QVBoxLayout()
        
        # Segmentasyon butonu
        self.segment_button = QPushButton("Start Segmentation")
        self.segment_button.setObjectName("segmentButton")
        self.segment_button.setMinimumHeight(40)
        self._apply_primary_button_style(self.segment_button)
        layout.addWidget(self.segment_button)
        
        # Alt kontroller
        sub_controls = QHBoxLayout()
        
        # Geri al
        self.undo_button = QPushButton("↶ Undo")
        self.undo_button.setObjectName("undoButton")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Undo last segmentation")
        sub_controls.addWidget(self.undo_button)
        
        # İleri al
        self.redo_button = QPushButton("↷ Redo")
        self.redo_button.setObjectName("redoButton")
        self.redo_button.setEnabled(False)
        self.redo_button.setToolTip("Redo segmentation")
        sub_controls.addWidget(self.redo_button)
        
        # Temizle
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setEnabled(False)
        self._apply_danger_button_style(self.clear_button)
        sub_controls.addWidget(self.clear_button)
        
        sub_controls.addStretch()
        layout.addLayout(sub_controls)
        
        group.setLayout(layout)
        return group
        
    def _create_method_selection(self) -> QGroupBox:
        """
        Segmentasyon yöntemi seçim grubunu oluşturur.
        
        Returns:
            QGroupBox: Yöntem grubu
        """
        group = QGroupBox("Segmentation Method")
        layout = QHBoxLayout()
        
        layout.addWidget(QLabel("Method:"))
        
        self.method_combo = QComboBox()
        self.method_combo.setObjectName("methodCombo")
        
        # Yöntemleri ekle
        method_items = [
            ("AI (AngioPy)", SegmentationMethod.AI_ANGIOPY),
            ("Traditional", SegmentationMethod.TRADITIONAL),
            ("Semi-Automatic", SegmentationMethod.SEMI_AUTOMATIC),
            ("Manual", SegmentationMethod.MANUAL),
            ("Hybrid", SegmentationMethod.HYBRID)
        ]
        
        for name, method in method_items:
            self.method_combo.addItem(name, method)
            
        # Varsayılan AI
        self.method_combo.setCurrentIndex(0)
        
        layout.addWidget(self.method_combo)
        layout.addStretch()
        
        group.setLayout(layout)
        return group
        
    def _create_parameters_group(self) -> QGroupBox:
        """
        Segmentasyon parametreleri grubunu oluşturur.
        
        Returns:
            QGroupBox: Parametreler grubu
        """
        group = QGroupBox("Parameters")
        layout = QVBoxLayout()
        
        # Post-processing seçenekleri
        self.post_process_cb = QCheckBox("Enable Post-Processing")
        self.post_process_cb.setChecked(True)
        self.post_process_cb.setObjectName("postProcessCheckbox")
        layout.addWidget(self.post_process_cb)
        
        self.extract_features_cb = QCheckBox("Extract Vessel Features")
        self.extract_features_cb.setChecked(True)
        self.extract_features_cb.setObjectName("extractFeaturesCheckbox")
        layout.addWidget(self.extract_features_cb)
        
        self.fill_holes_cb = QCheckBox("Fill Holes")
        self.fill_holes_cb.setChecked(True)
        self.fill_holes_cb.setObjectName("fillHolesCheckbox")
        layout.addWidget(self.fill_holes_cb)
        
        # Minimum damar boyutu
        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("Min Vessel Size:"))
        
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setObjectName("minSizeSpinBox")
        self.min_size_spin.setRange(0, 500)
        self.min_size_spin.setValue(50)
        self.min_size_spin.setSuffix(" pixels")
        min_size_layout.addWidget(self.min_size_spin)
        min_size_layout.addStretch()
        layout.addLayout(min_size_layout)
        
        # Yumuşatma iterasyonu
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothing:"))
        
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setObjectName("smoothingSlider")
        self.smoothing_slider.setRange(0, 5)
        self.smoothing_slider.setValue(2)
        self.smoothing_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.smoothing_slider.setTickInterval(1)
        smooth_layout.addWidget(self.smoothing_slider)
        
        self.smoothing_label = QLabel("2")
        self.smoothing_label.setMinimumWidth(20)
        smooth_layout.addWidget(self.smoothing_label)
        
        layout.addLayout(smooth_layout)
        
        group.setLayout(layout)
        return group
        
    def _create_refinement_controls(self) -> QGroupBox:
        """
        İyileştirme kontrollerini oluşturur.
        
        Returns:
            QGroupBox: İyileştirme grubu
        """
        group = QGroupBox("Refinement Tools")
        layout = QHBoxLayout()
        
        # Ekleme modu
        self.add_button = QPushButton("+ Add")
        self.add_button.setObjectName("addButton")
        self.add_button.setCheckable(True)
        self.add_button.setToolTip("Add to segmentation")
        self._apply_tool_button_style(self.add_button)
        layout.addWidget(self.add_button)
        
        # Çıkarma modu
        self.remove_button = QPushButton("- Remove")
        self.remove_button.setObjectName("removeButton")
        self.remove_button.setCheckable(True)
        self.remove_button.setToolTip("Remove from segmentation")
        self._apply_tool_button_style(self.remove_button)
        layout.addWidget(self.remove_button)
        
        # Fırça boyutu
        layout.addWidget(QLabel("Brush:"))
        
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setObjectName("brushSizeSpinBox")
        self.brush_size_spin.setRange(1, 50)
        self.brush_size_spin.setValue(10)
        self.brush_size_spin.setSuffix("px")
        layout.addWidget(self.brush_size_spin)
        
        layout.addStretch()
        
        group.setLayout(layout)
        group.setEnabled(False)  # Başlangıçta devre dışı
        self.refinement_group = group
        return group
        
    def _apply_primary_button_style(self, button: QPushButton):
        """Birincil buton stili uygular."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                border: none;
                font-size: 14px;
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
        
    def _apply_danger_button_style(self, button: QPushButton):
        """Tehlike butonu stili uygular."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton:disabled {
                background-color: #FFCDD2;
                color: #EF9A9A;
            }
        """)
        
    def _apply_tool_button_style(self, button: QPushButton):
        """Araç butonu stili uygular."""
        button.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5;
                color: #333;
                padding: 6px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #bbb;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                border-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #fafafa;
                color: #bbb;
                border-color: #eee;
            }
        """)
        
    def _connect_signals(self):
        """Signal bağlantılarını yapar."""
        # Ana kontroller
        self.segment_button.clicked.connect(self.segment_requested.emit)
        self.undo_button.clicked.connect(self.undo_requested.emit)
        self.redo_button.clicked.connect(self.redo_requested.emit)
        self.clear_button.clicked.connect(self.clear_requested.emit)
        
        # Yöntem değişimi
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        
        # Parametre değişimleri
        self.post_process_cb.stateChanged.connect(self._on_parameters_changed)
        self.extract_features_cb.stateChanged.connect(self._on_parameters_changed)
        self.fill_holes_cb.stateChanged.connect(self._on_parameters_changed)
        self.min_size_spin.valueChanged.connect(self._on_parameters_changed)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        
        # İyileştirme butonları
        self.add_button.toggled.connect(self._on_refine_mode_changed)
        self.remove_button.toggled.connect(self._on_refine_mode_changed)
        self.brush_size_spin.valueChanged.connect(self._on_parameters_changed)
        
    def _on_method_changed(self):
        """Yöntem değiştiğinde çağrılır."""
        method = self.method_combo.currentData()
        if method:
            self._current_method = method
            self.method_changed.emit(method)
            
            # Yönteme göre UI ayarları
            if method == SegmentationMethod.MANUAL:
                self.post_process_cb.setEnabled(False)
                self.extract_features_cb.setEnabled(False)
            else:
                self.post_process_cb.setEnabled(True)
                self.extract_features_cb.setEnabled(True)
                
    def _on_parameters_changed(self):
        """Parametreler değiştiğinde çağrılır."""
        params = self.get_parameters()
        self.parameters_changed.emit(params)
        
    def _on_smoothing_changed(self, value: int):
        """Yumuşatma değeri değiştiğinde çağrılır."""
        self.smoothing_label.setText(str(value))
        self._on_parameters_changed()
        
    def _on_refine_mode_changed(self):
        """İyileştirme modu değiştiğinde çağrılır."""
        if self.add_button.isChecked():
            self.remove_button.setChecked(False)
            self.refine_requested.emit('add')
        elif self.remove_button.isChecked():
            self.add_button.setChecked(False)
            self.refine_requested.emit('remove')
        else:
            self.refine_requested.emit('none')
            
    def get_parameters(self) -> Dict[str, Any]:
        """
        Mevcut parametreleri döndürür.
        
        Returns:
            Dict[str, Any]: Segmentasyon parametreleri
        """
        return {
            'post_process': self.post_process_cb.isChecked(),
            'extract_features': self.extract_features_cb.isChecked(),
            'fill_holes': self.fill_holes_cb.isChecked(),
            'min_vessel_size': self.min_size_spin.value(),
            'smoothing_iterations': self.smoothing_slider.value(),
            'brush_size': self.brush_size_spin.value()
        }
        
    def get_selected_method(self) -> SegmentationMethod:
        """
        Seçili segmentasyon yöntemini döndürür.
        
        Returns:
            SegmentationMethod: Seçili yöntem
        """
        return self._current_method
        
    def set_segmentation_active(self, active: bool):
        """
        Segmentasyon durumunu ayarlar.
        
        Args:
            active: Segmentasyon aktif mi?
        """
        self.segment_button.setEnabled(not active)
        self.method_combo.setEnabled(not active)
        
        if active:
            self.segment_button.setText("Segmenting...")
        else:
            self.segment_button.setText("Start Segmentation")
            
    def set_refinement_enabled(self, enabled: bool):
        """
        İyileştirme kontrollerini etkinleştirir/devre dışı bırakır.
        
        Args:
            enabled: Etkin mi?
        """
        self.refinement_group.setEnabled(enabled)
        
    def set_undo_enabled(self, enabled: bool):
        """Undo butonunu etkinleştirir/devre dışı bırakır."""
        self.undo_button.setEnabled(enabled)
        
    def set_redo_enabled(self, enabled: bool):
        """Redo butonunu etkinleştirir/devre dışı bırakır."""
        self.redo_button.setEnabled(enabled)
        
    def set_clear_enabled(self, enabled: bool):
        """Clear butonunu etkinleştirir/devre dışı bırakır."""
        self.clear_button.setEnabled(enabled)