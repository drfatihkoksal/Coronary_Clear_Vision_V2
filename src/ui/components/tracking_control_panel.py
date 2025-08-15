"""
Tracking Control Panel Component

İzleme kontrollerini içeren panel.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Dict, Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QColor
import logging

from src.domain.models.tracking_models import TrackingParameters, TrackedPoint, TrackingStatus

logger = logging.getLogger(__name__)


class TrackingControlPanel(QWidget):
    """
    İzleme kontrol paneli.

    Bu panel:
    - İzleme parametreleri ayarları
    - Nokta ekleme/çıkarma
    - İzleme başlatma/durdurma
    - Nokta listesi yönetimi

    Signals:
        add_point_requested: Nokta ekleme istendi
        remove_point_requested: Nokta silme istendi (point_id)
        tracking_started: İzleme başlatıldı
        tracking_stopped: İzleme durduruldu
        parameters_changed: Parametreler değişti
        point_selected: Nokta seçildi (point_id)
    """

    # Signals
    add_point_requested = pyqtSignal()
    remove_point_requested = pyqtSignal(str)  # point_id
    tracking_started = pyqtSignal()
    tracking_stopped = pyqtSignal()
    parameters_changed = pyqtSignal(TrackingParameters)
    point_selected = pyqtSignal(str)  # point_id

    def __init__(self, parent: Optional[QWidget] = None):
        """
        TrackingControlPanel constructor.

        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self._tracked_points: Dict[str, TrackedPoint] = {}
        self._is_tracking = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Nokta yönetimi
        self._create_point_management_group(layout)

        # İzleme parametreleri
        self._create_parameters_group(layout)

        # Nokta listesi
        self._create_points_table(layout)

        # Kontrol butonları
        self._create_control_buttons(layout)

        # Spacer
        layout.addStretch()

    def _create_point_management_group(self, parent_layout: QVBoxLayout):
        """Nokta yönetimi grubu oluştur."""
        group = QGroupBox("Nokta Yönetimi")
        layout = QHBoxLayout()

        # Nokta ekle butonu
        self.add_point_btn = QPushButton("Nokta Ekle")
        self.add_point_btn.setIcon(QIcon.fromTheme("list-add"))
        layout.addWidget(self.add_point_btn)

        # Nokta sil butonu
        self.remove_point_btn = QPushButton("Nokta Sil")
        self.remove_point_btn.setIcon(QIcon.fromTheme("list-remove"))
        self.remove_point_btn.setEnabled(False)
        layout.addWidget(self.remove_point_btn)

        # Tümünü temizle
        self.clear_all_btn = QPushButton("Tümünü Temizle")
        self.clear_all_btn.setIcon(QIcon.fromTheme("edit-clear"))
        layout.addWidget(self.clear_all_btn)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _create_parameters_group(self, parent_layout: QVBoxLayout):
        """İzleme parametreleri grubu oluştur."""
        group = QGroupBox("İzleme Parametreleri")
        layout = QVBoxLayout()

        # İzleme yöntemi
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Yöntem:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["Şablon Eşleme", "Optik Akış", "Kalman Filtresi", "Parçacık Filtresi"]
        )
        self.method_combo.setToolTip("İzleme algoritması seçimi")
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # Şablon boyutu
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Şablon Boyutu:"))
        self.template_size_spin = QSpinBox()
        self.template_size_spin.setRange(11, 51)
        self.template_size_spin.setSingleStep(2)
        self.template_size_spin.setValue(21)
        self.template_size_spin.setSuffix(" px")
        self.template_size_spin.setToolTip("İzleme şablonu pencere boyutu")
        template_layout.addWidget(self.template_size_spin)
        layout.addLayout(template_layout)

        # Arama yarıçapı
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Arama Yarıçapı:"))
        self.search_radius_spin = QSpinBox()
        self.search_radius_spin.setRange(10, 100)
        self.search_radius_spin.setValue(30)
        self.search_radius_spin.setSuffix(" px")
        self.search_radius_spin.setToolTip("Maksimum hareket mesafesi")
        search_layout.addWidget(self.search_radius_spin)
        layout.addLayout(search_layout)

        # Güven eşiği
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Güven Eşiği:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setToolTip("Minimum kabul edilebilir güven skoru")
        confidence_layout.addWidget(self.confidence_spin)
        layout.addLayout(confidence_layout)

        # Gelişmiş seçenekler
        advanced_layout = QVBoxLayout()

        self.adaptive_template_check = QCheckBox("Adaptif Şablon")
        self.adaptive_template_check.setChecked(True)
        self.adaptive_template_check.setToolTip("Şablonu otomatik güncelle")
        advanced_layout.addWidget(self.adaptive_template_check)

        self.motion_prediction_check = QCheckBox("Hareket Tahmini")
        self.motion_prediction_check.setChecked(True)
        self.motion_prediction_check.setToolTip("Gelecek pozisyonları tahmin et")
        advanced_layout.addWidget(self.motion_prediction_check)

        layout.addLayout(advanced_layout)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _create_points_table(self, parent_layout: QVBoxLayout):
        """İzlenen noktalar tablosu oluştur."""
        group = QGroupBox("İzlenen Noktalar")
        layout = QVBoxLayout()

        # Tablo
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(5)
        self.points_table.setHorizontalHeaderLabels(["ID", "İsim", "Durum", "Güven", "Frame"])

        # Sütun genişlikleri
        header = self.points_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.points_table.setColumnWidth(0, 50)
        self.points_table.setColumnWidth(1, 100)
        self.points_table.setColumnWidth(2, 80)
        self.points_table.setColumnWidth(3, 60)
        self.points_table.setColumnWidth(4, 60)

        # Seçim modu
        self.points_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.points_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        layout.addWidget(self.points_table)

        # İstatistikler
        stats_layout = QHBoxLayout()
        self.total_points_label = QLabel("Toplam: 0")
        self.active_points_label = QLabel("Aktif: 0")
        self.lost_points_label = QLabel("Kayıp: 0")

        stats_layout.addWidget(self.total_points_label)
        stats_layout.addWidget(self.active_points_label)
        stats_layout.addWidget(self.lost_points_label)
        stats_layout.addStretch()

        layout.addLayout(stats_layout)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _create_control_buttons(self, parent_layout: QVBoxLayout):
        """Kontrol butonları oluştur."""
        layout = QHBoxLayout()

        # Başlat/Durdur butonu
        self.start_stop_btn = QPushButton("İzlemeyi Başlat")
        self.start_stop_btn.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_stop_btn.setEnabled(False)
        layout.addWidget(self.start_stop_btn)

        # Sıfırla butonu
        self.reset_btn = QPushButton("Sıfırla")
        self.reset_btn.setIcon(QIcon.fromTheme("view-refresh"))
        layout.addWidget(self.reset_btn)

        parent_layout.addLayout(layout)

    def _connect_signals(self):
        """Sinyal bağlantılarını yap."""
        # Butonlar
        self.add_point_btn.clicked.connect(self._on_add_point_clicked)
        self.remove_point_btn.clicked.connect(self._on_remove_point_clicked)
        self.clear_all_btn.clicked.connect(self._on_clear_all_clicked)
        self.start_stop_btn.clicked.connect(self._on_start_stop_clicked)
        self.reset_btn.clicked.connect(self._on_reset_clicked)

        # Parametre değişimleri
        self.method_combo.currentIndexChanged.connect(self._on_parameters_changed)
        self.template_size_spin.valueChanged.connect(self._on_parameters_changed)
        self.search_radius_spin.valueChanged.connect(self._on_parameters_changed)
        self.confidence_spin.valueChanged.connect(self._on_parameters_changed)
        self.adaptive_template_check.toggled.connect(self._on_parameters_changed)
        self.motion_prediction_check.toggled.connect(self._on_parameters_changed)

        # Tablo seçimi
        self.points_table.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def _on_add_point_clicked(self):
        """Nokta ekle butonu tıklandı."""
        if not self._is_tracking:
            self.add_point_requested.emit()

    def _on_remove_point_clicked(self):
        """Nokta sil butonu tıklandı."""
        selected_row = self.points_table.currentRow()
        if selected_row >= 0:
            point_id = self.points_table.item(selected_row, 0).text()
            self.remove_point_requested.emit(point_id)

    def _on_clear_all_clicked(self):
        """Tümünü temizle butonu tıklandı."""
        if self._tracked_points:
            reply = QMessageBox.question(
                self,
                "Tümünü Temizle",
                "Tüm noktalar silinecek. Emin misiniz?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                for point_id in list(self._tracked_points.keys()):
                    self.remove_point_requested.emit(point_id)

    def _on_start_stop_clicked(self):
        """Başlat/Durdur butonu tıklandı."""
        if self._is_tracking:
            self.tracking_stopped.emit()
        else:
            self.tracking_started.emit()

    def _on_reset_clicked(self):
        """Sıfırla butonu tıklandı."""
        if self._is_tracking:
            self.tracking_stopped.emit()

        # Nokta geçmişlerini sıfırla
        for point in self._tracked_points.values():
            point.history.clear()
            point.frame_number = 0
            point.status = TrackingStatus.MANUAL

        self._update_table()

    def _on_parameters_changed(self):
        """Parametreler değişti."""
        params = self.get_parameters()
        self.parameters_changed.emit(params)

    def _on_selection_changed(self):
        """Tablo seçimi değişti."""
        selected_row = self.points_table.currentRow()
        if selected_row >= 0:
            point_id = self.points_table.item(selected_row, 0).text()
            self.point_selected.emit(point_id)
            self.remove_point_btn.setEnabled(True)
        else:
            self.remove_point_btn.setEnabled(False)

    def add_tracked_point(self, point: TrackedPoint):
        """
        İzlenen nokta ekle.

        Args:
            point: İzlenecek nokta
        """
        self._tracked_points[point.id] = point
        self._update_table()
        self._update_button_states()

    def remove_tracked_point(self, point_id: str):
        """
        İzlenen nokta kaldır.

        Args:
            point_id: Nokta ID'si
        """
        if point_id in self._tracked_points:
            del self._tracked_points[point_id]
            self._update_table()
            self._update_button_states()

    def update_tracked_point(self, point: TrackedPoint):
        """
        İzlenen nokta güncelle.

        Args:
            point: Güncellenmiş nokta
        """
        if point.id in self._tracked_points:
            self._tracked_points[point.id] = point
            self._update_table_row(point)

    def get_tracked_points(self) -> List[TrackedPoint]:
        """
        Tüm izlenen noktaları döndür.

        Returns:
            List[TrackedPoint]: İzlenen noktalar
        """
        return list(self._tracked_points.values())

    def get_parameters(self) -> TrackingParameters:
        """
        Mevcut parametreleri döndür.

        Returns:
            TrackingParameters: İzleme parametreleri
        """
        # Tek sayı kontrolü
        template_size = self.template_size_spin.value()
        if template_size % 2 == 0:
            template_size += 1

        return TrackingParameters(
            template_size=template_size,
            search_radius=self.search_radius_spin.value(),
            confidence_threshold=self.confidence_spin.value(),
            adaptive_template=self.adaptive_template_check.isChecked(),
            enable_prediction=self.motion_prediction_check.isChecked(),
        )

    def set_tracking_state(self, is_tracking: bool):
        """
        İzleme durumunu ayarla.

        Args:
            is_tracking: İzleme aktif mi?
        """
        self._is_tracking = is_tracking

        if is_tracking:
            self.start_stop_btn.setText("İzlemeyi Durdur")
            self.start_stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
            self.add_point_btn.setEnabled(False)
            self.clear_all_btn.setEnabled(False)
        else:
            self.start_stop_btn.setText("İzlemeyi Başlat")
            self.start_stop_btn.setIcon(QIcon.fromTheme("media-playback-start"))
            self.add_point_btn.setEnabled(True)
            self.clear_all_btn.setEnabled(True)

        self._update_button_states()

    def _update_table(self):
        """Tabloyu güncelle."""
        self.points_table.setRowCount(len(self._tracked_points))

        for row, point in enumerate(self._tracked_points.values()):
            self._update_table_row(point, row)

        self._update_statistics()

    def _update_table_row(self, point: TrackedPoint, row: Optional[int] = None):
        """
        Tablo satırını güncelle.

        Args:
            point: İzlenen nokta
            row: Satır numarası
        """
        if row is None:
            # Satırı bul
            for r in range(self.points_table.rowCount()):
                if self.points_table.item(r, 0).text() == point.id:
                    row = r
                    break
            else:
                return

        # ID
        id_item = QTableWidgetItem(point.id)
        id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.points_table.setItem(row, 0, id_item)

        # İsim
        name_item = QTableWidgetItem(point.name)
        self.points_table.setItem(row, 1, name_item)

        # Durum
        status_item = QTableWidgetItem(self._get_status_text(point.status))
        status_item.setForeground(self._get_status_color(point.status))
        status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.points_table.setItem(row, 2, status_item)

        # Güven
        confidence_text = f"{point.confidence:.2f}" if point.confidence > 0 else "-"
        confidence_item = QTableWidgetItem(confidence_text)
        confidence_item.setFlags(confidence_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.points_table.setItem(row, 3, confidence_item)

        # Frame
        frame_item = QTableWidgetItem(str(point.frame_number))
        frame_item.setFlags(frame_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.points_table.setItem(row, 4, frame_item)

        self._update_statistics()

    def _update_statistics(self):
        """İstatistikleri güncelle."""
        total = len(self._tracked_points)
        active = sum(1 for p in self._tracked_points.values() if p.status == TrackingStatus.ACTIVE)
        lost = sum(1 for p in self._tracked_points.values() if p.status == TrackingStatus.LOST)

        self.total_points_label.setText(f"Toplam: {total}")
        self.active_points_label.setText(f"Aktif: {active}")
        self.lost_points_label.setText(f"Kayıp: {lost}")

    def _update_button_states(self):
        """Buton durumlarını güncelle."""
        has_points = len(self._tracked_points) > 0
        self.start_stop_btn.setEnabled(has_points)
        self.reset_btn.setEnabled(has_points and not self._is_tracking)

    def _get_status_text(self, status: TrackingStatus) -> str:
        """Durum metni döndür."""
        status_texts = {
            TrackingStatus.ACTIVE: "Aktif",
            TrackingStatus.LOST: "Kayıp",
            TrackingStatus.OCCLUDED: "Gizli",
            TrackingStatus.MANUAL: "Manuel",
            TrackingStatus.PREDICTED: "Tahmin",
            TrackingStatus.VALIDATED: "Doğrulandı",
        }
        return status_texts.get(status, "Bilinmiyor")

    def _get_status_color(self, status: TrackingStatus) -> QColor:
        """Durum rengi döndür."""
        status_colors = {
            TrackingStatus.ACTIVE: QColor(0, 150, 0),
            TrackingStatus.LOST: QColor(200, 0, 0),
            TrackingStatus.OCCLUDED: QColor(200, 100, 0),
            TrackingStatus.MANUAL: QColor(0, 0, 200),
            TrackingStatus.PREDICTED: QColor(150, 150, 0),
            TrackingStatus.VALIDATED: QColor(0, 100, 0),
        }
        return status_colors.get(status, QColor(0, 0, 0))
