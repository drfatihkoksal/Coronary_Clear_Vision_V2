"""
DICOM Info Display Component

DICOM metadata ve bilgilerini gösteren bileşen.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import Optional, Dict
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont, QIcon
import logging

from src.domain.models.dicom_models import DicomStudy, DicomSeries

logger = logging.getLogger(__name__)


class DicomInfoDisplay(QWidget):
    """
    DICOM bilgi gösterimi.

    Bu widget:
    - Hasta bilgileri
    - Çalışma bilgileri
    - Seri bilgileri
    - Teknik parametreler
    - DICOM tag'leri

    Signals:
        tag_selected: DICOM tag seçildi (tag_name, tag_value)
        copy_requested: Kopyalama istendi
    """

    # Signals
    tag_selected = pyqtSignal(str, str)  # tag_name, tag_value
    copy_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """
        DicomInfoDisplay constructor.

        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self._current_study: Optional[DicomStudy] = None
        self._current_series: Optional[DicomSeries] = None
        self._setup_ui()

    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Genel bilgiler sekmesi
        self.general_tab = self._create_general_tab()
        self.tab_widget.addTab(self.general_tab, "Genel Bilgiler")

        # Teknik bilgiler sekmesi
        self.technical_tab = self._create_technical_tab()
        self.tab_widget.addTab(self.technical_tab, "Teknik Bilgiler")

        # DICOM tag'leri sekmesi
        self.tags_tab = self._create_tags_tab()
        self.tab_widget.addTab(self.tags_tab, "DICOM Tag'leri")

        # İstatistikler sekmesi
        self.stats_tab = self._create_stats_tab()
        self.tab_widget.addTab(self.stats_tab, "İstatistikler")

        layout.addWidget(self.tab_widget)

    def _create_general_tab(self) -> QWidget:
        """Genel bilgiler sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Hasta bilgileri
        patient_group = QGroupBox("Hasta Bilgileri")
        patient_layout = QVBoxLayout()

        self.patient_id_label = QLabel("ID: -")
        self.patient_name_label = QLabel("Ad: -")
        self.patient_age_label = QLabel("Yaş: -")
        self.patient_sex_label = QLabel("Cinsiyet: -")

        patient_layout.addWidget(self.patient_id_label)
        patient_layout.addWidget(self.patient_name_label)
        patient_layout.addWidget(self.patient_age_label)
        patient_layout.addWidget(self.patient_sex_label)

        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)

        # Çalışma bilgileri
        study_group = QGroupBox("Çalışma Bilgileri")
        study_layout = QVBoxLayout()

        self.study_uid_label = QLabel("UID: -")
        self.study_uid_label.setWordWrap(True)
        self.study_date_label = QLabel("Tarih: -")
        self.study_time_label = QLabel("Saat: -")
        self.study_desc_label = QLabel("Açıklama: -")
        self.accession_label = QLabel("Erişim No: -")
        self.referring_physician_label = QLabel("Yönlendiren: -")

        study_layout.addWidget(self.study_uid_label)
        study_layout.addWidget(self.study_date_label)
        study_layout.addWidget(self.study_time_label)
        study_layout.addWidget(self.study_desc_label)
        study_layout.addWidget(self.accession_label)
        study_layout.addWidget(self.referring_physician_label)

        study_group.setLayout(study_layout)
        layout.addWidget(study_group)

        # Seri bilgileri
        series_group = QGroupBox("Seri Bilgileri")
        series_layout = QVBoxLayout()

        self.series_table = QTableWidget()
        self.series_table.setColumnCount(5)
        self.series_table.setHorizontalHeaderLabels(
            ["No", "Açıklama", "Modalite", "Frame", "Projeksiyon"]
        )

        # Sütun genişlikleri
        header = self.series_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.series_table.setColumnWidth(0, 40)
        self.series_table.setColumnWidth(1, 150)
        self.series_table.setColumnWidth(2, 60)
        self.series_table.setColumnWidth(3, 60)

        series_layout.addWidget(self.series_table)
        series_group.setLayout(series_layout)
        layout.addWidget(series_group)

        layout.addStretch()

        return widget

    def _create_technical_tab(self) -> QWidget:
        """Teknik bilgiler sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Görüntü parametreleri
        image_group = QGroupBox("Görüntü Parametreleri")
        image_layout = QVBoxLayout()

        self.image_table = QTableWidget()
        self.image_table.setColumnCount(2)
        self.image_table.setHorizontalHeaderLabels(["Parametre", "Değer"])
        self.image_table.horizontalHeader().setStretchLastSection(True)

        image_layout.addWidget(self.image_table)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Projeksiyon parametreleri
        projection_group = QGroupBox("Projeksiyon Parametreleri")
        projection_layout = QVBoxLayout()

        self.projection_table = QTableWidget()
        self.projection_table.setColumnCount(2)
        self.projection_table.setHorizontalHeaderLabels(["Parametre", "Değer"])
        self.projection_table.horizontalHeader().setStretchLastSection(True)

        projection_layout.addWidget(self.projection_table)
        projection_group.setLayout(projection_layout)
        layout.addWidget(projection_group)

        # Cihaz bilgileri
        equipment_group = QGroupBox("Cihaz Bilgileri")
        equipment_layout = QVBoxLayout()

        self.equipment_text = QTextEdit()
        self.equipment_text.setReadOnly(True)
        self.equipment_text.setMaximumHeight(100)

        equipment_layout.addWidget(self.equipment_text)
        equipment_group.setLayout(equipment_layout)
        layout.addWidget(equipment_group)

        layout.addStretch()

        return widget

    def _create_tags_tab(self) -> QWidget:
        """DICOM tag'leri sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Arama
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Ara:"))
        # TODO: Arama kutusu eklenebilir
        layout.addLayout(search_layout)

        # Tag ağacı
        self.tags_tree = QTreeWidget()
        self.tags_tree.setHeaderLabels(["Tag", "VR", "Değer"])
        self.tags_tree.setAlternatingRowColors(True)

        # Sütun genişlikleri
        self.tags_tree.header().resizeSection(0, 200)
        self.tags_tree.header().resizeSection(1, 50)

        layout.addWidget(self.tags_tree)

        # Kopyala butonu
        copy_layout = QHBoxLayout()
        copy_layout.addStretch()
        self.copy_tags_btn = QPushButton("Seçili Tag'i Kopyala")
        self.copy_tags_btn.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_tags_btn.clicked.connect(self.copy_requested.emit)
        copy_layout.addWidget(self.copy_tags_btn)

        layout.addLayout(copy_layout)

        return widget

    def _create_stats_tab(self) -> QWidget:
        """İstatistikler sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Piksel istatistikleri
        pixel_group = QGroupBox("Piksel İstatistikleri")
        pixel_layout = QVBoxLayout()

        self.pixel_stats_table = QTableWidget()
        self.pixel_stats_table.setColumnCount(2)
        self.pixel_stats_table.setHorizontalHeaderLabels(["İstatistik", "Değer"])
        self.pixel_stats_table.horizontalHeader().setStretchLastSection(True)

        pixel_layout.addWidget(self.pixel_stats_table)
        pixel_group.setLayout(pixel_layout)
        layout.addWidget(pixel_group)

        # Histogram
        histogram_group = QGroupBox("Histogram Bilgileri")
        histogram_layout = QVBoxLayout()

        self.histogram_info = QTextEdit()
        self.histogram_info.setReadOnly(True)
        self.histogram_info.setMaximumHeight(150)

        histogram_layout.addWidget(self.histogram_info)
        histogram_group.setLayout(histogram_layout)
        layout.addWidget(histogram_group)

        # Dosya bilgileri
        file_group = QGroupBox("Dosya Bilgileri")
        file_layout = QVBoxLayout()

        self.file_info_text = QTextEdit()
        self.file_info_text.setReadOnly(True)
        self.file_info_text.setMaximumHeight(100)
        self.file_info_text.setFont(QFont("Monospace", 9))

        file_layout.addWidget(self.file_info_text)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        layout.addStretch()

        return widget

    def set_study(self, study: DicomStudy):
        """
        DICOM çalışmasını ayarla.

        Args:
            study: DICOM çalışması
        """
        self._current_study = study
        self._update_general_info()
        self._update_series_table()
        self._update_file_info()

    def set_current_series(self, series: DicomSeries):
        """
        Mevcut seriyi ayarla.

        Args:
            series: DICOM serisi
        """
        self._current_series = series
        self._update_technical_info()
        self._update_tags_tree()

    def update_pixel_statistics(self, stats: Dict[str, float]):
        """
        Piksel istatistiklerini güncelle.

        Args:
            stats: İstatistik sözlüğü
        """
        self.pixel_stats_table.setRowCount(len(stats))

        for i, (key, value) in enumerate(stats.items()):
            # İstatistik adı
            stat_name = {
                "min": "Minimum",
                "max": "Maksimum",
                "mean": "Ortalama",
                "std": "Standart Sapma",
                "median": "Medyan",
                "percentile_5": "%5 Yüzdelik",
                "percentile_95": "%95 Yüzdelik",
            }.get(key, key)

            self.pixel_stats_table.setItem(i, 0, QTableWidgetItem(stat_name))
            self.pixel_stats_table.setItem(i, 1, QTableWidgetItem(f"{value:.2f}"))

    def _update_general_info(self):
        """Genel bilgileri güncelle."""
        if not self._current_study:
            return

        # Hasta bilgileri
        patient = self._current_study.patient_info
        self.patient_id_label.setText(f"ID: {patient.patient_id}")
        self.patient_name_label.setText(f"Ad: {patient.patient_name}")
        self.patient_age_label.setText(f"Yaş: {patient.age or '-'}")

        sex_text = {"M": "Erkek", "F": "Kadın", "O": "Diğer"}.get(patient.sex, patient.sex)
        self.patient_sex_label.setText(f"Cinsiyet: {sex_text}")

        # Çalışma bilgileri
        study = self._current_study.study_info
        self.study_uid_label.setText(f"UID: {study.study_instance_uid}")

        if study.study_date:
            self.study_date_label.setText(f"Tarih: {study.study_date.strftime('%d.%m.%Y')}")
        else:
            self.study_date_label.setText("Tarih: -")

        self.study_time_label.setText(f"Saat: {study.study_time or '-'}")
        self.study_desc_label.setText(f"Açıklama: {study.study_description or '-'}")
        self.accession_label.setText(f"Erişim No: {study.accession_number or '-'}")
        self.referring_physician_label.setText(f"Yönlendiren: {study.referring_physician or '-'}")

    def _update_series_table(self):
        """Seri tablosunu güncelle."""
        if not self._current_study:
            self.series_table.setRowCount(0)
            return

        self.series_table.setRowCount(self._current_study.num_series)

        for i, series in enumerate(self._current_study.series_list):
            # Seri no
            self.series_table.setItem(i, 0, QTableWidgetItem(str(series.info.series_number)))

            # Açıklama
            desc = series.info.series_description or f"Seri {i+1}"
            self.series_table.setItem(i, 1, QTableWidgetItem(desc))

            # Modalite
            self.series_table.setItem(i, 2, QTableWidgetItem(series.info.modality.value))

            # Frame sayısı
            self.series_table.setItem(i, 3, QTableWidgetItem(str(series.num_frames)))

            # Projeksiyon
            if series.projection_info:
                proj = series.projection_info.angle_description
            else:
                proj = "-"
            self.series_table.setItem(i, 4, QTableWidgetItem(proj))

    def _update_technical_info(self):
        """Teknik bilgileri güncelle."""
        if not self._current_series:
            return

        # Görüntü parametreleri
        params = []

        # Boyutlar
        if self._current_series.frames:
            shape = self._current_series.frames[0].shape
            params.append(("Boyut", f"{shape[1]} x {shape[0]} piksel"))

        # Piksel aralığı
        if self._current_series.pixel_spacing:
            spacing = self._current_series.pixel_spacing
            params.append(("Piksel Aralığı", f"{spacing.average_spacing:.3f} mm/piksel"))
            params.append(("İzotropik", "Evet" if spacing.is_isotropic else "Hayır"))

        # Frame bilgileri
        params.append(("Frame Sayısı", str(self._current_series.num_frames)))
        params.append(("Frame Hızı", f"{self._current_series.frame_rate:.1f} fps"))
        params.append(("Süre", f"{self._current_series.duration:.1f} saniye"))

        # Pencere/seviye
        wl = self._current_series.window_level
        params.append(("Pencere Merkezi", str(int(wl.center))))
        params.append(("Pencere Genişliği", str(int(wl.width))))

        # Tabloya ekle
        self.image_table.setRowCount(len(params))
        for i, (param, value) in enumerate(params):
            self.image_table.setItem(i, 0, QTableWidgetItem(param))
            self.image_table.setItem(i, 1, QTableWidgetItem(value))

        # Projeksiyon parametreleri
        proj_params = []
        if self._current_series.projection_info:
            proj = self._current_series.projection_info
            proj_params.append(("Birincil Açı", f"{proj.primary_angle:.1f}°"))
            proj_params.append(("İkincil Açı", f"{proj.secondary_angle:.1f}°"))
            proj_params.append(("Açı Açıklaması", proj.angle_description))

            if proj.table_height is not None:
                proj_params.append(("Masa Yüksekliği", f"{proj.table_height:.1f} mm"))
            if proj.distance_source_to_detector is not None:
                proj_params.append(
                    ("Kaynak-Detektör", f"{proj.distance_source_to_detector:.1f} mm")
                )
            if proj.distance_source_to_patient is not None:
                proj_params.append(("Kaynak-Hasta", f"{proj.distance_source_to_patient:.1f} mm"))

        self.projection_table.setRowCount(len(proj_params))
        for i, (param, value) in enumerate(proj_params):
            self.projection_table.setItem(i, 0, QTableWidgetItem(param))
            self.projection_table.setItem(i, 1, QTableWidgetItem(value))

    def _update_tags_tree(self):
        """DICOM tag ağacını güncelle."""
        # TODO: Gerçek DICOM tag'lerini göster
        self.tags_tree.clear()

        # Örnek tag'ler
        if self._current_series:
            # Patient grubu
            patient_item = QTreeWidgetItem(["(0010,0000) Patient"])
            patient_item.addChild(
                QTreeWidgetItem(
                    [
                        "(0010,0010) PatientName",
                        "PN",
                        (
                            self._current_study.patient_info.patient_name
                            if self._current_study
                            else ""
                        ),
                    ]
                )
            )
            patient_item.addChild(
                QTreeWidgetItem(
                    [
                        "(0010,0020) PatientID",
                        "LO",
                        self._current_study.patient_info.patient_id if self._current_study else "",
                    ]
                )
            )
            self.tags_tree.addTopLevelItem(patient_item)

            # Study grubu
            study_item = QTreeWidgetItem(["(0020,0000) Study"])
            if self._current_study:
                study_item.addChild(
                    QTreeWidgetItem(
                        [
                            "(0020,000D) StudyInstanceUID",
                            "UI",
                            self._current_study.study_info.study_instance_uid,
                        ]
                    )
                )
            self.tags_tree.addTopLevelItem(study_item)

            # Series grubu
            series_item = QTreeWidgetItem(["(0020,0000) Series"])
            series_item.addChild(
                QTreeWidgetItem(
                    [
                        "(0020,000E) SeriesInstanceUID",
                        "UI",
                        self._current_series.info.series_instance_uid,
                    ]
                )
            )
            series_item.addChild(
                QTreeWidgetItem(
                    ["(0008,0060) Modality", "CS", self._current_series.info.modality.value]
                )
            )
            self.tags_tree.addTopLevelItem(series_item)

            # Tüm item'ları genişlet
            self.tags_tree.expandAll()

    def _update_file_info(self):
        """Dosya bilgilerini güncelle."""
        if not self._current_study:
            self.file_info_text.clear()
            return

        info_lines = []

        if self._current_study.file_path:
            info_lines.append(f"Dosya: {self._current_study.file_path.name}")
            info_lines.append(f"Yol: {self._current_study.file_path.parent}")

            # Dosya boyutu
            try:
                size = self._current_study.file_path.stat().st_size
                size_mb = size / (1024 * 1024)
                info_lines.append(f"Boyut: {size_mb:.1f} MB")
            except:
                pass

        info_lines.append(f"DICOMDIR: {'Evet' if self._current_study.is_dicomdir else 'Hayır'}")
        info_lines.append(f"Toplam Seri: {self._current_study.num_series}")
        info_lines.append(f"Toplam Frame: {self._current_study.total_frames}")

        self.file_info_text.setPlainText("\n".join(info_lines))

    def get_selected_tag(self) -> Optional[Tuple[str, str]]:
        """
        Seçili DICOM tag'ini döndür.

        Returns:
            Optional[Tuple[str, str]]: (tag_name, tag_value) veya None
        """
        current = self.tags_tree.currentItem()
        if current and current.columnCount() >= 3:
            tag_name = current.text(0)
            tag_value = current.text(2)
            return (tag_name, tag_value)
        return None
