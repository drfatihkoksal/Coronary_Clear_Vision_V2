"""
Calibration Control Panel Component

Kalibrasyon kontrolleri için UI bileşeni.
Single Responsibility: Sadece kullanıcı kontrollerini yönetir.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import pyqtSignal
from typing import Optional

from src.domain.models.calibration_models import CalibrationMethod, CatheterSize


class CalibrationControlPanel(QWidget):
    """
    Kalibrasyon kontrol paneli.

    Kalibrasyon yöntemi seçimi ve başlatma kontrollerini sağlar.
    Business logic içermez, sadece UI kontrollerini yönetir.

    Signals:
        calibration_requested: Kalibrasyon istendi (method)
        method_changed: Kalibrasyon yöntemi değişti
        catheter_size_changed: Kateter boyutu değişti
    """

    # Signals
    calibration_requested = pyqtSignal(CalibrationMethod)
    method_changed = pyqtSignal(CalibrationMethod)
    catheter_size_changed = pyqtSignal(CatheterSize)

    def __init__(self, parent=None):
        """
        Control panel'i başlatır.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        self._current_method = CalibrationMethod.CATHETER

    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Başlık
        title = QLabel("Calibration Settings")
        title.setObjectName("calibrationTitle")
        title.setStyleSheet(
            """
            QLabel#calibrationTitle {
                font-size: 14px;
                font-weight: bold;
                color: #1565C0;
                padding: 5px;
            }
        """
        )
        layout.addWidget(title)

        # Yöntem seçimi
        method_group = self._create_method_selection()
        layout.addWidget(method_group)

        # Kateter boyutu seçimi
        self.catheter_group = self._create_catheter_size_selection()
        layout.addWidget(self.catheter_group)

        # Kontrol butonları
        control_layout = self._create_control_buttons()
        layout.addLayout(control_layout)

        # Bilgi metni
        self.info_label = QLabel("")
        self.info_label.setObjectName("calibrationInfo")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            """
            QLabel#calibrationInfo {
                font-size: 11px;
                color: #666;
                padding: 5px;
                background-color: #f5f5f5;
                border-radius: 3px;
                min-height: 40px;
            }
        """
        )
        self._update_info_text()
        layout.addWidget(self.info_label)

        layout.addStretch()
        self.setLayout(layout)

    def _create_method_selection(self) -> QGroupBox:
        """
        Kalibrasyon yöntemi seçim grubunu oluşturur.

        Returns:
            QGroupBox: Yöntem seçim grubu
        """
        group = QGroupBox("Calibration Method")
        layout = QVBoxLayout()

        # Radio butonları için grup
        self.method_group = QButtonGroup()

        # Kateter kalibrasyonu
        self.catheter_radio = QRadioButton("Catheter Calibration")
        self.catheter_radio.setObjectName("catheterRadio")
        self.catheter_radio.setChecked(True)
        self.catheter_radio.setToolTip("Automatic calibration using catheter diameter")
        self.method_group.addButton(self.catheter_radio, 0)
        layout.addWidget(self.catheter_radio)

        # Manuel kalibrasyon
        self.manual_radio = QRadioButton("Manual Calibration")
        self.manual_radio.setObjectName("manualRadio")
        self.manual_radio.setToolTip("Manual calibration by selecting two points")
        self.method_group.addButton(self.manual_radio, 1)
        layout.addWidget(self.manual_radio)

        # DICOM metadata (devre dışı)
        self.dicom_radio = QRadioButton("DICOM Metadata")
        self.dicom_radio.setObjectName("dicomRadio")
        self.dicom_radio.setEnabled(False)
        self.dicom_radio.setToolTip("Extract calibration from DICOM metadata (if available)")
        self.method_group.addButton(self.dicom_radio, 2)
        layout.addWidget(self.dicom_radio)

        group.setLayout(layout)
        return group

    def _create_catheter_size_selection(self) -> QGroupBox:
        """
        Kateter boyutu seçim grubunu oluşturur.

        Returns:
            QGroupBox: Kateter boyutu grubu
        """
        group = QGroupBox("Catheter Size")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Size:"))

        self.catheter_size_combo = QComboBox()
        self.catheter_size_combo.setObjectName("catheterSizeCombo")

        # Kateter boyutlarını ekle
        for size in CatheterSize:
            self.catheter_size_combo.addItem(str(size), size)

        # Varsayılan olarak 6F seç
        default_index = self.catheter_size_combo.findText("6F (2.00mm)")
        if default_index >= 0:
            self.catheter_size_combo.setCurrentIndex(default_index)

        layout.addWidget(self.catheter_size_combo)
        layout.addStretch()

        group.setLayout(layout)
        return group

    def _create_control_buttons(self) -> QHBoxLayout:
        """
        Kontrol butonlarını oluşturur.

        Returns:
            QHBoxLayout: Buton layout'u
        """
        layout = QHBoxLayout()

        # Kalibrasyon başlat butonu
        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.setObjectName("calibrateButton")
        self.calibrate_button.setMinimumHeight(35)
        self._apply_primary_button_style(self.calibrate_button)
        layout.addWidget(self.calibrate_button)

        # Kalibrasyon temizle butonu
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("clearCalibrationButton")
        self.clear_button.setEnabled(False)
        self._apply_secondary_button_style(self.clear_button)
        layout.addWidget(self.clear_button)

        layout.addStretch()

        return layout

    def _apply_primary_button_style(self, button: QPushButton):
        """Birincil buton stili uygular."""
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #1976D2;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BBBBBB;
                color: #666666;
            }
        """
        )

    def _apply_secondary_button_style(self, button: QPushButton):
        """İkincil buton stili uygular."""
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #757575;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #424242;
            }
            QPushButton:disabled {
                background-color: #E0E0E0;
                color: #999999;
            }
        """
        )

    def _connect_signals(self):
        """Signal bağlantılarını yapar."""
        # Yöntem değişimi
        self.method_group.buttonClicked.connect(self._on_method_changed)

        # Kateter boyutu değişimi
        self.catheter_size_combo.currentIndexChanged.connect(self._on_catheter_size_changed)

        # Kalibrasyon başlat
        self.calibrate_button.clicked.connect(self._on_calibrate_clicked)

    def _on_method_changed(self):
        """Kalibrasyon yöntemi değiştiğinde çağrılır."""
        if self.catheter_radio.isChecked():
            self._current_method = CalibrationMethod.CATHETER
            self.catheter_group.setEnabled(True)
        elif self.manual_radio.isChecked():
            self._current_method = CalibrationMethod.MANUAL
            self.catheter_group.setEnabled(False)
        elif self.dicom_radio.isChecked():
            self._current_method = CalibrationMethod.DICOM_METADATA
            self.catheter_group.setEnabled(False)

        self._update_info_text()
        self.method_changed.emit(self._current_method)

    def _on_catheter_size_changed(self, index: int):
        """Kateter boyutu değiştiğinde çağrılır."""
        if index >= 0:
            catheter_size = self.catheter_size_combo.itemData(index)
            if catheter_size:
                self.catheter_size_changed.emit(catheter_size)

    def _on_calibrate_clicked(self):
        """Kalibrasyon butonu tıklandığında çağrılır."""
        self.calibration_requested.emit(self._current_method)

    def _update_info_text(self):
        """Bilgi metnini günceller."""
        info_texts = {
            CalibrationMethod.CATHETER: (
                "Click on the catheter in the image. The system will "
                "automatically detect and measure its width for calibration."
            ),
            CalibrationMethod.MANUAL: (
                "Click two points with a known distance between them. "
                "You will be asked to enter the actual distance in mm."
            ),
            CalibrationMethod.DICOM_METADATA: (
                "Calibration will be extracted from DICOM metadata if available. "
                "This is the most accurate method when supported."
            ),
        }

        self.info_label.setText(info_texts.get(self._current_method, ""))

    def get_selected_method(self) -> CalibrationMethod:
        """
        Seçili kalibrasyon yöntemini döndürür.

        Returns:
            CalibrationMethod: Seçili yöntem
        """
        return self._current_method

    def get_selected_catheter_size(self) -> Optional[CatheterSize]:
        """
        Seçili kateter boyutunu döndürür.

        Returns:
            Optional[CatheterSize]: Kateter boyutu veya None
        """
        if self._current_method != CalibrationMethod.CATHETER:
            return None

        return self.catheter_size_combo.currentData()

    def set_calibration_active(self, active: bool):
        """
        Kalibrasyon durumunu ayarlar.

        Args:
            active: Kalibrasyon aktif mi?
        """
        self.calibrate_button.setEnabled(not active)
        self.clear_button.setEnabled(active)
        self.method_group.setEnabled(not active)
        self.catheter_group.setEnabled(
            not active and self._current_method == CalibrationMethod.CATHETER
        )

        if active:
            self.calibrate_button.setText("Calibrating...")
        else:
            self.calibrate_button.setText("Start Calibration")

    def enable_dicom_calibration(self, enable: bool):
        """
        DICOM kalibrasyon seçeneğini etkinleştirir/devre dışı bırakır.

        Args:
            enable: Etkin mi?
        """
        self.dicom_radio.setEnabled(enable)
