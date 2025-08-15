"""
Calibration Progress Indicator Component

Kalibrasyon işlemi sırasında ilerleme durumunu gösteren UI bileşeni.
Single Responsibility: Sadece ilerleme durumunu görselleştirir.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from typing import Optional


class CalibrationProgressIndicator(QWidget):
    """
    Kalibrasyon ilerleme göstergesi.

    Progress bar ve durum mesajları ile kalibrasyon
    sürecini görselleştirir.

    Signals:
        cancel_requested: İptal butonu tıklandığında
    """

    # Signals
    cancel_requested = pyqtSignal()

    # İlerleme aşamaları
    STAGES = {
        "initializing": (0, 10, "Initializing calibration..."),
        "segmenting": (10, 40, "Detecting catheter..."),
        "measuring": (40, 70, "Measuring catheter width..."),
        "calculating": (70, 90, "Calculating calibration factor..."),
        "validating": (90, 100, "Validating results..."),
        "complete": (100, 100, "Calibration complete!"),
    }

    def __init__(self, parent=None):
        """
        Progress indicator'ı başlatır.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._init_ui()
        self._current_stage = None
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_progress)
        self.hide()  # Başlangıçta gizli

    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(8)

        # Başlık ve iptal butonu
        header_layout = QHBoxLayout()

        self.title_label = QLabel("Calibration in Progress")
        self.title_label.setObjectName("calibrationProgressTitle")
        self.title_label.setStyleSheet(
            """
            QLabel#calibrationProgressTitle {
                font-size: 13px;
                font-weight: bold;
                color: #1565C0;
            }
        """
        )
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("calibrationCancelButton")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self._apply_cancel_button_style()
        header_layout.addWidget(self.cancel_button)

        layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("calibrationProgressBar")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(22)
        self._apply_progress_style()
        layout.addWidget(self.progress_bar)

        # Durum mesajı
        self.status_label = QLabel("")
        self.status_label.setObjectName("calibrationStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            """
            QLabel#calibrationStatusLabel {
                font-size: 11px;
                color: #666;
                padding: 2px;
            }
        """
        )
        layout.addWidget(self.status_label)

        # Detay mesajı (opsiyonel)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("calibrationDetailLabel")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(
            """
            QLabel#calibrationDetailLabel {
                font-size: 10px;
                color: #888;
                font-style: italic;
                padding: 2px;
            }
        """
        )
        self.detail_label.hide()
        layout.addWidget(self.detail_label)

        self.setLayout(layout)

    def _apply_progress_style(self):
        """Progress bar stilini uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#calibrationProgressBar {
                border: 2px solid #1976D2;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #E3F2FD;
            }
            QProgressBar#calibrationProgressBar::chunk {
                background-color: #1976D2;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

    def _apply_cancel_button_style(self):
        """İptal butonu stilini uygular."""
        self.cancel_button.setStyleSheet(
            """
            QPushButton#calibrationCancelButton {
                background-color: transparent;
                color: #F44336;
                font-size: 11px;
                padding: 4px 8px;
                border: 1px solid #F44336;
                border-radius: 3px;
            }
            QPushButton#calibrationCancelButton:hover {
                background-color: #F44336;
                color: white;
            }
            QPushButton#calibrationCancelButton:pressed {
                background-color: #D32F2F;
            }
            QPushButton#calibrationCancelButton:disabled {
                color: #CCC;
                border-color: #CCC;
            }
        """
        )

    def show_progress(self, initial_message: str = "Starting calibration..."):
        """
        Progress göstergesini gösterir.

        Args:
            initial_message: Başlangıç mesajı
        """
        self.progress_bar.setValue(0)
        self.status_label.setText(initial_message)
        self.detail_label.hide()
        self.cancel_button.setEnabled(True)
        self._current_stage = None
        self._apply_progress_style()
        self.show()

    def update_stage(self, stage: str, detail: Optional[str] = None):
        """
        Kalibrasyon aşamasını günceller.

        Args:
            stage: Aşama adı (STAGES içinde tanımlı)
            detail: Ek detay mesajı
        """
        if stage not in self.STAGES:
            return

        self._current_stage = stage
        start_val, end_val, message = self.STAGES[stage]

        # Progress animasyonu
        self._animate_progress(start_val, end_val)

        # Durum mesajı
        self.status_label.setText(message)

        # Detay mesajı
        if detail:
            self.detail_label.setText(detail)
            self.detail_label.show()
        else:
            self.detail_label.hide()

        # Tamamlandıysa
        if stage == "complete":
            self._set_complete_style()
            self.cancel_button.setEnabled(False)

    def _animate_progress(self, start_value: int, end_value: int):
        """
        Progress bar'ı animasyonlu günceller.

        Args:
            start_value: Başlangıç değeri
            end_value: Bitiş değeri
        """
        # QPropertyAnimation kullan
        animation = QPropertyAnimation(self.progress_bar, b"value")
        animation.setDuration(500)  # 500ms
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        animation.start()

    def set_indeterminate(self, message: str = "Processing..."):
        """
        Belirsiz ilerleme moduna geçer.

        Args:
            message: Durum mesajı
        """
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)  # Belirsiz mod
        self._pulse_timer.start(100)  # Pulse animasyonu

    def _pulse_progress(self):
        """Belirsiz mod için pulse animasyonu."""
        # Qt'nin built-in animasyonunu kullan

    def set_progress(self, value: int, message: Optional[str] = None):
        """
        Manuel ilerleme değeri ayarlar.

        Args:
            value: İlerleme değeri (0-100)
            message: Durum mesajı
        """
        self._pulse_timer.stop()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)

        if message:
            self.status_label.setText(message)

    def _set_complete_style(self):
        """Tamamlanma durumu için stil uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#calibrationProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #E8F5E9;
            }
            QProgressBar#calibrationProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

        self.title_label.setText("Calibration Complete")
        self.title_label.setStyleSheet(
            """
            QLabel#calibrationProgressTitle {
                font-size: 13px;
                font-weight: bold;
                color: #2E7D32;
            }
        """
        )

    def set_error_state(self, error_message: str):
        """
        Hata durumunu gösterir.

        Args:
            error_message: Hata mesajı
        """
        self._pulse_timer.stop()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.progress_bar.setStyleSheet(
            """
            QProgressBar#calibrationProgressBar {
                border: 2px solid #F44336;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #FFEBEE;
                color: #F44336;
            }
        """
        )

        self.title_label.setText("Calibration Failed")
        self.title_label.setStyleSheet(
            """
            QLabel#calibrationProgressTitle {
                font-size: 13px;
                font-weight: bold;
                color: #C62828;
            }
        """
        )

        self.status_label.setText("Calibration failed")
        self.detail_label.setText(error_message)
        self.detail_label.show()
        self.cancel_button.setEnabled(False)

    def hide_progress(self):
        """Progress göstergesini gizler."""
        self._pulse_timer.stop()
        self.hide()

    def reset(self):
        """Progress göstergesini sıfırlar."""
        self._pulse_timer.stop()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText("")
        self.detail_label.setText("")
        self.detail_label.hide()
        self.cancel_button.setEnabled(True)
        self._apply_progress_style()
        self.title_label.setText("Calibration in Progress")
        self._current_stage = None

    def is_active(self) -> bool:
        """
        Kalibrasyon aktif mi?

        Returns:
            bool: Aktifse True
        """
        return self.isVisible() and self._current_stage != "complete"
