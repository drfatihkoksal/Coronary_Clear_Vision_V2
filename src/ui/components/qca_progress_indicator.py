"""
QCA Progress Indicator Component

QCA analizi sırasında ilerleme durumunu gösteren UI bileşeni.
Single Responsibility: Sadece ilerleme durumunu görselleştirir.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from typing import Optional


class QCAProgressIndicator(QWidget):
    """
    QCA analizi ilerleme göstergesi.

    Progress bar, durum mesajı ve iptal butonu içerir.
    Hem tekli hem de ardışık analiz ilerlemesini destekler.

    Signals:
        cancel_requested: İptal butonu tıklandığında
    """

    # Signals
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        """
        Progress indicator'ı başlatır.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._init_ui()
        self._is_sequential = False
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._update_animation)
        self.hide()  # Başlangıçta gizli

    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(8)

        # Ana progress layout
        progress_layout = QHBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("qcaProgressBar")
        self.progress_bar.setMinimumHeight(25)
        self._apply_progress_style()
        progress_layout.addWidget(self.progress_bar, 1)

        # İptal butonu
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("qcaCancelButton")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self._apply_cancel_button_style()
        progress_layout.addWidget(self.cancel_button)

        layout.addLayout(progress_layout)

        # Durum mesajları layout
        status_layout = QHBoxLayout()

        # Ana durum mesajı
        self.status_label = QLabel("")
        self.status_label.setObjectName("qcaStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.status_label)

        # Detay mesajı (sağda)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("qcaDetailLabel")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.detail_label.setStyleSheet(
            """
            QLabel#qcaDetailLabel {
                font-size: 11px;
                color: #666;
                font-style: italic;
            }
        """
        )
        status_layout.addWidget(self.detail_label)

        layout.addLayout(status_layout)

        # Sequential analiz için ek bilgi
        self.sequential_info_label = QLabel("")
        self.sequential_info_label.setObjectName("qcaSequentialInfo")
        self.sequential_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequential_info_label.setStyleSheet(
            """
            QLabel#qcaSequentialInfo {
                font-size: 12px;
                color: #1976D2;
                font-weight: bold;
                padding: 5px;
                background-color: #E3F2FD;
                border-radius: 3px;
            }
        """
        )
        self.sequential_info_label.hide()
        layout.addWidget(self.sequential_info_label)

        self.setLayout(layout)

    def _apply_progress_style(self):
        """Progress bar stilini uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#qcaProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #f0f0f0;
            }
            QProgressBar#qcaProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

    def _apply_cancel_button_style(self):
        """İptal butonu stilini uygular."""
        self.cancel_button.setStyleSheet(
            """
            QPushButton#qcaCancelButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
                border: none;
                min-width: 60px;
            }
            QPushButton#qcaCancelButton:hover {
                background-color: #D32F2F;
            }
            QPushButton#qcaCancelButton:pressed {
                background-color: #B71C1C;
            }
            QPushButton#qcaCancelButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """
        )

    def show_progress(self, message: str = "Initializing analysis...", is_sequential: bool = False):
        """
        Progress göstergesini gösterir.

        Args:
            message: Başlangıç mesajı
            is_sequential: Ardışık analiz mi?
        """
        self._is_sequential = is_sequential
        self.progress_bar.setValue(0)
        self.status_label.setText(message)
        self.detail_label.setText("")
        self.cancel_button.setEnabled(True)

        if is_sequential:
            self.sequential_info_label.show()
            self.sequential_info_label.setText("Sequential Analysis in Progress")
            self.progress_bar.setFormat("%v / %m frames (%p%)")
        else:
            self.sequential_info_label.hide()
            self.progress_bar.setFormat("%p%")

        self._apply_progress_style()  # Stili sıfırla
        self.show()

    def update_progress(
        self,
        value: int,
        maximum: int = 100,
        message: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        """
        İlerleme durumunu günceller.

        Args:
            value: Mevcut değer
            maximum: Maksimum değer
            message: Durum mesajı (opsiyonel)
            detail: Detay mesajı (opsiyonel)
        """
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

        if message:
            self.status_label.setText(message)

        if detail:
            self.detail_label.setText(detail)

        # Tamamlandıysa özel görünüm
        if value >= maximum:
            self._set_complete_style()
            self.cancel_button.setEnabled(False)

    def update_sequential_progress(
        self, current_frame: int, total_frames: int, current_status: str = "Analyzing..."
    ):
        """
        Ardışık analiz ilerlemesini günceller.

        Args:
            current_frame: Mevcut frame
            total_frames: Toplam frame sayısı
            current_status: Mevcut durum
        """
        self.update_progress(
            current_frame,
            total_frames,
            message=current_status,
            detail=f"Processing frame {current_frame} of {total_frames}",
        )

        # Sequential bilgiyi güncelle
        percent = (current_frame / total_frames * 100) if total_frames > 0 else 0
        self.sequential_info_label.setText(f"Sequential Analysis: {percent:.0f}% Complete")

    def set_indeterminate(self, message: str = "Processing..."):
        """
        Belirsiz ilerleme moduna geçer.

        Args:
            message: Durum mesajı
        """
        self.progress_bar.setRange(0, 0)  # Belirsiz mod
        self.status_label.setText(message)
        self._animation_timer.start(100)  # Animasyon başlat

    def _update_animation(self):
        """Belirsiz mod animasyonunu günceller."""
        current_text = self.detail_label.text()
        if current_text.endswith("..."):
            self.detail_label.setText("Processing")
        else:
            self.detail_label.setText(current_text + ".")

    def _set_complete_style(self):
        """Tamamlanma durumu için stil uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#qcaProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #E8F5E9;
            }
            QProgressBar#qcaProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

        if self._is_sequential:
            self.sequential_info_label.setText("Sequential Analysis Complete!")
            self.sequential_info_label.setStyleSheet(
                """
                QLabel#qcaSequentialInfo {
                    font-size: 12px;
                    color: #2E7D32;
                    font-weight: bold;
                    padding: 5px;
                    background-color: #C8E6C9;
                    border-radius: 3px;
                }
            """
            )

    def set_error_state(self, error_message: str):
        """
        Hata durumunu gösterir.

        Args:
            error_message: Hata mesajı
        """
        self._animation_timer.stop()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.progress_bar.setStyleSheet(
            """
            QProgressBar#qcaProgressBar {
                border: 2px solid #F44336;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                background-color: #FFEBEE;
                color: #F44336;
            }
        """
        )

        self.status_label.setText("Analysis Failed")
        self.detail_label.setText(error_message)
        self.cancel_button.setEnabled(False)

        if self._is_sequential:
            self.sequential_info_label.setText("Sequential Analysis Failed")
            self.sequential_info_label.setStyleSheet(
                """
                QLabel#qcaSequentialInfo {
                    font-size: 12px;
                    color: #C62828;
                    font-weight: bold;
                    padding: 5px;
                    background-color: #FFCDD2;
                    border-radius: 3px;
                }
            """
            )

    def hide_progress(self):
        """Progress göstergesini gizler."""
        self._animation_timer.stop()
        self.hide()

    def reset(self):
        """Progress göstergesini sıfırlar."""
        self._animation_timer.stop()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)
        self.status_label.setText("")
        self.detail_label.setText("")
        self.sequential_info_label.setText("")
        self.cancel_button.setEnabled(True)
        self._apply_progress_style()
        self._is_sequential = False
