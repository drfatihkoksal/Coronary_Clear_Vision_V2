"""
RWS Progress Indicator Component

RWS analizi sırasında ilerleme durumunu gösteren UI bileşeni.
Single Responsibility: Sadece ilerleme durumunu görselleştirir.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel
from PyQt6.QtCore import Qt
from src.models.rws_models import RWSAnalysisProgress


class RWSProgressIndicator(QWidget):
    """
    RWS analizi ilerleme göstergesi.

    Progress bar ve durum mesajı gösterir.
    Analiz mantığından bağımsızdır (SRP).
    """

    def __init__(self, parent=None):
        """
        Progress indicator'ı başlatır.

        Args:
            parent: Parent widget (isteğe bağlı)
        """
        super().__init__(parent)
        self._init_ui()
        self.hide()  # Başlangıçta gizli

    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("rwsProgressBar")
        self.progress_bar.setTextVisible(True)
        self._apply_progress_style()
        layout.addWidget(self.progress_bar)

        # Durum mesajı
        self.status_label = QLabel("")
        self.status_label.setObjectName("rwsStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            """
            QLabel#rwsStatusLabel {
                font-size: 12px;
                color: #666;
                padding: 2px;
            }
        """
        )
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _apply_progress_style(self):
        """Progress bar stilini uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#rwsProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #f0f0f0;
            }
            QProgressBar#rwsProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

    def update_progress(self, progress: RWSAnalysisProgress):
        """
        İlerleme durumunu günceller.

        Args:
            progress: İlerleme bilgisi
        """
        # Progress bar değerini güncelle
        self.progress_bar.setValue(progress.percentage)

        # Durum mesajını güncelle
        self.status_label.setText(progress.status)

        # Tamamlandıysa veya iptal edildiyse özel görünüm
        if progress.is_complete:
            self._set_complete_style()
        elif progress.is_cancelled:
            self._set_cancelled_style()

    def _set_complete_style(self):
        """Tamamlanma durumu için stil uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#rwsProgressBar {
                border: 1px solid #4CAF50;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #E8F5E9;
            }
            QProgressBar#rwsProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

    def _set_cancelled_style(self):
        """İptal durumu için stil uygular."""
        self.progress_bar.setStyleSheet(
            """
            QProgressBar#rwsProgressBar {
                border: 1px solid #F44336;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #FFEBEE;
            }
            QProgressBar#rwsProgressBar::chunk {
                background-color: #F44336;
                border-radius: 3px;
                margin: 1px;
            }
        """
        )

    def show_progress(self, initial_status: str = "Starting analysis..."):
        """
        Progress göstergesini gösterir ve başlangıç durumunu ayarlar.

        Args:
            initial_status: Başlangıç durum mesajı
        """
        self.progress_bar.setValue(0)
        self.status_label.setText(initial_status)
        self._apply_progress_style()  # Stili sıfırla
        self.show()

    def hide_progress(self):
        """Progress göstergesini gizler."""
        self.hide()

    def reset(self):
        """Progress göstergesini sıfırlar."""
        self.progress_bar.setValue(0)
        self.status_label.setText("")
        self._apply_progress_style()

    def set_indeterminate(self, is_indeterminate: bool = True):
        """
        Progress bar'ı belirsiz moda alır.

        Sürenin belli olmadığı durumlarda kullanılır.

        Args:
            is_indeterminate: True ise belirsiz mod
        """
        if is_indeterminate:
            self.progress_bar.setRange(0, 0)  # Belirsiz mod
        else:
            self.progress_bar.setRange(0, 100)  # Normal mod
