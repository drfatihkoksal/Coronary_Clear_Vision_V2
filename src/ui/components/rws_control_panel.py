"""
RWS Control Panel Component

RWS analizi kontrol butonları ve beat seçimi için ayrı bir UI bileşeni.
Single Responsibility: Sadece kullanıcı kontrollerini yönetir.
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, 
                            QLabel, QComboBox)
from PyQt6.QtCore import pyqtSignal
from typing import List


class RWSControlPanel(QWidget):
    """
    RWS analizi için kontrol paneli bileşeni.
    
    Bu bileşen sadece kullanıcı kontrollerini (butonlar, seçimler) yönetir.
    Analiz mantığı veya sonuç gösterimi içermez (SRP).
    
    Signals:
        analyze_requested: Analiz butonu tıklandığında yayınlanır
        beat_changed: Beat seçimi değiştiğinde yayınlanır (int: beat index)
    """
    
    # Signals
    analyze_requested = pyqtSignal()
    beat_changed = pyqtSignal(int)
    
    def __init__(self, parent=None):
        """
        Control panel'i başlatır.
        
        Args:
            parent: Parent widget (isteğe bağlı)
        """
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        """
        UI bileşenlerini oluşturur ve düzenler.
        
        Private method - sadece sınıf içinde kullanılır.
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Analiz butonu
        self.analyze_button = QPushButton("Analyze RWS")
        self.analyze_button.setObjectName("rwsAnalyzeButton")  # CSS için
        self._apply_button_style()
        layout.addWidget(self.analyze_button)
        
        # Beat seçimi
        layout.addWidget(QLabel("Beat:"))
        self.beat_combo = QComboBox()
        self.beat_combo.setMinimumWidth(100)
        self.beat_combo.setObjectName("rwsBeatCombo")
        layout.addWidget(self.beat_combo)
        
        # Sağa yaslamak için stretch ekle
        layout.addStretch()
        
        self.setLayout(layout)
        
    def _apply_button_style(self):
        """
        Analiz butonuna stil uygular.
        
        Stil kodunu ayrı tutarak okunabilirliği artırır.
        """
        self.analyze_button.setStyleSheet("""
            QPushButton#rwsAnalyzeButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton#rwsAnalyzeButton:hover {
                background-color: #45a049;
            }
            QPushButton#rwsAnalyzeButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#rwsAnalyzeButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
    def _connect_signals(self):
        """
        Widget içi signal bağlantılarını yapar.
        
        Signal-slot bağlantılarını tek yerde toplar.
        """
        self.analyze_button.clicked.connect(self.analyze_requested.emit)
        self.beat_combo.currentIndexChanged.connect(self._on_beat_changed)
        
    def _on_beat_changed(self, index: int):
        """
        Beat seçimi değiştiğinde çağrılır.
        
        Args:
            index: Seçilen beat'in combo box indeksi
        """
        if index >= 0:  # Geçerli bir seçim varsa
            self.beat_changed.emit(index)
            
    def set_beat_options(self, beat_names: List[str]):
        """
        Mevcut beat seçeneklerini ayarlar.
        
        Args:
            beat_names: Beat isimleri listesi
        """
        # Mevcut seçimi sakla
        current_text = self.beat_combo.currentText()
        
        # Listeyi güncelle
        self.beat_combo.clear()
        self.beat_combo.addItems(beat_names)
        
        # Önceki seçimi korumaya çalış
        index = self.beat_combo.findText(current_text)
        if index >= 0:
            self.beat_combo.setCurrentIndex(index)
            
    def get_selected_beat_index(self) -> int:
        """
        Seçili beat indeksini döndürür.
        
        Returns:
            int: Seçili beat indeksi, seçim yoksa -1
        """
        return self.beat_combo.currentIndex()
        
    def set_enabled(self, enabled: bool):
        """
        Kontrol panel'i etkinleştirir/devre dışı bırakır.
        
        Args:
            enabled: True ise etkin, False ise devre dışı
        """
        self.analyze_button.setEnabled(enabled)
        self.beat_combo.setEnabled(enabled)
        
    def set_analyze_button_text(self, text: str):
        """
        Analiz buton metnini değiştirir.
        
        Args:
            text: Yeni buton metni
        """
        self.analyze_button.setText(text)