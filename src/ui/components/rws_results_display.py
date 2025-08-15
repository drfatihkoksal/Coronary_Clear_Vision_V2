"""
RWS Results Display Component

RWS analiz sonuçlarını gösteren UI bileşeni.
Single Responsibility: Sadece sonuçları görselleştirir, analiz yapmaz.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt6.QtCore import Qt
from typing import Optional, List, Tuple
from src.models.rws_models import RWSAnalysisResult, RiskLevel


class RWSResultsDisplay(QWidget):
    """
    RWS analiz sonuçlarını gösteren widget.

    Bu bileşen sadece sonuç gösterimi ile ilgilenir.
    Veri işleme veya analiz mantığı içermez (SRP).
    """

    # Risk renkleri sabitleri
    RISK_COLORS = {
        RiskLevel.LOW: "#4CAF50",  # Yeşil
        RiskLevel.MODERATE: "#FF9800",  # Turuncu
        RiskLevel.HIGH: "#F44336",  # Kırmızı
        RiskLevel.UNKNOWN: "#666666",  # Gri
    }

    def __init__(self, parent=None):
        """
        Results display widget'ını başlatır.

        Args:
            parent: Parent widget (isteğe bağlı)
        """
        super().__init__(parent)
        self._init_ui()
        self._current_result: Optional[RWSAnalysisResult] = None

    def _init_ui(self):
        """UI bileşenlerini oluşturur ve düzenler."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Ana sonuçlar grubu
        self.results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()

        # Anahtar metrikler
        results_layout.addLayout(self._create_key_metrics_layout())

        # Yorum alanı
        self.interpretation_label = self._create_interpretation_label()
        results_layout.addWidget(self.interpretation_label)

        # Detaylı sonuçlar tablosu
        self.results_table = self._create_results_table()
        results_layout.addWidget(self.results_table)

        self.results_group.setLayout(results_layout)
        layout.addWidget(self.results_group)

        self.setLayout(layout)

    def _create_key_metrics_layout(self) -> QHBoxLayout:
        """
        Anahtar metrikleri gösteren layout oluşturur.

        Returns:
            QHBoxLayout: Metrik etiketlerini içeren layout
        """
        layout = QHBoxLayout()

        # Max RWS
        self.rws_max_label = QLabel("Max RWS: --")
        self.rws_max_label.setObjectName("rwsMaxLabel")
        self.rws_max_label.setStyleSheet(
            """
            QLabel#rwsMaxLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """
        )
        layout.addWidget(self.rws_max_label)

        # Stenosis RWS
        self.rws_stenosis_label = QLabel("Stenosis RWS: --")
        self.rws_stenosis_label.setObjectName("rwsStenosisLabel")
        self.rws_stenosis_label.setStyleSheet(
            """
            QLabel#rwsStenosisLabel {
                font-size: 14px;
                padding: 5px;
            }
        """
        )
        layout.addWidget(self.rws_stenosis_label)

        # Risk seviyesi
        self.risk_label = QLabel("Risk: --")
        self.risk_label.setObjectName("rwsRiskLabel")
        layout.addWidget(self.risk_label)

        layout.addStretch()
        return layout

    def _create_interpretation_label(self) -> QLabel:
        """
        Klinik yorum gösterimi için etiket oluşturur.

        Returns:
            QLabel: Yorum etiketi
        """
        label = QLabel("")
        label.setWordWrap(True)
        label.setObjectName("rwsInterpretationLabel")
        label.setStyleSheet(
            """
            QLabel#rwsInterpretationLabel {
                font-size: 12px;
                color: #666;
                padding: 10px;
                background-color: #f5f5f5;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
        """
        )
        return label

    def _create_results_table(self) -> QTableWidget:
        """
        Detaylı sonuçlar için tablo oluşturur.

        Returns:
            QTableWidget: Sonuç tablosu
        """
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Parameter", "Value"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setMaximumHeight(200)
        table.setAlternatingRowColors(True)
        table.setObjectName("rwsResultsTable")

        # Tablo stili
        table.setStyleSheet(
            """
            QTableWidget#rwsResultsTable {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QTableWidget#rwsResultsTable::item {
                padding: 5px;
            }
        """
        )

        return table

    def display_results(self, result: RWSAnalysisResult):
        """
        RWS analiz sonuçlarını gösterir.

        Args:
            result: Gösterilecek analiz sonuçları
        """
        self._current_result = result

        # Anahtar metrikleri güncelle
        self._update_key_metrics(result)

        # Yorumu güncelle
        self.interpretation_label.setText(result.interpretation)

        # Detay tablosunu güncelle
        self._update_results_table(result)

    def _update_key_metrics(self, result: RWSAnalysisResult):
        """
        Anahtar metrik göstergelerini günceller.

        Args:
            result: Analiz sonuçları
        """
        # Max RWS
        self.rws_max_label.setText(f"Max RWS: {result.rws_max}%")

        # Stenosis RWS
        self.rws_stenosis_label.setText(f"Stenosis RWS: {result.rws_stenosis}%")

        # Risk seviyesi - renkli gösterim
        risk_color = self.RISK_COLORS.get(result.risk_level, "#666")
        self.risk_label.setText(f"Risk: {result.risk_level.value.upper()}")
        self.risk_label.setStyleSheet(
            f"""
            QLabel#rwsRiskLabel {{
                font-size: 14px;
                font-weight: bold;
                color: {risk_color};
                padding: 5px 10px;
                border: 2px solid {risk_color};
                border-radius: 4px;
                background-color: {risk_color}20;
            }}
        """
        )

    def _update_results_table(self, result: RWSAnalysisResult):
        """
        Detaylı sonuçlar tablosunu günceller.

        Args:
            result: Analiz sonuçları
        """
        # Tabloyu temizle
        self.results_table.setRowCount(0)

        # Tablo verilerini hazırla
        table_data: List[Tuple[str, str]] = [
            ("Maximum RWS", f"{result.rws_max:.2f}%"),
            ("RWS at Stenosis", f"{result.rws_stenosis:.2f}%"),
            (
                "Max RWS Location",
                f"Index {result.rws_max_location}" if result.rws_max_location >= 0 else "N/A",
            ),
            ("Diameter Change", f"{result.diameter_change_mm:.3f} mm"),
            ("Reference Diameter", f"{result.reference_diameter_mm:.2f} mm"),
            (
                "End-Diastole Frame",
                str(result.end_diastole_frame) if result.end_diastole_frame >= 0 else "N/A",
            ),
            (
                "End-Systole Frame",
                str(result.end_systole_frame) if result.end_systole_frame >= 0 else "N/A",
            ),
        ]

        # Metadata varsa ekle
        if result.metadata:
            for key, value in result.metadata.items():
                table_data.append((key, str(value)))

        # Tabloya verileri ekle
        for param, value in table_data:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            param_item = QTableWidgetItem(param)
            param_item.setFlags(param_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(row, 0, param_item)

            value_item = QTableWidgetItem(value)
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(row, 1, value_item)

    def clear_results(self):
        """Gösterilen sonuçları temizler."""
        self._current_result = None

        # Metrikleri sıfırla
        self.rws_max_label.setText("Max RWS: --")
        self.rws_stenosis_label.setText("Stenosis RWS: --")
        self.risk_label.setText("Risk: --")
        self.risk_label.setStyleSheet(
            """
            QLabel#rwsRiskLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """
        )

        # Yorumu temizle
        self.interpretation_label.setText("")

        # Tabloyu temizle
        self.results_table.setRowCount(0)

    def get_current_result(self) -> Optional[RWSAnalysisResult]:
        """
        Şu anda gösterilen sonucu döndürür.

        Returns:
            Optional[RWSAnalysisResult]: Mevcut sonuç veya None
        """
        return self._current_result
