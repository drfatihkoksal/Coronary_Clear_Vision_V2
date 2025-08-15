"""
QCA Graph Widget Component

QCA analiz sonuçlarının grafiksel gösterimini sağlayan UI bileşeni.
Matplotlib kullanarak çap profili, stenoz bölgeleri ve zamansal değişimleri görselleştirir.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from PyQt6.QtCore import pyqtSignal
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from typing import Optional

from src.domain.models.qca_models import QCAAnalysisResult, QCASequentialResult


class QCAGraphWidget(QWidget):
    """
    QCA sonuçlarının grafiksel gösterimi için widget.

    Çap profili, stenoz görselleştirmesi ve zamansal analizleri destekler.
    Interactive zoom/pan özellikleri içerir.

    Signals:
        graph_clicked: Grafik üzerinde tıklama (position_mm)
        export_requested: Grafik export istendi
    """

    # Signals
    graph_clicked = pyqtSignal(float)  # position in mm
    export_requested = pyqtSignal()

    # Grafik tipleri
    GRAPH_TYPES = {
        "diameter_profile": "Diameter Profile",
        "area_profile": "Area Profile",
        "temporal_diameter": "Temporal Diameter",
        "stenosis_map": "Stenosis Map",
        "confidence_plot": "Confidence Plot",
    }

    def __init__(self, parent=None):
        """
        Graph widget'ını başlatır.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._current_result: Optional[QCAAnalysisResult] = None
        self._sequential_result: Optional[QCASequentialResult] = None
        self._init_ui()

    def _init_ui(self):
        """UI bileşenlerini oluşturur."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Kontrol toolbar
        toolbar_layout = self._create_toolbar()
        layout.addLayout(toolbar_layout)

        # Matplotlib figure ve canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("qcaGraphCanvas")

        # Mouse events
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        layout.addWidget(self.canvas)

        # Navigation toolbar
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.nav_toolbar)

        self.setLayout(layout)

        # İlk grafiği çiz (boş)
        self._draw_empty_graph()

    def _create_toolbar(self) -> QHBoxLayout:
        """
        Grafik kontrol toolbar'ını oluşturur.

        Returns:
            QHBoxLayout: Toolbar layout
        """
        layout = QHBoxLayout()

        # Grafik tipi seçimi
        layout.addWidget(QLabel("Graph Type:"))

        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(list(self.GRAPH_TYPES.values()))
        self.graph_type_combo.currentTextChanged.connect(self._on_graph_type_changed)
        self.graph_type_combo.setObjectName("qcaGraphType")
        layout.addWidget(self.graph_type_combo)

        layout.addStretch()

        # Export butonu
        self.export_button = QPushButton("Export Graph")
        self.export_button.clicked.connect(self.export_requested.emit)
        self.export_button.setObjectName("qcaExportGraph")
        layout.addWidget(self.export_button)

        # Clear butonu
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_graph)
        self.clear_button.setObjectName("qcaClearGraph")
        layout.addWidget(self.clear_button)

        return layout

    def _draw_empty_graph(self):
        """Boş grafik çizer."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No data to display",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def display_result(self, result: QCAAnalysisResult):
        """
        Tekli QCA sonucunu görselleştirir.

        Args:
            result: QCA analiz sonucu
        """
        self._current_result = result
        self._sequential_result = None

        if not result.is_successful or not result.measurements:
            self._draw_empty_graph()
            return

        # Seçili grafik tipine göre çiz
        graph_type = self._get_selected_graph_type()
        self._draw_graph(graph_type)

    def display_sequential_result(self, result: QCASequentialResult):
        """
        Ardışık QCA sonuçlarını görselleştirir.

        Args:
            result: Ardışık analiz sonucu
        """
        self._sequential_result = result
        self._current_result = None

        if result.frame_count == 0:
            self._draw_empty_graph()
            return

        # Zamansal grafik tipine geç
        self.graph_type_combo.setCurrentText(self.GRAPH_TYPES["temporal_diameter"])

    def _get_selected_graph_type(self) -> str:
        """Seçili grafik tipini döndürür."""
        current_text = self.graph_type_combo.currentText()
        for key, value in self.GRAPH_TYPES.items():
            if value == current_text:
                return key
        return "diameter_profile"

    def _on_graph_type_changed(self):
        """Grafik tipi değiştiğinde çağrılır."""
        if self._current_result or self._sequential_result:
            graph_type = self._get_selected_graph_type()
            self._draw_graph(graph_type)

    def _draw_graph(self, graph_type: str):
        """
        Belirtilen tipte grafik çizer.

        Args:
            graph_type: Çizilecek grafik tipi
        """
        self.figure.clear()

        if graph_type == "diameter_profile":
            self._draw_diameter_profile()
        elif graph_type == "area_profile":
            self._draw_area_profile()
        elif graph_type == "temporal_diameter":
            self._draw_temporal_diameter()
        elif graph_type == "stenosis_map":
            self._draw_stenosis_map()
        elif graph_type == "confidence_plot":
            self._draw_confidence_plot()

        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_diameter_profile(self):
        """Çap profili grafiği çizer."""
        if not self._current_result or not self._current_result.measurements:
            return

        ax = self.figure.add_subplot(111)
        measurements = self._current_result.measurements

        # Veri hazırlama
        positions = [m.position_mm for m in measurements]
        diameters = [m.diameter_mm for m in measurements]

        # Ana çap profili
        ax.plot(positions, diameters, "b-", linewidth=2, label="Diameter")
        ax.scatter(positions, diameters, c="blue", s=30, alpha=0.6)

        # Ortalama çap çizgisi
        mean_diameter = np.mean(diameters)
        ax.axhline(
            y=mean_diameter,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_diameter:.2f} mm",
        )

        # Stenoz bölgesi varsa işaretle
        if self._current_result.stenosis_data:
            stenosis = self._current_result.stenosis_data
            ax.axvspan(
                stenosis.location_mm - stenosis.length_mm / 2,
                stenosis.location_mm + stenosis.length_mm / 2,
                alpha=0.3,
                color="red",
                label="Stenosis Region",
            )

            # MLD işareti
            ax.axhline(
                y=stenosis.minimal_lumen_diameter_mm,
                color="red",
                linestyle=":",
                label=f"MLD: {stenosis.minimal_lumen_diameter_mm:.2f} mm",
            )

        # Grafik özellikleri
        ax.set_xlabel("Position (mm)", fontsize=12)
        ax.set_ylabel("Diameter (mm)", fontsize=12)
        ax.set_title("Vessel Diameter Profile", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Y ekseni sınırları
        y_margin = (max(diameters) - min(diameters)) * 0.1
        ax.set_ylim(min(diameters) - y_margin, max(diameters) + y_margin)

    def _draw_area_profile(self):
        """Alan profili grafiği çizer."""
        if not self._current_result or not self._current_result.measurements:
            return

        ax = self.figure.add_subplot(111)
        measurements = self._current_result.measurements

        positions = [m.position_mm for m in measurements]
        areas = [m.area_mm2 for m in measurements]

        # Alan profili
        ax.fill_between(
            positions, 0, areas, alpha=0.5, color="skyblue", label="Cross-sectional Area"
        )
        ax.plot(positions, areas, "b-", linewidth=2)

        # Ortalama alan
        mean_area = np.mean(areas)
        ax.axhline(
            y=mean_area,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_area:.2f} mm²",
        )

        # Grafik özellikleri
        ax.set_xlabel("Position (mm)", fontsize=12)
        ax.set_ylabel("Area (mm²)", fontsize=12)
        ax.set_title("Vessel Cross-sectional Area Profile", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _draw_temporal_diameter(self):
        """Zamansal çap değişimi grafiği çizer."""
        if not self._sequential_result:
            return

        ax = self.figure.add_subplot(111)

        # Çap eğrisini al
        frames, diameters = self._sequential_result.get_diameter_curve()

        if not frames:
            ax.text(
                0.5,
                0.5,
                "No temporal data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Ana çap eğrisi
        ax.plot(
            frames, diameters, "b-", linewidth=2, marker="o", markersize=5, label="Mean Diameter"
        )

        # İstatistikler
        mean_diam = np.mean(diameters)
        std_diam = np.std(diameters)

        ax.axhline(y=mean_diam, color="green", linestyle="--", label=f"Mean: {mean_diam:.2f} mm")
        ax.fill_between(
            frames,
            mean_diam - std_diam,
            mean_diam + std_diam,
            alpha=0.2,
            color="green",
            label=f"±1 SD",
        )

        # Pulsatilite
        if "pulsatility_index" in self._sequential_result.temporal_analysis:
            pi = self._sequential_result.temporal_analysis["pulsatility_index"]
            ax.text(
                0.02,
                0.98,
                f"Pulsatility Index: {pi:.3f}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Grafik özellikleri
        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Mean Diameter (mm)", fontsize=12)
        ax.set_title("Temporal Diameter Variation", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _draw_stenosis_map(self):
        """Stenoz haritası çizer."""
        if not self._current_result or not self._current_result.measurements:
            return

        ax = self.figure.add_subplot(111)
        measurements = self._current_result.measurements

        positions = [m.position_mm for m in measurements]
        diameters = [m.diameter_mm for m in measurements]

        # Referans çapı hesapla (en geniş %20'lik bölge)
        sorted_diams = sorted(diameters, reverse=True)
        ref_diameter = np.mean(sorted_diams[: max(1, len(sorted_diams) // 5)])

        # Stenoz yüzdelerini hesapla
        stenosis_percents = [(1 - d / ref_diameter) * 100 for d in diameters]

        # Renk haritası
        colors = []
        for sp in stenosis_percents:
            if sp < 25:
                colors.append("green")
            elif sp < 50:
                colors.append("yellow")
            elif sp < 70:
                colors.append("orange")
            else:
                colors.append("red")

        # Bar grafik
        bars = ax.bar(positions, stenosis_percents, width=0.5, color=colors, alpha=0.7)

        # Eşik çizgileri
        ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% Stenosis")
        ax.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="70% Stenosis")

        # Grafik özellikleri
        ax.set_xlabel("Position (mm)", fontsize=12)
        ax.set_ylabel("Stenosis (%)", fontsize=12)
        ax.set_title("Stenosis Severity Map", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right")

    def _draw_confidence_plot(self):
        """Güven değerleri grafiği çizer."""
        if not self._current_result or not self._current_result.measurements:
            return

        ax = self.figure.add_subplot(111)
        measurements = self._current_result.measurements

        positions = [m.position_mm for m in measurements]
        confidences = [m.confidence * 100 for m in measurements]  # Yüzde olarak

        # Güven değerleri
        ax.plot(
            positions, confidences, "g-", linewidth=2, marker="s", markersize=6, label="Confidence"
        )

        # Renk kodlaması
        for i, (pos, conf) in enumerate(zip(positions, confidences)):
            if conf >= 80:
                color = "green"
            elif conf >= 60:
                color = "orange"
            else:
                color = "red"
            ax.scatter(pos, conf, c=color, s=50, zorder=5)

        # Eşik çizgisi
        ax.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="High Confidence (>80%)")
        ax.axhline(
            y=60, color="orange", linestyle="--", alpha=0.5, label="Medium Confidence (>60%)"
        )

        # Grafik özellikleri
        ax.set_xlabel("Position (mm)", fontsize=12)
        ax.set_ylabel("Confidence (%)", fontsize=12)
        ax.set_title("Measurement Confidence Plot", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def _on_canvas_click(self, event):
        """
        Canvas üzerinde tıklama eventi.

        Args:
            event: Matplotlib mouse event
        """
        if event.inaxes and event.xdata is not None:
            # X koordinatını mm cinsinden yayınla
            self.graph_clicked.emit(float(event.xdata))

    def clear_graph(self):
        """Grafiği temizler."""
        self._current_result = None
        self._sequential_result = None
        self._draw_empty_graph()

    def export_graph(self, filepath: str, dpi: int = 300):
        """
        Grafiği dosyaya kaydeder.

        Args:
            filepath: Kayıt yolu
            dpi: Çözünürlük (dots per inch)
        """
        self.figure.savefig(filepath, dpi=dpi, bbox_inches="tight")

    def get_graph_type(self) -> str:
        """
        Mevcut grafik tipini döndürür.

        Returns:
            str: Grafik tipi
        """
        return self._get_selected_graph_type()
