"""
ECG Signal Display Component

ECG sinyalini görüntüleyen bileşen.
PyQtGraph kullanarak yüksek performanslı görselleştirme.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import Optional, List, Tuple
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
import logging

from src.domain.models.ecg_models import ECGSignal, RPeak, CardiacPhaseAnalysis, CardiacPhase

logger = logging.getLogger(__name__)


class ECGSignalDisplay(QWidget):
    """
    ECG sinyal görüntüleme bileşeni.

    Bu bileşen:
    - ECG sinyalini çizer
    - R-peak'leri işaretler
    - Kardiyak fazları gösterir
    - Zoom ve pan desteği sağlar
    - Crosshair ile detaylı inceleme

    Signals:
        time_clicked: Tıklanan zaman noktası
        rpeak_clicked: Tıklanan R-peak indeksi
        phase_clicked: Tıklanan kardiyak faz
    """

    # Signals
    time_clicked = pyqtSignal(float)  # time in seconds
    rpeak_clicked = pyqtSignal(int)  # peak index
    phase_clicked = pyqtSignal(str)  # phase code

    def __init__(self, parent: Optional[QWidget] = None):
        """
        ECGSignalDisplay constructor.

        Args:
            parent: Ana widget
        """
        super().__init__(parent)

        # Veri
        self._ecg_signal: Optional[ECGSignal] = None
        self._r_peaks: List[RPeak] = []
        self._phase_analysis: Optional[CardiacPhaseAnalysis] = None

        # Görüntüleme elemanları
        self._signal_curve: Optional[pg.PlotDataItem] = None
        self._rpeak_scatter: Optional[pg.ScatterPlotItem] = None
        self._phase_regions: List[pg.LinearRegionItem] = []
        self._phase_lines: List[pg.InfiniteLine] = []

        # Ayarlar
        self._show_r_peaks = True
        self._show_phases = False
        self._phase_colors = {
            CardiacPhase.D1: "#9C27B0",  # Mid-diastole - Mor
            CardiacPhase.D2: "#4CAF50",  # End-diastole - Yeşil
            CardiacPhase.S1: "#2196F3",  # Early-systole - Mavi
            CardiacPhase.S2: "#FF9800",  # End-systole - Turuncu
        }

        self._setup_ui()
        logger.info("ECGSignalDisplay initialized")

    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # PyQtGraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Eksen etiketleri
        self.plot_widget.setLabel("left", "Amplitude", units="mV")
        self.plot_widget.setLabel("bottom", "Time", units="s")

        # Eksen fontları
        font = self.plot_widget.getAxis("left").label.font()
        font.setPointSize(12)
        self.plot_widget.getAxis("left").label.setFont(font)
        self.plot_widget.getAxis("left").setTickFont(font)
        self.plot_widget.getAxis("bottom").label.setFont(font)
        self.plot_widget.getAxis("bottom").setTickFont(font)

        # Mouse interaction
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Crosshair
        self._vline = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("b", width=1, style=Qt.PenStyle.DashLine)
        )
        self._hline = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("b", width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self._vline)
        self.plot_widget.addItem(self._hline)

        # Mouse move proxy
        self._mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        layout.addWidget(self.plot_widget)

    def set_ecg_signal(self, signal: ECGSignal):
        """
        ECG sinyalini ayarla.

        Args:
            signal: ECG sinyali
        """
        self._ecg_signal = signal
        self._clear_plot()

        if signal is None:
            return

        # Zaman ekseni
        time_array = signal.time_array

        # Sinyal çiz
        self._signal_curve = self.plot_widget.plot(
            time_array, signal.data, pen=pg.mkPen("k", width=2), name="ECG"
        )

        # Otomatik ölçekleme
        self.plot_widget.autoRange()

        logger.debug(f"ECG signal displayed: {len(signal.data)} samples")

    def set_r_peaks(self, r_peaks: List[RPeak]):
        """
        R-peak'leri ayarla.

        Args:
            r_peaks: R-peak listesi
        """
        self._r_peaks = r_peaks
        self._update_rpeak_display()

    def set_cardiac_phases(self, phase_analysis: CardiacPhaseAnalysis):
        """
        Kardiyak fazları ayarla.

        Args:
            phase_analysis: Kardiyak faz analizi
        """
        self._phase_analysis = phase_analysis
        if self._show_phases:
            self._update_phase_display()

    def set_show_r_peaks(self, show: bool):
        """R-peak gösterimini aç/kapa."""
        self._show_r_peaks = show
        self._update_rpeak_display()

    def set_show_phases(self, show: bool):
        """Kardiyak faz gösterimini aç/kapa."""
        self._show_phases = show
        self._update_phase_display()

    def add_time_marker(self, time: float, color: str = "r", label: Optional[str] = None):
        """
        Belirli bir zamana marker ekle.

        Args:
            time: Zaman (saniye)
            color: Marker rengi
            label: Marker etiketi
        """
        marker = pg.InfiniteLine(
            pos=time,
            angle=90,
            pen=pg.mkPen(color, width=2),
            movable=False,
            label=label,
            labelOpts={"position": 0.9, "color": color},
        )
        self.plot_widget.addItem(marker)

    def clear_markers(self):
        """Tüm marker'ları temizle."""
        # Time marker'ları temizle
        for item in self.plot_widget.items():
            if isinstance(item, pg.InfiniteLine) and item not in [self._vline, self._hline]:
                self.plot_widget.removeItem(item)

    def get_visible_time_range(self) -> Tuple[float, float]:
        """
        Görünür zaman aralığını döndür.

        Returns:
            Tuple[float, float]: (başlangıç, bitiş) zamanları
        """
        vb = self.plot_widget.plotItem.vb
        x_range = vb.viewRange()[0]
        return x_range[0], x_range[1]

    def set_visible_time_range(self, start: float, end: float):
        """
        Görünür zaman aralığını ayarla.

        Args:
            start: Başlangıç zamanı (saniye)
            end: Bitiş zamanı (saniye)
        """
        self.plot_widget.setXRange(start, end)

    def zoom_to_r_peak(self, peak_index: int, window_seconds: float = 1.0):
        """
        Belirli bir R-peak'e zoom yap.

        Args:
            peak_index: R-peak indeksi
            window_seconds: Pencere genişliği (saniye)
        """
        if 0 <= peak_index < len(self._r_peaks):
            peak_time = self._r_peaks[peak_index].time
            half_window = window_seconds / 2
            self.set_visible_time_range(peak_time - half_window, peak_time + half_window)

    def _clear_plot(self):
        """Plot'u temizle."""
        # Sinyal eğrisini kaldır
        if self._signal_curve:
            self.plot_widget.removeItem(self._signal_curve)
            self._signal_curve = None

        # R-peak'leri kaldır
        if self._rpeak_scatter:
            self.plot_widget.removeItem(self._rpeak_scatter)
            self._rpeak_scatter = None

        # Faz görüntülerini kaldır
        self._clear_phase_display()

    def _update_rpeak_display(self):
        """R-peak gösterimini güncelle."""
        # Eski scatter'ı kaldır
        if self._rpeak_scatter:
            self.plot_widget.removeItem(self._rpeak_scatter)
            self._rpeak_scatter = None

        if not self._show_r_peaks or not self._r_peaks or not self._ecg_signal:
            return

        # R-peak konumları ve değerleri
        peak_times = [p.time for p in self._r_peaks]
        peak_values = [p.amplitude for p in self._r_peaks]

        # Scatter plot oluştur
        self._rpeak_scatter = pg.ScatterPlotItem(
            x=peak_times,
            y=peak_values,
            pen=None,
            symbol="o",
            symbolBrush="r",
            symbolSize=10,
            symbolPen=pg.mkPen("darkred", width=2),
            name="R-peaks",
        )

        # Tıklanabilir yap
        self._rpeak_scatter.sigClicked.connect(self._on_rpeak_clicked)

        self.plot_widget.addItem(self._rpeak_scatter)

    def _update_phase_display(self):
        """Kardiyak faz gösterimini güncelle."""
        # Eski gösterimleri temizle
        self._clear_phase_display()

        if not self._show_phases or not self._phase_analysis:
            return

        # Her döngü için fazları göster
        for cycle in self._phase_analysis.cycles:
            # Sistol bölgesi (S1'den S2'ye)
            s1_phase = cycle.get_phase(CardiacPhase.S1)
            s2_phase = cycle.get_phase(CardiacPhase.S2)

            if s1_phase and s2_phase:
                systole_region = pg.LinearRegionItem(
                    [s1_phase.time, s2_phase.time],
                    brush=pg.mkBrush(255, 82, 82, 30),  # Kırmızımsı
                    movable=False,
                    pen=pg.mkPen(None),
                )
                systole_region.setZValue(-10)
                self.plot_widget.addItem(systole_region)
                self._phase_regions.append(systole_region)

            # Faz çizgileri
            for phase_type, phase_info in cycle.phases.items():
                color = self._phase_colors.get(phase_type, "#000000")

                phase_line = pg.InfiniteLine(
                    pos=phase_info.time,
                    angle=90,
                    pen=pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine),
                    movable=False,
                    label=phase_type.value.upper(),
                    labelOpts={"position": 0.95, "color": color},
                )

                self.plot_widget.addItem(phase_line)
                self._phase_lines.append(phase_line)

    def _clear_phase_display(self):
        """Faz gösterimlerini temizle."""
        # Bölgeleri kaldır
        for region in self._phase_regions:
            self.plot_widget.removeItem(region)
        self._phase_regions.clear()

        # Çizgileri kaldır
        for line in self._phase_lines:
            self.plot_widget.removeItem(line)
        self._phase_lines.clear()

    def _on_mouse_moved(self, evt):
        """Mouse hareket eventi."""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self._vline.setPos(mouse_point.x())
            self._hline.setPos(mouse_point.y())

    def _on_mouse_clicked(self, evt):
        """Mouse tıklama eventi."""
        if evt.button() == Qt.MouseButton.LeftButton:
            pos = evt.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                self.time_clicked.emit(mouse_point.x())

    def _on_rpeak_clicked(self, plot, points):
        """R-peak tıklama eventi."""
        if points:
            # İlk tıklanan point
            point = points[0]

            # Hangi R-peak'e tıklandı?
            for i, peak in enumerate(self._r_peaks):
                if abs(peak.time - point.pos().x()) < 0.01:  # 10ms tolerans
                    self.rpeak_clicked.emit(i)
                    break
