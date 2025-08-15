"""
ECG Phase Indicator Component

Kardiyak faz gösterge bileşeni.
Mevcut kardiyak fazı görsel olarak gösterir.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QLinearGradient
import logging

from src.domain.models.ecg_models import CardiacPhase

logger = logging.getLogger(__name__)


class PhaseIndicatorWidget(QWidget):
    """
    Kardiyak faz göstergesi.

    Döngüsel bir gösterge ile mevcut kardiyak fazı
    ve geçişleri görselleştirir.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        PhaseIndicatorWidget constructor.

        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self.setFixedSize(200, 200)

        # Mevcut faz
        self._current_phase: Optional[CardiacPhase] = None
        self._phase_progress: float = 0.0  # 0-1 arası

        # Renk şeması
        self._phase_colors = {
            CardiacPhase.D1: QColor("#9C27B0"),  # Mid-diastole - Mor
            CardiacPhase.D2: QColor("#4CAF50"),  # End-diastole - Yeşil
            CardiacPhase.S1: QColor("#2196F3"),  # Early-systole - Mavi
            CardiacPhase.S2: QColor("#FF9800"),  # End-systole - Turuncu
        }

        # Faz açıları (derece)
        self._phase_angles = {
            CardiacPhase.D2: 0,  # Üst (12 saat)
            CardiacPhase.S1: 90,  # Sağ (3 saat)
            CardiacPhase.S2: 180,  # Alt (6 saat)
            CardiacPhase.D1: 270,  # Sol (9 saat)
        }

        # Animasyon
        self._rotation_angle = 0
        self._animation = QPropertyAnimation(self, b"rotation_angle")
        self._animation.setDuration(1000)

        logger.info("PhaseIndicatorWidget initialized")

    @pyqtProperty(int)
    def rotation_angle(self):
        """Rotasyon açısı property."""
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        """Rotasyon açısı setter."""
        self._rotation_angle = value
        self.update()

    def set_phase(self, phase: CardiacPhase, progress: float = 0.5):
        """
        Kardiyak fazı ayarla.

        Args:
            phase: Kardiyak faz
            progress: Faz içindeki ilerleme (0-1)
        """
        self._current_phase = phase
        self._phase_progress = max(0.0, min(1.0, progress))
        self.update()

    def animate_to_phase(self, phase: CardiacPhase):
        """
        Belirli bir faza animasyonlu geçiş.

        Args:
            phase: Hedef faz
        """
        if phase in self._phase_angles:
            target_angle = self._phase_angles[phase]

            # En kısa yolu bul
            current = self._rotation_angle % 360
            diff = target_angle - current

            if abs(diff) > 180:
                if diff > 0:
                    diff -= 360
                else:
                    diff += 360

            self._animation.setStartValue(self._rotation_angle)
            self._animation.setEndValue(self._rotation_angle + diff)
            self._animation.start()

            self._current_phase = phase

    def paintEvent(self, event):
        """Paint event."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Merkez ve yarıçap
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 20

        # Arka plan dairesi
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.setBrush(QBrush(QColor(250, 250, 250)))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # Faz bölümleri
        self._draw_phase_segments(painter, center_x, center_y, radius)

        # Merkez gösterge
        self._draw_center_indicator(painter, center_x, center_y, radius)

        # Faz etiketleri
        self._draw_phase_labels(painter, center_x, center_y, radius)

        # Mevcut faz göstergesi
        if self._current_phase:
            self._draw_current_phase_indicator(painter, center_x, center_y, radius)

    def _draw_phase_segments(self, painter: QPainter, cx: int, cy: int, radius: int):
        """Faz bölümlerini çiz."""
        # Her faz 90 derece
        phases = [CardiacPhase.D2, CardiacPhase.S1, CardiacPhase.S2, CardiacPhase.D1]

        for i, phase in enumerate(phases):
            start_angle = i * 90 * 16  # Qt açıları 1/16 derece
            span_angle = 90 * 16

            # Gradient oluştur
            gradient = QLinearGradient(cx, cy - radius, cx, cy + radius)
            color = self._phase_colors[phase]
            gradient.setColorAt(0, color.lighter(150))
            gradient.setColorAt(1, color)

            painter.setPen(QPen(Qt.PenStyle.NoPen))
            painter.setBrush(QBrush(gradient))

            # Pasta dilimi çiz
            painter.drawPie(
                cx - radius, cy - radius, radius * 2, radius * 2, start_angle, span_angle
            )

    def _draw_center_indicator(self, painter: QPainter, cx: int, cy: int, radius: int):
        """Merkez göstergeyi çiz."""
        # Beyaz daire
        inner_radius = radius // 3
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(Qt.GlobalColor.white))
        painter.drawEllipse(
            cx - inner_radius, cy - inner_radius, inner_radius * 2, inner_radius * 2
        )

        # Dönen gösterge
        if self._current_phase:
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(self._rotation_angle)

            # Ok çiz
            painter.setPen(QPen(Qt.GlobalColor.black, 3))
            painter.drawLine(0, 0, 0, -inner_radius + 5)

            # Ok başı
            painter.setBrush(QBrush(Qt.GlobalColor.black))
            points = [(0, -inner_radius + 5), (-5, -inner_radius + 15), (5, -inner_radius + 15)]
            painter.drawPolygon([painter.worldTransform().map(x, y) for x, y in points])

            painter.restore()

    def _draw_phase_labels(self, painter: QPainter, cx: int, cy: int, radius: int):
        """Faz etiketlerini çiz."""
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)

        labels = {
            CardiacPhase.D2: ("D2", "End-diastole"),
            CardiacPhase.S1: ("S1", "Early-systole"),
            CardiacPhase.S2: ("S2", "End-systole"),
            CardiacPhase.D1: ("D1", "Mid-diastole"),
        }

        # Etiket konumları
        positions = {
            CardiacPhase.D2: (cx, cy - radius - 30),
            CardiacPhase.S1: (cx + radius + 20, cy),
            CardiacPhase.S2: (cx, cy + radius + 30),
            CardiacPhase.D1: (cx - radius - 60, cy),
        }

        for phase, (code, name) in labels.items():
            pos_x, pos_y = positions[phase]
            color = self._phase_colors[phase]

            # Kod
            painter.setPen(QPen(color.darker(120)))
            painter.drawText(pos_x - 20, pos_y - 10, 40, 20, Qt.AlignmentFlag.AlignCenter, code)

            # İsim (küçük font)
            small_font = QFont("Arial", 9)
            painter.setFont(small_font)
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(pos_x - 40, pos_y + 5, 80, 20, Qt.AlignmentFlag.AlignCenter, name)

            painter.setFont(font)

    def _draw_current_phase_indicator(self, painter: QPainter, cx: int, cy: int, radius: int):
        """Mevcut faz göstergesini çiz."""
        if not self._current_phase:
            return

        # Dış halka
        painter.setPen(QPen(self._phase_colors[self._current_phase], 4))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(cx - radius - 5, cy - radius - 5, (radius + 5) * 2, (radius + 5) * 2)


class ECGPhaseIndicator(QWidget):
    """
    ECG faz gösterge paneli.

    Kardiyak faz göstergesi ve bilgi panelini içerir.

    Signals:
        phase_clicked: Faza tıklandı
    """

    # Signals
    phase_clicked = pyqtSignal(str)  # phase code

    def __init__(self, parent: Optional[QWidget] = None):
        """
        ECGPhaseIndicator constructor.

        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self._setup_ui()

        # Mevcut bilgiler
        self._current_phase: Optional[CardiacPhase] = None
        self._heart_rate: float = 0.0
        self._cycle_number: int = 0

        logger.info("ECGPhaseIndicator initialized")

    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Başlık
        title_label = QLabel("Kardiyak Faz")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)

        # Faz göstergesi
        self.phase_widget = PhaseIndicatorWidget()
        layout.addWidget(self.phase_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Bilgi paneli
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.Box)
        info_layout = QVBoxLayout(info_frame)

        # Mevcut faz
        phase_layout = QHBoxLayout()
        phase_layout.addWidget(QLabel("Faz:"))
        self.phase_label = QLabel("--")
        self.phase_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        phase_layout.addWidget(self.phase_label)
        phase_layout.addStretch()
        info_layout.addLayout(phase_layout)

        # Faz açıklaması
        self.phase_desc_label = QLabel("--")
        self.phase_desc_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.phase_desc_label)

        # Kalp hızı
        hr_layout = QHBoxLayout()
        hr_layout.addWidget(QLabel("Kalp Hızı:"))
        self.hr_label = QLabel("-- BPM")
        hr_layout.addWidget(self.hr_label)
        hr_layout.addStretch()
        info_layout.addLayout(hr_layout)

        # Döngü numarası
        cycle_layout = QHBoxLayout()
        cycle_layout.addWidget(QLabel("Döngü:"))
        self.cycle_label = QLabel("#--")
        cycle_layout.addWidget(self.cycle_label)
        cycle_layout.addStretch()
        info_layout.addLayout(cycle_layout)

        layout.addWidget(info_frame)

        # Spacer
        layout.addStretch()

    def set_phase(
        self,
        phase: CardiacPhase,
        cycle_number: int = 0,
        heart_rate: float = 0.0,
        progress: float = 0.5,
    ):
        """
        Kardiyak fazı ayarla.

        Args:
            phase: Kardiyak faz
            cycle_number: Döngü numarası
            heart_rate: Kalp hızı (BPM)
            progress: Faz içindeki ilerleme (0-1)
        """
        self._current_phase = phase
        self._cycle_number = cycle_number
        self._heart_rate = heart_rate

        # Göstergeyi güncelle
        self.phase_widget.set_phase(phase, progress)

        # Etiketleri güncelle
        self._update_labels()

    def animate_phase_transition(self, from_phase: CardiacPhase, to_phase: CardiacPhase):
        """
        Faz geçişini animasyonlu göster.

        Args:
            from_phase: Başlangıç fazı
            to_phase: Hedef faz
        """
        self.phase_widget.animate_to_phase(to_phase)
        self._current_phase = to_phase
        self._update_labels()

    def _update_labels(self):
        """Bilgi etiketlerini güncelle."""
        if self._current_phase:
            # Faz kodu
            self.phase_label.setText(self._current_phase.value.upper())

            # Faz rengi
            color = self.phase_widget._phase_colors.get(self._current_phase, QColor(0, 0, 0))
            self.phase_label.setStyleSheet(
                f"font-weight: bold; font-size: 14px; color: {color.name()};"
            )

            # Faz açıklaması
            descriptions = {
                CardiacPhase.D1: "Ventrikül gevşemesi ve dolumu",
                CardiacPhase.D2: "Ventrikül dolumu tamamlandı",
                CardiacPhase.S1: "Ventrikül kasılması başlıyor",
                CardiacPhase.S2: "Ventrikül kasılması sona eriyor",
            }
            desc = descriptions.get(self._current_phase, "")
            self.phase_desc_label.setText(desc)
        else:
            self.phase_label.setText("--")
            self.phase_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.phase_desc_label.setText("--")

        # Kalp hızı
        if self._heart_rate > 0:
            self.hr_label.setText(f"{self._heart_rate:.0f} BPM")
        else:
            self.hr_label.setText("-- BPM")

        # Döngü numarası
        if self._cycle_number > 0:
            self.cycle_label.setText(f"#{self._cycle_number}")
        else:
            self.cycle_label.setText("#--")

    def clear(self):
        """Göstergeyi temizle."""
        self._current_phase = None
        self._heart_rate = 0.0
        self._cycle_number = 0
        self._update_labels()
