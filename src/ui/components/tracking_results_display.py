"""
Tracking Results Display Component

İzleme sonuçlarını görselleştiren bileşen.
Clean Architecture prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Dict, Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QTextCharFormat, QColor
import numpy as np
import logging
from datetime import datetime

from src.domain.models.tracking_models import (
    TrackingResult, TrackingSession, TrackedPoint,
    TrackingStatus, Point2D
)

logger = logging.getLogger(__name__)


class TrackingResultsDisplay(QWidget):
    """
    İzleme sonuçları gösterimi.
    
    Bu widget:
    - Anlık izleme sonuçları
    - İzleme geçmişi
    - Hareket yörüngeleri
    - Performans metrikleri
    
    Signals:
        export_requested: Dışa aktarma istendi
        trajectory_hover: Yörünge üzerine gelindi (point_id, frame_number)
    """
    
    # Signals
    export_requested = pyqtSignal()
    trajectory_hover = pyqtSignal(str, int)  # point_id, frame_number
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        TrackingResultsDisplay constructor.
        
        Args:
            parent: Ana widget
        """
        super().__init__(parent)
        self._current_result: Optional[TrackingResult] = None
        self._session: Optional[TrackingSession] = None
        self._performance_history: List[float] = []
        self._setup_ui()
        self._setup_timer()
        
    def _setup_ui(self):
        """UI bileşenlerini oluştur."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        
        # Anlık sonuçlar sekmesi
        self.instant_tab = self._create_instant_results_tab()
        self.tab_widget.addTab(self.instant_tab, "Anlık Sonuçlar")
        
        # Yörünge analizi sekmesi
        self.trajectory_tab = self._create_trajectory_tab()
        self.tab_widget.addTab(self.trajectory_tab, "Yörünge Analizi")
        
        # Performans sekmesi
        self.performance_tab = self._create_performance_tab()
        self.tab_widget.addTab(self.performance_tab, "Performans")
        
        layout.addWidget(self.tab_widget)
        
        # Dışa aktarma butonu
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("Sonuçları Dışa Aktar")
        self.export_btn.setIcon(QIcon.fromTheme("document-save"))
        self.export_btn.clicked.connect(self.export_requested.emit)
        export_layout.addWidget(self.export_btn)
        
        layout.addLayout(export_layout)
        
    def _create_instant_results_tab(self) -> QWidget:
        """Anlık sonuçlar sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Durum göstergesi
        status_group = QGroupBox("İzleme Durumu")
        status_layout = QVBoxLayout()
        
        # Ana durum
        self.status_label = QLabel("İzleme Bekliyor...")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.status_label.setFont(font)
        status_layout.addWidget(self.status_label)
        
        # Detaylı bilgiler
        info_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame: -")
        self.success_rate_label = QLabel("Başarı: -%")
        self.processing_time_label = QLabel("İşlem: - ms")
        
        info_layout.addWidget(self.frame_label)
        info_layout.addWidget(self.success_rate_label)
        info_layout.addWidget(self.processing_time_label)
        info_layout.addStretch()
        
        status_layout.addLayout(info_layout)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Nokta durumları tablosu
        points_group = QGroupBox("Nokta Durumları")
        points_layout = QVBoxLayout()
        
        self.instant_table = QTableWidget()
        self.instant_table.setColumnCount(6)
        self.instant_table.setHorizontalHeaderLabels([
            "Nokta", "X", "Y", "Hız", "Güven", "Durum"
        ])
        
        # Sütun genişlikleri
        header = self.instant_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.instant_table.setColumnWidth(0, 80)
        self.instant_table.setColumnWidth(1, 60)
        self.instant_table.setColumnWidth(2, 60)
        self.instant_table.setColumnWidth(3, 60)
        self.instant_table.setColumnWidth(4, 60)
        
        points_layout.addWidget(self.instant_table)
        points_group.setLayout(points_layout)
        layout.addWidget(points_group)
        
        return widget
        
    def _create_trajectory_tab(self) -> QWidget:
        """Yörünge analizi sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Yörünge istatistikleri
        stats_group = QGroupBox("Yörünge İstatistikleri")
        stats_layout = QVBoxLayout()
        
        self.trajectory_table = QTableWidget()
        self.trajectory_table.setColumnCount(7)
        self.trajectory_table.setHorizontalHeaderLabels([
            "Nokta", "Toplam Mesafe", "Ortalama Hız", "Max Hız", 
            "Yön Değişimi", "Düzgünlük", "Frame Sayısı"
        ])
        
        # Sütun genişlikleri
        header = self.trajectory_table.horizontalHeader()
        header.setStretchLastSection(False)
        for i in range(7):
            self.trajectory_table.setColumnWidth(i, 100)
        
        stats_layout.addWidget(self.trajectory_table)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Hareket karakteristiği
        char_group = QGroupBox("Hareket Karakteristiği")
        char_layout = QVBoxLayout()
        
        self.motion_text = QTextEdit()
        self.motion_text.setReadOnly(True)
        self.motion_text.setMaximumHeight(150)
        
        char_layout.addWidget(self.motion_text)
        char_group.setLayout(char_layout)
        layout.addWidget(char_group)
        
        return widget
        
    def _create_performance_tab(self) -> QWidget:
        """Performans sekmesi oluştur."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performans metrikleri
        metrics_group = QGroupBox("Performans Metrikleri")
        metrics_layout = QVBoxLayout()
        
        # Özet bilgiler
        summary_layout = QHBoxLayout()
        
        self.avg_time_label = QLabel("Ortalama İşlem: - ms")
        self.fps_label = QLabel("FPS: -")
        self.cpu_label = QLabel("CPU: -%")
        self.memory_label = QLabel("Bellek: - MB")
        
        summary_layout.addWidget(self.avg_time_label)
        summary_layout.addWidget(self.fps_label)
        summary_layout.addWidget(self.cpu_label)
        summary_layout.addWidget(self.memory_label)
        summary_layout.addStretch()
        
        metrics_layout.addLayout(summary_layout)
        
        # Detaylı metrikler
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.performance_text.setFont(QFont("Monospace", 9))
        
        metrics_layout.addWidget(self.performance_text)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        return widget
        
    def _setup_timer(self):
        """Güncelleme zamanlayıcısı kur."""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_performance_display)
        self.update_timer.setInterval(1000)  # 1 saniye
        
    def update_result(self, result: TrackingResult):
        """
        İzleme sonucunu güncelle.
        
        Args:
            result: İzleme sonucu
        """
        self._current_result = result
        
        # Performans geçmişi
        self._performance_history.append(result.processing_time_ms)
        if len(self._performance_history) > 100:
            self._performance_history.pop(0)
        
        # Anlık sonuçları güncelle
        self._update_instant_display()
        
        # Timer'ı başlat
        if not self.update_timer.isActive():
            self.update_timer.start()
            
    def update_session(self, session: TrackingSession):
        """
        İzleme oturumunu güncelle.
        
        Args:
            session: İzleme oturumu
        """
        self._session = session
        self._update_trajectory_display()
        
    def clear_results(self):
        """Sonuçları temizle."""
        self._current_result = None
        self._session = None
        self._performance_history.clear()
        
        # Gösterimleri temizle
        self.status_label.setText("İzleme Bekliyor...")
        self.frame_label.setText("Frame: -")
        self.success_rate_label.setText("Başarı: -%")
        self.processing_time_label.setText("İşlem: - ms")
        
        self.instant_table.setRowCount(0)
        self.trajectory_table.setRowCount(0)
        self.motion_text.clear()
        self.performance_text.clear()
        
        # Timer'ı durdur
        self.update_timer.stop()
        
    def _update_instant_display(self):
        """Anlık sonuç gösterimini güncelle."""
        if not self._current_result:
            return
            
        result = self._current_result
        
        # Durum
        if result.success:
            self.status_label.setText("İzleme Aktif")
            self.status_label.setStyleSheet("color: green;")
        else:
            error_msg = result.error_message or "Bilinmeyen hata"
            self.status_label.setText(f"Hata: {error_msg}")
            self.status_label.setStyleSheet("color: red;")
            
        # Bilgiler
        self.frame_label.setText(f"Frame: {result.frame_number}")
        self.success_rate_label.setText(f"Başarı: {result.success_rate*100:.0f}%")
        self.processing_time_label.setText(f"İşlem: {result.processing_time_ms:.1f} ms")
        
        # Nokta tablosu
        self.instant_table.setRowCount(len(result.tracked_points))
        
        for row, point in enumerate(result.tracked_points):
            # Nokta adı
            self.instant_table.setItem(row, 0, QTableWidgetItem(point.name))
            
            # Pozisyon
            self.instant_table.setItem(row, 1, QTableWidgetItem(f"{point.current_position.x:.1f}"))
            self.instant_table.setItem(row, 2, QTableWidgetItem(f"{point.current_position.y:.1f}"))
            
            # Hız
            velocity = point.get_velocity()
            if velocity:
                speed = np.sqrt(velocity.x**2 + velocity.y**2)
                self.instant_table.setItem(row, 3, QTableWidgetItem(f"{speed:.1f}"))
            else:
                self.instant_table.setItem(row, 3, QTableWidgetItem("-"))
                
            # Güven
            self.instant_table.setItem(row, 4, QTableWidgetItem(f"{point.confidence:.2f}"))
            
            # Durum
            status_item = QTableWidgetItem(self._get_status_text(point.status))
            status_item.setForeground(self._get_status_color(point.status))
            self.instant_table.setItem(row, 5, status_item)
            
    def _update_trajectory_display(self):
        """Yörünge analizini güncelle."""
        if not self._session:
            return
            
        # Yörünge istatistikleri tablosu
        points = self._session.get_all_points()
        self.trajectory_table.setRowCount(len(points))
        
        motion_analysis = []
        
        for row, point in enumerate(points):
            trajectory = point.get_trajectory()
            
            if len(trajectory) < 2:
                continue
                
            # İstatistikleri hesapla
            stats = self._calculate_trajectory_stats(trajectory)
            
            # Tabloya ekle
            self.trajectory_table.setItem(row, 0, QTableWidgetItem(point.name))
            self.trajectory_table.setItem(row, 1, QTableWidgetItem(f"{stats['total_distance']:.1f} px"))
            self.trajectory_table.setItem(row, 2, QTableWidgetItem(f"{stats['avg_speed']:.1f} px/f"))
            self.trajectory_table.setItem(row, 3, QTableWidgetItem(f"{stats['max_speed']:.1f} px/f"))
            self.trajectory_table.setItem(row, 4, QTableWidgetItem(f"{stats['direction_changes']}"))
            self.trajectory_table.setItem(row, 5, QTableWidgetItem(f"{stats['smoothness']:.2f}"))
            self.trajectory_table.setItem(row, 6, QTableWidgetItem(f"{len(trajectory)}"))
            
            # Hareket karakteristiği analizi
            motion_type = self._analyze_motion_type(trajectory)
            motion_analysis.append(f"{point.name}: {motion_type}")
            
        # Hareket karakteristiği metni
        self.motion_text.clear()
        self.motion_text.append("Hareket Analizi:\n")
        for analysis in motion_analysis:
            self.motion_text.append(f"• {analysis}")
            
    def _update_performance_display(self):
        """Performans gösterimini güncelle."""
        if not self._performance_history:
            return
            
        # Ortalama işlem süresi
        avg_time = np.mean(self._performance_history)
        self.avg_time_label.setText(f"Ortalama İşlem: {avg_time:.1f} ms")
        
        # FPS
        if avg_time > 0:
            fps = 1000 / avg_time
            self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Detaylı metrikler
        self.performance_text.clear()
        self.performance_text.append("Performans İstatistikleri\n" + "="*30 + "\n")
        
        if self._performance_history:
            self.performance_text.append(f"Ortalama: {np.mean(self._performance_history):.1f} ms")
            self.performance_text.append(f"Minimum: {np.min(self._performance_history):.1f} ms")
            self.performance_text.append(f"Maksimum: {np.max(self._performance_history):.1f} ms")
            self.performance_text.append(f"Std Sapma: {np.std(self._performance_history):.1f} ms")
            
        if self._current_result:
            self.performance_text.append(f"\nSon Frame:")
            self.performance_text.append(f"İzlenen Nokta: {len(self._current_result.tracked_points)}")
            self.performance_text.append(f"Başarısız Nokta: {len(self._current_result.failed_points)}")
            
    def _calculate_trajectory_stats(self, trajectory: List[Point2D]) -> Dict:
        """
        Yörünge istatistiklerini hesapla.
        
        Args:
            trajectory: Nokta listesi
            
        Returns:
            Dict: İstatistikler
        """
        # Mesafe ve hız hesaplama
        distances = []
        speeds = []
        
        for i in range(1, len(trajectory)):
            dist = trajectory[i].distance_to(trajectory[i-1])
            distances.append(dist)
            speeds.append(dist)  # frame başına piksel
            
        # Yön değişimi sayısı
        direction_changes = 0
        if len(trajectory) >= 3:
            for i in range(2, len(trajectory)):
                v1 = (trajectory[i-1].x - trajectory[i-2].x,
                      trajectory[i-1].y - trajectory[i-2].y)
                v2 = (trajectory[i].x - trajectory[i-1].x,
                      trajectory[i].y - trajectory[i-1].y)
                      
                # Vektör çarpımı ile yön değişimi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross) > 0.1:  # Eşik
                    direction_changes += 1
                    
        # Düzgünlük (hız varyansı)
        smoothness = 1.0 / (1.0 + np.std(speeds)) if speeds else 0.0
        
        return {
            'total_distance': sum(distances),
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'direction_changes': direction_changes,
            'smoothness': smoothness
        }
        
    def _analyze_motion_type(self, trajectory: List[Point2D]) -> str:
        """
        Hareket tipini analiz et.
        
        Args:
            trajectory: Nokta listesi
            
        Returns:
            str: Hareket tipi açıklaması
        """
        if len(trajectory) < 10:
            return "Yetersiz veri"
            
        # Hız profili
        speeds = []
        for i in range(1, len(trajectory)):
            dist = trajectory[i].distance_to(trajectory[i-1])
            speeds.append(dist)
            
        avg_speed = np.mean(speeds)
        speed_std = np.std(speeds)
        
        # Hareket karakteristiği
        if speed_std < avg_speed * 0.2:
            motion = "Düzgün hareket"
        elif speed_std < avg_speed * 0.5:
            motion = "Hafif değişken hareket"
        else:
            motion = "Yüksek değişken hareket"
            
        # Periyodiklik kontrolü
        # Basit FFT analizi
        if len(speeds) >= 20:
            fft = np.fft.fft(speeds)
            power = np.abs(fft[1:len(fft)//2])
            if len(power) > 0:
                max_power = np.max(power)
                mean_power = np.mean(power)
                if max_power > mean_power * 3:
                    motion += " (Periyodik)"
                    
        return motion
        
    def _get_status_text(self, status: TrackingStatus) -> str:
        """Durum metni döndür."""
        status_texts = {
            TrackingStatus.ACTIVE: "Aktif",
            TrackingStatus.LOST: "Kayıp",
            TrackingStatus.OCCLUDED: "Gizli",
            TrackingStatus.MANUAL: "Manuel",
            TrackingStatus.PREDICTED: "Tahmin",
            TrackingStatus.VALIDATED: "Doğrulandı"
        }
        return status_texts.get(status, "Bilinmiyor")
        
    def _get_status_color(self, status: TrackingStatus) -> QColor:
        """Durum rengi döndür."""
        status_colors = {
            TrackingStatus.ACTIVE: QColor(0, 150, 0),
            TrackingStatus.LOST: QColor(200, 0, 0),
            TrackingStatus.OCCLUDED: QColor(200, 100, 0),
            TrackingStatus.MANUAL: QColor(0, 0, 200),
            TrackingStatus.PREDICTED: QColor(150, 150, 0),
            TrackingStatus.VALIDATED: QColor(0, 100, 0)
        }
        return status_colors.get(status, QColor(0, 0, 0))