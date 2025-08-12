"""
RWS (Radial Wall Strain) Analysis Data Models

Bu modül RWS analizi için kullanılan veri yapılarını tanımlar.
Clean architecture prensipleri doğrultusunda, business logic'ten bağımsız
düz veri yapıları (Data Transfer Objects - DTOs) olarak tasarlanmıştır.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class RiskLevel(Enum):
    """
    RWS risk seviyeleri enumerasyonu.
    
    Klinik çalışmalara göre:
    - LOW: RWS < 10% (düşük plak yırtılma riski)
    - MODERATE: 10% <= RWS < 14.25% (orta risk)
    - HIGH: RWS >= 14.25% (yüksek plak yırtılma riski)
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass
class RWSAnalysisResult:
    """
    RWS analiz sonuçlarını kapsayan veri yapısı.
    
    Attributes:
        success (bool): Analiz başarılı mı tamamlandı
        rws_max (float): Maksimum RWS değeri (yüzde olarak)
        rws_stenosis (float): Stenoz bölgesindeki RWS değeri
        rws_max_location (int): Maksimum RWS'nin bulunduğu indeks
        diameter_change_mm (float): Çap değişimi (mm cinsinden)
        reference_diameter_mm (float): Referans çap değeri (mm)
        end_diastole_frame (int): Diyastol sonu frame numarası
        end_systole_frame (int): Sistol sonu frame numarası
        risk_level (RiskLevel): Hesaplanan risk seviyesi
        interpretation (str): Sonuçların klinik yorumu
        error (Optional[str]): Hata durumunda hata mesajı
        rws_values (Optional[List[float]]): Tüm RWS değerleri listesi
        metadata (Dict): Ek metadata bilgileri
    """
    success: bool
    rws_max: float = 0.0
    rws_stenosis: float = 0.0
    rws_max_location: int = -1
    diameter_change_mm: float = 0.0
    reference_diameter_mm: float = 0.0
    end_diastole_frame: int = -1
    end_systole_frame: int = -1
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    interpretation: str = ""
    error: Optional[str] = None
    rws_values: Optional[List[float]] = None
    metadata: Dict = None
    
    def __post_init__(self):
        """Veri doğrulama ve varsayılan değer atamaları"""
        if self.metadata is None:
            self.metadata = {}
            
        # Risk seviyesini RWS değerine göre otomatik hesapla
        if self.success and self.risk_level == RiskLevel.UNKNOWN:
            self.risk_level = self._calculate_risk_level()
            
    def _calculate_risk_level(self) -> RiskLevel:
        """
        RWS değerine göre risk seviyesini hesaplar.
        
        Returns:
            RiskLevel: Hesaplanan risk seviyesi
        """
        if self.rws_max < 10.0:
            return RiskLevel.LOW
        elif self.rws_max < 14.25:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH


@dataclass
class RWSAnalysisProgress:
    """
    RWS analiz ilerlemesini takip eden veri yapısı.
    
    Attributes:
        status (str): Mevcut durum mesajı
        percentage (int): İlerleme yüzdesi (0-100)
        is_complete (bool): Analiz tamamlandı mı
        is_cancelled (bool): Analiz iptal edildi mi
    """
    status: str
    percentage: int
    is_complete: bool = False
    is_cancelled: bool = False
    
    def __post_init__(self):
        """Değer doğrulaması"""
        # Yüzde değerini 0-100 aralığında tut
        self.percentage = max(0, min(100, self.percentage))


@dataclass
class RWSAnalysisRequest:
    """
    RWS analiz isteği parametrelerini kapsayan veri yapısı.
    
    Attributes:
        qca_results_by_frame (Dict): Frame bazlı QCA sonuçları
        cardiac_phase_info (Dict): Kardiyak faz bilgileri
        selected_beat (Optional[int]): Seçili kalp atışı indeksi
        analysis_options (Dict): Ek analiz seçenekleri
    """
    qca_results_by_frame: Dict
    cardiac_phase_info: Dict
    selected_beat: Optional[int] = None
    analysis_options: Dict = None
    
    def __post_init__(self):
        """Varsayılan değer atamaları"""
        if self.analysis_options is None:
            self.analysis_options = {}