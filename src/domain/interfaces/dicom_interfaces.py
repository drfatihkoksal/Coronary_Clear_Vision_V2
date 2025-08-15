"""
DICOM Interface Definitions

DICOM işlemleri için kullanılan arayüzleri tanımlar.
Dependency Inversion Principle (DIP) uygulaması için abstraksiyonlar sağlar.
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np

from src.domain.models.dicom_models import (
    DicomStudy,
    DicomSeries,
    DicomLoadRequest,
    DicomLoadResult,
    DicomWindowLevel,
    DicomProjectionInfo,
    DicomModality,
)


class IDicomReader(Protocol):
    """
    DICOM okuyucu protokolü.

    Farklı DICOM okuma implementasyonları için.
    """

    def read_file(self, file_path: Path) -> DicomLoadResult:
        """
        DICOM dosyası oku.

        Args:
            file_path: Dosya yolu

        Returns:
            DicomLoadResult: Okuma sonucu
        """
        ...

    def read_dicomdir(self, dicomdir_path: Path) -> DicomLoadResult:
        """
        DICOMDIR oku.

        Args:
            dicomdir_path: DICOMDIR yolu

        Returns:
            DicomLoadResult: Okuma sonucu
        """
        ...

    def validate_file(self, file_path: Path) -> bool:
        """
        DICOM dosyasını doğrula.

        Args:
            file_path: Dosya yolu

        Returns:
            bool: Geçerli mi?
        """
        ...

    @property
    def supported_formats(self) -> List[str]:
        """Desteklenen formatlar"""
        ...


class IDicomLoader(ABC):
    """
    DICOM yükleyici abstract base class.

    Ana DICOM yükleme iş mantığını tanımlar.
    """

    @abstractmethod
    def load(self, request: DicomLoadRequest) -> DicomLoadResult:
        """
        DICOM yükle.

        Args:
            request: Yükleme isteği

        Returns:
            DicomLoadResult: Yükleme sonucu
        """

    @abstractmethod
    def load_folder(self, folder_path: Path) -> List[DicomLoadResult]:
        """
        Klasörden DICOM'ları yükle.

        Args:
            folder_path: Klasör yolu

        Returns:
            List[DicomLoadResult]: Yükleme sonuçları
        """

    @abstractmethod
    def get_projections(self, file_path: Path) -> List[DicomProjectionInfo]:
        """
        Mevcut projeksiyonları al.

        Args:
            file_path: DICOM veya DICOMDIR yolu

        Returns:
            List[DicomProjectionInfo]: Projeksiyon listesi
        """


class IDicomWindowLevelProvider(Protocol):
    """
    Pencere/seviye sağlayıcı protokolü.

    Farklı modaliteler için pencere/seviye preset'leri.
    """

    def get_presets(self, modality: DicomModality) -> Dict[str, DicomWindowLevel]:
        """
        Modaliteye göre preset'leri al.

        Args:
            modality: DICOM modalitesi

        Returns:
            Dict[str, DicomWindowLevel]: Preset haritası
        """
        ...

    def get_default(self, modality: DicomModality) -> DicomWindowLevel:
        """
        Varsayılan pencere/seviye al.

        Args:
            modality: DICOM modalitesi

        Returns:
            DicomWindowLevel: Varsayılan ayarlar
        """
        ...

    def calculate_auto(self, pixel_array: np.ndarray) -> DicomWindowLevel:
        """
        Otomatik pencere/seviye hesapla.

        Args:
            pixel_array: Piksel verisi

        Returns:
            DicomWindowLevel: Hesaplanan ayarlar
        """
        ...


class IDicomFrameExtractor(Protocol):
    """
    DICOM frame çıkarıcı protokolü.

    Multi-frame DICOM'lardan frame çıkarma.
    """

    def extract_frame(
        self, series: DicomSeries, frame_index: int, apply_window_level: bool = True
    ) -> Optional[np.ndarray]:
        """
        Frame çıkar.

        Args:
            series: DICOM serisi
            frame_index: Frame indeksi
            apply_window_level: Pencere/seviye uygulansın mı?

        Returns:
            Optional[np.ndarray]: Frame verisi
        """
        ...

    def extract_all_frames(
        self, series: DicomSeries, apply_window_level: bool = True
    ) -> List[np.ndarray]:
        """
        Tüm frame'leri çıkar.

        Args:
            series: DICOM serisi
            apply_window_level: Pencere/seviye uygulansın mı?

        Returns:
            List[np.ndarray]: Frame listesi
        """
        ...

    def extract_time_range(
        self,
        series: DicomSeries,
        start_time: float,
        end_time: float,
        apply_window_level: bool = True,
    ) -> List[np.ndarray]:
        """
        Zaman aralığındaki frame'leri çıkar.

        Args:
            series: DICOM serisi
            start_time: Başlangıç zamanı (saniye)
            end_time: Bitiş zamanı (saniye)
            apply_window_level: Pencere/seviye uygulansın mı?

        Returns:
            List[np.ndarray]: Frame listesi
        """
        ...


class IDicomExporter(Protocol):
    """
    DICOM dışa aktarıcı protokolü.

    DICOM verilerini farklı formatlara aktarma.
    """

    def export_frame(self, frame: np.ndarray, output_path: Path, format: str = "png") -> bool:
        """
        Frame'i dışa aktar.

        Args:
            frame: Frame verisi
            output_path: Çıktı yolu
            format: Çıktı formatı

        Returns:
            bool: Başarılı mı?
        """
        ...

    def export_series_as_video(
        self, series: DicomSeries, output_path: Path, fps: int = 30, codec: str = "mp4v"
    ) -> bool:
        """
        Seriyi video olarak dışa aktar.

        Args:
            series: DICOM serisi
            output_path: Çıktı yolu
            fps: Frame hızı
            codec: Video codec'i

        Returns:
            bool: Başarılı mı?
        """
        ...

    def export_study_report(
        self, study: DicomStudy, output_path: Path, include_images: bool = True
    ) -> bool:
        """
        Çalışma raporu oluştur.

        Args:
            study: DICOM çalışması
            output_path: Çıktı yolu
            include_images: Görüntüler dahil edilsin mi?

        Returns:
            bool: Başarılı mı?
        """
        ...

    @property
    def supported_formats(self) -> List[str]:
        """Desteklenen dışa aktarma formatları"""
        ...


class IDicomAnonymizer(Protocol):
    """
    DICOM anonimleştirici protokolü.

    Hasta bilgilerini anonimleştirme.
    """

    def anonymize_study(self, study: DicomStudy, preserve_dates: bool = False) -> DicomStudy:
        """
        Çalışmayı anonimleştir.

        Args:
            study: DICOM çalışması
            preserve_dates: Tarihleri koru

        Returns:
            DicomStudy: Anonimleştirilmiş çalışma
        """
        ...

    def get_anonymization_rules(self) -> Dict[str, str]:
        """
        Anonimleştirme kurallarını al.

        Returns:
            Dict[str, str]: Tag -> değişim kuralları
        """
        ...

    def set_anonymization_rule(self, tag: str, rule: str):
        """
        Anonimleştirme kuralı ekle.

        Args:
            tag: DICOM tag'i
            rule: Kural (remove, replace, hash, vb.)
        """
        ...


class IDicomMetadataExtractor(Protocol):
    """
    DICOM metadata çıkarıcı protokolü.

    DICOM başlık bilgilerini çıkarma.
    """

    def extract_patient_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Hasta bilgilerini çıkar.

        Args:
            file_path: DICOM dosyası

        Returns:
            Dict[str, Any]: Hasta bilgileri
        """
        ...

    def extract_study_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Çalışma bilgilerini çıkar.

        Args:
            file_path: DICOM dosyası

        Returns:
            Dict[str, Any]: Çalışma bilgileri
        """
        ...

    def extract_equipment_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Cihaz bilgilerini çıkar.

        Args:
            file_path: DICOM dosyası

        Returns:
            Dict[str, Any]: Cihaz bilgileri
        """
        ...

    def extract_custom_tags(self, file_path: Path, tags: List[str]) -> Dict[str, Any]:
        """
        Özel tag'leri çıkar.

        Args:
            file_path: DICOM dosyası
            tags: Tag listesi

        Returns:
            Dict[str, Any]: Tag değerleri
        """
        ...


class IDicomPixelProcessor(Protocol):
    """
    DICOM piksel işleyici protokolü.

    Piksel verisi dönüşümleri.
    """

    def normalize_pixels(
        self, pixel_array: np.ndarray, output_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """
        Piksel değerlerini normalize et.

        Args:
            pixel_array: Piksel verisi
            output_range: Çıktı aralığı

        Returns:
            np.ndarray: Normalize edilmiş veri
        """
        ...

    def apply_lut(self, pixel_array: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        Look-up table uygula.

        Args:
            pixel_array: Piksel verisi
            lut: Look-up table

        Returns:
            np.ndarray: Dönüştürülmüş veri
        """
        ...

    def convert_to_hounsfield(
        self, pixel_array: np.ndarray, slope: float, intercept: float
    ) -> np.ndarray:
        """
        Hounsfield birimlerine çevir (CT için).

        Args:
            pixel_array: Piksel verisi
            slope: Rescale slope
            intercept: Rescale intercept

        Returns:
            np.ndarray: HU değerleri
        """
        ...

    def enhance_contrast(self, pixel_array: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Kontrast iyileştirme.

        Args:
            pixel_array: Piksel verisi
            method: İyileştirme yöntemi

        Returns:
            np.ndarray: İyileştirilmiş veri
        """
        ...


class IDicomValidator(Protocol):
    """
    DICOM doğrulayıcı protokolü.

    DICOM dosya ve veri doğrulama.
    """

    def validate_file_format(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Dosya formatını doğrula.

        Args:
            file_path: Dosya yolu

        Returns:
            Tuple: (geçerli mi, hata/uyarı listesi)
        """
        ...

    def validate_required_tags(
        self, file_path: Path, required_tags: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Gerekli tag'leri doğrula.

        Args:
            file_path: Dosya yolu
            required_tags: Gerekli tag listesi

        Returns:
            Tuple: (geçerli mi, eksik tag listesi)
        """
        ...

    def validate_pixel_data(self, study: DicomStudy) -> Tuple[bool, List[str]]:
        """
        Piksel verisini doğrula.

        Args:
            study: DICOM çalışması

        Returns:
            Tuple: (geçerli mi, hata listesi)
        """
        ...

    def check_integrity(self, file_path: Path) -> bool:
        """
        Dosya bütünlüğünü kontrol et.

        Args:
            file_path: Dosya yolu

        Returns:
            bool: Bütünlük sağlam mı?
        """
        ...


class IDicomCacheManager(Protocol):
    """
    DICOM önbellek yöneticisi protokolü.

    Performans için önbellekleme.
    """

    def cache_study(self, study: DicomStudy):
        """
        Çalışmayı önbelleğe al.

        Args:
            study: DICOM çalışması
        """
        ...

    def get_cached_study(self, study_uid: str) -> Optional[DicomStudy]:
        """
        Önbellekten çalışma al.

        Args:
            study_uid: Çalışma UID

        Returns:
            Optional[DicomStudy]: Önbellekteki çalışma
        """
        ...

    def cache_frame(self, series_uid: str, frame_index: int, frame_data: np.ndarray):
        """
        Frame'i önbelleğe al.

        Args:
            series_uid: Seri UID
            frame_index: Frame indeksi
            frame_data: Frame verisi
        """
        ...

    def get_cached_frame(self, series_uid: str, frame_index: int) -> Optional[np.ndarray]:
        """
        Önbellekten frame al.

        Args:
            series_uid: Seri UID
            frame_index: Frame indeksi

        Returns:
            Optional[np.ndarray]: Önbellekteki frame
        """
        ...

    def clear_cache(self):
        """Önbelleği temizle."""
        ...

    def get_cache_size(self) -> int:
        """
        Önbellek boyutunu al.

        Returns:
            int: Bayt cinsinden boyut
        """
        ...

    def set_cache_limit(self, size_mb: int):
        """
        Önbellek limitini ayarla.

        Args:
            size_mb: MB cinsinden limit
        """
        ...
