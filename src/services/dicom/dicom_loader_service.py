"""
DICOM Loader Service

DICOM dosyalarını yükleyen ana servis.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Optional, Dict
from pathlib import Path
import logging
from datetime import datetime
import pydicom
import numpy as np

from src.domain.models.dicom_models import (
    DicomStudy,
    DicomSeries,
    DicomFrame,
    DicomLoadRequest,
    DicomLoadResult,
    DicomPatientInfo,
    DicomStudyInfo,
    DicomSeriesInfo,
    DicomImageInfo,
    DicomProjectionInfo,
    DicomPixelSpacing,
    DicomWindowLevel,
    DicomModality,
    ProjectionType,
)
from src.domain.interfaces.dicom_interfaces import (
    IDicomLoader,
    IDicomReader,
    IDicomWindowLevelProvider,
    IDicomValidator,
)

logger = logging.getLogger(__name__)


class DicomLoaderService(IDicomLoader):
    """
    DICOM yükleme servisi implementasyonu.

    Bu servis:
    - DICOM dosyalarını okur
    - DICOMDIR'leri işler
    - Multi-frame desteği sağlar
    - Metadata çıkarımı yapar
    - Validasyon gerçekleştirir
    """

    def __init__(
        self,
        reader: Optional[IDicomReader] = None,
        window_provider: Optional[IDicomWindowLevelProvider] = None,
        validator: Optional[IDicomValidator] = None,
    ):
        """
        DicomLoaderService constructor.

        Args:
            reader: DICOM okuyucu
            window_provider: Pencere/seviye sağlayıcı
            validator: DICOM doğrulayıcı
        """
        self._reader = reader or PydicomReader()
        self._window_provider = window_provider or DefaultWindowLevelProvider()
        self._validator = validator

        logger.info("DicomLoaderService initialized")

    def load(self, request: DicomLoadRequest) -> DicomLoadResult:
        """
        DICOM yükle.

        Args:
            request: Yükleme isteği

        Returns:
            DicomLoadResult: Yükleme sonucu
        """
        start_time = datetime.now()

        try:
            # Dosya doğrulama
            if self._validator and not self._validator.validate_file_format(request.file_path)[0]:
                return DicomLoadResult(
                    success=False, error_message=f"Invalid DICOM file: {request.file_path}"
                )

            # DICOMDIR kontrolü
            if request.file_path.name == "DICOMDIR":
                result = self._load_dicomdir(request)
            else:
                result = self._load_single_file(request)

            # Süre hesapla
            load_time = (datetime.now() - start_time).total_seconds() * 1000
            result.load_time_ms = load_time

            logger.info(f"DICOM loaded in {load_time:.1f}ms: {request.file_path}")

            return result

        except Exception as e:
            logger.error(f"Error loading DICOM: {str(e)}")
            return DicomLoadResult(
                success=False,
                error_message=str(e),
                load_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def load_folder(self, folder_path: Path) -> List[DicomLoadResult]:
        """
        Klasörden DICOM'ları yükle.

        Args:
            folder_path: Klasör yolu

        Returns:
            List[DicomLoadResult]: Yükleme sonuçları
        """
        results = []

        try:
            # DICOM dosyalarını bul
            dicom_files = self._find_dicom_files(folder_path)

            logger.info(f"Found {len(dicom_files)} DICOM files in {folder_path}")

            # Her dosyayı yükle
            for file_path in dicom_files:
                request = DicomLoadRequest(file_path=file_path, load_pixel_data=True)
                result = self.load(request)
                results.append(result)

        except Exception as e:
            logger.error(f"Error loading folder: {str(e)}")

        return results

    def get_projections(self, file_path: Path) -> List[DicomProjectionInfo]:
        """
        Mevcut projeksiyonları al.

        Args:
            file_path: DICOM veya DICOMDIR yolu

        Returns:
            List[DicomProjectionInfo]: Projeksiyon listesi
        """
        projections = []

        try:
            if file_path.name == "DICOMDIR":
                projections = self._get_dicomdir_projections(file_path)
            else:
                # Tek dosya için projeksiyon bilgisi
                projection = self._extract_projection_info(file_path)
                if projection:
                    projections.append(projection)

        except Exception as e:
            logger.error(f"Error getting projections: {str(e)}")

        return projections

    def _load_single_file(self, request: DicomLoadRequest) -> DicomLoadResult:
        """
        Tek DICOM dosyası yükle.

        Args:
            request: Yükleme isteği

        Returns:
            DicomLoadResult: Yükleme sonucu
        """
        try:
            # Dosyayı oku
            dataset = pydicom.dcmread(str(request.file_path), force=True)

            # Patient bilgileri
            patient_info = self._extract_patient_info(dataset)

            # Study bilgileri
            study_info = self._extract_study_info(dataset)

            # Series bilgileri
            series_info = self._extract_series_info(dataset)

            # Görüntü bilgileri
            self._extract_image_info(dataset)

            # Piksel aralığı
            pixel_spacing = self._extract_pixel_spacing(dataset)

            # Pencere/seviye
            window_level = self._extract_window_level(dataset, series_info.modality)

            # Projeksiyon bilgileri
            projection_info = self._extract_projection_info_from_dataset(dataset)

            # Frame'leri yükle
            frames = []
            if request.load_pixel_data:
                frames = self._load_frames(dataset, request.specific_frames)

            # Frame rate
            frame_rate = self._extract_frame_rate(dataset)

            # Seri oluştur
            series = DicomSeries(
                info=series_info,
                frames=frames,
                pixel_spacing=pixel_spacing,
                window_level=window_level,
                projection_info=projection_info,
                frame_rate=frame_rate,
            )

            # Çalışma oluştur
            study = DicomStudy(
                patient_info=patient_info,
                study_info=study_info,
                series_list=[series],
                file_path=request.file_path,
                is_dicomdir=False,
            )

            # Uyarıları topla
            warnings = []
            if not pixel_spacing:
                warnings.append("No pixel spacing information found")
            if len(frames) == 0 and request.load_pixel_data:
                warnings.append("No pixel data loaded")

            return DicomLoadResult(success=True, study=study, warnings=warnings)

        except Exception as e:
            logger.error(f"Error loading single file: {str(e)}")
            return DicomLoadResult(success=False, error_message=str(e))

    def _load_dicomdir(self, request: DicomLoadRequest) -> DicomLoadResult:
        """
        DICOMDIR yükle.

        Args:
            request: Yükleme isteği

        Returns:
            DicomLoadResult: Yükleme sonucu
        """
        try:
            # DICOMDIR oku
            dicomdir = pydicom.dcmread(str(request.file_path))
            base_dir = request.file_path.parent

            # İlk hasta/çalışma bilgilerini al
            patient_info = None
            study_info = None
            series_list = []

            # DICOMDIR kayıtlarını işle
            for patient_record in dicomdir.patient_records:
                if patient_info is None:
                    patient_info = DicomPatientInfo(
                        patient_id=getattr(patient_record, "PatientID", "Unknown"),
                        patient_name=str(getattr(patient_record, "PatientName", "Anonymous")),
                    )

                for study_record in patient_record.children:
                    if study_info is None:
                        study_info = DicomStudyInfo(
                            study_instance_uid=getattr(study_record, "StudyInstanceUID", ""),
                            study_description=getattr(study_record, "StudyDescription", ""),
                        )

                    for series_record in study_record.children:
                        # İlk görüntüyü bul
                        for image_record in series_record.children:
                            if hasattr(image_record, "ReferencedFileID"):
                                file_path = base_dir
                                for part in image_record.ReferencedFileID:
                                    file_path = file_path / part

                                if file_path.exists():
                                    # Seriyi yükle
                                    series_request = DicomLoadRequest(
                                        file_path=file_path,
                                        load_pixel_data=request.load_pixel_data,
                                        specific_frames=request.specific_frames,
                                    )

                                    series_result = self._load_single_file(series_request)
                                    if series_result.success and series_result.study:
                                        series_list.extend(series_result.study.series_list)

                                    break  # Sadece ilk görüntü

            if not series_list:
                return DicomLoadResult(
                    success=False, error_message="No valid series found in DICOMDIR"
                )

            # Çalışma oluştur
            study = DicomStudy(
                patient_info=patient_info or DicomPatientInfo(patient_id="Unknown"),
                study_info=study_info or DicomStudyInfo(study_instance_uid=""),
                series_list=series_list,
                file_path=request.file_path,
                is_dicomdir=True,
            )

            return DicomLoadResult(success=True, study=study)

        except Exception as e:
            logger.error(f"Error loading DICOMDIR: {str(e)}")
            return DicomLoadResult(success=False, error_message=str(e))

    def _find_dicom_files(self, folder_path: Path) -> List[Path]:
        """
        Klasörde DICOM dosyalarını bul.

        Args:
            folder_path: Klasör yolu

        Returns:
            List[Path]: DICOM dosya listesi
        """
        dicom_files = []

        # Bilinen uzantılar
        extensions = [".dcm", ".DCM", ".dicom", ".DICOM"]

        # Uzantılı dosyalar
        for ext in extensions:
            dicom_files.extend(folder_path.glob(f"*{ext}"))

        # Uzantısız dosyaları kontrol et
        for file_path in folder_path.iterdir():
            if file_path.is_file() and not file_path.suffix:
                # DICOM header kontrolü
                try:
                    with open(file_path, "rb") as f:
                        f.seek(128)
                        if f.read(4) == b"DICM":
                            dicom_files.append(file_path)
                except:
                    pass

        return sorted(dicom_files)

    def _extract_patient_info(self, dataset: pydicom.Dataset) -> DicomPatientInfo:
        """Hasta bilgilerini çıkar."""
        return DicomPatientInfo(
            patient_id=getattr(dataset, "PatientID", "Unknown"),
            patient_name=str(getattr(dataset, "PatientName", "Anonymous")),
            sex=getattr(dataset, "PatientSex", "O"),
            age=getattr(dataset, "PatientAge", None),
        )

    def _extract_study_info(self, dataset: pydicom.Dataset) -> DicomStudyInfo:
        """Çalışma bilgilerini çıkar."""
        study_date = None
        if hasattr(dataset, "StudyDate"):
            try:
                study_date = datetime.strptime(dataset.StudyDate, "%Y%m%d").date()
            except:
                pass

        return DicomStudyInfo(
            study_instance_uid=getattr(dataset, "StudyInstanceUID", ""),
            study_date=study_date,
            study_time=getattr(dataset, "StudyTime", None),
            study_description=getattr(dataset, "StudyDescription", ""),
            accession_number=getattr(dataset, "AccessionNumber", ""),
            referring_physician=str(getattr(dataset, "ReferringPhysicianName", "")),
        )

    def _extract_series_info(self, dataset: pydicom.Dataset) -> DicomSeriesInfo:
        """Seri bilgilerini çıkar."""
        modality = DicomModality.from_string(getattr(dataset, "Modality", "OTHER"))

        return DicomSeriesInfo(
            series_instance_uid=getattr(dataset, "SeriesInstanceUID", ""),
            series_number=int(getattr(dataset, "SeriesNumber", 0)),
            series_description=getattr(dataset, "SeriesDescription", ""),
            modality=modality,
            body_part=getattr(dataset, "BodyPartExamined", ""),
            patient_position=getattr(dataset, "PatientPosition", ""),
        )

    def _extract_image_info(self, dataset: pydicom.Dataset) -> DicomImageInfo:
        """Görüntü bilgilerini çıkar."""
        return DicomImageInfo(
            rows=int(getattr(dataset, "Rows", 0)),
            columns=int(getattr(dataset, "Columns", 0)),
            bits_allocated=int(getattr(dataset, "BitsAllocated", 16)),
            bits_stored=int(getattr(dataset, "BitsStored", 12)),
            pixel_representation=int(getattr(dataset, "PixelRepresentation", 0)),
            photometric_interpretation=getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"),
            samples_per_pixel=int(getattr(dataset, "SamplesPerPixel", 1)),
        )

    def _extract_pixel_spacing(self, dataset: pydicom.Dataset) -> Optional[DicomPixelSpacing]:
        """Piksel aralığını çıkar."""
        # PixelSpacing tercih edilir
        if hasattr(dataset, "PixelSpacing"):
            spacing = dataset.PixelSpacing
            if isinstance(spacing, (list, pydicom.multival.MultiValue)) and len(spacing) >= 2:
                return DicomPixelSpacing(
                    row_spacing=float(spacing[0]), column_spacing=float(spacing[1])
                )

        # ImagerPixelSpacing alternatif
        if hasattr(dataset, "ImagerPixelSpacing"):
            spacing = dataset.ImagerPixelSpacing
            if isinstance(spacing, (list, pydicom.multival.MultiValue)) and len(spacing) >= 2:
                return DicomPixelSpacing(
                    row_spacing=float(spacing[0]), column_spacing=float(spacing[1])
                )

        return None

    def _extract_window_level(
        self, dataset: pydicom.Dataset, modality: DicomModality
    ) -> DicomWindowLevel:
        """Pencere/seviye ayarlarını çıkar."""
        # Dataset'ten al
        center = getattr(dataset, "WindowCenter", None)
        width = getattr(dataset, "WindowWidth", None)

        # Liste ise ilk değeri al
        if isinstance(center, (list, pydicom.multival.MultiValue)):
            center = float(center[0]) if center else None
        if isinstance(width, (list, pydicom.multival.MultiValue)):
            width = float(width[0]) if width else None

        # Değerler varsa kullan
        if center is not None and width is not None:
            return DicomWindowLevel(center=float(center), width=float(width), name="DICOM")

        # Yoksa modaliteye göre varsayılan
        if self._window_provider:
            return self._window_provider.get_default(modality)

        # Fallback
        return DicomWindowLevel(center=128, width=256, name="Default")

    def _extract_projection_info_from_dataset(
        self, dataset: pydicom.Dataset
    ) -> Optional[DicomProjectionInfo]:
        """Dataset'ten projeksiyon bilgilerini çıkar."""
        primary_angle = None
        secondary_angle = None

        # Açıları al
        if hasattr(dataset, "PositionerPrimaryAngle"):
            try:
                primary_angle = float(dataset.PositionerPrimaryAngle)
            except:
                pass

        if hasattr(dataset, "PositionerSecondaryAngle"):
            try:
                secondary_angle = float(dataset.PositionerSecondaryAngle)
            except:
                pass

        # Açılar yoksa None döndür
        if primary_angle is None and secondary_angle is None:
            return None

        # Projeksiyon tipi belirle
        projection_type = ProjectionType.UNKNOWN
        if primary_angle is not None and secondary_angle is not None:
            projection_type = ProjectionType.from_angles(primary_angle, secondary_angle)

        return DicomProjectionInfo(
            primary_angle=primary_angle or 0.0,
            secondary_angle=secondary_angle or 0.0,
            projection_type=projection_type,
            view_position=getattr(dataset, "ViewPosition", ""),
            table_height=getattr(dataset, "TableHeight", None),
            distance_source_to_detector=getattr(dataset, "DistanceSourceToDetector", None),
            distance_source_to_patient=getattr(dataset, "DistanceSourceToPatient", None),
        )

    def _extract_projection_info(self, file_path: Path) -> Optional[DicomProjectionInfo]:
        """Dosyadan projeksiyon bilgilerini çıkar."""
        try:
            dataset = pydicom.dcmread(str(file_path), stop_before_pixels=True)
            return self._extract_projection_info_from_dataset(dataset)
        except:
            return None

    def _load_frames(
        self, dataset: pydicom.Dataset, specific_frames: Optional[List[int]] = None
    ) -> List[DicomFrame]:
        """Frame'leri yükle."""
        frames = []

        try:
            # Piksel verisini al
            pixel_array = dataset.pixel_array

            # Multi-frame kontrolü
            if len(pixel_array.shape) == 3:
                # Multi-frame
                num_frames = pixel_array.shape[0]
                frame_indices = specific_frames or list(range(num_frames))

                for i in frame_indices:
                    if 0 <= i < num_frames:
                        frame = DicomFrame(
                            index=i,
                            pixel_array=pixel_array[i],
                            timestamp=self._calculate_frame_timestamp(dataset, i),
                        )
                        frames.append(frame)
            else:
                # Single frame
                frame = DicomFrame(index=0, pixel_array=pixel_array, timestamp=0.0)
                frames.append(frame)

        except Exception as e:
            logger.error(f"Error loading frames: {str(e)}")

        return frames

    def _calculate_frame_timestamp(self, dataset: pydicom.Dataset, frame_index: int) -> float:
        """Frame zaman damgasını hesapla."""
        # FrameTime varsa kullan
        if hasattr(dataset, "FrameTime"):
            frame_time_ms = float(dataset.FrameTime)
            return frame_index * frame_time_ms / 1000.0

        # FrameTimeVector varsa kullan
        if hasattr(dataset, "FrameTimeVector"):
            vector = dataset.FrameTimeVector
            if frame_index < len(vector):
                return float(vector[frame_index]) / 1000.0

        # Varsayılan 30 fps
        return frame_index / 30.0

    def _extract_frame_rate(self, dataset: pydicom.Dataset) -> float:
        """Frame hızını çıkar."""
        # FrameTime'dan hesapla
        if hasattr(dataset, "FrameTime"):
            frame_time_ms = float(dataset.FrameTime)
            if frame_time_ms > 0:
                return 1000.0 / frame_time_ms

        # CineRate varsa kullan
        if hasattr(dataset, "CineRate"):
            return float(dataset.CineRate)

        # Varsayılan
        return 30.0

    def _get_dicomdir_projections(self, dicomdir_path: Path) -> List[DicomProjectionInfo]:
        """DICOMDIR'den projeksiyonları al."""
        projections = []

        try:
            dicomdir = pydicom.dcmread(str(dicomdir_path))
            base_dir = dicomdir_path.parent

            # Benzersiz projeksiyonları topla
            projection_map = {}

            for patient_record in dicomdir.patient_records:
                for study_record in patient_record.children:
                    for series_record in study_record.children:
                        for image_record in series_record.children:
                            if hasattr(image_record, "ReferencedFileID"):
                                file_path = base_dir
                                for part in image_record.ReferencedFileID:
                                    file_path = file_path / part

                                if file_path.exists():
                                    projection = self._extract_projection_info(file_path)
                                    if projection:
                                        # Benzersiz anahtar
                                        key = f"{projection.primary_angle:.1f}_{projection.secondary_angle:.1f}"
                                        if key not in projection_map:
                                            projection_map[key] = projection
                                            projections.append(projection)
                                    break  # Seri başına bir görüntü

        except Exception as e:
            logger.error(f"Error getting DICOMDIR projections: {str(e)}")

        return projections


class PydicomReader(IDicomReader):
    """
    Pydicom tabanlı DICOM okuyucu.

    IDicomReader protokolünü implement eder.
    """

    def read_file(self, file_path: Path) -> DicomLoadResult:
        """DICOM dosyası oku."""
        loader = DicomLoaderService(reader=self)
        request = DicomLoadRequest(file_path=file_path)
        return loader.load(request)

    def read_dicomdir(self, dicomdir_path: Path) -> DicomLoadResult:
        """DICOMDIR oku."""
        loader = DicomLoaderService(reader=self)
        request = DicomLoadRequest(file_path=dicomdir_path)
        return loader.load(request)

    def validate_file(self, file_path: Path) -> bool:
        """DICOM dosyasını doğrula."""
        try:
            # DICOM header kontrolü
            with open(file_path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except:
            return False

    @property
    def supported_formats(self) -> List[str]:
        """Desteklenen formatlar."""
        return [".dcm", ".dicom", "DICOMDIR"]


class DefaultWindowLevelProvider(IDicomWindowLevelProvider):
    """
    Varsayılan pencere/seviye sağlayıcı.

    Farklı modaliteler için preset'ler.
    """

    def get_presets(self, modality: DicomModality) -> Dict[str, DicomWindowLevel]:
        """Modaliteye göre preset'leri al."""
        if modality in [DicomModality.XA, DicomModality.XRF]:
            return {
                "Default": DicomWindowLevel(300, 600, "Default"),
                "Angio": DicomWindowLevel(300, 600, "Angio"),
                "Bone": DicomWindowLevel(400, 1500, "Bone"),
                "Soft": DicomWindowLevel(40, 400, "Soft Tissue"),
            }
        elif modality == DicomModality.CT:
            return {
                "Default": DicomWindowLevel(40, 400, "Default"),
                "Lung": DicomWindowLevel(-600, 1500, "Lung"),
                "Bone": DicomWindowLevel(300, 1500, "Bone"),
                "Brain": DicomWindowLevel(40, 80, "Brain"),
                "Abdomen": DicomWindowLevel(40, 350, "Abdomen"),
            }
        else:
            return {"Default": DicomWindowLevel(128, 256, "Default")}

    def get_default(self, modality: DicomModality) -> DicomWindowLevel:
        """Varsayılan pencere/seviye al."""
        presets = self.get_presets(modality)
        return presets.get("Default", DicomWindowLevel(128, 256, "Default"))

    def calculate_auto(self, pixel_array: np.ndarray) -> DicomWindowLevel:
        """Otomatik pencere/seviye hesapla."""
        # Basit histogram tabanlı hesaplama
        p5 = np.percentile(pixel_array, 5)
        p95 = np.percentile(pixel_array, 95)

        center = (p5 + p95) / 2
        width = p95 - p5

        return DicomWindowLevel(center, width, "Auto")
