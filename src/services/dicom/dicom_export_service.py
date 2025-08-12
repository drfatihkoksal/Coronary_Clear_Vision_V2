"""
DICOM Export Service

DICOM verilerini dışa aktarma servisi.
Clean Architecture ve SOLID prensiplerine uygun tasarlanmıştır.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import cv2
import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import json

from src.domain.models.dicom_models import (
    DicomStudy, DicomSeries, DicomFrame,
    DicomWindowLevel
)
from src.domain.interfaces.dicom_interfaces import IDicomExporter
from src.services.dicom.dicom_processor_service import DicomProcessorService

logger = logging.getLogger(__name__)


class DicomExportService(IDicomExporter):
    """
    DICOM dışa aktarma servisi.
    
    Bu servis:
    - Frame'leri görüntü olarak kaydeder
    - Video dışa aktarımı yapar
    - Rapor oluşturur
    - Farklı formatları destekler
    """
    
    def __init__(self, processor: Optional[DicomProcessorService] = None):
        """
        DicomExportService constructor.
        
        Args:
            processor: Görüntü işleme servisi
        """
        self._processor = processor or DicomProcessorService()
        self._supported_image_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        self._supported_video_formats = ['mp4', 'avi', 'mov']
        
        logger.info("DicomExportService initialized")
    
    def export_frame(self,
                    frame: np.ndarray,
                    output_path: Path,
                    format: str = "png") -> bool:
        """
        Frame'i dışa aktar.
        
        Args:
            frame: Frame verisi
            output_path: Çıktı yolu
            format: Çıktı formatı
            
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Format kontrolü
            format = format.lower()
            if format not in self._supported_image_formats:
                logger.error(f"Unsupported image format: {format}")
                return False
            
            # Dizin oluştur
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Frame'i işle
            processed_frame = self._prepare_frame_for_export(frame)
            
            # Kaydet
            if format in ['jpg', 'jpeg']:
                # JPEG için kalite ayarı
                cv2.imwrite(str(output_path), processed_frame, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif format == 'png':
                # PNG için sıkıştırma
                cv2.imwrite(str(output_path), processed_frame,
                           [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                # Diğer formatlar
                cv2.imwrite(str(output_path), processed_frame)
            
            logger.info(f"Frame exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting frame: {str(e)}")
            return False
    
    def export_series_as_video(self,
                              series: DicomSeries,
                              output_path: Path,
                              fps: int = 30,
                              codec: str = "mp4v") -> bool:
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
        try:
            # Format kontrolü
            suffix = output_path.suffix.lower().lstrip('.')
            if suffix not in self._supported_video_formats:
                logger.error(f"Unsupported video format: {suffix}")
                return False
            
            # Dizin oluştur
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # İlk frame'den boyut al
            if series.num_frames == 0:
                logger.error("No frames to export")
                return False
            
            first_frame = self._processor.extract_frame(series, 0)
            if first_frame is None:
                logger.error("Could not extract first frame")
                return False
            
            height, width = first_frame.shape[:2]
            
            # Video writer oluştur
            fourcc = self._get_fourcc(codec)
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
                isColor=len(first_frame.shape) == 3
            )
            
            if not writer.isOpened():
                logger.error("Failed to open video writer")
                return False
            
            # Frame'leri yaz
            progress_interval = max(1, series.num_frames // 10)
            
            for i in range(series.num_frames):
                # Frame'i al
                frame = self._processor.extract_frame(series, i)
                if frame is None:
                    logger.warning(f"Could not extract frame {i}")
                    continue
                
                # İşle ve yaz
                processed = self._prepare_frame_for_export(frame)
                
                # Grayscale ise RGB'ye çevir
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                
                writer.write(processed)
                
                # İlerleme
                if i % progress_interval == 0:
                    logger.debug(f"Video export progress: {i}/{series.num_frames}")
            
            # Kapat
            writer.release()
            
            logger.info(f"Video exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting video: {str(e)}")
            return False
    
    def export_study_report(self,
                           study: DicomStudy,
                           output_path: Path,
                           include_images: bool = True) -> bool:
        """
        Çalışma raporu oluştur.
        
        Args:
            study: DICOM çalışması
            output_path: Çıktı yolu
            include_images: Görüntüler dahil edilsin mi?
            
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Dizin oluştur
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format kontrolü
            if output_path.suffix.lower() == '.pdf':
                return self._export_pdf_report(study, output_path, include_images)
            elif output_path.suffix.lower() == '.html':
                return self._export_html_report(study, output_path, include_images)
            elif output_path.suffix.lower() == '.json':
                return self._export_json_report(study, output_path)
            else:
                # Varsayılan HTML
                output_path = output_path.with_suffix('.html')
                return self._export_html_report(study, output_path, include_images)
                
        except Exception as e:
            logger.error(f"Error exporting study report: {str(e)}")
            return False
    
    @property
    def supported_formats(self) -> List[str]:
        """Desteklenen dışa aktarma formatları."""
        return self._supported_image_formats + self._supported_video_formats + ['pdf', 'html', 'json']
    
    def _prepare_frame_for_export(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame'i dışa aktarım için hazırla.
        
        Args:
            frame: Ham frame verisi
            
        Returns:
            np.ndarray: İşlenmiş frame
        """
        # Kopya oluştur
        processed = frame.copy()
        
        # Float ise uint8'e çevir
        if processed.dtype != np.uint8:
            if processed.dtype in [np.float32, np.float64]:
                # 0-1 aralığında varsay
                if processed.max() <= 1.0:
                    processed = (processed * 255).astype(np.uint8)
                else:
                    # Normalize et
                    processed = self._processor.normalize_pixels(processed, (0, 255))
                    processed = processed.astype(np.uint8)
            else:
                # Diğer integer tipler
                processed = self._processor.normalize_pixels(processed, (0, 255))
                processed = processed.astype(np.uint8)
        
        return processed
    
    def _get_fourcc(self, codec: str) -> int:
        """
        Codec string'inden fourcc kodu al.
        
        Args:
            codec: Codec adı
            
        Returns:
            int: FourCC kodu
        """
        codec_map = {
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v'),
            'h264': cv2.VideoWriter_fourcc(*'H264'),
            'xvid': cv2.VideoWriter_fourcc(*'XVID'),
            'mjpg': cv2.VideoWriter_fourcc(*'MJPG'),
            'divx': cv2.VideoWriter_fourcc(*'DIVX')
        }
        
        return codec_map.get(codec.lower(), cv2.VideoWriter_fourcc(*'mp4v'))
    
    def _export_html_report(self,
                           study: DicomStudy,
                           output_path: Path,
                           include_images: bool) -> bool:
        """HTML raporu oluştur."""
        try:
            # HTML şablonu
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DICOM Study Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .series-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .thumbnail {{
            display: inline-block;
            margin: 10px;
            text-align: center;
        }}
        .thumbnail img {{
            max-width: 200px;
            max-height: 200px;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DICOM Çalışma Raporu</h1>
        <p>Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Hasta Bilgileri</h2>
        <table>
            <tr><th>Alan</th><th>Değer</th></tr>
            <tr><td>Hasta ID</td><td>{study.patient_info.patient_id}</td></tr>
            <tr><td>Hasta Adı</td><td>{study.patient_info.patient_name}</td></tr>
            <tr><td>Cinsiyet</td><td>{study.patient_info.sex}</td></tr>
            <tr><td>Yaş</td><td>{study.patient_info.age or 'N/A'}</td></tr>
        </table>
        
        <h2>Çalışma Bilgileri</h2>
        <table>
            <tr><th>Alan</th><th>Değer</th></tr>
            <tr><td>Study UID</td><td>{study.study_info.study_instance_uid}</td></tr>
            <tr><td>Tarih</td><td>{study.study_info.study_date or 'N/A'}</td></tr>
            <tr><td>Açıklama</td><td>{study.study_info.study_description}</td></tr>
            <tr><td>Seri Sayısı</td><td>{study.num_series}</td></tr>
            <tr><td>Toplam Frame</td><td>{study.total_frames}</td></tr>
        </table>
"""
            
            # Her seri için bölüm
            for i, series in enumerate(study.series_list):
                html_content += f"""
        <div class="series-section">
            <h3>Seri {i+1}: {series.info.series_description}</h3>
            <table>
                <tr><th>Alan</th><th>Değer</th></tr>
                <tr><td>Series UID</td><td>{series.info.series_instance_uid}</td></tr>
                <tr><td>Modalite</td><td>{series.info.modality.value}</td></tr>
                <tr><td>Frame Sayısı</td><td>{series.num_frames}</td></tr>
                <tr><td>Frame Rate</td><td>{series.frame_rate:.1f} fps</td></tr>
                <tr><td>Süre</td><td>{series.duration:.1f} saniye</td></tr>
"""
                
                # Piksel aralığı
                if series.pixel_spacing:
                    html_content += f"""
                <tr><td>Piksel Aralığı</td><td>{series.pixel_spacing.average_spacing:.3f} mm/piksel</td></tr>
"""
                
                # Projeksiyon bilgileri
                if series.projection_info:
                    html_content += f"""
                <tr><td>Projeksiyon</td><td>{series.projection_info.angle_description}</td></tr>
"""
                
                html_content += "</table>"
                
                # Görüntüler
                if include_images and series.num_frames > 0:
                    html_content += "<h4>Örnek Görüntüler</h4><div>"
                    
                    # İlk, orta ve son frame
                    sample_indices = [0]
                    if series.num_frames > 2:
                        sample_indices.append(series.num_frames // 2)
                        sample_indices.append(series.num_frames - 1)
                    
                    for idx in sample_indices:
                        # Frame'i dışa aktar
                        frame = self._processor.extract_frame(series, idx)
                        if frame is not None:
                            # Thumbnail oluştur
                            thumbnail = self._processor.create_thumbnail(frame, 200)
                            
                            # Base64 encode
                            import base64
                            from io import BytesIO
                            
                            # PIL Image'e çevir
                            pil_image = Image.fromarray(thumbnail)
                            
                            # Base64 string
                            buffer = BytesIO()
                            pil_image.save(buffer, format='PNG')
                            img_base64 = base64.b64encode(buffer.getvalue()).decode()
                            
                            html_content += f"""
                    <div class="thumbnail">
                        <img src="data:image/png;base64,{img_base64}" alt="Frame {idx}">
                        <p>Frame {idx}</p>
                    </div>
"""
                    
                    html_content += "</div>"
                
                html_content += "</div>"
            
            # HTML kapat
            html_content += """
    </div>
</body>
</html>
"""
            
            # Dosyaya yaz
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting HTML report: {str(e)}")
            return False
    
    def _export_pdf_report(self,
                          study: DicomStudy,
                          output_path: Path,
                          include_images: bool) -> bool:
        """PDF raporu oluştur."""
        try:
            # PDF için reportlab kullan
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            # Doküman oluştur
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Başlık
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#333333'),
                spaceAfter=30
            )
            story.append(Paragraph("DICOM Çalışma Raporu", title_style))
            story.append(Spacer(1, 12))
            
            # Tarih
            story.append(Paragraph(f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Hasta bilgileri
            story.append(Paragraph("Hasta Bilgileri", styles['Heading2']))
            patient_data = [
                ['Alan', 'Değer'],
                ['Hasta ID', study.patient_info.patient_id],
                ['Hasta Adı', study.patient_info.patient_name],
                ['Cinsiyet', study.patient_info.sex],
                ['Yaş', study.patient_info.age or 'N/A']
            ]
            
            patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 20))
            
            # Çalışma bilgileri
            story.append(Paragraph("Çalışma Bilgileri", styles['Heading2']))
            study_data = [
                ['Alan', 'Değer'],
                ['Study UID', study.study_info.study_instance_uid[:30] + '...'],
                ['Tarih', str(study.study_info.study_date) if study.study_info.study_date else 'N/A'],
                ['Açıklama', study.study_info.study_description],
                ['Seri Sayısı', str(study.num_series)],
                ['Toplam Frame', str(study.total_frames)]
            ]
            
            study_table = Table(study_data, colWidths=[2*inch, 4*inch])
            study_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(study_table)
            
            # Sayfa sonu
            doc.build(story)
            
            logger.info(f"PDF report exported to: {output_path}")
            return True
            
        except ImportError:
            logger.error("reportlab package not installed. Cannot export PDF.")
            return False
        except Exception as e:
            logger.error(f"Error exporting PDF report: {str(e)}")
            return False
    
    def _export_json_report(self, study: DicomStudy, output_path: Path) -> bool:
        """JSON raporu oluştur."""
        try:
            # Study'yi dictionary'e çevir
            report_data = study.to_dict()
            
            # Ek bilgiler
            report_data['export_date'] = datetime.now().isoformat()
            report_data['export_version'] = '1.0'
            
            # JSON olarak kaydet
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON report exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting JSON report: {str(e)}")
            return False