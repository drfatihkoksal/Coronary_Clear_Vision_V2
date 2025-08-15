"""
QCA Report Generation Module
Creates comprehensive PDF reports for QCA analysis results
"""

import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.enums import TA_CENTER
import io
from PIL import Image as PILImage
import cv2
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class QCAReportGenerator:
    """Generate comprehensive QCA analysis reports"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    @staticmethod
    def _safe_format_number(value, default=0):
        """Safely format a number, handling None values"""
        return value if value is not None else default

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                textColor=colors.HexColor("#1976D2"),
                spaceAfter=30,
                alignment=TA_CENTER,
            )
        )

        # Subtitle style
        self.styles.add(
            ParagraphStyle(
                name="Subtitle",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#424242"),
                spaceBefore=20,
                spaceAfter=10,
            )
        )

        # Info style
        self.styles.add(
            ParagraphStyle(
                name="Info",
                parent=self.styles["Normal"],
                fontSize=11,
                textColor=colors.HexColor("#616161"),
                leftIndent=20,
            )
        )

    def generate_report(
        self,
        qca_results: Dict,
        patient_info: Dict,
        output_path: str,
        angiogram_image: Optional[np.ndarray] = None,
        with_overlay: bool = True,
    ) -> bool:
        """
        Generate comprehensive QCA report

        Args:
            qca_results: QCA analysis results
            patient_info: Patient information
            output_path: Path to save PDF report
            angiogram_image: Original angiogram image
            with_overlay: Include image with QCA overlay

        Returns:
            Success status
        """
        try:
            # Create document with reduced margins for more content space
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=50,
                leftMargin=50,
                topMargin=50,
                bottomMargin=36,
            )

            # Container for elements
            elements = []

            # Add header
            elements.extend(self._create_header(patient_info))

            # Add summary section
            elements.extend(self._create_summary_section(qca_results))

            # Add measurements table
            elements.extend(self._create_measurements_table(qca_results))

            # Add stenosis classification
            elements.extend(self._create_stenosis_classification(qca_results))

            # Add diameter graph
            if qca_results.get("profile_data"):
                elements.extend(self._create_diameter_graph(qca_results["profile_data"]))

            # Add images if available
            if angiogram_image is not None:
                elements.append(PageBreak())
                elements.extend(
                    self._create_image_section(angiogram_image, qca_results, with_overlay)
                )

            # Add technical details
            elements.extend(self._create_technical_section(qca_results))

            # Add footer
            elements.extend(self._create_footer())

            # Build PDF
            doc.build(elements)

            logger.info(f"QCA report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate QCA report: {e}")
            return False

    def _create_header(self, patient_info: Dict) -> list:
        """Create report header"""
        elements = []

        # Title
        elements.append(
            Paragraph("Quantitative Coronary Angiography (QCA) Report", self.styles["CustomTitle"])
        )

        # Patient information table
        patient_data = [
            [
                "Patient ID:",
                patient_info.get("patient_id", "N/A"),
                "Study Date:",
                patient_info.get("study_date", datetime.now().strftime("%Y-%m-%d")),
            ],
            [
                "Patient Name:",
                patient_info.get("patient_name", "N/A"),
                "Accession #:",
                patient_info.get("accession_number", "N/A"),
            ],
            [
                "Date of Birth:",
                patient_info.get("birth_date", "N/A"),
                "Referring Physician:",
                patient_info.get("referring_physician", "N/A"),
            ],
        ]

        patient_table = Table(patient_data, colWidths=[2 * inch, 2 * inch, 2 * inch, 2 * inch])
        patient_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("BACKGROUND", (1, 0), (1, -1), colors.white),
                    ("BACKGROUND", (3, 0), (3, -1), colors.white),
                    ("TEXTCOLOR", (1, 0), (1, -1), colors.black),
                    ("TEXTCOLOR", (3, 0), (3, -1), colors.black),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTNAME", (3, 0), (3, -1), "Helvetica"),
                ]
            )
        )

        elements.append(patient_table)
        elements.append(Spacer(1, 0.5 * inch))

        return elements

    def _create_summary_section(self, qca_results: Dict) -> list:
        """Create summary section"""
        elements = []

        elements.append(Paragraph("Analysis Summary", self.styles["Subtitle"]))

        # Determine stenosis severity
        percent_stenosis = qca_results.get("percent_stenosis", 0) or 0
        if percent_stenosis < 50:
            severity = "Mild"
            color = colors.green
        elif percent_stenosis < 70:
            severity = "Moderate"
            color = colors.orange
        else:
            severity = "Severe"
            color = colors.red

        # Summary text
        summary_text = f"""
        <font color='{color.hexval()}' size='14'><b>Stenosis Severity: {severity}</b></font><br/>
        <font size='12'>Percent Diameter Stenosis: <b>{percent_stenosis:.1f}%</b></font><br/>
        <font size='12'>Percent Area Stenosis: <b>{(qca_results.get('percent_area_stenosis', 0) or 0):.1f}%</b></font>
        """

        elements.append(Paragraph(summary_text, self.styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_measurements_table(self, qca_results: Dict) -> list:
        """Create detailed measurements table"""
        elements = []

        elements.append(Paragraph("Detailed Measurements", self.styles["Subtitle"]))

        # Measurement data
        measurements = [
            ["Parameter", "Value", "Unit"],
            ["Reference Diameter", f"{(qca_results.get('reference_diameter', 0) or 0):.2f}", "mm"],
            ["Minimal Lumen Diameter (MLD)", f"{(qca_results.get('mld', 0) or 0):.2f}", "mm"],
            ["Proximal Reference", f"{(qca_results.get('proximal_reference', 0) or 0):.2f}", "mm"],
            ["Distal Reference", f"{(qca_results.get('distal_reference', 0) or 0):.2f}", "mm"],
            [
                "Percent Diameter Stenosis",
                f"{(qca_results.get('percent_stenosis', 0) or 0):.1f}",
                "%",
            ],
            [
                "Percent Area Stenosis",
                f"{(qca_results.get('percent_area_stenosis', 0) or 0):.1f}",
                "%",
            ],
            ["Minimal Lumen Area (MLA)", f"{(qca_results.get('mla', 0) or 0):.2f}", "mm²"],
            ["Reference Area", f"{(qca_results.get('reference_area', 0) or 0):.2f}", "mm²"],
            [
                "Lesion Length",
                (
                    f"{(qca_results.get('lesion_length', 0) or 0):.1f}"
                    if qca_results.get("lesion_length") is not None
                    else "N/A"
                ),
                "mm",
            ],
        ]

        # Create table
        table = Table(measurements, colWidths=[3 * inch, 1.5 * inch, 1 * inch])
        table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1976D2")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    # Data rows
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F5F5F5")],
                    ),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_stenosis_classification(self, qca_results: Dict) -> list:
        """Create stenosis classification section"""
        elements = []

        elements.append(Paragraph("Clinical Classification", self.styles["Subtitle"]))

        percent_stenosis = qca_results.get("percent_stenosis", 0) or 0

        # Classification based on AHA/ACC guidelines
        classifications = [
            (0, 25, "Minimal coronary atherosclerosis", colors.green),
            (25, 50, "Mild coronary stenosis", colors.yellowgreen),
            (50, 70, "Moderate coronary stenosis", colors.orange),
            (70, 90, "Severe coronary stenosis", colors.orangered),
            (90, 99, "Subtotal occlusion", colors.red),
            (99, 100, "Total occlusion", colors.darkred),
        ]

        for min_val, max_val, description, color in classifications:
            if min_val <= percent_stenosis < max_val:
                classification_text = f"""
                <font color='{color.hexval()}' size='12'><b>{description}</b></font><br/>
                <font size='10'>({min_val}% - {max_val}% diameter stenosis)</font>
                """
                elements.append(Paragraph(classification_text, self.styles["Info"]))
                break

        # Add clinical significance
        if percent_stenosis >= 70:
            significance = "Hemodynamically significant stenosis - Consider intervention"
        elif percent_stenosis >= 50:
            significance = "Borderline significant - Consider functional assessment (FFR/iFR)"
        else:
            significance = "Not hemodynamically significant"

        elements.append(Spacer(1, 0.1 * inch))
        elements.append(
            Paragraph(f"<b>Clinical Significance:</b> {significance}", self.styles["Info"])
        )
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_diameter_graph(self, profile_data: Dict) -> list:
        """Create diameter profile graph"""
        elements = []

        elements.append(Paragraph("Vessel Diameter Profile", self.styles["Subtitle"]))

        # Create drawing with larger size for better visibility
        drawing = Drawing(500, 250)

        # Set white background
        from reportlab.graphics.shapes import Rect

        drawing.add(Rect(0, 0, 500, 250, fillColor=colors.white, strokeColor=None))

        # Create line plot
        lp = LinePlot()
        lp.x = 60
        lp.y = 60
        lp.height = 150
        lp.width = 380

        # Prepare data
        distances = profile_data.get("distances", [])
        diameters = profile_data.get("diameters", [])

        if distances and diameters:
            lp.data = [list(zip(distances, diameters))]

            # Configure plot
            lp.lines[0].strokeColor = colors.blue
            lp.lines[0].strokeWidth = 2.5

            # X-axis
            lp.xValueAxis.valueMin = 0
            lp.xValueAxis.valueMax = max(distances) if distances else 1
            lp.xValueAxis.valueStep = max(5, int(max(distances) / 10)) if distances else 1
            lp.xValueAxis.labelTextFormat = "%0.0f"
            lp.xValueAxis.labels.boxAnchor = "n"
            lp.xValueAxis.labels.dy = -8
            lp.xValueAxis.labels.fontSize = 10
            lp.xValueAxis.strokeColor = colors.black
            lp.xValueAxis.strokeWidth = 1

            # Y-axis
            lp.yValueAxis.valueMin = 0
            lp.yValueAxis.valueMax = max(diameters) * 1.2 if diameters else 5
            lp.yValueAxis.valueStep = 0.5
            lp.yValueAxis.labelTextFormat = "%0.1f"
            lp.yValueAxis.labels.fontSize = 10
            lp.yValueAxis.strokeColor = colors.black
            lp.yValueAxis.strokeWidth = 1

            # Add grid lines
            lp.xValueAxis.visibleGrid = 1
            lp.xValueAxis.gridStrokeColor = colors.lightgrey
            lp.xValueAxis.gridStrokeWidth = 0.5
            lp.yValueAxis.visibleGrid = 1
            lp.yValueAxis.gridStrokeColor = colors.lightgrey
            lp.yValueAxis.gridStrokeWidth = 0.5

            drawing.add(lp)

            # Add axis labels
            from reportlab.graphics.shapes import String

            drawing.add(String(250, 25, "Distance (mm)", textAnchor="middle", fontSize=12))

            # Vertical label for Y-axis
            from reportlab.graphics.shapes import Group

            y_label = String(0, 0, "Diameter (mm)", fontSize=12, textAnchor="middle")
            y_label_group = Group(y_label)
            y_label_group.rotate(90)
            y_label_group.shift(25, 135)
            drawing.add(y_label_group)

            elements.append(drawing)

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_image_section(
        self, angiogram_image: np.ndarray, qca_results: Dict, with_overlay: bool
    ) -> list:
        """Create image section with angiogram"""
        elements = []

        elements.append(Paragraph("Angiographic Images", self.styles["Subtitle"]))

        # Convert numpy array to PIL Image
        if len(angiogram_image.shape) == 2:
            pil_image = PILImage.fromarray(angiogram_image, mode="L")
        else:
            pil_image = PILImage.fromarray(cv2.cvtColor(angiogram_image, cv2.COLOR_BGR2RGB))

        # Add overlay if requested
        # Note: centerline now comes from segmentation, not QCA results
        if with_overlay:
            # Create overlay image
            overlay_image = self._create_overlay_image(angiogram_image, qca_results)
            pil_image = PILImage.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))

        # Save to buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Create reportlab Image with better sizing
        # Calculate proper size maintaining aspect ratio
        img_width = 7 * inch  # Increased for better visibility
        aspect_ratio = pil_image.height / pil_image.width
        img_height = img_width * aspect_ratio

        # Limit height if too tall
        if img_height > 5 * inch:
            img_height = 5 * inch
            img_width = img_height / aspect_ratio

        img = Image(img_buffer, width=img_width, height=img_height)
        elements.append(img)

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_overlay_image(self, image: np.ndarray, qca_results: Dict) -> np.ndarray:
        """Create image with QCA overlay"""
        # Convert to color if grayscale
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        # Don't draw centerline here - it should come from segmentation overlay
        # centerline = qca_results.get('centerline')
        # if centerline is not None:
        #     for i in range(len(centerline) - 1):
        #         pt1 = (int(centerline[i][1]), int(centerline[i][0]))
        #         pt2 = (int(centerline[i+1][1]), int(centerline[i+1][0]))
        #         cv2.line(overlay, pt1, pt2, (0, 255, 255), 2)

        # Mark MLD location
        mld_location = qca_results.get("mld_location")
        if mld_location is not None:
            center = (int(mld_location[1]), int(mld_location[0]))
            cv2.circle(overlay, center, 10, (0, 0, 255), 3)

            # Add text
            percent_stenosis = qca_results.get("percent_stenosis", 0) or 0
            text = f"{percent_stenosis:.1f}%"
            cv2.putText(
                overlay,
                text,
                (center[0] + 15, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Mark reference points
        if centerline is not None:
            prox_idx = qca_results.get("proximal_ref_index")
            dist_idx = qca_results.get("distal_ref_index")

            if prox_idx is not None and prox_idx < len(centerline):
                pt = (int(centerline[prox_idx][1]), int(centerline[prox_idx][0]))
                cv2.circle(overlay, pt, 8, (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    "P",
                    (pt[0] - 5, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            if dist_idx is not None and dist_idx < len(centerline):
                pt = (int(centerline[dist_idx][1]), int(centerline[dist_idx][0]))
                cv2.circle(overlay, pt, 8, (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    "D",
                    (pt[0] - 5, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        return overlay

    def _create_technical_section(self, qca_results: Dict) -> list:
        """Create technical details section"""
        elements = []

        elements.append(Paragraph("Technical Details", self.styles["Subtitle"]))

        tech_details = []

        # Calibration info
        calibration_factor = qca_results.get("calibration_factor")
        if calibration_factor is not None and calibration_factor != 0:
            tech_details.append(f"Calibration Factor: {calibration_factor:.5f} mm/pixel")

        # Segmentation method
        method = qca_results.get("segmentation_method", "Unknown")
        tech_details.append(f"Segmentation Method: {method}")

        # Analysis date/time
        tech_details.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Software version
        tech_details.append("Software: Coronary Clear Vision v1.0")

        for detail in tech_details:
            elements.append(Paragraph(f"• {detail}", self.styles["Info"]))

        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_footer(self) -> list:
        """Create report footer"""
        elements = []

        elements.append(Spacer(1, 0.5 * inch))

        disclaimer = """
        <font size='9' color='#666666'>
        <b>Disclaimer:</b> This report is generated by automated QCA analysis software.
        Results should be interpreted by a qualified physician in conjunction with clinical findings.
        Measurements are estimates and may vary based on image quality and calibration accuracy.
        </font>
        """

        elements.append(Paragraph(disclaimer, self.styles["Normal"]))

        return elements
