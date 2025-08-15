"""
RWS report generation module
"""

import logging
from datetime import datetime
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from .models import RWSAnalysisData
from .visualizer import RWSVisualizer

logger = logging.getLogger(__name__)


class RWSReportGenerator:
    """Generate reports for RWS analysis"""

    def __init__(self, analysis_data: RWSAnalysisData):
        self.data = analysis_data
        self.visualizer = RWSVisualizer(analysis_data)

    def generate_pdf_report(self, output_path: str, include_patient_info: bool = True) -> bool:
        """
        Generate PDF report with RWS analysis results

        Args:
            output_path: Path for output PDF file
            include_patient_info: Whether to include patient information

        Returns:
            True if successful, False otherwise
        """
        if not self.data.success:
            logger.error("No valid RWS results to generate report")
            return False

        try:
            # Generate visualizations
            figures = self.visualizer.generate_all_visualizations()

            # Create PDF
            with PdfPages(output_path) as pdf:
                # Title page
                self._create_title_page(pdf, include_patient_info)

                # Summary visualization page
                summary_fig = self.visualizer.create_summary_plot()
                pdf.savefig(summary_fig, bbox_inches="tight")
                plt.close(summary_fig)

                # Add detailed visualization pages
                for fig_name, fig in figures.items():
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                # Add metadata
                d = pdf.infodict()
                d["Title"] = "RWS Analysis Report"
                d["Author"] = "Coronary Clear Vision"
                d["Subject"] = "Radial Wall Strain Analysis"
                d["Keywords"] = "RWS, QCA, Coronary Analysis"
                d["CreationDate"] = datetime.now()

            logger.info(f"PDF report generated: {output_path}")
            return True

        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            return False

    def _create_title_page(self, pdf: PdfPages, include_patient_info: bool):
        """Create title page for PDF report"""
        fig_title = plt.figure(figsize=(8.5, 11))
        fig_title.text(
            0.5,
            0.9,
            "Radial Wall Strain (RWS) Analysis Report",
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
        )

        # Build report text
        report_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Patient information
        if include_patient_info and self.data.patient_info:
            patient = self.data.patient_info
            report_text += "Patient Information:\n"
            report_text += f"  ID: {patient.patient_id}\n"
            report_text += f"  Name: {patient.patient_name}\n"
            report_text += f"  Study Date: {patient.study_date}\n\n"

        # Summary results
        report_text += "RWS Analysis Summary:\n"
        report_text += f"  RWS: {self.data.rws_result.rws_percentage}%\n"
        report_text += f"  Minimum MLD: {self.data.rws_result.min_mld:.3f} mm (Frame {self.data.rws_result.min_mld_frame + 1})\n"
        report_text += f"  Maximum MLD: {self.data.rws_result.max_mld:.3f} mm (Frame {self.data.rws_result.max_mld_frame + 1})\n"
        report_text += f"  MLD Variation: {self.data.rws_result.mld_variation:.3f} mm\n"
        report_text += f"  Frames Analyzed: {self.data.num_frames_analyzed}\n"
        report_text += f"  Calibration Factor: {self.data.calibration_factor:.4f} mm/pixel\n\n"

        # Clinical interpretation
        report_text += "Clinical Interpretation:\n"
        interpretation = self.data.rws_result.get_clinical_interpretation()
        if self.data.rws_result.is_vulnerable:
            report_text += f"  ⚠️ {interpretation}\n"
            report_text += "  Recommendation: Consider further evaluation\n\n"
        else:
            report_text += f"  ✓ {interpretation}\n\n"

        # RWS Calculation Details
        report_text += "RWS Calculation:\n"
        report_text += "  Formula: (MLDmax - MLDmin) / MLDmax × 100%\n"
        report_text += f"  Calculation: ({self.data.rws_result.max_mld:.3f} - {self.data.rws_result.min_mld:.3f}) / "
        report_text += f"{self.data.rws_result.max_mld:.3f} × 100%\n"
        report_text += f"  Result: {self.data.rws_result.rws_percentage}%\n"

        fig_title.text(
            0.1, 0.85, report_text, ha="left", va="top", fontsize=10, family="monospace", wrap=True
        )

        plt.axis("off")
        pdf.savefig(fig_title, bbox_inches="tight")
        plt.close(fig_title)

    def export_to_excel(self, output_path: str) -> bool:
        """
        Export RWS results to Excel file

        Args:
            output_path: Path for output Excel file

        Returns:
            True if successful, False otherwise
        """
        if not self.data.success:
            logger.error("No valid RWS results to export")
            return False

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Summary sheet
                summary_df = self._create_summary_dataframe()
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # MLD Values sheet
                mld_df = self._create_mld_dataframe()
                mld_df.to_excel(writer, sheet_name="MLD Diameter Changes", index=False)

                # Statistics sheet
                stats_df = self._create_statistics_dataframe()
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)

                # Patient info sheet (if available)
                if self.data.patient_info:
                    patient_df = self._create_patient_dataframe()
                    patient_df.to_excel(writer, sheet_name="Patient Info", index=False)

            logger.info(f"Excel report exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Excel export error: {e}")
            return False

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary dataframe"""
        summary_data = {
            "Parameter": [
                "RWS (%)",
                "Minimum MLD (mm)",
                "Maximum MLD (mm)",
                "MLD Variation (mm)",
                "Min MLD Frame",
                "Max MLD Frame",
                "Frames Analyzed",
                "Calibration Factor (mm/pixel)",
                "Clinical Interpretation",
                "Analysis Timestamp",
            ],
            "Value": [
                f"{self.data.rws_result.rws_percentage:.1f}",
                f"{self.data.rws_result.min_mld:.3f}",
                f"{self.data.rws_result.max_mld:.3f}",
                f"{self.data.rws_result.mld_variation:.3f}",
                self.data.rws_result.min_mld_frame + 1,
                self.data.rws_result.max_mld_frame + 1,
                self.data.num_frames_analyzed,
                f"{self.data.calibration_factor:.4f}",
                self.data.rws_result.get_clinical_interpretation(),
                self.data.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ],
        }
        return pd.DataFrame(summary_data)

    def _create_mld_dataframe(self) -> pd.DataFrame:
        """Create MLD values dataframe"""
        mld_data = []
        for frame_idx in sorted(self.data.mld_info_by_frame.keys()):
            info = self.data.mld_info_by_frame[frame_idx]
            mld_data.append(
                {
                    "Frame": info.frame_number,
                    "MLD Value (mm)": f"{info.mld_value:.3f}",
                    "MLD Position (index)": info.mld_index if info.mld_index is not None else "N/A",
                    "Frame Type": self._get_frame_type(frame_idx),
                }
            )
        return pd.DataFrame(mld_data)

    def _create_statistics_dataframe(self) -> pd.DataFrame:
        """Create statistics dataframe"""
        from ..rws.calculator import RWSCalculator

        stats = RWSCalculator.get_statistics(self.data.mld_info_by_frame)

        stats_data = {
            "Statistic": [
                "Mean MLD (mm)",
                "Std Dev MLD (mm)",
                "Median MLD (mm)",
                "Q1 MLD (mm)",
                "Q3 MLD (mm)",
                "IQR MLD (mm)",
                "Coefficient of Variation (%)",
            ],
            "Value": [
                f"{stats['mean_mld']:.3f}",
                f"{stats['std_mld']:.3f}",
                f"{stats['median_mld']:.3f}",
                f"{stats['q1_mld']:.3f}",
                f"{stats['q3_mld']:.3f}",
                f"{stats['iqr_mld']:.3f}",
                f"{stats['cv_mld']:.1f}",
            ],
        }
        return pd.DataFrame(stats_data)

    def _create_patient_dataframe(self) -> pd.DataFrame:
        """Create patient info dataframe"""
        patient = self.data.patient_info
        patient_data = {
            "Field": ["Patient ID", "Patient Name", "Study Date", "Study Description", "Physician"],
            "Value": [
                patient.patient_id,
                patient.patient_name,
                patient.study_date,
                patient.study_description or "N/A",
                patient.physician or "N/A",
            ],
        }
        return pd.DataFrame(patient_data)

    def _get_frame_type(self, frame_idx: int) -> str:
        """Get frame type description"""
        if frame_idx == self.data.rws_result.min_mld_frame:
            return "Minimum MLD"
        elif frame_idx == self.data.rws_result.max_mld_frame:
            return "Maximum MLD"
        return "Normal"

    def get_database_record(self) -> Dict:
        """Get RWS results formatted for database storage"""
        if not self.data.success:
            return None

        # Prepare data for database
        db_record = {
            "timestamp": self.data.timestamp.isoformat(),
            "rws": self.data.rws_result.rws_percentage,
            "mld_min_value": self.data.rws_result.min_mld,
            "mld_max_value": self.data.rws_result.max_mld,
            "mld_variation": self.data.rws_result.mld_variation,
            "mld_min_frame": self.data.rws_result.min_mld_frame,
            "mld_max_frame": self.data.rws_result.max_mld_frame,
            "num_frames": self.data.num_frames_analyzed,
            "calibration_factor": self.data.calibration_factor,
            "clinical_interpretation": self.data.rws_result.get_clinical_interpretation(),
            "is_vulnerable": self.data.rws_result.is_vulnerable,
            "beat_frames": ",".join(map(str, self.data.beat_frames)),
        }

        # Add patient info if available
        if self.data.patient_info:
            db_record.update(
                {
                    "patient_id": self.data.patient_info.patient_id,
                    "patient_name": self.data.patient_info.patient_name,
                    "study_date": self.data.patient_info.study_date,
                }
            )

        return db_record
