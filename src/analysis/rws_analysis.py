"""
Radial Wall Strain (RWS) Analysis Module
Calculates RWS from sequential QCA measurements across cardiac phases
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

logger = logging.getLogger(__name__)


class RWSAnalysis:
    """
    Analyzes Radial Wall Strain from sequential QCA data.
    RWS = (Dmax - Dmin) / Dmax × 100%
    """

    def __init__(self):
        """Initialize RWS Analysis"""
        self.qca_results = {}
        self.rws_results = {}
        self.beat_info = {}
        self.calibration_factor = None

    def analyze_beat(
        self,
        qca_results: Dict[int, Dict],
        beat_frames: List[int],
        calibration_factor: float,
        patient_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Analyze RWS for a single cardiac beat.

        Args:
            qca_results: Dictionary of QCA results keyed by frame index
            beat_frames: List of frame indices for the beat
            calibration_factor: Calibration factor in mm/pixel
            patient_info: Optional patient information

        Returns:
            Dictionary containing RWS analysis results
        """
        try:
            # Store calibration factor
            self.calibration_factor = calibration_factor

            # Filter valid QCA results for the beat
            valid_results = {}
            for frame, qca in qca_results.items():
                if frame in beat_frames and qca.get("success", False):
                    valid_results[frame] = qca
                    # Log available keys for debugging
                    logger.debug(f"Frame {frame} QCA keys: {list(qca.keys())}")

            logger.info(
                f"Found {len(valid_results)} valid QCA results out of {len(beat_frames)} beat frames"
            )

            if len(valid_results) < 2:
                return {"success": False, "error": "Insufficient valid QCA data for RWS analysis"}

            # Extract diameter data along vessel
            diameter_profiles = self._extract_diameter_profiles(valid_results)

            if not diameter_profiles:
                return {
                    "success": False,
                    "error": "Failed to extract diameter profiles from QCA results. Make sure QCA analysis includes diameter measurements.",
                }

            logger.info(f"Successfully extracted {len(diameter_profiles)} diameter profiles")

            # Calculate RWS from MLD values
            rws_result = self._calculate_rws_from_mld_values(diameter_profiles, valid_results)

            # Compile results
            results = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "beat_frames": beat_frames,
                "num_frames_analyzed": len(valid_results),
                "calibration_factor": calibration_factor,
                "rws_at_mld": rws_result["rws"],
                "mld_min_value": rws_result["min_mld"],
                "mld_max_value": rws_result["max_mld"],
                "mld_min_frame": rws_result["min_mld_frame"],
                "mld_max_frame": rws_result["max_mld_frame"],
                "mld_min_index": rws_result["min_mld_index"],
                "mld_max_index": rws_result["max_mld_index"],
                "mld_values_by_frame": rws_result["mld_info_by_frame"],
                "diameter_profiles": diameter_profiles,
                "patient_info": patient_info,
            }

            # Store results
            self.rws_results = results

            logger.info(f"RWS Analysis completed: RWS = {rws_result['rws']}%")

            return results

        except Exception as e:
            logger.error(f"RWS analysis error: {e}")
            return {"success": False, "error": str(e)}

    def _extract_diameter_profiles(self, qca_results: Dict[int, Dict]) -> Dict[int, np.ndarray]:
        """Extract diameter measurements along the vessel for each frame"""
        diameter_profiles = {}

        for frame_idx, qca_data in qca_results.items():
            # Check for diameter data in different possible keys
            diameters = None

            # Try 'diameters_mm' first (already in mm)
            if "diameters_mm" in qca_data and qca_data["diameters_mm"] is not None:
                diameters = np.array(qca_data["diameters_mm"])
                logger.debug(f"Frame {frame_idx}: Found diameters_mm with {len(diameters)} points")
            # Try 'diameters_pixels' and convert to mm
            elif "diameters_pixels" in qca_data and qca_data["diameters_pixels"] is not None:
                diameters = np.array(qca_data["diameters_pixels"])
                if self.calibration_factor:
                    diameters = diameters * self.calibration_factor
                    logger.debug(
                        f"Frame {frame_idx}: Found diameters_pixels, converted to mm with {len(diameters)} points"
                    )
                else:
                    logger.warning(
                        f"Frame {frame_idx}: Found diameters_pixels but no calibration factor"
                    )
            # Try 'diameters' (might be in pixels or mm)
            elif "diameters" in qca_data and qca_data["diameters"] is not None:
                diameters = np.array(qca_data["diameters"])
                # If values are too large (>50), assume they're in pixels and need conversion
                if np.max(diameters) > 50 and self.calibration_factor:
                    logger.debug(f"Frame {frame_idx}: Converting pixel diameters to mm")
                    diameters = diameters * self.calibration_factor
                logger.debug(f"Frame {frame_idx}: Found diameters with {len(diameters)} points")
            # Try 'profile_data' which contains diameter information
            elif "profile_data" in qca_data and qca_data["profile_data"] is not None:
                profile = qca_data["profile_data"]
                if "diameters" in profile:
                    diameters = np.array(profile["diameters"])
                    logger.debug(
                        f"Frame {frame_idx}: Found diameters in profile_data with {len(diameters)} points"
                    )

            if diameters is not None and len(diameters) > 0:
                diameter_profiles[frame_idx] = diameters
            else:
                logger.warning(f"Frame {frame_idx}: No diameter data found")

        logger.info(f"Extracted diameter profiles for {len(diameter_profiles)} frames")
        return diameter_profiles

    def _calculate_rws_from_mld_values(
        self, diameter_profiles: Dict[int, np.ndarray], qca_results: Dict
    ) -> Dict:
        """Calculate RWS from the widest and narrowest MLD values across all frames"""
        if not qca_results:
            raise ValueError("No QCA results available")

        # Collect all MLD values from each frame
        mld_values = []
        mld_info_by_frame = {}

        for frame_idx, qca_data in qca_results.items():
            if "mld" in qca_data and qca_data["mld"] is not None:
                mld_value = float(qca_data["mld"])
                mld_values.append(mld_value)
                mld_info_by_frame[frame_idx] = {
                    "mld_value": mld_value,
                    "mld_index": qca_data.get("mld_index", None),
                }
                logger.debug(f"Frame {frame_idx}: MLD = {mld_value:.2f}mm")

        if not mld_values:
            raise ValueError("No MLD values found in QCA results")

        # Find the widest (max) and narrowest (min) MLD
        max_mld = max(mld_values)
        min_mld = min(mld_values)

        # Find which frames have these values (use first occurrence if multiple)
        max_mld_frame = None
        min_mld_frame = None
        max_mld_index = None
        min_mld_index = None

        for frame_idx, info in mld_info_by_frame.items():
            if info["mld_value"] == max_mld and max_mld_frame is None:
                max_mld_frame = frame_idx
                max_mld_index = info["mld_index"]
            if info["mld_value"] == min_mld and min_mld_frame is None:
                min_mld_frame = frame_idx
                min_mld_index = info["mld_index"]

        # Calculate RWS: (MLDmax - MLDmin) / MLDmin × 100% (CORRECT FORMULA)
        if min_mld > 0:
            rws = ((max_mld - min_mld) / min_mld) * 100
        else:
            rws = 0.0

        logger.info(
            f"RWS Calculation: MLDmax={max_mld:.2f}mm (Frame {max_mld_frame+1}), MLDmin={min_mld:.2f}mm (Frame {min_mld_frame+1})"
        )
        logger.info(f"RWS = ({max_mld:.2f} - {min_mld:.2f}) / {min_mld:.2f} × 100% = {rws}%")

        return {
            "rws": rws,
            "max_mld": max_mld,
            "min_mld": min_mld,
            "max_mld_frame": max_mld_frame,
            "min_mld_frame": min_mld_frame,
            "max_mld_index": max_mld_index,
            "min_mld_index": min_mld_index,
            "mld_values": mld_values,
            "mld_info_by_frame": mld_info_by_frame,
        }

    def _find_valid_mld(
        self, mld_candidates: list, proximal_ref: int, distal_ref: int, diameter_profiles: Dict
    ) -> Optional[int]:
        """Find a valid MLD that meets distance and position criteria"""
        if not mld_candidates:
            return None

        # Sort candidates by diameter (smallest first)
        sorted_candidates = sorted(mld_candidates, key=lambda x: x[1])

        for mld_idx, mld_value in sorted_candidates:
            if self._validate_mld_position(mld_idx, proximal_ref, distal_ref):
                return mld_idx

        return None

    def _validate_mld_position(
        self, mld_idx: int, proximal_ref: Optional[int], distal_ref: Optional[int]
    ) -> bool:
        """Validate MLD position relative to reference points"""
        MIN_DISTANCE = 20  # Minimum distance in pixels

        # If no reference points, accept any MLD
        if proximal_ref is None or distal_ref is None:
            return True

        # Ensure correct order (handle if they're reversed)
        min_ref = min(proximal_ref, distal_ref)
        max_ref = max(proximal_ref, distal_ref)

        # Check if MLD is between proximal and distal references
        if not (min_ref < mld_idx < max_ref):
            logger.debug(f"MLD at {mld_idx} is not between references ({min_ref}, {max_ref})")
            return False

        # Check minimum distance from proximal reference
        if abs(mld_idx - proximal_ref) < MIN_DISTANCE:
            logger.debug(f"MLD at {mld_idx} is too close to proximal reference {proximal_ref}")
            return False

        # Check minimum distance from distal reference
        if abs(mld_idx - distal_ref) < MIN_DISTANCE:
            logger.debug(f"MLD at {mld_idx} is too close to distal reference {distal_ref}")
            return False

        return True

    def generate_visualizations(self) -> Dict:
        """Generate RWS visualization plots focused on MLD"""
        if not self.rws_results or not self.rws_results.get("success"):
            return {"success": False, "error": "No valid RWS results to visualize"}

        try:
            figures = {}

            # 1. MLD Diameter Variation Plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            diameter_profiles = self.rws_results["diameter_profiles"]

            # Plot diameter profiles for each frame
            min_frame = self.rws_results["mld_min_frame"]
            max_frame = self.rws_results["mld_max_frame"]

            mld_info = self.rws_results["mld_values_by_frame"]

            for frame_idx, diameters in diameter_profiles.items():
                positions = np.arange(len(diameters))
                if frame_idx == min_frame:
                    ax1.plot(
                        positions,
                        diameters,
                        "r-",
                        linewidth=2,
                        label=f"Min MLD (Frame {frame_idx + 1})",
                    )
                    # Mark MLD position for min frame
                    if frame_idx in mld_info and mld_info[frame_idx]["mld_index"] is not None:
                        mld_idx = mld_info[frame_idx]["mld_index"]
                        ax1.plot(mld_idx, diameters[mld_idx], "ro", markersize=10)
                elif frame_idx == max_frame:
                    ax1.plot(
                        positions,
                        diameters,
                        "g-",
                        linewidth=2,
                        label=f"Max MLD (Frame {frame_idx + 1})",
                    )
                    # Mark MLD position for max frame
                    if frame_idx in mld_info and mld_info[frame_idx]["mld_index"] is not None:
                        mld_idx = mld_info[frame_idx]["mld_index"]
                        ax1.plot(mld_idx, diameters[mld_idx], "go", markersize=10)
                else:
                    ax1.plot(positions, diameters, alpha=0.3, color="gray")

            ax1.set_xlabel("Position along vessel (pixels)")
            ax1.set_ylabel("Diameter (mm)")
            ax1.set_title("Diameter Variations Across Cardiac Cycle")
            ax1.grid(True, alpha=0.3)
            if len(diameter_profiles) <= 10:  # Only show legend if not too many frames
                ax1.legend()

            figures["diameter_variations"] = fig1

            # 2. MLD Values Across Frames Plot
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # Extract MLD values from each frame
            mld_info = self.rws_results["mld_values_by_frame"]
            frame_indices = sorted(mld_info.keys())
            mld_values = []

            for frame_idx in frame_indices:
                mld_values.append(mld_info[frame_idx]["mld_value"])

            # Convert frame indices to 1-based for display
            frame_numbers = [idx + 1 for idx in frame_indices]

            # Plot MLD values
            ax2.plot(frame_numbers, mld_values, "b-", linewidth=2, marker="o", markersize=6)

            # Mark min and max MLD points
            mld_min_val = self.rws_results.get("mld_min_value")
            mld_max_val = self.rws_results.get("mld_max_value")
            rws_val = self.rws_results.get("rws_at_mld", 0)

            if min_frame is not None and min_frame in frame_indices and mld_min_val is not None:
                min_idx = frame_indices.index(min_frame)
                ax2.plot(
                    frame_numbers[min_idx],
                    mld_values[min_idx],
                    "ro",
                    markersize=12,
                    label=f"Min MLD: {mld_min_val:.2f}mm (Frame {min_frame + 1})",
                )

            if max_frame is not None and max_frame in frame_indices and mld_max_val is not None:
                max_idx = frame_indices.index(max_frame)
                ax2.plot(
                    frame_numbers[max_idx],
                    mld_values[max_idx],
                    "go",
                    markersize=12,
                    label=f"Max MLD: {mld_max_val:.2f}mm (Frame {max_frame + 1})",
                )

            # Add RWS value as title
            title_text = (
                f"MLD Diameter Variation - RWS = {rws_val}%"
                if rws_val is not None
                else "MLD Diameter Variation - RWS = N/A"
            )
            ax2.set_title(title_text)
            ax2.set_xlabel("Frame Number")
            ax2.set_ylabel("MLD Diameter (mm)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Set x-axis to show frame numbers as integers
            ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            figures["mld_variation"] = fig2

            # 3. Frame-specific diameter profiles for min and max MLD
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

            # Min MLD frame profile
            if min_frame is not None and min_frame in diameter_profiles:
                diameters = diameter_profiles[min_frame]
                positions = np.arange(len(diameters))
                ax3a.plot(positions, diameters, "r-", linewidth=2)
                if min_frame in mld_info and mld_info[min_frame]["mld_index"] is not None:
                    mld_idx = mld_info[min_frame]["mld_index"]
                    ax3a.plot(mld_idx, diameters[mld_idx], "ro", markersize=10)
                    ax3a.annotate(
                        f"MLD: {diameters[mld_idx]:.2f}mm",
                        xy=(mld_idx, diameters[mld_idx]),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )
                ax3a.set_title(f"Minimum MLD Frame {min_frame + 1}")
                ax3a.set_xlabel("Position (pixels)")
                ax3a.set_ylabel("Diameter (mm)")
                ax3a.grid(True, alpha=0.3)

            # Max MLD frame profile
            if max_frame is not None and max_frame in diameter_profiles:
                diameters = diameter_profiles[max_frame]
                positions = np.arange(len(diameters))
                ax3b.plot(positions, diameters, "g-", linewidth=2)
                if max_frame in mld_info and mld_info[max_frame]["mld_index"] is not None:
                    mld_idx = mld_info[max_frame]["mld_index"]
                    ax3b.plot(mld_idx, diameters[mld_idx], "go", markersize=10)
                    ax3b.annotate(
                        f"MLD: {diameters[mld_idx]:.2f}mm",
                        xy=(mld_idx, diameters[mld_idx]),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc="green", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )
                ax3b.set_title(f"Maximum MLD Frame {max_frame + 1}")
                ax3b.set_xlabel("Position (pixels)")
                ax3b.set_ylabel("Diameter (mm)")
                ax3b.grid(True, alpha=0.3)

            fig3.suptitle("Diameter Profiles: Min vs Max MLD Frames")
            figures["frame_profiles"] = fig3

            return {"success": True, "figures": figures}

        except Exception as e:
            logger.error(f"Visualization generation error: {e}")
            return {"success": False, "error": str(e)}

    def generate_pdf_report(self, output_path: str, include_patient_info: bool = True) -> bool:
        """Generate PDF report with RWS analysis results"""
        if not self.rws_results or not self.rws_results.get("success"):
            logger.error("No valid RWS results to generate report")
            return False

        try:
            # Generate visualizations
            viz_result = self.generate_visualizations()
            if not viz_result.get("success"):
                logger.error(f"Failed to generate visualizations: {viz_result.get('error')}")
                return False

            figures = viz_result["figures"]

            # Create PDF
            with PdfPages(output_path) as pdf:
                # Title page
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

                # Report info
                report_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                if include_patient_info and self.rws_results.get("patient_info"):
                    patient = self.rws_results["patient_info"]
                    report_text += "Patient Information:\n"
                    report_text += f"  ID: {patient.get('id', 'N/A')}\n"
                    report_text += f"  Name: {patient.get('name', 'N/A')}\n"
                    report_text += f"  Study Date: {patient.get('study_date', 'N/A')}\n\n"

                # Summary results
                report_text += "RWS Analysis Summary:\n"
                # Safe formatting with None checks
                rws_val = self.rws_results.get("rws_at_mld", 0)
                report_text += f"  RWS: {rws_val}%\n" if rws_val is not None else "  RWS: N/A\n"

                min_frame_ui = (
                    self.rws_results["mld_min_frame"] + 1
                    if self.rws_results["mld_min_frame"] is not None
                    else "N/A"
                )
                max_frame_ui = (
                    self.rws_results["mld_max_frame"] + 1
                    if self.rws_results["mld_max_frame"] is not None
                    else "N/A"
                )

                mld_min = self.rws_results.get("mld_min_value")
                mld_max = self.rws_results.get("mld_max_value")
                cal_factor = self.rws_results.get("calibration_factor")

                if mld_min is not None:
                    report_text += f"  Minimum MLD: {mld_min:.2f} mm (Frame {min_frame_ui})\n"
                else:
                    report_text += f"  Minimum MLD: N/A (Frame {min_frame_ui})\n"

                if mld_max is not None:
                    report_text += f"  Maximum MLD: {mld_max:.2f} mm (Frame {max_frame_ui})\n"
                else:
                    report_text += f"  Maximum MLD: N/A (Frame {max_frame_ui})\n"

                if mld_max is not None and mld_min is not None:
                    report_text += f"  MLD Variation: {mld_max - mld_min:.2f} mm\n"
                else:
                    report_text += f"  MLD Variation: N/A\n"

                report_text += (
                    f"  Frames Analyzed: {self.rws_results.get('num_frames_analyzed', 0)}\n"
                )

                if cal_factor is not None:
                    report_text += f"  Calibration Factor: {cal_factor:.4f} mm/pixel\n\n"
                else:
                    report_text += f"  Calibration Factor: N/A\n\n"

                # Clinical interpretation
                report_text += "Clinical Interpretation:\n"
                if rws_val is not None and rws_val > 12:
                    report_text += (
                        "  ⚠️ HIGH RWS at MLD (>12%): Indicates potential plaque vulnerability\n"
                    )
                    report_text += "  Recommendation: Consider further evaluation\n"
                elif rws_val is not None:
                    report_text += (
                        "  ✓ NORMAL RWS at MLD (<12%): Indicates stable plaque characteristics\n\n"
                    )
                else:
                    report_text += "  Unable to determine - insufficient data\n\n"

                # RWS Calculation Details
                report_text += "RWS Calculation:\n"
                report_text += f"  Formula: (MLDmax - MLDmin) / MLDmax × 100%\n"

                if mld_max is not None and mld_min is not None and mld_max > 0:
                    report_text += (
                        f"  Calculation: ({mld_max:.2f} - {mld_min:.2f}) / {mld_max:.2f} × 100%\n"
                    )
                    if rws_val is not None:
                        report_text += f"  Result: {rws_val}%\n"
                    else:
                        report_text += f"  Result: N/A\n"
                else:
                    report_text += f"  Calculation: N/A (insufficient data)\n"
                    report_text += f"  Result: N/A\n"

                fig_title.text(
                    0.1,
                    0.85,
                    report_text,
                    ha="left",
                    va="top",
                    fontsize=10,
                    family="monospace",
                    wrap=True,
                )

                plt.axis("off")
                pdf.savefig(fig_title, bbox_inches="tight")
                plt.close(fig_title)

                # Add visualization pages
                for fig_name, fig in figures.items():
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

                # Metadata
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

    def export_to_excel(self, output_path: str) -> bool:
        """Export RWS results to Excel file"""
        if not self.rws_results or not self.rws_results.get("success"):
            logger.error("No valid RWS results to export")
            return False

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Summary sheet
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
                        "Analysis Timestamp",
                    ],
                    "Value": [
                        self.rws_results.get("rws_at_mld", "N/A"),
                        self.rws_results.get("mld_min_value", "N/A"),
                        self.rws_results.get("mld_max_value", "N/A"),
                        (
                            self.rws_results.get("mld_max_value", 0)
                            - self.rws_results.get("mld_min_value", 0)
                            if self.rws_results.get("mld_max_value") is not None
                            and self.rws_results.get("mld_min_value") is not None
                            else "N/A"
                        ),
                        (
                            self.rws_results["mld_min_frame"] + 1
                            if self.rws_results.get("mld_min_frame") is not None
                            else "N/A"
                        ),
                        (
                            self.rws_results["mld_max_frame"] + 1
                            if self.rws_results.get("mld_max_frame") is not None
                            else "N/A"
                        ),
                        self.rws_results.get("num_frames_analyzed", 0),
                        self.rws_results.get("calibration_factor", "N/A"),
                        self.rws_results.get("timestamp", "N/A"),
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

                # MLD Values sheet
                mld_info = self.rws_results["mld_values_by_frame"]

                # Extract MLD values across frames
                mld_data = []
                for frame_idx in sorted(mld_info.keys()):
                    mld_data.append(
                        {
                            "Frame": frame_idx + 1,  # 1-based for UI
                            "MLD Value (mm)": mld_info[frame_idx]["mld_value"],
                            "MLD Position (index)": (
                                mld_info[frame_idx]["mld_index"]
                                if mld_info[frame_idx]["mld_index"] is not None
                                else "N/A"
                            ),
                        }
                    )

                if mld_data:
                    df_mld = pd.DataFrame(mld_data)
                    df_mld.to_excel(writer, sheet_name="MLD Diameter Changes", index=False)

                # Diameter Profiles sheet (if not too large)
                diameter_profiles = self.rws_results["diameter_profiles"]
                if len(diameter_profiles) <= 20:  # Limit to prevent huge files
                    diameter_data = {}
                    for frame_idx, diameters in diameter_profiles.items():
                        diameter_data[f"Frame {frame_idx}"] = diameters
                    df_diameters = pd.DataFrame(diameter_data)
                    df_diameters.to_excel(writer, sheet_name="Diameter Profiles", index=False)

            logger.info(f"Excel report exported: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Excel export error: {e}")
            return False

    def get_database_record(self) -> Dict:
        """Get RWS results formatted for database storage"""
        if not self.rws_results or not self.rws_results.get("success"):
            return None

        # Prepare data for database
        db_record = {
            "timestamp": self.rws_results["timestamp"],
            "rws": self.rws_results["rws_at_mld"],
            "mld_min_value": self.rws_results["mld_min_value"],
            "mld_max_value": self.rws_results["mld_max_value"],
            "mld_variation": self.rws_results["mld_max_value"] - self.rws_results["mld_min_value"],
            "mld_min_frame": self.rws_results["mld_min_frame"],
            "mld_max_frame": self.rws_results["mld_max_frame"],
            "num_frames": self.rws_results["num_frames_analyzed"],
            "calibration_factor": self.rws_results["calibration_factor"],
            "clinical_interpretation": (
                "High RWS - Vulnerable"
                if self.rws_results["rws_at_mld"] > 12
                else "Normal RWS - Stable"
            ),
            "beat_frames": ",".join(map(str, self.rws_results["beat_frames"])),
        }

        # Add patient info if available
        if self.rws_results.get("patient_info"):
            db_record.update(
                {
                    "patient_id": self.rws_results["patient_info"].get("id"),
                    "patient_name": self.rws_results["patient_info"].get("name"),
                    "study_date": self.rws_results["patient_info"].get("study_date"),
                }
            )

        return db_record
