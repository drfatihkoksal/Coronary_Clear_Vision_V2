"""
RWS visualization module
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .models import RWSAnalysisData

logger = logging.getLogger(__name__)


class RWSVisualizer:
    """Generate visualizations for RWS analysis"""

    def __init__(self, analysis_data: RWSAnalysisData):
        self.data = analysis_data

    def generate_all_visualizations(self) -> Dict[str, plt.Figure]:
        """Generate all RWS visualization plots"""
        if not self.data.success:
            return {}

        figures = {}

        try:
            # Generate individual visualizations
            figures["diameter_variations"] = self._create_diameter_variation_plot()
            figures["mld_variation"] = self._create_mld_variation_plot()
            figures["frame_profiles"] = self._create_frame_comparison_plot()

            return figures

        except Exception as e:
            logger.error(f"Visualization generation error: {e}")
            return figures

    def _create_diameter_variation_plot(self) -> plt.Figure:
        """Create diameter variations across cardiac cycle plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        diameter_profiles = self.data.diameter_profiles
        min_frame = self.data.rws_result.min_mld_frame
        max_frame = self.data.rws_result.max_mld_frame
        mld_info = self.data.mld_info_by_frame

        # Plot diameter profiles for each frame
        for frame_idx, diameters in diameter_profiles.items():
            positions = np.arange(len(diameters))

            if frame_idx == min_frame:
                ax.plot(
                    positions,
                    diameters,
                    "r-",
                    linewidth=2,
                    label=f"Min MLD (Frame {frame_idx + 1})",
                )
                # Mark MLD position
                if frame_idx in mld_info and mld_info[frame_idx].mld_index is not None:
                    mld_idx = mld_info[frame_idx].mld_index
                    ax.plot(mld_idx, diameters[mld_idx], "ro", markersize=10)

            elif frame_idx == max_frame:
                ax.plot(
                    positions,
                    diameters,
                    "g-",
                    linewidth=2,
                    label=f"Max MLD (Frame {frame_idx + 1})",
                )
                # Mark MLD position
                if frame_idx in mld_info and mld_info[frame_idx].mld_index is not None:
                    mld_idx = mld_info[frame_idx].mld_index
                    ax.plot(mld_idx, diameters[mld_idx], "go", markersize=10)
            else:
                ax.plot(positions, diameters, alpha=0.3, color="gray")

        ax.set_xlabel("Position along vessel (pixels)")
        ax.set_ylabel("Diameter (mm)")
        ax.set_title("Diameter Variations Across Cardiac Cycle")
        ax.grid(True, alpha=0.3)
        if len(diameter_profiles) <= 10:
            ax.legend()

        return fig

    def _create_mld_variation_plot(self) -> plt.Figure:
        """Create MLD values across frames plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract MLD values from each frame
        mld_info = self.data.mld_info_by_frame
        frame_indices = sorted(mld_info.keys())
        mld_values = [mld_info[idx].mld_value for idx in frame_indices]
        frame_numbers = [idx + 1 for idx in frame_indices]

        # Plot MLD values
        ax.plot(frame_numbers, mld_values, "b-", linewidth=2, marker="o", markersize=6)

        # Mark min and max MLD points
        min_frame = self.data.rws_result.min_mld_frame
        max_frame = self.data.rws_result.max_mld_frame

        if min_frame in frame_indices:
            min_idx = frame_indices.index(min_frame)
            ax.plot(
                frame_numbers[min_idx],
                mld_values[min_idx],
                "ro",
                markersize=12,
                label=f"Min MLD: {self.data.rws_result.min_mld:.3f}mm",
            )

        if max_frame in frame_indices:
            max_idx = frame_indices.index(max_frame)
            ax.plot(
                frame_numbers[max_idx],
                mld_values[max_idx],
                "go",
                markersize=12,
                label=f"Max MLD: {self.data.rws_result.max_mld:.3f}mm",
            )

        # Add title with RWS value
        ax.set_title(f"MLD Diameter Variation - RWS = {self.data.rws_result.rws_percentage}%")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("MLD Diameter (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add annotation for clinical interpretation
        interpretation = self.data.rws_result.get_clinical_interpretation()
        ax.text(
            0.02,
            0.98,
            interpretation,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="yellow" if self.data.rws_result.is_vulnerable else "lightgreen",
                alpha=0.7,
            ),
        )

        return fig

    def _create_frame_comparison_plot(self) -> plt.Figure:
        """Create side-by-side comparison of min and max MLD frames"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        min_frame = self.data.rws_result.min_mld_frame
        max_frame = self.data.rws_result.max_mld_frame
        mld_info = self.data.mld_info_by_frame
        diameter_profiles = self.data.diameter_profiles

        # Plot min MLD frame
        if min_frame in diameter_profiles:
            diameters = diameter_profiles[min_frame]
            positions = np.arange(len(diameters))
            ax1.plot(positions, diameters, "r-", linewidth=2)

            if min_frame in mld_info and mld_info[min_frame].mld_index is not None:
                mld_idx = mld_info[min_frame].mld_index
                ax1.plot(mld_idx, diameters[mld_idx], "ro", markersize=10)
                ax1.annotate(
                    f"MLD: {diameters[mld_idx]:.3f}mm",
                    xy=(mld_idx, diameters[mld_idx]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            ax1.set_title(f"Minimum MLD Frame {min_frame + 1}")
            ax1.set_xlabel("Position (pixels)")
            ax1.set_ylabel("Diameter (mm)")
            ax1.grid(True, alpha=0.3)

        # Plot max MLD frame
        if max_frame in diameter_profiles:
            diameters = diameter_profiles[max_frame]
            positions = np.arange(len(diameters))
            ax2.plot(positions, diameters, "g-", linewidth=2)

            if max_frame in mld_info and mld_info[max_frame].mld_index is not None:
                mld_idx = mld_info[max_frame].mld_index
                ax2.plot(mld_idx, diameters[mld_idx], "go", markersize=10)
                ax2.annotate(
                    f"MLD: {diameters[mld_idx]:.3f}mm",
                    xy=(mld_idx, diameters[mld_idx]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="green", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            ax2.set_title(f"Maximum MLD Frame {max_frame + 1}")
            ax2.set_xlabel("Position (pixels)")
            ax2.set_ylabel("Diameter (mm)")
            ax2.grid(True, alpha=0.3)

        fig.suptitle("Diameter Profiles: Min vs Max MLD Frames")
        return fig

    def create_summary_plot(self) -> plt.Figure:
        """Create a single summary plot with key information"""
        fig = plt.figure(figsize=(12, 8))

        # Create grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Top row - full width
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
        ax3 = fig.add_subplot(gs[1, 1])  # Bottom right

        # 1. MLD variation over time (top)
        mld_info = self.data.mld_info_by_frame
        frame_indices = sorted(mld_info.keys())
        mld_values = [mld_info[idx].mld_value for idx in frame_indices]
        frame_numbers = [idx + 1 for idx in frame_indices]

        ax1.plot(frame_numbers, mld_values, "b-", linewidth=2, marker="o")
        ax1.set_title(
            f"RWS Analysis Summary - RWS = {self.data.rws_result.rws_percentage}%", fontsize=14
        )
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("MLD (mm)")
        ax1.grid(True, alpha=0.3)

        # 2. Statistics box (bottom left)
        ax2.axis("off")
        stats_text = f"""RWS Analysis Results:
        
RWS: {self.data.rws_result.rws_percentage}%
Min MLD: {self.data.rws_result.min_mld:.3f} mm (Frame {self.data.rws_result.min_mld_frame + 1})
Max MLD: {self.data.rws_result.max_mld:.3f} mm (Frame {self.data.rws_result.max_mld_frame + 1})
MLD Variation: {self.data.rws_result.mld_variation:.3f} mm

Clinical Interpretation:
{self.data.rws_result.get_clinical_interpretation()}
"""
        ax2.text(
            0.1,
            0.9,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )

        # 3. RWS gauge (bottom right)
        self._create_rws_gauge(ax3, self.data.rws_result.rws_percentage)

        return fig

    def _create_rws_gauge(self, ax, rws_value: float):
        """Create a gauge visualization for RWS value"""
        # Create semi-circle gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1

        # Background arc
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, "lightgray", linewidth=20)

        # Color zones
        # Green zone (0-12%)
        theta_green = np.linspace(0, np.pi * 0.12 / 20, 50)
        x_green = r * np.cos(theta_green)
        y_green = r * np.sin(theta_green)
        ax.plot(x_green, y_green, "green", linewidth=20, alpha=0.6)

        # Red zone (>12%)
        theta_red = np.linspace(np.pi * 0.12 / 20, np.pi, 50)
        x_red = r * np.cos(theta_red)
        y_red = r * np.sin(theta_red)
        ax.plot(x_red, y_red, "red", linewidth=20, alpha=0.6)

        # Indicator needle
        angle = np.pi * (1 - rws_value / 20)  # Scale to 0-20% range
        x_needle = [0, 0.8 * np.cos(angle)]
        y_needle = [0, 0.8 * np.sin(angle)]
        ax.plot(x_needle, y_needle, "black", linewidth=3)
        ax.plot(0, 0, "ko", markersize=10)

        # Labels
        ax.text(0, -0.3, f"{rws_value}%", ha="center", fontsize=16, fontweight="bold")
        ax.text(-1.2, -0.1, "0%", ha="center")
        ax.text(1.2, -0.1, "20%", ha="center")
        ax.text(0, 1.2, "10%", ha="center")

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("RWS Gauge", fontsize=12)
