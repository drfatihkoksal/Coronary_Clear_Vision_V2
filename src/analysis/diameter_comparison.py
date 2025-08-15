"""
Diameter measurement method comparison tool
Compares different diameter measurement approaches for accuracy
"""

import numpy as np
import logging
from typing import Dict, Optional
import matplotlib.pyplot as plt
from .vessel_diameter_advanced import AdvancedDiameterMeasurement
from .vessel_diameter_matlab_inspired import MatlabInspiredDiameterMeasurement
from .vessel_diameter_simple import measure_vessel_diameter_simple

logger = logging.getLogger(__name__)


class DiameterMethodComparison:
    """Compare different diameter measurement methods"""

    def __init__(self):
        self.advanced = AdvancedDiameterMeasurement()
        self.matlab = MatlabInspiredDiameterMeasurement()

    def compare_methods(
        self, mask: np.ndarray, centerline: np.ndarray, ground_truth: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compare all diameter measurement methods

        Args:
            mask: Binary vessel mask
            centerline: Vessel centerline (y, x format)
            ground_truth: Optional ground truth diameters

        Returns:
            Comparison results
        """
        logger.info("Starting diameter method comparison")

        results = {}

        # Method 1: Simple
        logger.info("Testing simple method...")
        try:
            simple_diameters = measure_vessel_diameter_simple(mask, centerline)
            results["simple"] = {
                "diameters": simple_diameters,
                "valid_count": np.sum(simple_diameters > 0),
                "mean": (
                    np.mean(simple_diameters[simple_diameters > 0])
                    if np.any(simple_diameters > 0)
                    else 0
                ),
                "std": (
                    np.std(simple_diameters[simple_diameters > 0])
                    if np.any(simple_diameters > 0)
                    else 0
                ),
            }
        except Exception as e:
            logger.error(f"Simple method failed: {e}")
            results["simple"] = {"error": str(e)}

        # Method 2: Advanced
        logger.info("Testing advanced method...")
        try:
            advanced_result = self.advanced.measure_diameter_profile(
                mask=mask, centerline=centerline, method="combined"
            )
            results["advanced"] = {
                "diameters": advanced_result["diameters_pixels"],
                "statistics": advanced_result["statistics"],
                "valid_count": advanced_result["statistics"]["valid_count"],
                "mean": advanced_result["statistics"]["mean"],
                "std": advanced_result["statistics"]["std"],
            }
        except Exception as e:
            logger.error(f"Advanced method failed: {e}")
            results["advanced"] = {"error": str(e)}

        # Method 3: MATLAB-inspired
        logger.info("Testing MATLAB-inspired method...")
        try:
            matlab_result = self.matlab.measure_diameter_profile(mask=mask, centerline=centerline)
            results["matlab"] = {
                "diameters": matlab_result["diameters_pixels"],
                "statistics": matlab_result["statistics"],
                "valid_count": matlab_result["statistics"]["valid_count"],
                "mean": matlab_result["statistics"]["mean"],
                "std": matlab_result["statistics"]["std"],
            }
        except Exception as e:
            logger.error(f"MATLAB method failed: {e}")
            results["matlab"] = {"error": str(e)}

        # Compare with ground truth if available
        if ground_truth is not None:
            results["ground_truth_comparison"] = self._compare_with_ground_truth(
                results, ground_truth
            )

        # Cross-method comparison
        results["cross_comparison"] = self._cross_compare_methods(results)

        return results

    def _compare_with_ground_truth(self, results: Dict, ground_truth: np.ndarray) -> Dict:
        """Compare each method with ground truth"""
        comparison = {}

        for method_name, method_result in results.items():
            if isinstance(method_result, dict) and "diameters" in method_result:
                diameters = method_result["diameters"]

                # Ensure same length
                min_len = min(len(diameters), len(ground_truth))
                diameters = diameters[:min_len]
                gt = ground_truth[:min_len]

                # Calculate metrics
                valid_mask = (diameters > 0) & (gt > 0)
                if np.any(valid_mask):
                    mae = np.mean(np.abs(diameters[valid_mask] - gt[valid_mask]))
                    rmse = np.sqrt(np.mean((diameters[valid_mask] - gt[valid_mask]) ** 2))
                    mape = (
                        np.mean(np.abs((diameters[valid_mask] - gt[valid_mask]) / gt[valid_mask]))
                        * 100
                    )

                    comparison[method_name] = {
                        "mae": float(mae),
                        "rmse": float(rmse),
                        "mape": float(mape),
                        "correlation": float(
                            np.corrcoef(diameters[valid_mask], gt[valid_mask])[0, 1]
                        ),
                    }

        return comparison

    def _cross_compare_methods(self, results: Dict) -> Dict:
        """Compare methods against each other"""
        cross_comp = {}

        methods = ["simple", "advanced", "matlab"]
        valid_methods = [m for m in methods if m in results and "diameters" in results[m]]

        for i, method1 in enumerate(valid_methods):
            for method2 in valid_methods[i + 1 :]:
                d1 = results[method1]["diameters"]
                d2 = results[method2]["diameters"]

                # Ensure same length
                min_len = min(len(d1), len(d2))
                d1 = d1[:min_len]
                d2 = d2[:min_len]

                # Calculate correlation and differences
                valid_mask = (d1 > 0) & (d2 > 0)
                if np.any(valid_mask):
                    correlation = np.corrcoef(d1[valid_mask], d2[valid_mask])[0, 1]
                    mean_diff = np.mean(d1[valid_mask] - d2[valid_mask])
                    std_diff = np.std(d1[valid_mask] - d2[valid_mask])

                    cross_comp[f"{method1}_vs_{method2}"] = {
                        "correlation": float(correlation),
                        "mean_difference": float(mean_diff),
                        "std_difference": float(std_diff),
                    }

        return cross_comp

    def plot_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Plot comparison of different methods"""
        plt.figure(figsize=(12, 8))

        # Plot diameters from each method
        methods = ["simple", "advanced", "matlab"]
        colors = ["blue", "red", "green"]

        for method, color in zip(methods, colors):
            if method in results and "diameters" in results[method]:
                diameters = results[method]["diameters"]
                valid_mask = diameters > 0
                indices = np.where(valid_mask)[0]
                if len(indices) > 0:
                    plt.plot(
                        indices,
                        diameters[valid_mask],
                        label=f"{method.capitalize()} (μ={results[method]['mean']:.1f})",
                        color=color,
                        alpha=0.7,
                    )

        plt.xlabel("Centerline Position")
        plt.ylabel("Diameter (pixels)")
        plt.title("Diameter Measurement Method Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def generate_report(self, results: Dict) -> str:
        """Generate text report of comparison results"""
        report = []
        report.append("=" * 60)
        report.append("DIAMETER MEASUREMENT METHOD COMPARISON")
        report.append("=" * 60)

        # Individual method results
        for method in ["simple", "advanced", "matlab"]:
            if method in results:
                report.append(f"\n{method.upper()} METHOD:")
                if "error" in results[method]:
                    report.append(f"  Error: {results[method]['error']}")
                else:
                    report.append(f"  Valid measurements: {results[method]['valid_count']}")
                    report.append(f"  Mean diameter: {results[method]['mean']:.2f} pixels")
                    report.append(f"  Std deviation: {results[method]['std']:.2f} pixels")

        # Cross-comparison
        if "cross_comparison" in results:
            report.append("\nCROSS-METHOD COMPARISON:")
            for comparison, metrics in results["cross_comparison"].items():
                report.append(f"\n  {comparison}:")
                report.append(f"    Correlation: {metrics['correlation']:.3f}")
                report.append(f"    Mean difference: {metrics['mean_difference']:.2f} pixels")
                report.append(f"    Std difference: {metrics['std_difference']:.2f} pixels")

        # Ground truth comparison
        if "ground_truth_comparison" in results:
            report.append("\nGROUND TRUTH COMPARISON:")
            for method, metrics in results["ground_truth_comparison"].items():
                report.append(f"\n  {method}:")
                report.append(f"    MAE: {metrics['mae']:.2f} pixels")
                report.append(f"    RMSE: {metrics['rmse']:.2f} pixels")
                report.append(f"    MAPE: {metrics['mape']:.1f}%")
                report.append(f"    Correlation: {metrics['correlation']:.3f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
