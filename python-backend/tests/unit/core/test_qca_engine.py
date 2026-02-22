import numpy as np
from app.core.qca_engine import compute_qca_measurements


class TestQCAEngine:
    def test_uniform_vessel(self):
        """A uniform-width vessel should have constant diameters and 0% stenosis."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255  # 14px wide uniform vessel
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]

        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.3, method="threshold")
        assert result["mld_mm"] > 0
        assert result["diameter_stenosis_pct"] < 30  # Should be very low for uniform

    def test_stenotic_vessel(self):
        """A vessel with a narrowing should show significant DS%."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        # Wide parts
        mask[20:44, 10:50] = 255  # 24px wide
        mask[20:44, 80:120] = 255  # 24px wide
        # Narrow part (stenosis)
        mask[28:36, 50:80] = 255  # 8px wide

        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]
        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.3, method="threshold")

        assert result["diameter_stenosis_pct"] > 30  # Should show significant stenosis
        assert result["mld_mm"] < result["proximal_ref_mm"]

    def test_empty_inputs(self):
        result = compute_qca_measurements(np.zeros((64, 64), dtype=np.uint8), [], 0.3)
        assert result["mld_mm"] == 0.0

    def test_gaussian_method(self):
        """Gaussian method should produce valid results."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]

        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.3, method="gaussian")
        assert result["mld_mm"] > 0
        assert result["method"] == "gaussian"
        assert result["num_points"] == 50

    def test_parabolic_method(self):
        """Parabolic method should produce valid results."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]

        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.3, method="parabolic")
        assert result["mld_mm"] > 0
        assert result["method"] == "parabolic"

    def test_result_structure(self):
        """Result dict should have all expected keys."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]

        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.3)
        expected_keys = {
            "centerline", "diameter_profile_mm", "diameter_profile_px",
            "distances_mm", "mld_mm", "mld_px", "mld_index",
            "diameter_stenosis_pct", "proximal_ref_mm", "distal_ref_mm",
            "proximal_ref_index", "distal_ref_index", "lesion_length_mm",
            "pixel_spacing_mm", "num_points", "method", "vessel_length_mm",
        }
        assert set(result.keys()) == expected_keys

    def test_vessel_length(self):
        """Vessel length should be positive for valid input."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]

        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.3)
        assert result["vessel_length_mm"] > 0

    def test_zero_pixel_spacing(self):
        """Zero pixel spacing should return empty result."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]

        result = compute_qca_measurements(mask, centerline, pixel_spacing_mm=0.0)
        assert result["mld_mm"] == 0.0

    def test_single_point_centerline(self):
        """Single-point centerline should return empty result."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255

        result = compute_qca_measurements(mask, [(50.0, 32.0)], pixel_spacing_mm=0.3)
        assert result["mld_mm"] == 0.0
