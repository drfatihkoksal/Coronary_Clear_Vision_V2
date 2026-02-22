import numpy as np

from app.core.centerline_extractor import (
    extract_centerline,
    compute_perpendicular_diameters,
    measure_diameters_at_points,
)


class TestCenterlineExtractor:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = extract_centerline(mask)
        assert result == []

    def test_horizontal_vessel(self):
        """A horizontal strip should produce a roughly horizontal centerline."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[28:36, 5:60] = 255  # Horizontal vessel

        centerline = extract_centerline(mask, num_points=20)
        assert len(centerline) == 20

        # All y coords should be near center (28-36 range, center ~32)
        ys = [p[1] for p in centerline]
        assert all(26 < y < 38 for y in ys), f"Y coords out of range: {ys}"

    def test_vertical_vessel(self):
        """A vertical strip should produce a roughly vertical centerline."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:60, 28:36] = 255  # Vertical vessel

        centerline = extract_centerline(mask, num_points=20)
        assert len(centerline) == 20

        # All x coords should be near center
        xs = [p[0] for p in centerline]
        assert all(26 < x < 38 for x in xs), f"X coords out of range: {xs}"

    def test_diameter_measurement(self):
        """A vessel of known width should give consistent diameters."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255  # 14px wide horizontal vessel

        centerline = extract_centerline(mask, num_points=30)
        if len(centerline) >= 2:
            diameters = compute_perpendicular_diameters(centerline, mask)
            assert len(diameters) == len(centerline)
            # Most diameters should be close to 14 (+/-4 for edge effects)
            mid_diameters = diameters[5:-5]
            if mid_diameters:
                avg = np.mean(mid_diameters)
                assert 8 < avg < 20, f"Average diameter {avg} not close to expected ~14"

    def test_gaussian_diameter_measurement(self):
        """Perpendicular Gaussian fitting should give accurate diameters."""
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[25:39, 10:120] = 255  # 14px wide horizontal vessel

        # Horizontal centerline through vessel center
        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]
        diameters = measure_diameters_at_points(centerline, mask)
        assert len(diameters) == len(centerline)

        # Mid-vessel diameters should be close to 14px (FWHM of a 14px mask)
        mid = diameters[2:-2]
        avg = np.mean(mid)
        assert 10 < avg < 18, f"Average Gaussian diameter {avg} not close to 14"

    def test_gaussian_vs_edt_on_stenosis(self):
        """Gaussian measurement should detect narrowing in stenotic vessel."""
        mask = np.zeros((64, 200), dtype=np.uint8)
        for x in range(10, 190):
            # Vessel narrows from 14px to 6px at center, then widens back
            t = abs(x - 100) / 90.0  # 0 at center, 1 at edges
            half_width = int(3 + 4 * t)  # 3px at center (6px diam), 7px at edges (14px diam)
            mask[32 - half_width:32 + half_width, x] = 255

        centerline = [(float(x), 32.0) for x in range(15, 185, 2)]
        gauss_d = measure_diameters_at_points(centerline, mask)
        edt_d = compute_perpendicular_diameters(centerline, mask)

        # Both should detect narrowing at center
        gauss_min_idx = np.argmin(gauss_d)
        edt_min_idx = np.argmin(edt_d)
        n = len(centerline)
        # MLD should be near center (index ~42 of 85 points)
        assert n // 4 < gauss_min_idx < 3 * n // 4, f"Gaussian MLD at wrong position: {gauss_min_idx}"
        assert n // 4 < edt_min_idx < 3 * n // 4, f"EDT MLD at wrong position: {edt_min_idx}"

    def test_edt_cross_check_prevents_overestimate(self):
        """EDT cross-check should cap Gaussian when it overestimates.

        When the perpendicular crosses a bifurcation or the Gaussian fit
        goes wild, the EDT-based diameter provides an upper bound.
        """
        # Create a narrow vessel (6px wide) with a side branch
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[29:35, 10:120] = 255  # 6px wide horizontal vessel
        # Add a wide side branch at x=60 (creates a perpendicular artifact)
        mask[20:44, 58:62] = 255   # 24px wide vertical branch

        # Measure diameters along the main vessel centerline
        centerline = [(float(x), 32.0) for x in range(15, 115, 3)]
        diameters = measure_diameters_at_points(centerline, mask)

        # Without EDT cross-check, the Gaussian at x=60 would measure ~24px
        # (the branch width). With EDT cross-check, it should be capped
        # to a more reasonable value.
        mid = diameters[5:-5]
        assert all(d < 18 for d in mid), (
            f"EDT cross-check failed: some diameters too large (max={max(mid):.1f})"
        )

    def test_narrow_vessel_diameter_accuracy(self):
        """Diameter measurement should be reasonably accurate for narrow vessels.

        For a 4px wide vessel, the measured diameter should be within 40%
        of the true width. This tests the Gaussian+threshold fallback path.
        """
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[30:34, 10:120] = 255  # 4px wide horizontal vessel

        centerline = [(float(x), 32.0) for x in range(15, 115, 5)]
        diameters = measure_diameters_at_points(centerline, mask)

        mid = diameters[2:-2]
        avg = np.mean(mid)
        # 4px vessel: accept 2.5-6.0 range (measurement is hard for narrow vessels)
        assert 2.5 < avg < 6.0, (
            f"Narrow vessel avg diameter {avg:.1f} outside expected range"
        )
