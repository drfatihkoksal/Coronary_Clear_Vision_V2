import numpy as np
from app.core.qfr_calculator import (
    calculate_qfr,
    compute_reference_profile,
    resting_to_hyperemic_velocity,
    estimate_flow_velocity,
    smooth_diameter_profile,
    MIN_HYPEREMIC_VELOCITY,
)


def _make_straight_centerline(n_points: int, length_mm: float = 50.0):
    """Create a straight centerline along X-axis."""
    return np.column_stack([
        np.linspace(0, length_mm, n_points),
        np.zeros(n_points),
        np.zeros(n_points),
    ])


def _make_stenosis_profile(n, ref_d, mld, sten_start_frac=0.35, sten_len_frac=0.12):
    """Create a realistic smooth stenosis diameter profile."""
    d = np.full(n, ref_d)
    sten_start = int(n * sten_start_frac)
    sten_len = int(n * sten_len_frac)
    taper = int(sten_len * 0.3)
    for i in range(taper):
        frac = (i + 1) / (taper + 1)
        d[sten_start + i] = ref_d - (ref_d - mld) * frac
    d[sten_start + taper : sten_start + sten_len - taper] = mld
    for i in range(taper):
        frac = 1.0 - (i + 1) / (taper + 1)
        d[sten_start + sten_len - taper + i] = ref_d - (ref_d - mld) * frac
    return d


def test_healthy_vessel_qfr_near_one():
    """Uniform healthy vessel should have QFR close to 1.0."""
    centerline = _make_straight_centerline(50)
    diameters = np.full(50, 3.0)
    result = calculate_qfr(centerline, diameters)
    assert result["qfr"] > 0.9


def test_stenotic_vessel_lower_qfr():
    """Vessel with stenosis should have lower QFR."""
    centerline = _make_straight_centerline(50)
    diameters = np.array([3.0] * 20 + [1.0] * 10 + [3.0] * 20)
    result = calculate_qfr(centerline, diameters)
    assert result["qfr"] < 0.95
    assert result["disclaimer"] != ""


def test_empty_input():
    result = calculate_qfr(np.array([]), np.array([]))
    assert result["qfr"] == 0.0


def test_vessel_length_matches_centerline():
    """Vessel length should match the 3D arc length of the centerline."""
    centerline = _make_straight_centerline(100, length_mm=60.0)
    diameters = np.full(100, 3.0)
    result = calculate_qfr(centerline, diameters)
    assert abs(result["vessel_length_mm"] - 60.0) < 0.1


def test_modes_produce_results():
    """All three QFR modes should produce valid results."""
    centerline = _make_straight_centerline(50)
    diameters = np.array([3.0] * 20 + [1.5] * 10 + [3.0] * 20)
    for mode in ("fQFR", "cQFR", "aQFR"):
        result = calculate_qfr(centerline, diameters, mode=mode)
        assert 0 < result["qfr"] <= 1.0
        assert result["mode"] == mode
        assert result["vessel_length_mm"] > 0


def test_quadratic_velocity_conversion():
    """Quadratic formula: HFV = 0.10 + 1.55*CFV - 0.93*CFV².

    At CFV=0.15: HFV = 0.10 + 0.2325 - 0.020925 = 0.3116
    """
    v_hyp = resting_to_hyperemic_velocity(0.15)
    expected = 0.10 + 1.55 * 0.15 - 0.93 * 0.15 ** 2
    assert abs(v_hyp - expected) < 0.001


def test_quadratic_lower_than_power_law():
    """Quadratic formula should give lower hyperemic velocity than old power-law.

    Old power-law: V_hyp = 0.904 * V_rest^0.5
    New quadratic: V_hyp = 0.10 + 1.55*CFV - 0.93*CFV²

    At V_rest=0.15: old=0.350, new≈0.312
    """
    v_hyp = resting_to_hyperemic_velocity(0.15)
    old_power_law = 0.904 * 0.15 ** 0.5  # ≈ 0.350
    assert v_hyp < old_power_law


def test_cqfr_uses_timi():
    """cQFR with different TIMI frame counts should give different QFR."""
    centerline = _make_straight_centerline(50, length_mm=50.0)
    diameters = np.array([3.0] * 20 + [1.5] * 10 + [3.0] * 20)
    # Short TFC → fast flow → more pressure drop → lower QFR
    result_fast = calculate_qfr(centerline, diameters, mode="cQFR",
                                 timi_frame_count=5, frame_rate=15.0)
    # Long TFC → slow flow → less pressure drop → higher QFR
    result_slow = calculate_qfr(centerline, diameters, mode="cQFR",
                                 timi_frame_count=30, frame_rate=15.0)
    assert result_fast["qfr"] < result_slow["qfr"]


def test_fqfr_ignores_timi():
    """fQFR should give same result regardless of TIMI frame count."""
    centerline = _make_straight_centerline(50)
    diameters = np.array([3.0] * 20 + [1.5] * 10 + [3.0] * 20)
    result1 = calculate_qfr(centerline, diameters, mode="fQFR", timi_frame_count=5)
    result2 = calculate_qfr(centerline, diameters, mode="fQFR", timi_frame_count=30)
    assert result1["qfr"] == result2["qfr"]


def test_frame_rate_affects_velocity():
    """Different frame rates with same TFC should give different velocities.

    Uses TFC=5 so both velocities stay above the MIN_HYPEREMIC_VELOCITY floor.
    """
    centerline = _make_straight_centerline(50, length_mm=50.0)
    diameters = np.array([3.0] * 20 + [1.5] * 10 + [3.0] * 20)
    # Same TFC=5, but fps=15 → 0.33s transit vs fps=30 → 0.17s transit
    result_15fps = calculate_qfr(centerline, diameters, mode="cQFR",
                                  timi_frame_count=5, frame_rate=15.0)
    result_30fps = calculate_qfr(centerline, diameters, mode="cQFR",
                                  timi_frame_count=5, frame_rate=30.0)
    # 30fps → faster velocity → more pressure drop → lower QFR
    assert result_30fps["qfr"] < result_15fps["qfr"]


def test_result_contains_velocity_info():
    """Result should include velocity and TIMI debug info."""
    centerline = _make_straight_centerline(50)
    diameters = np.full(50, 3.0)
    result = calculate_qfr(centerline, diameters, mode="fQFR")
    assert "v_rest_m_s" in result
    assert "v_hyp_m_s" in result
    assert "timi_frame_count" in result
    assert "frame_rate" in result
    assert result["v_hyp_m_s"] == 0.35  # fQFR fixed


def test_multi_lesion_detection():
    """Two separate stenoses should both be detected and both lower QFR."""
    centerline = _make_straight_centerline(100, length_mm=80.0)
    # Two separate 1.0mm stenoses at positions 25 and 75
    diameters = np.array(
        [3.0] * 20 + [1.0] * 10 + [3.0] * 40 + [1.0] * 10 + [3.0] * 20
    )
    result = calculate_qfr(centerline, diameters)
    assert "num_lesions" in result
    assert result["num_lesions"] >= 2, f"Expected >=2 lesions, got {result['num_lesions']}"
    assert "lesions" in result
    assert len(result["lesions"]) >= 2


def test_multi_lesion_lower_qfr_than_single():
    """Two lesions should give lower QFR than a single lesion of same severity."""
    centerline = _make_straight_centerline(100, length_mm=80.0)
    # Single lesion
    diameters_single = np.array(
        [3.0] * 20 + [1.0] * 10 + [3.0] * 70
    )
    # Two lesions of same severity
    diameters_double = np.array(
        [3.0] * 20 + [1.0] * 10 + [3.0] * 40 + [1.0] * 10 + [3.0] * 20
    )
    result_single = calculate_qfr(centerline, diameters_single)
    result_double = calculate_qfr(centerline, diameters_double)
    assert result_double["qfr"] < result_single["qfr"], (
        f"Double lesion QFR ({result_double['qfr']}) should be lower than "
        f"single lesion QFR ({result_single['qfr']})"
    )


# =====================================================================
# New tests: velocity floor, smoothing, clinical ranges
# =====================================================================


def test_hyperemic_velocity_floor():
    """Hyperemic velocity should never fall below MIN_HYPEREMIC_VELOCITY.

    Even with very high TFC (slow contrast flow), the hyperemic velocity
    must stay above the clinical floor to avoid falsely high QFR.
    """
    # Very slow resting velocity → quadratic formula gives < 0.25
    v_hyp = resting_to_hyperemic_velocity(0.02)
    assert v_hyp >= MIN_HYPEREMIC_VELOCITY, (
        f"v_hyp={v_hyp:.4f} fell below floor {MIN_HYPEREMIC_VELOCITY}"
    )

    # Zero resting velocity
    v_hyp_zero = resting_to_hyperemic_velocity(0.0)
    assert v_hyp_zero >= MIN_HYPEREMIC_VELOCITY


def test_cqfr_velocity_floor_prevents_high_qfr():
    """With velocity floor, cQFR should not give near-normal QFR for 60% DS.

    Before the fix: cQFR with high TFC gave QFR > 0.90 for 60% DS.
    After the fix: velocity floor ensures sufficient pressure drop.
    """
    n_pts = 200
    cl = _make_straight_centerline(n_pts, 65.0)
    d = _make_stenosis_profile(n_pts, 3.0, 1.2)  # 60% DS

    # High TFC = 30 at 15fps → very slow resting velocity
    result = calculate_qfr(cl, d, mode="cQFR", timi_frame_count=30, frame_rate=15.0)

    # With floor, QFR should be noticeably below 1.0 for 60% DS
    assert result["qfr"] < 0.88, (
        f"cQFR={result['qfr']:.3f} too high for 60% DS with TFC=30. "
        f"v_hyp={result['v_hyp_m_s']:.4f}"
    )
    assert result["v_hyp_m_s"] >= MIN_HYPEREMIC_VELOCITY


def test_cqfr_severe_stenosis_below_threshold():
    """cQFR for 70% DS should be below ischemic threshold (0.80).

    Clinical validation: 70% DS typically has invasive FFR 0.50-0.70.
    """
    n_pts = 200
    cl = _make_straight_centerline(n_pts, 65.0)
    d = _make_stenosis_profile(n_pts, 3.0, 0.9)  # 70% DS

    # Even with relatively high TFC
    result = calculate_qfr(cl, d, mode="cQFR", timi_frame_count=20, frame_rate=15.0)
    assert result["qfr"] < 0.80, (
        f"cQFR={result['qfr']:.3f} should be < 0.80 for 70% DS"
    )


def test_fqfr_clinical_range_50pct():
    """50% DS with fQFR should give QFR 0.80-0.95.

    Clinical reference: 50% DS → FFR typically 0.80-0.90.
    """
    n_pts = 200
    cl = _make_straight_centerline(n_pts, 65.0)
    d = _make_stenosis_profile(n_pts, 3.0, 1.5)  # 50% DS
    result = calculate_qfr(cl, d, mode="fQFR")
    assert 0.80 <= result["qfr"] <= 0.95, (
        f"fQFR={result['qfr']:.3f} outside expected range for 50% DS"
    )


def test_fqfr_clinical_range_60pct():
    """60% DS with fQFR should give QFR 0.60-0.85.

    Clinical reference: 60% DS → FFR typically 0.65-0.80.
    """
    n_pts = 200
    cl = _make_straight_centerline(n_pts, 65.0)
    d = _make_stenosis_profile(n_pts, 3.0, 1.2)  # 60% DS
    result = calculate_qfr(cl, d, mode="fQFR")
    assert 0.60 <= result["qfr"] <= 0.85, (
        f"fQFR={result['qfr']:.3f} outside expected range for 60% DS"
    )


def test_smooth_diameter_profile_preserves_stenosis():
    """Smoothing should preserve the MLD location and approximate depth."""
    n = 200
    d = _make_stenosis_profile(n, 3.0, 1.2)
    smoothed = smooth_diameter_profile(d)

    # MLD should still be near the true minimum
    true_mld_idx = np.argmin(d)
    smoothed_mld_idx = np.argmin(smoothed)
    assert abs(true_mld_idx - smoothed_mld_idx) < 10, (
        f"MLD shifted from {true_mld_idx} to {smoothed_mld_idx}"
    )

    # MLD value should not increase by more than 15%
    true_mld = float(np.min(d))
    smoothed_mld = float(np.min(smoothed))
    assert smoothed_mld < true_mld * 1.15, (
        f"Smoothed MLD={smoothed_mld:.2f} too much higher than true={true_mld:.2f}"
    )


def test_smooth_diameter_profile_reduces_noise():
    """Smoothing should reduce noise without destroying the profile shape."""
    n = 200
    d = _make_stenosis_profile(n, 3.0, 1.2)
    # Add noise
    rng = np.random.default_rng(42)
    noisy = d + rng.normal(0, 0.15, n)
    noisy = np.clip(noisy, 0.3, 5.0)

    smoothed = smooth_diameter_profile(noisy)

    # Smoothed should be closer to the true profile than noisy is
    rmse_noisy = np.sqrt(np.mean((noisy - d) ** 2))
    rmse_smoothed = np.sqrt(np.mean((smoothed - d) ** 2))
    assert rmse_smoothed < rmse_noisy, (
        f"Smoothing made it worse: RMSE {rmse_noisy:.3f} → {rmse_smoothed:.3f}"
    )


def test_smooth_diameter_profile_short_array():
    """Smoothing should handle short arrays gracefully."""
    d_short = np.array([3.0, 2.0, 3.0])
    smoothed = smooth_diameter_profile(d_short)
    assert len(smoothed) == 3
    # Should return as-is for very short arrays
    np.testing.assert_array_almost_equal(smoothed, d_short)


# =====================================================================
# Focal stenosis tests — these catch the smoothing valley-erosion bug
# =====================================================================


def _make_focal_stenosis(n, ref_d, mld, focal_width_mm, vessel_len_mm=65.0):
    """Create a focal (short) stenosis profile."""
    d = np.full(n, ref_d)
    focal_pts = max(3, int(focal_width_mm / (vessel_len_mm / n)))
    center = n // 2
    half = focal_pts // 2
    taper = max(1, focal_pts // 4)
    for i in range(taper):
        frac = (i + 1) / (taper + 1)
        d[center - half + i] = ref_d - (ref_d - mld) * frac
        d[center + half - 1 - i] = ref_d - (ref_d - mld) * frac
    d[center - half + taper : center + half - taper] = mld
    return d


def test_smooth_preserves_focal_stenosis_mld():
    """Smoothing must not inflate focal stenosis MLD by more than 5%.

    Focal stenoses (2-3mm) are common in real angiograms. The previous
    Gaussian-only smoothing inflated their MLD by 30-55%, making severe
    stenoses appear normal. Valley-preserving smoothing must keep the
    median-filtered minimum.
    """
    n = 200
    for focal_mm in [2.0, 3.0, 5.0]:
        d = _make_focal_stenosis(n, 3.0, 0.9, focal_mm)
        orig_mld = float(np.min(d))
        smoothed = smooth_diameter_profile(d)
        smooth_mld = float(np.min(smoothed))
        inflation = (smooth_mld - orig_mld) / orig_mld
        assert inflation < 0.05, (
            f"Focal {focal_mm}mm stenosis: MLD inflated by {inflation*100:.1f}% "
            f"(orig={orig_mld:.3f} smooth={smooth_mld:.3f})"
        )


def test_focal_70pct_stenosis_detected():
    """2mm focal 70% DS must produce QFR < 0.80 (ischemic).

    This is the key regression test: with naive Gaussian smoothing,
    a 2mm focal 70% DS gave QFR=0.878 — falsely normal. With
    valley-preserving smoothing, it must be recognized as ischemic.
    """
    n = 200
    cl = _make_straight_centerline(n, 65.0)
    d = _make_focal_stenosis(n, 3.0, 0.9, focal_width_mm=2.0)
    result = calculate_qfr(cl, d, mode="fQFR")
    assert result["qfr"] < 0.80, (
        f"Focal 2mm 70% DS: QFR={result['qfr']:.3f} is falsely normal "
        f"(MLD={result['mld_mm']:.2f}mm, expected < 0.80)"
    )


def test_focal_60pct_stenosis_below_normal():
    """3mm focal 60% DS must produce QFR < 0.90.

    With naive smoothing, a 3mm focal 60% DS gave QFR=0.913 (normal).
    """
    n = 200
    cl = _make_straight_centerline(n, 65.0)
    d = _make_focal_stenosis(n, 3.0, 1.2, focal_width_mm=3.0)
    result = calculate_qfr(cl, d, mode="fQFR")
    assert result["qfr"] < 0.90, (
        f"Focal 3mm 60% DS: QFR={result['qfr']:.3f} is too high "
        f"(MLD={result['mld_mm']:.2f}mm, expected < 0.90)"
    )


def test_downstream_viscous_loss_preserved():
    """Turbulent loss must not destroy downstream viscous losses.

    With the old monotonicity-only propagation, viscous losses after
    the expansion zone were zeroed out (~1-2 mmHg lost). With explicit
    downstream propagation, the distal QFR should be slightly lower.
    """
    n = 200
    cl = _make_straight_centerline(n, 65.0)
    d = _make_stenosis_profile(n, 3.0, 1.5)  # 50% DS, moderate
    result = calculate_qfr(cl, d, mode="fQFR")
    # Pressure profile should be strictly decreasing (no flat regions)
    profile = np.array(result["pressure_profile"])
    # Check that distal 1/4 still shows viscous decline (not flat)
    distal_quarter = profile[3 * n // 4:]
    distal_drop = distal_quarter[0] - distal_quarter[-1]
    assert distal_drop > 0.001, (
        f"Distal pressure is flat (drop={distal_drop:.6f}), "
        "downstream viscous losses may be lost"
    )


# =====================================================================
# Tapering reference model tests
# =====================================================================


def _make_tapering_centerline_and_diameters(
    n_points: int,
    length_mm: float,
    prox_d: float,
    dist_d: float,
):
    """Create a straight vessel with linearly tapering healthy diameter."""
    cl = _make_straight_centerline(n_points, length_mm)
    d = np.linspace(prox_d, dist_d, n_points)
    return cl, d


def test_reference_profile_captures_taper():
    """Reference profile should reflect proximal/distal diameter difference.

    The "upper-half" sampling within each fifth shifts references slightly
    inward (proximal ~3.8, distal ~2.3 for a 4→2 taper), which is expected
    since healthy reference estimation intentionally excludes the smallest
    values in each segment to avoid stenosis contamination.
    """
    n = 200
    prox_d, dist_d = 4.0, 2.0
    d = np.linspace(prox_d, dist_d, n)
    ref_profile, mean_ref = compute_reference_profile(d)

    # Profile should taper: proximal > distal
    assert ref_profile[0] > ref_profile[-1], "Reference must taper prox > distal"
    # Proximal reference should be close to prox_d (within upper-half bias)
    assert abs(ref_profile[0] - prox_d) < 0.4, (
        f"Proximal reference {ref_profile[0]:.2f} too far from {prox_d}"
    )
    # Distal reference should be close to dist_d
    assert abs(ref_profile[-1] - dist_d) < 0.4, (
        f"Distal reference {ref_profile[-1]:.2f} too far from {dist_d}"
    )
    # Mean should be near midpoint
    assert abs(mean_ref - (prox_d + dist_d) / 2.0) < 0.3


def test_tapering_healthy_vessel_qfr_near_one():
    """A naturally tapering healthy vessel should still have QFR near 1.0.

    Before tapering fix: a 4mm→2mm vessel would have mean ref 3mm, and the
    distal 2mm segment would appear as ~33% stenosis. With tapering reference,
    no stenosis is detected and QFR stays near 1.0.
    """
    n = 200
    cl, d = _make_tapering_centerline_and_diameters(n, 65.0, 4.0, 2.5)
    result = calculate_qfr(cl, d, mode="fQFR")
    assert result["qfr"] > 0.90, (
        f"Tapering healthy vessel QFR={result['qfr']:.3f} should be > 0.90"
    )


def test_tapering_no_false_positive_distal():
    """Distal narrowing from taper should not be flagged as stenosis.

    With a flat scalar reference (mean of 4mm and 2mm = 3mm), the distal
    2mm segment would appear as 33% stenosis. With tapering reference,
    the local reference at the distal end is ~2mm, so no stenosis.
    """
    n = 200
    cl, d = _make_tapering_centerline_and_diameters(n, 65.0, 4.0, 2.0)
    result = calculate_qfr(cl, d, mode="fQFR")
    # All lesions should have low stenosis % (below 25%) — natural taper
    for lesion in result["lesions"]:
        assert lesion["stenosis_pct"] < 25.0, (
            f"False positive: distal lesion at {lesion['distance_mm']:.0f}mm "
            f"with DS={lesion['stenosis_pct']:.1f}% is likely a taper artifact"
        )


def test_tapering_proximal_stenosis_detected():
    """A stenosis in the wide proximal segment must be detected correctly.

    With a flat scalar reference of 3mm (mean of 4mm+2mm), a 3mm narrowing
    at the 4mm proximal segment would be missed (3mm == reference). With
    tapering reference, local reference at the proximal stenosis is ~4mm,
    so 3mm is a 25% DS — correctly detected.
    """
    n = 200
    cl = _make_straight_centerline(n, 65.0)
    # Tapering vessel: 4mm → 2mm, with a proximal stenosis at ~25%
    d = np.linspace(4.0, 2.0, n)
    # Add proximal stenosis: narrow to 2.5mm in the proximal region (37.5% DS)
    sten_start = int(n * 0.15)
    sten_len = int(n * 0.08)
    d[sten_start:sten_start + sten_len] = 2.5

    result = calculate_qfr(cl, d, mode="fQFR")
    # Should detect a lesion in the proximal region
    proximal_lesions = [l for l in result["lesions"] if l["distance_mm"] < 25.0]
    assert len(proximal_lesions) > 0, (
        "Proximal stenosis (4mm→2.5mm = 37.5% DS) not detected. "
        f"All lesions: {result['lesions']}"
    )
    # The detected DS% should reflect local reference (~4mm), not mean (3mm)
    for l in proximal_lesions:
        assert l["stenosis_pct"] > 25.0, (
            f"Proximal stenosis DS={l['stenosis_pct']:.1f}% too low — "
            "local reference should be ~4mm, not the 3mm mean"
        )


def test_tapering_distal_stenosis_accurate():
    """A stenosis in the narrow distal segment should use local reference.

    With a flat scalar reference of 3mm, a 1.0mm stenosis at the 2mm distal
    segment gives DS = (1-1.0/3.0)*100 = 67%. With tapering reference, local
    ref is ~2mm, giving DS = (1-1.0/2.0)*100 = 50%. Both detect the stenosis,
    but the tapering model gives the physiologically correct severity.
    """
    n = 200
    cl = _make_straight_centerline(n, 65.0)
    d = np.linspace(4.0, 2.0, n)
    # Add distal stenosis: narrow to 1.0mm
    sten_start = int(n * 0.75)
    sten_len = int(n * 0.08)
    d[sten_start:sten_start + sten_len] = 1.0

    result = calculate_qfr(cl, d, mode="fQFR")
    # Should detect a lesion in the distal region
    distal_lesions = [l for l in result["lesions"] if l["distance_mm"] > 40.0]
    assert len(distal_lesions) > 0, (
        f"Distal stenosis not detected. Lesions: {result['lesions']}"
    )
    # DS% should be ~50% (relative to local 2mm ref), not ~67% (relative to 3mm mean)
    for l in distal_lesions:
        assert 35.0 < l["stenosis_pct"] < 65.0, (
            f"Distal DS={l['stenosis_pct']:.1f}% — expected ~50% "
            "relative to local ~2mm reference"
        )
