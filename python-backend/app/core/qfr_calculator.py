"""QFR (Quantitative Flow Ratio) Calculator — Gould/Young-Tsai Model.

Implements the Gould/Young-Tsai pressure drop model with support for
fQFR, cQFR, and aQFR modes.

Pressure drop model (segmented Gould formula):
    ΔP = ΔP_viscous + ΔP_turbulent
    ΔP_viscous = (8μL / πr⁴) · Q           (Poiseuille friction, per segment)
    ΔP_turbulent = Kt · (ρ/2) · (1 - A_min/A_downstream)² · V_min²
                   (Borda-Carnot expansion loss at stenosis exit only)

Flow velocity modes (Tu et al., FAVOR Pilot 2016):
    fQFR: Fixed hyperemic velocity (0.35 m/s)
    cQFR: Contrast-flow TIMI → resting velocity → hyperemic via quadratic model
           HFV = a₀ + a₁·CFV + a₂·CFV²  (Tu et al., FAVOR I)
    aQFR: Adenosine-flow TIMI → direct hyperemic velocity

DISCLAIMER: Research implementation. Not validated for clinical use.

References:
- Tu et al., "Fractional Flow Reserve Calculation from 3-D QCA", JACC CI 2016
- Gould KL, "Pressure-flow characteristics of coronary stenoses", AJC 1978
- Young & Tsai, "Flow Characteristics in Models of Arterial Stenoses", JBiomech 1973
- Gibson et al., "TIMI Frame Count", Circulation 1996
"""
import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

logger = logging.getLogger(__name__)

# ===========================================================================
# Physical constants
# ===========================================================================
BLOOD_DENSITY = 1050.0       # kg/m³
BLOOD_VISCOSITY = 0.0035     # Pa·s (3.5 cP)
AORTIC_PRESSURE = 100.0      # mmHg (assumed mean aortic pressure)
PA_TO_MMHG = 133.322         # 1 mmHg = 133.322 Pa

# Gould/Young-Tsai coefficient
DEFAULT_KT = 1.52

# Flow velocity constants
FIXED_HYPEREMIC_VELOCITY = 0.35  # m/s (fQFR default)
FIXED_RESTING_VELOCITY = 0.14    # m/s (resting fallback)
DEFAULT_RESTING_VELOCITY = 0.17  # m/s (Gibson et al. typical FCV)
MAX_RESTING_VELOCITY = 0.30      # m/s
MIN_HYPEREMIC_VELOCITY = 0.25    # m/s (clinical floor — prevents underestimation)
MAX_HYPEREMIC_VELOCITY = 0.60    # m/s


# ===========================================================================
# Flow velocity estimation
# ===========================================================================

def estimate_resting_velocity(
    vessel_length_mm: float,
    tfc: float,
    frame_rate: float = 15.0,
) -> float:
    """Estimate resting contrast flow velocity from TIMI frame count.

    FCV = vessel_length / (TFC / frame_rate)
    """
    if tfc <= 0:
        return FIXED_RESTING_VELOCITY
    time_seconds = tfc / frame_rate
    return (vessel_length_mm / 1000.0) / time_seconds


def resting_to_hyperemic_velocity(v_rest: float) -> float:
    """Quadratic conversion from resting to hyperemic flow velocity.

    HFV = a₀ + a₁·CFV + a₂·CFV²  (Tu et al., FAVOR I study)

    The negative a₂ coefficient provides natural damping at high velocities,
    reflecting coronary autoregulation. The quadratic model is the validated
    formula from the original FAVOR pilot study.

    Coefficients: a₀=0.10, a₁=1.55, a₂=-0.93

    A clinical floor of MIN_HYPEREMIC_VELOCITY (0.25 m/s) is enforced to
    prevent unrealistically low hyperemic estimates when resting velocity
    is very low (e.g., high TFC at low frame rates). Without this floor,
    the turbulent pressure drop (∝ V²) becomes negligible, yielding
    falsely high QFR values for significant stenoses.
    """
    if v_rest <= 0:
        return FIXED_HYPEREMIC_VELOCITY

    # Tu et al. FAVOR I validated quadratic coefficients
    A0 = 0.10
    A1 = 1.55
    A2 = -0.93

    v_hyperemic = A0 + A1 * v_rest + A2 * v_rest ** 2
    return float(np.clip(v_hyperemic, MIN_HYPEREMIC_VELOCITY, MAX_HYPEREMIC_VELOCITY))


def estimate_flow_velocity(
    vessel_length_mm: float,
    tfc: float,
    frame_rate: float = 15.0,
    mode: str = "cQFR",
) -> tuple[float, float]:
    """Estimate (resting, hyperemic) flow velocity based on QFR mode.

    Returns:
        (v_rest, v_hyp) in m/s
    """
    if mode == "fQFR":
        return FIXED_RESTING_VELOCITY, FIXED_HYPEREMIC_VELOCITY

    elif mode == "cQFR":
        v_rest = estimate_resting_velocity(vessel_length_mm, tfc, frame_rate)
        v_rest = min(v_rest, MAX_RESTING_VELOCITY)
        v_hyp = resting_to_hyperemic_velocity(v_rest)
        v_hyp = min(v_hyp, MAX_HYPEREMIC_VELOCITY)
        return v_rest, v_hyp

    elif mode == "aQFR":
        v_hyp = estimate_resting_velocity(vessel_length_mm, tfc, frame_rate)
        v_hyp = min(v_hyp, MAX_HYPEREMIC_VELOCITY)
        v_rest = v_hyp / 2.5
        return v_rest, v_hyp

    return FIXED_RESTING_VELOCITY, FIXED_HYPEREMIC_VELOCITY


def estimate_default_tfc(vessel_length_mm: float, frame_rate: float = 15.0) -> float:
    """Estimate default TIMI frame count when T0/T1 are not set."""
    transit_time = (vessel_length_mm / 1000.0) / DEFAULT_RESTING_VELOCITY
    return max(transit_time * frame_rate, 3.0)


# ===========================================================================
# Diameter profile conditioning
# ===========================================================================

def smooth_diameter_profile(diameters_mm: np.ndarray) -> np.ndarray:
    """Valley-preserving smoothing of the diameter profile.

    Strategy:
    1. Median filter — removes impulse noise, preserves edges and valleys
    2. Gaussian smoothing — removes residual jitter from healthy segments
    3. Element-wise min(gaussian, median) — ensures the Gaussian never
       raises a valley above its median-filtered value

    The min-merge is critical because the QFR pressure drop scales as r⁴
    (viscous) and V²∝1/d⁴ (turbulent). Even a small MLD overestimate from
    Gaussian blurring of a focal stenosis (2-3mm) dramatically reduces the
    computed pressure drop. For a 6-point focal stenosis at 70% DS, naive
    Gaussian smoothing inflates MLD by ~55%, cutting turbulent loss by >80%.
    """
    n = len(diameters_mm)
    if n < 5:
        return diameters_mm

    # Step 1: Median filter — removes impulse noise while preserving edges
    # Kernel size scales with profile length but stays small (3-7)
    med_size = min(7, max(3, n // 40))
    if med_size % 2 == 0:
        med_size += 1
    d_median = median_filter(diameters_mm, size=med_size)

    # Step 2: Light Gaussian — removes residual jitter in healthy segments
    sigma = max(1.0, n / 100.0)
    d_smooth = gaussian_filter1d(d_median, sigma=sigma)

    # Step 3: Valley-preserving merge.
    # Take the minimum of Gaussian-smoothed and median-filtered values,
    # then ALSO cap at the original input. This is critical because the
    # median filter itself erases focal stenoses narrower than ~kernel/2
    # points, and the subsequent min(gaussian, median) can't recover them.
    # By also comparing against the original, we ensure that no valley
    # is ever raised above its raw measurement — even a single-point
    # focal stenosis is preserved. This is safe because downward noise
    # (apparent narrowing) is preferable to upward bias in QFR calculation
    # (Poiseuille r⁴ sensitivity means underestimating diameter is less
    # dangerous than overestimating it).
    smoothed = np.minimum(d_smooth, d_median)
    smoothed = np.minimum(smoothed, diameters_mm)

    # Preserve exact endpoint values (proximal/distal reference)
    smoothed[0] = diameters_mm[0]
    smoothed[-1] = diameters_mm[-1]

    return smoothed


# ===========================================================================
# Reference diameter estimation
# ===========================================================================

def compute_reference_profile(diameters_mm: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute tapering reference diameter profile from healthy segments.

    Coronary arteries naturally taper from proximal to distal (e.g. LAD:
    ~4mm proximal → ~2mm distal). A single scalar reference causes:
    - Proximal: stenosis underestimated (narrow reference vs wide vessel)
    - Distal: false positives (wide reference vs naturally narrow vessel)

    This function models the healthy vessel as a linear taper between
    proximal and distal healthy reference diameters, estimated from the
    upper-half diameters in each fifth of the vessel.

    Returns:
        (reference_profile_mm, mean_reference_mm)
        - reference_profile_mm: N-length array of expected healthy diameters
        - mean_reference_mm: scalar mean for volumetric flow computation
    """
    n = len(diameters_mm)
    n_fifth = max(1, n // 5)

    proximal = np.sort(diameters_mm[:n_fifth])
    proximal_rd = float(np.mean(proximal[-max(1, len(proximal) // 2):]))

    distal = np.sort(diameters_mm[-n_fifth:])
    distal_rd = float(np.mean(distal[-max(1, len(distal) // 2):]))

    reference_profile = np.linspace(proximal_rd, distal_rd, n)
    mean_reference = (proximal_rd + distal_rd) / 2.0

    return reference_profile, mean_reference


# ===========================================================================
# Pressure drop calculation (Gould/Young-Tsai)
# ===========================================================================

def detect_stenoses(
    diameters_mm: np.ndarray,
    reference_profile: np.ndarray,
    min_stenosis_pct: float = 20.0,
    min_separation: int = 10,
) -> list[int]:
    """Detect significant stenoses as local minima in the diameter profile.

    Args:
        diameters_mm: N-length diameter profile
        reference_profile: N-length tapering reference (healthy) diameters
        min_stenosis_pct: Minimum % diameter stenosis to qualify
        min_separation: Minimum index separation between distinct lesions

    Returns:
        List of indices where stenoses are detected (sorted by severity).
    """
    n = len(diameters_mm)
    if n < 3:
        return [int(np.argmin(diameters_mm))]

    # Smooth to avoid noise-induced minima
    sigma = max(2, n // 50)
    smoothed = gaussian_filter1d(diameters_mm.astype(np.float64), sigma=sigma)

    # Per-position threshold based on tapering reference
    threshold = reference_profile * (1.0 - min_stenosis_pct / 100.0)

    # Find all local minima below threshold
    candidates = []
    for i in range(1, n - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            if smoothed[i] < threshold[i]:
                candidates.append((i, float(smoothed[i])))

    if not candidates:
        # No significant stenosis found — use global MLD
        return [int(np.argmin(diameters_mm))]

    # Sort by severity (smallest diameter first)
    candidates.sort(key=lambda x: x[1])

    # Merge nearby stenoses (keep only the most severe within min_separation)
    selected: list[int] = []
    for idx, _ in candidates:
        if all(abs(idx - s) >= min_separation for s in selected):
            selected.append(idx)

    # Refine: snap to the actual minimum (unsmoothed) near each detected point
    refined = []
    for idx in selected:
        lo = max(0, idx - sigma)
        hi = min(n, idx + sigma + 1)
        local_min_idx = lo + int(np.argmin(diameters_mm[lo:hi]))
        refined.append(local_min_idx)

    return sorted(set(refined))


def calculate_pressure_drop(
    centerline_3d: np.ndarray,
    diameters_mm: np.ndarray,
    flow_velocity_m_s: float,
    kt: float = DEFAULT_KT,
    reference_profile: np.ndarray | None = None,
    mean_reference_mm: float | None = None,
    stenosis_indices_override: list[int] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate QFR from 3D vessel geometry using Gould/Young-Tsai model.

    Two components:
    1. Viscous losses (Poiseuille friction): per-segment along the vessel
    2. Turbulent/expansion losses (Borda-Carnot): at EACH detected stenosis

    Multiple lesions are supported — Borda-Carnot expansion loss is applied
    independently at every significant stenosis, not just the single MLD.

    Args:
        centerline_3d: Nx3 array of 3D points (mm)
        diameters_mm: N-length array of diameters (mm)
        flow_velocity_m_s: Hyperemic flow velocity (m/s)
        kt: Turbulent loss coefficient
        reference_profile: N-length tapering reference. If None, computed.
        mean_reference_mm: Scalar mean reference for Q. If None, computed.

    Returns:
        (qfr_value, qfr_profile, pressure_profile_mmhg)
    """
    n_points = len(centerline_3d)

    if reference_profile is None or mean_reference_mm is None:
        reference_profile, mean_reference_mm = compute_reference_profile(diameters_mm)

    # Convert mean reference to SI for volumetric flow (Q is conserved)
    d_ref_m = mean_reference_mm / 1000.0
    A_ref = np.pi * (d_ref_m / 2) ** 2
    Q = flow_velocity_m_s * A_ref  # volumetric flow rate (m³/s)

    # Segment lengths (mm)
    segment_lengths = np.sqrt(np.sum(np.diff(centerline_3d, axis=0) ** 2, axis=1))

    # Diameters to SI
    d_m = diameters_mm / 1000.0
    r_m = d_m / 2.0

    # ----- 1. Viscous losses (Poiseuille, per-segment) -----
    pressure_profile_pa = np.zeros(n_points)
    pressure_profile_pa[0] = AORTIC_PRESSURE * PA_TO_MMHG

    for i in range(n_points - 1):
        L = segment_lengths[i] / 1000.0  # mm → m
        r_avg = (r_m[i] + r_m[i + 1]) / 2.0
        if r_avg > 1e-10 and L > 0:
            dp_visc = (8.0 * BLOOD_VISCOSITY * L * Q) / (np.pi * r_avg ** 4)
        else:
            dp_visc = 0.0
        pressure_profile_pa[i + 1] = pressure_profile_pa[i] - dp_visc

    # ----- 2. Turbulent/expansion losses (Borda-Carnot at EACH stenosis) -----
    if stenosis_indices_override is not None:
        stenosis_indices = stenosis_indices_override
    else:
        stenosis_indices = detect_stenoses(diameters_mm, reference_profile)

    for sten_idx in stenosis_indices:
        mld = float(diameters_mm[sten_idx])

        # Downstream recovery: max diameter after this stenosis,
        # but before the next stenosis (or within 1/3 of vessel)
        next_sten = min(
            (s for s in stenosis_indices if s > sten_idx),
            default=n_points,
        )
        search_end = min(n_points, sten_idx + max(n_points // 3, 5), next_sten)
        if search_end <= sten_idx:
            search_end = min(n_points, sten_idx + 5)

        downstream_d = float(np.max(diameters_mm[sten_idx:search_end]))
        # Cap at local tapering reference (not a global scalar)
        local_ref = float(reference_profile[min(search_end - 1, n_points - 1)])
        downstream_d = min(downstream_d, local_ref)

        A_mld = np.pi * (mld / 2000.0) ** 2
        A_downstream = np.pi * (downstream_d / 2000.0) ** 2

        if A_mld > 1e-10 and A_downstream > A_mld:
            V_mld = Q / A_mld
            dp_turb = kt * 0.5 * BLOOD_DENSITY * (1.0 - A_mld / A_downstream) ** 2 * V_mld ** 2

            # Distribute turbulent loss: smooth ramp in expansion zone,
            # then full loss carried to all downstream points.
            # Previous bug: only the ramp was applied, so downstream
            # viscous losses were lost by the monotonicity enforcement.
            expansion_end = min(search_end, n_points)
            for i in range(sten_idx, expansion_end):
                frac = (i - sten_idx) / max(1, expansion_end - sten_idx - 1)
                pressure_profile_pa[i] -= dp_turb * frac
            # Full turbulent loss to all points beyond expansion zone
            if expansion_end < n_points:
                pressure_profile_pa[expansion_end:] -= dp_turb

    # Safety: ensure monotonically decreasing pressure
    for i in range(1, n_points):
        pressure_profile_pa[i] = min(pressure_profile_pa[i], pressure_profile_pa[i - 1])

    # Convert to mmHg
    pressure_profile_mmhg = pressure_profile_pa / PA_TO_MMHG
    qfr_profile = np.clip(pressure_profile_mmhg / AORTIC_PRESSURE, 0.0, 1.0)
    qfr_value = float(qfr_profile[-1])

    return qfr_value, qfr_profile, pressure_profile_mmhg


# ===========================================================================
# Main pipeline
# ===========================================================================

def calculate_qfr(
    centerline_3d: np.ndarray,
    diameters_mm: np.ndarray,
    kt: float = DEFAULT_KT,
    mode: str = "fQFR",
    timi_frame_count: float | None = None,
    frame_rate: float = 15.0,
    vessel_length_override_mm: float | None = None,
    stenosis_indices_override: list[int] | None = None,
) -> dict:
    """Full QFR calculation pipeline.

    Args:
        centerline_3d: Nx3 array of 3D centerline points (mm)
        diameters_mm: N-length array of diameters (mm)
        kt: Turbulent loss coefficient
        mode: QFR mode — "fQFR", "cQFR", or "aQFR"
        timi_frame_count: For cQFR/aQFR modes (averaged from both projections)
        frame_rate: Cine frame rate (fps)
        vessel_length_override_mm: If set, overrides the 3D-computed vessel length.
            Used when 2D projections give more reliable length estimates.

    Returns:
        Dict with qfr, pressure_profile, vessel_length_mm, etc.
    """
    if len(centerline_3d) < 2 or len(diameters_mm) < 2:
        return _empty_result(mode)

    centerline_3d = np.asarray(centerline_3d, dtype=np.float64)
    diameters_mm = np.asarray(diameters_mm, dtype=np.float64)

    # Step 0: Smooth diameter profile to reduce measurement noise.
    # Raw perpendicular Gaussian/EDT measurements have per-point jitter that
    # can shift the apparent MLD by 20-30%, which (due to r⁴ sensitivity)
    # changes the pressure drop dramatically.
    diameters_mm = smooth_diameter_profile(diameters_mm)

    # Step 1: Compute vessel length
    raw_3d_length = float(np.sum(
        np.sqrt(np.sum(np.diff(centerline_3d, axis=0) ** 2, axis=1))
    ))
    vessel_length_mm = vessel_length_override_mm if vessel_length_override_mm else raw_3d_length

    # Step 2: If vessel length is overridden (2D estimate is more accurate than
    # noisy 3D depth), scale the centerline so all segment lengths are correct.
    # This ensures the per-segment pressure drop calculation uses corrected lengths.
    if vessel_length_override_mm and raw_3d_length > 1e-6:
        scale = vessel_length_mm / raw_3d_length
        # Scale positions relative to first point
        origin = centerline_3d[0].copy()
        centerline_3d = origin + (centerline_3d - origin) * scale

    diffs = np.diff(centerline_3d, axis=0)
    seg_lens = np.sqrt(np.sum(diffs ** 2, axis=1))
    distances_mm = np.concatenate([[0.0], np.cumsum(seg_lens)])

    # Step 3: Determine TIMI frame count
    if timi_frame_count is not None and timi_frame_count > 0:
        tfc = timi_frame_count
    else:
        tfc = estimate_default_tfc(vessel_length_mm, frame_rate)

    # Step 4: Estimate flow velocity
    v_rest, v_hyp = estimate_flow_velocity(vessel_length_mm, tfc, frame_rate, mode)

    logger.info(
        "QFR[%s] velocity: TFC=%.1f fps=%.0f vessel=%.1fmm "
        "-> v_rest=%.4f v_hyp=%.4f m/s",
        mode, tfc, frame_rate, vessel_length_mm, v_rest, v_hyp,
    )

    # Step 5: Tapering reference profile
    reference_profile, mean_reference_mm = compute_reference_profile(diameters_mm)

    # Step 5b: Detect stenoses for reporting (use override if provided)
    if stenosis_indices_override is not None:
        stenosis_indices = stenosis_indices_override
        mld_mm = float(np.min([diameters_mm[i] for i in stenosis_indices_override]))
    else:
        stenosis_indices = detect_stenoses(diameters_mm, reference_profile)
        mld_mm = float(np.min(diameters_mm))

    # DS% using local tapering reference at MLD position
    mld_idx = int(np.argmin(diameters_mm))
    local_ref_at_mld = float(reference_profile[mld_idx])
    ds_pct = (1.0 - mld_mm / local_ref_at_mld) * 100.0 if local_ref_at_mld > 0 else 0.0
    logger.info(
        "QFR[%s] geometry: ref_mean=%.2fmm ref_local=%.2fmm MLD=%.2fmm DS=%.1f%% "
        "n_lesions=%d stenosis_idx=%s",
        mode, mean_reference_mm, local_ref_at_mld, mld_mm, ds_pct,
        len(stenosis_indices), stenosis_indices,
    )

    # Step 6: Pressure drop + QFR
    qfr_value, qfr_profile, pressure_profile_mmhg = calculate_pressure_drop(
        centerline_3d, diameters_mm, v_hyp, kt,
        reference_profile=reference_profile,
        mean_reference_mm=mean_reference_mm,
        stenosis_indices_override=stenosis_indices_override,
    )

    # Compute viscous-only QFR for diagnostic breakdown
    qfr_visc_only, _, _ = calculate_pressure_drop(
        centerline_3d, diameters_mm, v_hyp, kt,
        reference_profile=reference_profile,
        mean_reference_mm=mean_reference_mm,
        stenosis_indices_override=[],  # no turbulent loss
    )
    dp_total = AORTIC_PRESSURE * (1.0 - qfr_value)
    dp_viscous = AORTIC_PRESSURE * (1.0 - qfr_visc_only)
    dp_turbulent = dp_total - dp_viscous
    logger.info(
        "QFR[%s] result: QFR=%.3f dP_total=%.1fmmHg "
        "(viscous=%.1f + turbulent=%.1f)",
        mode, qfr_value, dp_total, dp_viscous, dp_turbulent,
    )

    # Flow rate for reporting
    A_ref = np.pi * (mean_reference_mm / 2000.0) ** 2
    flow_rate_ml_s = v_hyp * A_ref * 1e6  # m³/s → ml/s

    # Build per-lesion info with local tapering reference
    lesions = []
    for idx in stenosis_indices:
        d = float(diameters_mm[idx])
        local_ref = float(reference_profile[idx])
        ds_local = (1.0 - d / local_ref) * 100.0 if local_ref > 0 else 0.0
        dist = float(distances_mm[idx]) if idx < len(distances_mm) else 0.0
        lesions.append({
            "index": idx,
            "diameter_mm": round(d, 3),
            "stenosis_pct": round(ds_local, 1),
            "distance_mm": round(dist, 1),
        })

    return {
        "qfr": round(qfr_value, 3),
        "mode": mode,
        "pressure_profile": qfr_profile.tolist(),
        "diameters_mm": diameters_mm.tolist(),
        "distances_mm": distances_mm.tolist(),
        "reference_diameter_mm": round(mean_reference_mm, 3),
        "mld_mm": round(mld_mm, 3),
        "kt": kt,
        "flow_rate_ml_s": round(flow_rate_ml_s, 3),
        "vessel_length_mm": round(vessel_length_mm, 3),
        "v_rest_m_s": round(v_rest, 4),
        "v_hyp_m_s": round(v_hyp, 4),
        "timi_frame_count": round(tfc, 1),
        "frame_rate": frame_rate,
        "num_lesions": len(lesions),
        "lesions": lesions,
        "disclaimer": "Research implementation. Not validated for clinical decision-making.",
    }


def _empty_result(mode: str = "fQFR") -> dict:
    return {
        "qfr": 0.0,
        "mode": mode,
        "pressure_profile": [],
        "diameters_mm": [],
        "distances_mm": [],
        "reference_diameter_mm": 0.0,
        "mld_mm": 0.0,
        "kt": DEFAULT_KT,
        "flow_rate_ml_s": 0.0,
        "vessel_length_mm": 0.0,
        "disclaimer": "Research implementation. Not validated for clinical decision-making.",
    }
