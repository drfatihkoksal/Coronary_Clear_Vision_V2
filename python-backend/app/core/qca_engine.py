"""QCA Engine: Quantitative Coronary Angiography measurements.

Computes vessel diameter profile along centerline using perpendicular
cross-section sampling with sub-pixel accuracy.

Methods:
  - gaussian: Gaussian FWHM fitting on intensity/probability profile
  - parabolic: Parabolic fitting at half-height
  - threshold: Simple threshold crossing (fastest)

Sub-pixel accuracy via bilinear interpolation of probability/mask map.
If no probability map is provided, the binary mask is used as float32
and bilinear interpolation produces smooth edge transitions.

Ported from reference project (coronary_rws_analyser v1.1).
"""

import logging
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto point count
# ---------------------------------------------------------------------------

def _calculate_optimal_points(
    centerline: np.ndarray,
    pixel_spacing_mm: float,
    target_resolution_mm: float = 0.3,
    min_points: int = 20,
    max_points: int = 500,
) -> int:
    """Calculate optimal measurement point count from vessel length and pixel spacing.

    Uses ~0.3 mm spacing between measurement points (standard QCA resolution).
    """
    diffs = np.diff(centerline, axis=0)
    vessel_length_px = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
    vessel_length_mm = vessel_length_px * pixel_spacing_mm

    if vessel_length_mm <= 0 or target_resolution_mm <= 0:
        return min_points

    n = int(round(vessel_length_mm / target_resolution_mm))
    return max(min_points, min(n, max_points))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def compute_qca_measurements(
    mask: np.ndarray,
    centerline: list[tuple[float, float]],
    pixel_spacing_mm: float,
    method: Literal["gaussian", "parabolic", "threshold"] = "gaussian",
    num_points: int | None = None,
    probability_map: np.ndarray | None = None,
) -> dict:
    """Compute QCA measurements along a vessel centerline.

    Args:
        mask: Binary mask (HxW uint8, 0 or 255)
        centerline: Ordered (x, y) points along vessel center
        pixel_spacing_mm: mm per pixel
        method: Diameter fitting method
        num_points: Number of measurement points. None = auto-calculate.
        probability_map: Optional soft probability map for sub-pixel accuracy.
                         If None, mask is converted to float32 (bilinear
                         interpolation on edges provides smooth transitions).

    Returns:
        Dict with diameter profiles, MLD, reference diameters, stenosis, etc.
    """
    if len(centerline) < 2 or pixel_spacing_mm <= 0:
        return _empty_result(num_points or 0)

    pts = np.array(centerline)  # (N, 2) as (x, y)

    # Auto-calculate optimal point count if not specified
    if num_points is None:
        num_points = _calculate_optimal_points(pts, pixel_spacing_mm)

    # Resample centerline to num_points with equal arc-length spacing
    pts = _resample_points(pts, num_points)

    # Prepare probability map for sub-pixel sampling
    if probability_map is not None:
        prob = probability_map.astype(np.float32)
    else:
        prob = mask.astype(np.float32)

    # Compute tangent directions
    tangents = _compute_tangents(pts)

    # Measure diameter at each point using perpendicular cross-section
    diameters_px = np.array([
        max(_measure_diameter(pts[i], tangents[i], prob, method), 0.5)
        for i in range(len(pts))
    ])
    diameters_mm = diameters_px * pixel_spacing_mm

    # Compute cumulative distances along centerline
    diffs = np.diff(pts, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    distances_px = np.concatenate([[0], np.cumsum(segment_lengths)])
    distances_mm = distances_px * pixel_spacing_mm

    # Reference diameters: max diameter in first/last N/5
    ref_window = max(1, len(diameters_mm) // 5)

    prox_region = diameters_mm[:ref_window]
    prox_idx = int(np.argmax(prox_region))
    prox_ref_mm = float(prox_region[prox_idx])

    dist_region = diameters_mm[-ref_window:]
    dist_rel_idx = int(np.argmax(dist_region))
    dist_idx = len(diameters_mm) - ref_window + dist_rel_idx
    dist_ref_mm = float(dist_region[dist_rel_idx])

    # MLD: minimum between proximal and distal reference (constrained)
    search_start = prox_idx + 1
    search_end = dist_idx
    if search_start >= search_end:
        search_start = ref_window
        search_end = len(diameters_mm) - ref_window

    if search_start < search_end:
        search_region = diameters_mm[search_start:search_end]
        mld_idx = search_start + int(np.argmin(search_region))
    else:
        mld_idx = int(np.argmin(diameters_mm))
        mld_idx = max(prox_idx + 1, min(mld_idx, dist_idx - 1))
        mld_idx = max(0, min(mld_idx, len(diameters_mm) - 1))

    mld_mm = float(diameters_mm[mld_idx])
    mld_px = float(diameters_px[mld_idx])

    # Interpolated reference diameter at MLD position
    if dist_idx != prox_idx:
        t = (mld_idx - prox_idx) / (dist_idx - prox_idx)
        interp_ref = prox_ref_mm + t * (dist_ref_mm - prox_ref_mm)
    else:
        interp_ref = (prox_ref_mm + dist_ref_mm) / 2

    # Diameter stenosis (using interpolated reference)
    ds_pct = ((1 - mld_mm / interp_ref) * 100) if interp_ref > 0 else 0.0
    ds_pct = max(0.0, min(100.0, ds_pct))

    # Lesion length (multi-method)
    lesion_length_mm = _compute_lesion_length(
        pts, diameters_mm, interp_ref, pixel_spacing_mm,
        mld_idx=mld_idx, prox_idx=prox_idx, dist_idx=dist_idx,
    )

    return {
        "centerline": [{"x": float(p[0]), "y": float(p[1])} for p in pts],
        "diameter_profile_mm": diameters_mm.tolist(),
        "diameter_profile_px": diameters_px.tolist(),
        "distances_mm": distances_mm.tolist(),
        "mld_mm": round(mld_mm, 3),
        "mld_px": round(mld_px, 3),
        "mld_index": int(mld_idx),
        "diameter_stenosis_pct": round(ds_pct, 1),
        "proximal_ref_mm": round(prox_ref_mm, 3),
        "distal_ref_mm": round(dist_ref_mm, 3),
        "proximal_ref_index": int(prox_idx),
        "distal_ref_index": int(dist_idx),
        "interpolated_ref_mm": round(interp_ref, 3),
        "lesion_length_mm": round(lesion_length_mm, 3) if lesion_length_mm else None,
        "pixel_spacing_mm": pixel_spacing_mm,
        "num_points": num_points,
        "method": method,
        "vessel_length_mm": round(float(distances_mm[-1]), 3),
    }


# ---------------------------------------------------------------------------
# Diameter measurement (sub-pixel)
# ---------------------------------------------------------------------------

def _measure_diameter(
    point: np.ndarray,
    tangent: np.ndarray,
    prob_map: np.ndarray,
    method: str,
    max_radius: int = 50,
) -> float:
    """Measure vessel diameter at a point using perpendicular cross-section.

    Samples the probability/mask map along the perpendicular direction
    using bilinear interpolation for sub-pixel accuracy, then fits
    a model to estimate the vessel width.
    """
    # Perpendicular (normal) direction — tangent is (dx, dy), normal is (-dy, dx)
    normal = np.array([-tangent[1], tangent[0]])
    cx, cy = point  # centerline is (x, y)

    # Sample along perpendicular in both directions
    t_values = np.linspace(-max_radius, max_radius, 2 * max_radius + 1)
    sample_x = cx + t_values * normal[0]
    sample_y = cy + t_values * normal[1]

    # Clip to image bounds
    h, w = prob_map.shape
    valid = (
        (sample_y >= 0) & (sample_y < h - 1) &
        (sample_x >= 0) & (sample_x < w - 1)
    )
    if not valid.any():
        return 0.0

    # Bilinear interpolation for sub-pixel sampling
    profile = _bilinear_sample(prob_map, sample_y[valid], sample_x[valid])
    t_valid = t_values[valid]

    if len(profile) < 5:
        return _threshold_diameter(t_valid, profile)

    if method == "gaussian":
        return _gaussian_diameter(t_valid, profile)
    elif method == "parabolic":
        return _parabolic_diameter(t_valid, profile)
    else:
        return _threshold_diameter(t_valid, profile)


def _bilinear_sample(
    image: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation for sub-pixel sampling."""
    y0 = np.floor(y_coords).astype(int)
    x0 = np.floor(x_coords).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    h, w = image.shape
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)

    fy = y_coords - np.floor(y_coords)
    fx = x_coords - np.floor(x_coords)

    return (
        (1 - fx) * (1 - fy) * image[y0, x0] +
        (1 - fx) * fy * image[y1, x0] +
        fx * (1 - fy) * image[y0, x1] +
        fx * fy * image[y1, x1]
    )


def _gaussian_diameter(t: np.ndarray, profile: np.ndarray) -> float:
    """Fit Gaussian to profile → FWHM = 2.355 * sigma."""
    A_init = float(profile.max() - profile.min())
    mu_init = float(t[np.argmax(profile)])
    sigma_init = 5.0
    offset_init = float(profile.min())

    def gaussian(x, A, mu, sigma, offset):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

    try:
        popt, _ = curve_fit(
            gaussian, t, profile,
            p0=[A_init, mu_init, sigma_init, offset_init],
            maxfev=1000,
        )
        diameter = 2.355 * abs(popt[2])  # FWHM
        if diameter > len(t) or diameter < 1:
            return _threshold_diameter(t, profile)
        return diameter
    except (RuntimeError, ValueError):
        return _threshold_diameter(t, profile)


def _parabolic_diameter(t: np.ndarray, profile: np.ndarray) -> float:
    """Fit parabola to profile → diameter at half-height."""
    peak_idx = np.argmax(profile)
    peak_t = float(t[peak_idx])
    peak_val = float(profile[peak_idx])

    def parabola(x, a, mu, peak):
        return a * (x - mu) ** 2 + peak

    try:
        popt, _ = curve_fit(
            parabola, t, profile,
            p0=[-0.01, peak_t, peak_val],
            maxfev=1000,
        )
        a, mu, peak = popt
        if a >= 0:
            return _threshold_diameter(t, profile)
        threshold_val = 0.5 * peak
        delta_sq = (threshold_val - peak) / a
        if delta_sq < 0:
            return _threshold_diameter(t, profile)
        diameter = 2 * float(np.sqrt(delta_sq))
        if diameter > len(t) or diameter < 1:
            return _threshold_diameter(t, profile)
        return diameter
    except (RuntimeError, ValueError):
        return _threshold_diameter(t, profile)


def _threshold_diameter(t: np.ndarray, profile: np.ndarray, threshold: float = 0.5) -> float:
    """Simple threshold crossing: diameter = distance between first/last crossing."""
    if len(profile) == 0:
        return 0.0
    max_val = float(profile.max())
    threshold_val = threshold * max_val
    above = profile >= threshold_val
    if not above.any():
        return 0.0
    indices = np.where(above)[0]
    return abs(float(t[indices[-1]] - t[indices[0]]))


# ---------------------------------------------------------------------------
# Tangent, resampling, helpers
# ---------------------------------------------------------------------------

def _compute_tangents(pts: np.ndarray) -> np.ndarray:
    """Compute unit tangent vectors at each centerline point."""
    tangents = np.zeros_like(pts)
    for i in range(len(pts)):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == len(pts) - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        length = np.linalg.norm(t)
        tangents[i] = t / length if length > 1e-6 else np.array([1.0, 0.0])
    return tangents


def _resample_points(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample point array to n evenly-spaced points using arc-length."""
    if len(pts) <= 1 or n <= 1:
        return pts[:n] if len(pts) >= n else pts

    diffs = np.diff(pts, axis=0)
    lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumlen = np.concatenate([[0], np.cumsum(lengths)])
    total = cumlen[-1]
    if total == 0:
        return pts[:n] if len(pts) >= n else pts

    targets = np.linspace(0, total, n)
    x = np.interp(targets, cumlen, pts[:, 0])
    y = np.interp(targets, cumlen, pts[:, 1])
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Lesion length (multi-method, from reference)
# ---------------------------------------------------------------------------

def _compute_lesion_length(
    centerline: np.ndarray,
    diameters_mm: np.ndarray,
    interp_ref_mm: float,
    pixel_spacing_mm: float,
    mld_idx: int,
    prox_idx: int,
    dist_idx: int,
    threshold: float = 0.85,
) -> float | None:
    """Compute lesion length using multi-method approach.

    1. Primary: region where diameter < 85% of interpolated reference
    2. Fallback: half-height method around MLD
    3. Last resort: distance between prox and distal reference
    """
    def _arc_length(start: int, end: int) -> float:
        seg = centerline[start:end + 1]
        d = np.diff(seg, axis=0)
        return float(np.sum(np.sqrt(np.sum(d ** 2, axis=1)))) * pixel_spacing_mm

    # Method 1: threshold-based
    threshold_diam = threshold * interp_ref_mm
    below = diameters_mm < threshold_diam
    if below.any():
        indices = np.where(below)[0]
        if indices[-1] > indices[0]:
            return _arc_length(indices[0], indices[-1])

    # Method 2: half-height around MLD
    mld_diam = diameters_mm[mld_idx]
    half_height = (mld_diam + interp_ref_mm) / 2
    below_half = diameters_mm < half_height
    if below_half.any():
        indices = np.where(below_half)[0]
        if indices[-1] > indices[0]:
            return _arc_length(indices[0], indices[-1])

    # Method 3: prox-to-dist distance
    if prox_idx != dist_idx:
        s, e = min(prox_idx, dist_idx), max(prox_idx, dist_idx)
        return _arc_length(s, e)

    return None


# ---------------------------------------------------------------------------
# Empty result
# ---------------------------------------------------------------------------

def _empty_result(num_points: int) -> dict:
    return {
        "centerline": [],
        "diameter_profile_mm": [],
        "diameter_profile_px": [],
        "distances_mm": [],
        "mld_mm": 0.0,
        "mld_px": 0.0,
        "mld_index": 0,
        "diameter_stenosis_pct": 0.0,
        "proximal_ref_mm": 0.0,
        "distal_ref_mm": 0.0,
        "proximal_ref_index": 0,
        "distal_ref_index": 0,
        "interpolated_ref_mm": 0.0,
        "lesion_length_mm": None,
        "pixel_spacing_mm": 0.0,
        "num_points": num_points,
        "method": "threshold",
        "vessel_length_mm": 0.0,
    }
