"""RWS (Radial Wall Strain) Calculator.

RWS measures the change in vessel wall diameter between diastole and systole.
Formula: RWS% = ((D_diastole - D_systole) / D_diastole) * 100

Interpretation:
  - Normal: RWS < 8%
  - Intermediate: 8-12%
  - Vulnerable: 12-14%
  - High Risk: > 14%
"""
import numpy as np
from app.core.outlier_filter import hampel_filter, double_hampel_filter
from app.models.enums import RWSInterpretation, OutlierMethod


def calculate_rws(
    diameter_profiles: dict[int, list[float]],
    start_frame: int,
    end_frame: int,
    outlier_method: OutlierMethod = OutlierMethod.HAMPEL,
    vessel: str | None = None,
) -> dict:
    """Calculate RWS for a range of frames.

    Args:
        diameter_profiles: Map of frame_index -> diameter_profile_mm (list of diameters at each centerline point)
        start_frame: Start frame (diastole)
        end_frame: End frame (systole)
        outlier_method: Outlier filtering method
        vessel: Vessel name (optional)

    Returns:
        RWS result dict
    """
    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be < end_frame ({end_frame})")

    # Collect diameter profiles for the frame range
    frame_indices = sorted(k for k in diameter_profiles.keys() if start_frame <= k <= end_frame)

    if len(frame_indices) < 2:
        raise ValueError(f"Need at least 2 frames with QCA data in range [{start_frame}, {end_frame}]")

    # Build diameter matrix: (num_frames, num_points)
    num_points = len(diameter_profiles[frame_indices[0]])
    diameter_matrix = np.zeros((len(frame_indices), num_points))

    for i, fi in enumerate(frame_indices):
        profile = diameter_profiles[fi]
        if len(profile) != num_points:
            # Resample to match
            profile = np.interp(
                np.linspace(0, 1, num_points),
                np.linspace(0, 1, len(profile)),
                profile,
            ).tolist()
        diameter_matrix[i] = profile

    # Apply outlier filtering per measurement point
    filtered_matrix = diameter_matrix.copy()
    total_outliers = 0

    for j in range(num_points):
        time_series = diameter_matrix[:, j].tolist()

        if outlier_method == OutlierMethod.HAMPEL:
            filtered, outlier_idx = hampel_filter(time_series)
        elif outlier_method == OutlierMethod.DOUBLE_HAMPEL:
            filtered, outlier_idx = double_hampel_filter(time_series)
        else:
            filtered = time_series
            outlier_idx = []

        filtered_matrix[:, j] = filtered
        total_outliers += len(outlier_idx)

    # Compute RWS at each centerline point
    # Max diameter (diastole) and min diameter (systole)
    d_max = np.max(filtered_matrix, axis=0)  # Per point
    d_min = np.min(filtered_matrix, axis=0)  # Per point

    # RWS% = ((D_max - D_min) / D_max) * 100
    with np.errstate(divide='ignore', invalid='ignore'):
        rws_per_point = np.where(d_max > 0, (d_max - d_min) / d_max * 100, 0.0)

    # Apply physiological bounds
    rws_per_point = np.clip(rws_per_point, 0.0, 50.0)

    # Compute summary statistics
    # MLD point is where average diameter is smallest
    avg_diameters = np.mean(filtered_matrix, axis=0)
    mld_idx = int(np.argmin(avg_diameters))
    mld_rws = float(rws_per_point[mld_idx])

    # Proximal (first 1/3) and distal (last 1/3)
    third = max(1, num_points // 3)
    prox_rws = float(np.mean(rws_per_point[:third]))
    dist_rws = float(np.mean(rws_per_point[-third:]))
    avg_rws = float(np.mean(rws_per_point))

    interpretation = _interpret_rws(mld_rws)

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "num_frames_used": len(frame_indices),
        "mld_rws_pct": round(mld_rws, 2),
        "proximal_rws_pct": round(prox_rws, 2),
        "distal_rws_pct": round(dist_rws, 2),
        "average_rws_pct": round(avg_rws, 2),
        "interpretation": interpretation.value,
        "outlier_method": outlier_method.value,
        "outliers_removed": total_outliers,
        "vessel": vessel,
        "rws_profile": rws_per_point.tolist(),
        "frame_indices": frame_indices,
    }


def _interpret_rws(rws_pct: float) -> RWSInterpretation:
    """Interpret RWS percentage value."""
    if rws_pct < 8.0:
        return RWSInterpretation.NORMAL
    elif rws_pct < 12.0:
        return RWSInterpretation.INTERMEDIATE
    elif rws_pct < 14.0:
        return RWSInterpretation.VULNERABLE
    else:
        return RWSInterpretation.HIGH_RISK
