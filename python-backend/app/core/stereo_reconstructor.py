"""Stereo Reconstruction from dual angiographic projections.

Uses epipolar geometry with proper C-arm camera model to reconstruct
3D vessel centerline from two calibrated X-ray projections.

Key improvements over naive resampling:
- Proper C-arm geometry with primary (RAO/LAO) and secondary (CRA/CAU) angles
- Epipolar correspondence matching with direction detection
- Longest Increasing Subsequence filtering for monotonic ordering
- Coupled arc-length resampling that preserves correspondence
"""
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionParams:
    """Camera projection parameters extracted from DICOM metadata."""
    primary_angle: float    # RAO (-) / LAO (+) degrees
    secondary_angle: float  # Cranial (+) / Caudal (-) degrees
    sid: float              # Source-to-Image Distance (mm)
    sod: float              # Source-to-Object Distance (mm)
    pixel_spacing: float    # isocenter pixel spacing (mm)
    image_size: tuple[int, int]  # (width, height)


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of 3D stereo reconstruction."""
    points_3d: list[tuple[float, float, float]]
    t1_fractions: np.ndarray  # arc-length fractions on original centerline 1
    t2_fractions: np.ndarray  # arc-length fractions on original centerline 2


def reconstruct_3d_centerline(
    centerline_1: list[tuple[float, float]],
    centerline_2: list[tuple[float, float]],
    projection_params_1: ProjectionParams,
    projection_params_2: ProjectionParams,
) -> ReconstructionResult:
    """Reconstruct 3D centerline from two 2D projections using epipolar matching.

    Args:
        centerline_1: (x, y) points from projection 1
        centerline_2: (x, y) points from projection 2
        projection_params_1: Camera params for projection 1
        projection_params_2: Camera params for projection 2

    Returns:
        ReconstructionResult with 3D points and arc-length fractions for
        each projection (used to align diameter profiles).
    """
    if len(centerline_1) < 2 or len(centerline_2) < 2:
        return ReconstructionResult([], np.array([]), np.array([]))

    cl1 = np.array(centerline_1, dtype=np.float64)
    cl2 = np.array(centerline_2, dtype=np.float64)

    # Build camera matrices
    K1, R1, t1, P1 = build_camera_matrix(projection_params_1)
    K2, R2, t2, P2 = build_camera_matrix(projection_params_2)

    # Compute fundamental matrix for epipolar matching
    F = compute_fundamental_matrix(K1, R1, t1, K2, R2, t2)

    # Find correspondences via epipolar geometry
    matched1, matched2, t1_frac, t2_frac = find_epipolar_correspondences(cl1, cl2, F)

    # Triangulate each pair of corresponding points
    raw_3d = np.zeros((len(matched1), 3))
    for i in range(len(matched1)):
        raw_3d[i] = _triangulate_point(matched1[i], matched2[i], P1, P2)

    if len(raw_3d) < 5:
        points_3d = [(float(p[0]), float(p[1]), float(p[2])) for p in raw_3d]
        return ReconstructionResult(points_3d, t1_frac, t2_frac)

    # Filter outliers by reprojection error
    raw_3d, t1_frac, t2_frac = _filter_by_reprojection(
        raw_3d, matched1, matched2, P1, P2, t1_frac, t2_frac,
        max_reproj_px=5.0,
    )

    # Smooth 3D centerline to reduce depth noise from parallax errors
    raw_3d = _smooth_3d_centerline(raw_3d, window=11)

    points_3d = [(float(p[0]), float(p[1]), float(p[2])) for p in raw_3d]
    return ReconstructionResult(points_3d, t1_frac, t2_frac)


def compute_angular_separation(angle_1: float, angle_2: float) -> float:
    """Compute angular separation between two projection angles.

    For accurate reconstruction, separation should be 25-40 degrees.
    """
    diff = abs(angle_1 - angle_2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def build_camera_matrix(
    params: ProjectionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build camera projection matrix from C-arm geometry parameters.

    C-arm geometry:
    - Primary angle (RAO/LAO): rotation around patient's AP axis (Y)
    - Secondary angle (CRA/CAU): rotation around patient's LR axis (X)
    - R = R_gantry^T (camera rotates WITH the gantry)

    Returns:
        (K, R, t, P) - intrinsic, rotation, translation, projection matrix
    """
    w, h = params.image_size
    cx = w / 2.0
    cy = h / 2.0

    # Focal length in pixels: fx = SOD / isocenter_pixel_spacing
    fx = params.sod / params.pixel_spacing
    fy = fx

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])

    # Gantry rotations
    alpha = np.radians(params.primary_angle)   # RAO/LAO around Y
    beta = np.radians(params.secondary_angle)  # CRA/CAU around X

    Ry = np.array([
        [np.cos(alpha), 0, np.sin(alpha)],
        [0, 1, 0],
        [-np.sin(alpha), 0, np.cos(alpha)],
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(beta), -np.sin(beta)],
        [0, np.sin(beta),  np.cos(beta)],
    ])

    # R_gantry rotates C-arm from AP; world-to-camera is R_gantry^T
    R_gantry = Rx @ Ry
    R = R_gantry.T

    # Camera starts at (0, 0, -SOD) in AP, then rotates by gantry
    initial_pos = np.array([[0], [0], [-params.sod]])
    camera_pos_world = R_gantry @ initial_pos
    t = -R @ camera_pos_world

    Rt = np.hstack([R, t])
    P = K @ Rt

    return K, R, t, P


def compute_fundamental_matrix(
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    K2: np.ndarray, R2: np.ndarray, t2: np.ndarray,
) -> np.ndarray:
    """Compute fundamental matrix F such that x2^T F x1 = 0."""
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    R12 = R2 @ R1.T
    t12 = t2 - R12 @ t1
    t12_flat = t12.flatten()

    # Skew-symmetric matrix of t12
    tx = np.array([
        [0, -t12_flat[2], t12_flat[1]],
        [t12_flat[2], 0, -t12_flat[0]],
        [-t12_flat[1], t12_flat[0], 0],
    ])

    E = tx @ R12
    F = K2_inv.T @ E @ K1_inv
    return F


def find_epipolar_correspondences(
    centerline1: np.ndarray,
    centerline2: np.ndarray,
    F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find point correspondences using dense epipolar matching.

    Strategy:
    1. Detect direction (forward vs reversed) using epipolar lines
    2. For each c1 point, find closest c2 point by epipolar distance
    3. Extract longest monotonically increasing subsequence (LIS)
    4. Coupled resampling using shared arc-length parameterization

    Returns:
        (matched_pts1, matched_pts2, t1_fractions, t2_fractions)
        where t1/t2_fractions are normalized arc-length positions [0..1]
        on the original centerlines, used for aligned diameter interpolation.
    """
    n1 = len(centerline1)
    n2 = len(centerline2)

    # Compute arc-length parameterization for both original centerlines
    arc_c1 = _arc_length_fractions(centerline1)
    arc_c2 = _arc_length_fractions(centerline2)

    # Step 1: Check direction
    score_fwd = _direction_score(centerline1, centerline2, F)
    score_rev = _direction_score(centerline1, centerline2[::-1], F)

    if score_rev > score_fwd:
        centerline2 = centerline2[::-1].copy()
        arc_c2 = 1.0 - arc_c2[::-1]  # reverse arc-length fractions

    # Step 2: Vectorized epipolar distance matrix
    c1_hom = np.column_stack([centerline1, np.ones(n1)])  # n1x3
    c2_hom = np.column_stack([centerline2, np.ones(n2)])  # n2x3

    # Epipolar lines in view2 for each c1 point: l2 = F @ x1
    l2_all = (F @ c1_hom.T).T  # (n1, 3)
    line_norms = np.sqrt(l2_all[:, 0]**2 + l2_all[:, 1]**2)
    line_norms = np.maximum(line_norms, 1e-10)

    # Distance matrix: epipolar_dists[i, j] = dist from c2[j] to epipolar line of c1[i]
    signed_dists = c2_hom @ l2_all.T  # (n2, n1)
    epipolar_dists = np.abs(signed_dists).T / line_norms[:, np.newaxis]  # (n1, n2)

    # Step 3: Arc-length regularized matching
    # Pure epipolar argmin can grossly mismatch proximal↔distal when the
    # epipolar line is nearly tangent to the vessel. Adding an arc-length
    # consistency cost prevents c1[30%] from matching c2[70%].
    arc_diff = np.abs(arc_c1[:, np.newaxis] - arc_c2[np.newaxis, :])  # (n1, n2)
    ARC_WEIGHT = 100.0  # pixels per unit arc-length difference
    cost = epipolar_dists + ARC_WEIGHT * arc_diff

    raw_matches = np.argmin(cost, axis=1)  # (n1,)
    raw_min_dists = epipolar_dists[np.arange(n1), raw_matches]

    # Step 3b: Bidirectional consistency — keep only mutual best matches
    # Forward: c1→c2 already computed. Backward: c2→c1.
    bwd_matches = np.argmin(cost, axis=0)  # (n2,) — for each c2, best c1
    mutual_mask = np.array([bwd_matches[raw_matches[i]] == i for i in range(n1)])

    # Step 4: Filter by max epipolar distance AND mutual consistency
    MAX_EPIPOLAR_DIST = 50.0  # pixels (tighter than before)
    valid_mask = (raw_min_dists < MAX_EPIPOLAR_DIST) & mutual_mask
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 5:
        # Fallback to uniform resampling
        n_pts = min(n1, n2)
        t_uniform = np.linspace(0, 1, n_pts)
        return _resample(centerline1, n_pts), _resample(centerline2, n_pts), t_uniform, t_uniform

    valid_c2_matches = raw_matches[valid_indices]

    # Step 5: Longest increasing subsequence (O(n log n))
    lis_pos = _longest_increasing_subsequence(valid_c2_matches)

    mono_c1_idx = valid_indices[lis_pos]
    mono_c2_idx = valid_c2_matches[lis_pos]

    if len(mono_c1_idx) < 5:
        n_pts = min(n1, n2)
        t_uniform = np.linspace(0, 1, n_pts)
        return _resample(centerline1, n_pts), _resample(centerline2, n_pts), t_uniform, t_uniform

    # Step 6: Coupled resampling using c1's arc length as shared parameter
    matched1_raw = centerline1[mono_c1_idx]
    matched2_raw = centerline2[mono_c2_idx]

    # Arc-length fractions of matched points on original centerlines
    t1_matched = arc_c1[mono_c1_idx]
    t2_matched = arc_c2[mono_c2_idx]

    diffs1 = np.diff(matched1_raw, axis=0)
    arc1 = np.concatenate([[0], np.cumsum(np.sqrt(np.sum(diffs1**2, axis=1)))])
    total_arc = arc1[-1]

    if total_arc < 1e-6:
        n_pts = min(n1, n2)
        t_uniform = np.linspace(0, 1, n_pts)
        return _resample(centerline1, n_pts), _resample(centerline2, n_pts), t_uniform, t_uniform

    arc1_norm = arc1 / total_arc

    # Output: adaptive sampling based on arc length in pixels
    # Aim for ~1.0 pixel spacing for maximum fidelity to the segmentation mask
    n_output = int(np.ceil(total_arc))
    # Safety: ensure at least 50 points (unless extremely short) to avoid degeneracies
    n_output = max(n_output, min(50, n1, n2))
    t_uniform = np.linspace(0, 1, n_output)

    # Resample BOTH curves using c1's arc-length (preserves correspondence)
    out1_x = np.interp(t_uniform, arc1_norm, matched1_raw[:, 0])
    out1_y = np.interp(t_uniform, arc1_norm, matched1_raw[:, 1])
    out2_x = np.interp(t_uniform, arc1_norm, matched2_raw[:, 0])
    out2_y = np.interp(t_uniform, arc1_norm, matched2_raw[:, 1])

    # Resample arc-length fractions using same shared parameter
    t1_out = np.interp(t_uniform, arc1_norm, t1_matched)
    t2_out = np.interp(t_uniform, arc1_norm, t2_matched)

    return (
        np.column_stack([out1_x, out1_y]),
        np.column_stack([out2_x, out2_y]),
        t1_out,
        t2_out,
    )


def _direction_score(
    c1: np.ndarray, c2: np.ndarray, F: np.ndarray,
) -> float:
    """Score how well c2's direction matches c1 using epipolar geometry."""
    n1, n2 = len(c1), len(c2)
    c2_hom = np.column_stack([c2, np.ones(n2)])

    sample_indices = np.linspace(0, n1 - 1, min(10, n1), dtype=int)
    match_positions = []

    for i in sample_indices:
        x1_hom = np.array([c1[i, 0], c1[i, 1], 1.0])
        l2 = F @ x1_hom
        line_norm = np.sqrt(l2[0]**2 + l2[1]**2)
        if line_norm < 1e-10:
            continue
        dists = np.abs(c2_hom @ l2) / line_norm
        match_positions.append(np.argmin(dists))

    if len(match_positions) < 2:
        return 0.0

    return sum(1 for a, b in zip(match_positions, match_positions[1:]) if b > a)


def _longest_increasing_subsequence(arr: np.ndarray) -> np.ndarray:
    """Find indices of the longest strictly increasing subsequence.
    O(n log n) patience sorting algorithm.
    """
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int)

    tails = []
    tail_idx = []
    pred = [-1] * n

    for i in range(n):
        val = arr[i]
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < val:
                lo = mid + 1
            else:
                hi = mid

        if lo == len(tails):
            tails.append(val)
            tail_idx.append(i)
        else:
            tails[lo] = val
            tail_idx[lo] = i

        pred[i] = tail_idx[lo - 1] if lo > 0 else -1

    lis_len = len(tails)
    result = [0] * lis_len
    k = tail_idx[-1]
    for j in range(lis_len - 1, -1, -1):
        result[j] = k
        k = pred[k]

    return np.array(result)


def _triangulate_point(
    pt1: np.ndarray, pt2: np.ndarray,
    P1: np.ndarray, P2: np.ndarray,
) -> np.ndarray:
    """Triangulate a 3D point from two 2D observations (DLT method)."""
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]

    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]

    return X


def _arc_length_fractions(pts: np.ndarray) -> np.ndarray:
    """Compute normalized arc-length fractions [0..1] for each point."""
    diffs = np.diff(pts, axis=0)
    lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumlen = np.concatenate([[0], np.cumsum(lengths)])
    total = cumlen[-1]
    if total < 1e-10:
        return np.linspace(0, 1, len(pts))
    return cumlen / total


def _filter_by_reprojection(
    pts_3d: np.ndarray,
    pts_2d_1: np.ndarray,
    pts_2d_2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    t1_frac: np.ndarray,
    t2_frac: np.ndarray,
    max_reproj_px: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove 3D points with high reprojection error."""
    n = len(pts_3d)
    hom = np.column_stack([pts_3d, np.ones(n)])  # Nx4

    proj1 = (P1 @ hom.T).T  # Nx3
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    proj2 = (P2 @ hom.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]

    err1 = np.sqrt(np.sum((proj1 - pts_2d_1) ** 2, axis=1))
    err2 = np.sqrt(np.sum((proj2 - pts_2d_2) ** 2, axis=1))
    max_err = np.maximum(err1, err2)

    keep = max_err < max_reproj_px
    if keep.sum() < 5:
        return pts_3d, t1_frac, t2_frac

    return pts_3d[keep], t1_frac[keep], t2_frac[keep]


def _smooth_3d_centerline(pts: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth 3D centerline to reduce depth noise from stereo parallax.

    With typical angiographic angular separations (25-40°), depth (Z) is
    poorly constrained: ~1mm error per pixel of matching inaccuracy.
    This causes the 3D path length to be inflated by noisy Z zigzag.

    Strategy: XY light moving average (preserves vessel curvature from 2D),
    Z aggressive moving average (removes parallax noise).
    """
    n = len(pts)
    if n < 5:
        return pts

    smoothed = pts.copy()

    # XY: light smoothing (preserves vessel curvature from 2D views)
    xy_window = min(window, n)
    if xy_window >= 3:
        half = xy_window // 2
        for dim in range(2):
            padded = np.pad(pts[:, dim], half, mode='edge')
            kernel = np.ones(xy_window) / xy_window
            smoothed[:, dim] = np.convolve(padded, kernel, mode='valid')

    # Z: very aggressive smoothing (depth is poorly constrained)
    z_window = min(n - 1 if n % 2 == 0 else n, max(31, n // 3))
    if z_window % 2 == 0:
        z_window -= 1
    z_window = max(3, z_window)
    half_z = z_window // 2
    padded_z = np.pad(pts[:, 2], half_z, mode='edge')
    kernel_z = np.ones(z_window) / z_window
    smoothed[:, 2] = np.convolve(padded_z, kernel_z, mode='valid')

    # Preserve exact endpoints
    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]

    return smoothed


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample 2D point array to n evenly-spaced points."""
    if len(pts) == n:
        return pts
    diffs = np.diff(pts, axis=0)
    lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumlen = np.concatenate([[0], np.cumsum(lengths)])
    total = cumlen[-1]
    if total == 0:
        return pts[:n] if len(pts) >= n else np.tile(pts[0], (n, 1))
    targets = np.linspace(0, total, n)
    x = np.interp(targets, cumlen, pts[:, 0])
    y = np.interp(targets, cumlen, pts[:, 1])
    return np.column_stack([x, y])
