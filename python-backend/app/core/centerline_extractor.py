"""
Vessel Centerline Extraction

Extracts sub-pixel accurate centerlines from segmentation masks.
Critical for QCA diameter profiling - centerline defines measurement axis.

Methods:
1. Skeleton-based (fast ~50ms): Morphological skeletonization + ordering
2. Distance Transform (sub-pixel): EDT ridge detection
3. Minimum Cost Path (guided): Dijkstra through seed points
4. MCP-Auto: Preprocessed mask + EDT endpoints + MCP (recommended, default)

Ported from reference project (coronary_rws_analyser v1.1).
"""

import logging
from typing import Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
from skimage.graph import route_through_array

logger = logging.getLogger(__name__)


class CenterlineExtractor:
    """Extract vessel centerline from segmentation mask.

    Centerline extraction is essential for:
    - QCA diameter measurement perpendicular to vessel axis
    - Seed point generation for tracking propagation
    - Vessel length calculation

    All internal coordinates are (y, x) = (row, col).
    Public API functions at module level convert to (x, y) for the frontend.
    """

    def __init__(self):
        self.last_centerline: np.ndarray | None = None
        self.last_diameter_map: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        mask: np.ndarray,
        method: str = "skeleton",
        seed_points: list[tuple[float, float]] | None = None,
        probability_map: np.ndarray | None = None,
        num_points: int | None = None,
    ) -> np.ndarray:
        """Extract centerline from segmentation mask.

        Args:
            mask: Binary segmentation mask (any dtype, non-zero = vessel)
            method: "skeleton" (uses mcp_auto), "distance", "mcp", "mcp_auto",
                    "skeleton_legacy"
            seed_points: Optional (x, y) points for guided extraction
            probability_map: Optional probability map for better accuracy
            num_points: Number of points to sample (for "distance" method)

        Returns:
            Centerline points as (N, 2) array of (y, x) coordinates.
        """
        if method == "skeleton":
            return self.extract_mcp_auto(mask, seed_points, probability_map)
        elif method == "distance":
            points, _ = self.extract_distance_transform(mask, num_points or 50)
            return points
        elif method == "mcp":
            if seed_points is None or len(seed_points) < 2:
                raise ValueError("MCP method requires at least 2 seed points")
            return self.extract_minimum_cost_path(mask, seed_points)
        elif method == "mcp_auto":
            return self.extract_mcp_auto(mask, seed_points, probability_map)
        elif method == "skeleton_legacy":
            return self.extract_skeleton_based(mask, seed_points, probability_map)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Use 'skeleton', 'distance', 'mcp', or 'mcp_auto'"
            )

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def extract_skeleton_based(
        self,
        mask: np.ndarray,
        seed_points: list[tuple[float, float]] | None = None,
        probability_map: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract centerline using morphological skeletonization.

        Fast method (~50ms) suitable for real-time tracking.
        """
        binary_mask = (mask > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            return np.array([])

        if probability_map is not None:
            high_prob = (probability_map > 0.3).astype(np.uint8)
            filtered = binary_mask * high_prob
            if filtered.sum() > 0:
                binary_mask = filtered

        skeleton = skeletonize(binary_mask).astype(np.uint8)
        if skeleton.sum() == 0:
            return np.array([])

        endpoints = self._find_skeleton_endpoints(skeleton)

        if len(endpoints) < 2 and seed_points and len(seed_points) >= 2:
            skel_coords = np.column_stack(np.where(skeleton > 0))
            start_idx = self._find_nearest_point(
                skel_coords, (seed_points[0][1], seed_points[0][0])
            )
            end_idx = self._find_nearest_point(
                skel_coords, (seed_points[-1][1], seed_points[-1][0])
            )
            endpoints = [tuple(skel_coords[start_idx]), tuple(skel_coords[end_idx])]

        if len(endpoints) < 2:
            skel_coords = np.column_stack(np.where(skeleton > 0))
            if len(skel_coords) > 0:
                endpoints = [tuple(skel_coords[0]), tuple(skel_coords[-1])]
            else:
                return np.array([])

        ordered = self._order_skeleton_points(skeleton, endpoints[0])
        self.last_centerline = ordered
        return ordered

    def extract_distance_transform(
        self, mask: np.ndarray, num_points: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract centerline using distance transform for sub-pixel accuracy.

        Returns:
            (centerline_points (N,2) (y,x), diameter_values)
        """
        binary_mask = (mask > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            return np.array([]), np.array([])

        edt = ndimage.distance_transform_edt(binary_mask)
        self.last_diameter_map = edt * 2

        skeleton = skeletonize(binary_mask).astype(np.uint8)
        if skeleton.sum() == 0:
            return np.array([]), np.array([])

        endpoints = self._find_skeleton_endpoints(skeleton)
        if len(endpoints) < 2:
            skel_coords = np.column_stack(np.where(skeleton > 0))
            endpoints = [tuple(skel_coords[0]), tuple(skel_coords[-1])]

        ordered = self._order_skeleton_points(skeleton, endpoints[0])

        if len(ordered) > num_points:
            indices = np.linspace(0, len(ordered) - 1, num_points, dtype=int)
            sampled = ordered[indices]
        else:
            sampled = ordered

        diameters = np.array([
            edt[int(y), int(x)] * 2 for y, x in sampled
        ])

        self.last_centerline = sampled
        return sampled, diameters

    def extract_minimum_cost_path(
        self,
        mask: np.ndarray,
        seed_points: list[tuple[float, float]],
        smooth_sigma: float = 1.0,
    ) -> np.ndarray:
        """Extract centerline using minimum cost path through seed points.

        Uses Dijkstra with cost = 1 / distance_transform.
        """
        if len(seed_points) < 2:
            raise ValueError("At least 2 seed points required")

        binary_mask = (mask > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            return np.array([])

        edt = ndimage.distance_transform_edt(binary_mask)
        cost_map = 1.0 / (edt + 0.1)

        seed_coords = [(int(y), int(x)) for x, y in seed_points]

        full_path: list = []
        for i in range(len(seed_coords) - 1):
            start, end = seed_coords[i], seed_coords[i + 1]
            try:
                indices, _ = route_through_array(
                    cost_map, start, end, fully_connected=True
                )
                if i == 0:
                    full_path.extend(indices)
                else:
                    full_path.extend(indices[1:])
            except Exception as e:
                logger.warning("Path finding failed: %s, using linear interpolation", e)
                segment = self._linear_interpolate(start, end)
                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])

        path_array = np.array(full_path)

        if smooth_sigma > 0 and len(path_array) > 3:
            path_array = self._smooth_path(path_array, smooth_sigma)

        self.last_centerline = path_array
        return path_array

    def extract_mcp_auto(
        self,
        mask: np.ndarray,
        seed_points: list[tuple[float, float]] | None = None,
        probability_map: np.ndarray | None = None,
        smooth_sigma: float = 1.0,
    ) -> np.ndarray:
        """Extract centerline using preprocessed mask + EDT endpoints + MCP.

        Recommended method — no skeletonization, robust to noise.

        Pipeline:
        1. Preprocess mask: keep largest component, remove branches
        2. Find endpoints using EDT heatmap (two furthest high-EDT points)
        3. Use seed points if provided, otherwise use EDT endpoints
        4. Extract centerline using Minimum Cost Path
        """
        binary_mask = (mask > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            return np.array([])

        if probability_map is not None:
            high_prob = (probability_map > 0.3).astype(np.uint8)
            filtered = binary_mask * high_prob
            if filtered.sum() > 0:
                binary_mask = filtered

        preprocessed = self._preprocess_mask(binary_mask)
        if preprocessed.sum() == 0:
            preprocessed = binary_mask

        if seed_points and len(seed_points) >= 2:
            start_yx = (int(seed_points[0][1]), int(seed_points[0][0]))
            end_yx = (int(seed_points[-1][1]), int(seed_points[-1][0]))
        else:
            start_yx, end_yx = self._find_endpoints_edt(preprocessed)

        if start_yx is None or end_yx is None:
            return np.array([])

        edt = ndimage.distance_transform_edt(preprocessed)
        self.last_diameter_map = edt * 2
        cost_map = 1.0 / (edt + 0.1)

        try:
            path_indices, _ = route_through_array(
                cost_map, start_yx, end_yx, fully_connected=True
            )
            path_array = np.array(path_indices)
        except Exception as e:
            logger.warning("MCP path finding failed: %s, linear fallback", e)
            path_array = np.array(self._linear_interpolate(start_yx, end_yx))

        if smooth_sigma > 0 and len(path_array) > 3:
            path_array = self._smooth_path(path_array, smooth_sigma)

        self.last_centerline = path_array
        return path_array

    # ------------------------------------------------------------------
    # Mask preprocessing
    # ------------------------------------------------------------------

    def _preprocess_mask(
        self,
        mask: np.ndarray,
        prune_length: int = 10,
    ) -> np.ndarray:
        """Remove islands and small branches from mask.

        1. Keep largest connected component
        2. Morphological opening / closing
        3. Branch pruning via EDT thresholding
        """
        if mask.sum() == 0:
            return mask

        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask

        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        if len(sizes) == 0:
            return mask

        largest_idx = int(np.argmax(sizes)) + 1
        clean = (labeled == largest_idx).astype(np.uint8)

        if CV2_AVAILABLE:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        else:
            struct = ndimage.generate_binary_structure(2, 1)
            clean = ndimage.binary_opening(clean, struct).astype(np.uint8)
            clean = ndimage.binary_closing(clean, struct).astype(np.uint8)

        if prune_length > 0 and clean.sum() > 100:
            temp_skel = skeletonize(clean).astype(np.uint8)
            if temp_skel.sum() > 0:
                branch_pts = self._find_branch_points(temp_skel)
                if len(branch_pts) > 0:
                    edt = ndimage.distance_transform_edt(clean)
                    thick_thr = (
                        np.percentile(edt[edt > 0], 30) if edt.max() > 0 else 0
                    )
                    main = (edt >= thick_thr).astype(np.uint8)
                    if CV2_AVAILABLE:
                        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        main = cv2.dilate(main, k5, iterations=2)
                    else:
                        main = ndimage.binary_dilation(main, iterations=2).astype(
                            np.uint8
                        )
                    clean = (clean & main).astype(np.uint8)

        return clean

    # ------------------------------------------------------------------
    # Endpoint & branch detection
    # ------------------------------------------------------------------

    def _find_endpoints_edt(
        self, mask: np.ndarray, percentile: float = 70
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """Find two vessel endpoints using EDT heatmap (furthest-apart high-EDT points)."""
        if mask.sum() == 0:
            return None, None

        edt = ndimage.distance_transform_edt(mask)
        if edt.max() == 0:
            return None, None

        threshold = np.percentile(edt[edt > 0], percentile)
        high_coords = np.column_stack(np.where(edt >= threshold))

        if len(high_coords) < 2:
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) < 2:
                return None, None
            high_coords = coords

        if len(high_coords) > 100:
            idx = np.linspace(0, len(high_coords) - 1, 100, dtype=int)
            sample = high_coords[idx]
        else:
            sample = high_coords

        max_dist = 0
        best_start: tuple[int, int] | None = None
        best_end: tuple[int, int] | None = None
        for i, p1 in enumerate(sample):
            for p2 in sample[i + 1 :]:
                d = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                if d > max_dist:
                    max_dist = d
                    best_start = tuple(p1)
                    best_end = tuple(p2)

        return best_start, best_end

    def _find_skeleton_endpoints(
        self, skeleton: np.ndarray
    ) -> list[tuple[int, int]]:
        """Find pixels with exactly 1 neighbor in skeleton."""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(
            skeleton.astype(int), kernel, mode="constant"
        )
        coords = np.column_stack(np.where((skeleton > 0) & (neighbor_count == 1)))
        return [tuple(c) for c in coords]

    def _find_branch_points(
        self, skeleton: np.ndarray
    ) -> list[tuple[int, int]]:
        """Find pixels with 3+ neighbors (branch junctions)."""
        if skeleton.sum() == 0:
            return []
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        nc = ndimage.convolve(skeleton.astype(int), kernel, mode="constant")
        coords = np.column_stack(np.where((skeleton > 0) & (nc >= 3)))
        return [tuple(c) for c in coords]

    # ------------------------------------------------------------------
    # Point ordering & interpolation
    # ------------------------------------------------------------------

    def _order_skeleton_points(
        self, skeleton: np.ndarray, start: tuple[int, int]
    ) -> np.ndarray:
        """Nearest-neighbor traversal from *start* through all skeleton pixels."""
        skel_coords = np.column_stack(np.where(skeleton > 0))
        skel_list = [tuple(c) for c in skel_coords]
        if not skel_list:
            return np.array([])

        ordered = [start]
        remaining = [p for p in skel_list if p != start]
        current = start

        while remaining:
            dists = [abs(p[0] - current[0]) + abs(p[1] - current[1]) for p in remaining]
            ni = int(np.argmin(dists))
            nearest = remaining[ni]
            ordered.append(nearest)
            current = nearest
            remaining.pop(ni)

        return np.array(ordered)

    @staticmethod
    def _find_nearest_point(points: np.ndarray, target: tuple[float, float]) -> int:
        return int(np.argmin(np.sum((points - np.array(target)) ** 2, axis=1)))

    @staticmethod
    def _linear_interpolate(
        start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        y1, x1 = start
        y2, x2 = end
        n = max(2, int(max(abs(y2 - y1), abs(x2 - x1))))
        pts: list[tuple[int, int]] = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            pts.append((int(y1 + t * (y2 - y1)), int(x1 + t * (x2 - x1))))
        return pts

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_path(path: np.ndarray, sigma: float) -> np.ndarray:
        if len(path) < 3:
            return path
        sy = ndimage.gaussian_filter1d(path[:, 0].astype(float), sigma=sigma)
        sx = ndimage.gaussian_filter1d(path[:, 1].astype(float), sigma=sigma)
        return np.column_stack([sy, sx])

    def smooth_bspline(
        self,
        points: np.ndarray,
        smoothing: float = 0.0,
        num_output_points: int | None = None,
    ) -> np.ndarray:
        """B-spline interpolation for sub-pixel smooth coordinates."""
        if len(points) < 4:
            return points
        try:
            tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothing, k=3)
            u_new = np.linspace(0, 1, num_output_points or len(points))
            return np.column_stack(splev(u_new, tck))
        except Exception as e:
            logger.warning("B-spline smoothing failed: %s", e)
            return points

    # ------------------------------------------------------------------
    # Diameter profile
    # ------------------------------------------------------------------

    def get_diameter_profile(
        self,
        mask: np.ndarray,
        centerline: np.ndarray | None = None,
        num_points: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """EDT-based diameter profile along centerline.

        Returns:
            (sampled_centerline (N,2) (y,x), diameters in pixels)
        """
        if centerline is None:
            centerline = self.last_centerline
        if centerline is None or len(centerline) == 0:
            return np.array([]), np.array([])

        binary = (mask > 0).astype(np.uint8)
        edt = ndimage.distance_transform_edt(binary)

        if len(centerline) > num_points:
            idx = np.linspace(0, len(centerline) - 1, num_points, dtype=int)
            sampled = centerline[idx]
        else:
            sampled = centerline

        diameters = np.array([
            edt[int(y), int(x)] * 2
            for y, x in sampled
            if 0 <= int(y) < edt.shape[0] and 0 <= int(x) < edt.shape[1]
        ])

        return sampled[: len(diameters)], diameters

    def generate_seed_points(
        self, centerline: np.ndarray, num_seeds: int = 3
    ) -> list[tuple[float, float]]:
        """Generate evenly-spaced (x, y) seeds from centerline for tracking."""
        if len(centerline) < num_seeds:
            return [(float(x), float(y)) for y, x in centerline]
        idx = np.linspace(0, len(centerline) - 1, num_seeds, dtype=int)
        return [(float(centerline[i][1]), float(centerline[i][0])) for i in idx]


# ======================================================================
# Singleton
# ======================================================================

_extractor: CenterlineExtractor | None = None


def get_extractor() -> CenterlineExtractor:
    global _extractor
    if _extractor is None:
        _extractor = CenterlineExtractor()
    return _extractor


# ======================================================================
# Module-level convenience functions (used by segmentation_service)
# ======================================================================


def extract_centerline(
    mask: np.ndarray,
    method: str = "skeleton",
    num_points: int | None = None,
    seed_points: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Extract ordered centerline points from a binary mask.

    Wrapper around CenterlineExtractor for backward compatibility.

    Args:
        mask: Binary mask (HxW, uint8, 0 or 255)
        method: Extraction method
        num_points: Number of resampled centerline points.
                    None = keep natural resolution (B-spline smooth only).
        seed_points: Optional (x, y) seed points

    Returns:
        List of (x, y) coordinates, ordered along vessel direction
    """
    if mask is None or mask.max() == 0:
        return []

    ext = get_extractor()
    cl = ext.extract(mask, method=method, seed_points=seed_points)

    if cl is None or len(cl) == 0:
        return []

    # B-spline smooth; resample to num_points if specified, otherwise keep count
    if len(cl) >= 4:
        cl = ext.smooth_bspline(cl, smoothing=0.0, num_output_points=num_points)
    elif num_points is not None and len(cl) > num_points:
        idx = np.linspace(0, len(cl) - 1, num_points, dtype=int)
        cl = cl[idx]

    # Convert (y, x) -> (x, y) for the API
    return [(float(x), float(y)) for y, x in cl]


def compute_perpendicular_diameters(
    centerline: list[tuple[float, float]],
    mask: np.ndarray,
) -> list[float]:
    """Compute vessel diameter at each centerline point using EDT.

    Args:
        centerline: List of (x, y) coordinates
        mask: Binary mask (0 or 255)

    Returns:
        Diameter in pixels at each point
    """
    if len(centerline) < 2:
        return []

    binary = (mask > 127).astype(np.uint8)
    edt = ndimage.distance_transform_edt(binary)

    diameters: list[float] = []
    h, w = mask.shape
    for x, y in centerline:
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < h and 0 <= ix < w:
            diameters.append(float(edt[iy, ix] * 2))
        else:
            diameters.append(0.0)

    return diameters


# ======================================================================
# Perpendicular Gaussian diameter measurement (sub-pixel accurate)
# ======================================================================


def _bilinear_sample(
    image: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation for sub-pixel sampling."""
    h, w = image.shape
    y0 = np.floor(y_coords).astype(int)
    x0 = np.floor(x_coords).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)

    fy = y_coords - np.floor(y_coords)
    fx = x_coords - np.floor(x_coords)

    return (
        (1 - fx) * (1 - fy) * image[y0, x0]
        + (1 - fx) * fy * image[y1, x0]
        + fx * (1 - fy) * image[y0, x1]
        + fx * fy * image[y1, x1]
    )


def _threshold_diameter(t: np.ndarray, profile: np.ndarray, threshold: float = 0.5) -> float:
    """Simple threshold-based diameter: distance between half-max crossings."""
    if len(profile) == 0:
        return 0.0
    max_val = profile.max()
    if max_val <= 0:
        return 0.0
    above = t[profile >= threshold * max_val]
    if len(above) < 2:
        return 0.0
    return float(above[-1] - above[0])


def _fit_gaussian_diameter(t: np.ndarray, profile: np.ndarray) -> float:
    """Fit Gaussian to intensity profile and return FWHM as diameter.

    Model: f(t) = A * exp(-((t - mu)^2) / (2 * sigma^2)) + offset
    Diameter = FWHM = 2.355 * |sigma|

    Uses bounded optimization and cross-validates against threshold
    estimate to prevent wild overestimates from poorly constrained fits.
    """
    from scipy.optimize import curve_fit

    if len(profile) < 5:
        return _threshold_diameter(t, profile)

    # Get threshold-based estimate for initialization and validation
    d_threshold = _threshold_diameter(t, profile)

    A_init = float(profile.max() - profile.min())
    mu_init = float(t[np.argmax(profile)])
    # Initialize sigma from threshold estimate (much better than fixed 5.0)
    sigma_init = max(1.0, d_threshold / 2.355) if d_threshold > 0 else 3.0
    offset_init = float(profile.min())

    def gaussian(x, A, mu, sigma, offset):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + offset

    try:
        t_range = float(t[-1] - t[0])
        popt, _ = curve_fit(
            gaussian, t, profile,
            p0=[A_init, mu_init, sigma_init, offset_init],
            bounds=(
                [0.0, t[0], 0.3, -0.1],                      # lower
                [A_init * 2 + 0.1, t[-1], t_range / 2, 0.5],  # upper
            ),
            maxfev=1000,
        )
        diameter = 2.355 * abs(popt[2])  # FWHM = 2.355 * |sigma|

        # Sanity checks: reject clearly wrong fits
        if diameter > t_range or diameter < 1:
            return d_threshold if d_threshold > 0 else 0.0

        # Cross-validate: if Gaussian FWHM disagrees with threshold by >60%,
        # prefer threshold (more robust for binary masks with sharp edges)
        if d_threshold > 0:
            ratio = diameter / d_threshold
            if ratio > 1.6 or ratio < 0.5:
                return d_threshold

        return diameter
    except Exception:
        return d_threshold if d_threshold > 0 else 0.0


def _measure_diameter_at_point(
    prob_map: np.ndarray,
    cx: float,
    cy: float,
    normal_x: float,
    normal_y: float,
    max_radius: int = 50,
) -> float:
    """Measure diameter along perpendicular at a single point.

    Samples the probability map along the normal direction using bilinear
    interpolation, then fits a Gaussian to extract FWHM.

    Args:
        prob_map: Float probability map (mask as float32)
        cx, cy: Center point (x, y)
        normal_x, normal_y: Perpendicular direction (normalized)
        max_radius: Maximum search radius in pixels

    Returns:
        Diameter in pixels
    """
    h, w = prob_map.shape
    t_values = np.linspace(-max_radius, max_radius, 2 * max_radius + 1)
    sample_x = cx + t_values * normal_x
    sample_y = cy + t_values * normal_y

    valid = (
        (sample_y >= 0) & (sample_y < h - 1)
        & (sample_x >= 0) & (sample_x < w - 1)
    )
    if not valid.any():
        return 0.0

    profile = _bilinear_sample(prob_map, sample_y[valid], sample_x[valid])
    t_valid = t_values[valid]

    return _fit_gaussian_diameter(t_valid, profile)


def measure_diameters_at_points(
    points_xy: list[tuple[float, float]],
    mask: np.ndarray,
    max_radius: int = 50,
    probability_map: np.ndarray | None = None,
) -> list[float]:
    """Measure vessel diameter at specific points using perpendicular Gaussian fitting.

    For each point, computes the tangent from neighboring points, samples
    the mask along the perpendicular direction with bilinear interpolation,
    and fits a Gaussian to determine the vessel width (FWHM).

    An EDT (Euclidean Distance Transform) cross-check prevents the Gaussian
    from overestimating past the actual mask boundary. The EDT gives the
    inscribed circle diameter, which is a strict lower bound. If the Gaussian
    FWHM exceeds 2× the EDT diameter, the EDT value is used instead.

    Args:
        points_xy: List of (x, y) measurement points, ordered along vessel
        mask: Binary segmentation mask (0 or 255)
        max_radius: Maximum perpendicular search radius in pixels
        probability_map: Optional soft probability map (float32, 0..1).
            When provided, used directly for Gaussian fitting instead of
            the binary mask. This preserves sub-pixel edge information
            from the segmentation model's sigmoid/softmax output.
            When None, a Gaussian-blurred version of the binary mask is
            used to create soft edges for the fitter.

    Returns:
        List of diameters in pixels at each point
    """
    if len(points_xy) < 2:
        return []

    if probability_map is not None:
        prob_map = probability_map.astype(np.float32)
    else:
        # Apply Gaussian blur to binary mask to create soft edges.
        # A hard binary (0/1) forces the Gaussian fitter to fit a bell
        # curve to a step function, limiting sub-pixel accuracy. Blurring
        # with sigma=1.5 creates a smooth gradient at vessel boundaries
        # that gives the fitter meaningful edge information, similar to
        # what a raw probability map would provide.
        from scipy.ndimage import gaussian_filter
        binary_float = (mask > 127).astype(np.float32)
        prob_map = gaussian_filter(binary_float, sigma=1.5)

    pts = np.array(points_xy, dtype=np.float64)  # Nx2, columns are (x, y)
    n = len(pts)

    # EDT for cross-validation (inscribed circle diameter)
    binary = (mask > 127).astype(np.uint8)
    edt = ndimage.distance_transform_edt(binary)
    h, w = mask.shape

    diameters: list[float] = []
    for i in range(n):
        # Tangent from ±K neighbors (K=3 for stability)
        k = min(3, i, n - 1 - i)
        if k == 0:
            if i == 0 and n > 1:
                tangent = pts[1] - pts[0]
            elif i == n - 1 and n > 1:
                tangent = pts[-1] - pts[-2]
            else:
                tangent = np.array([1.0, 0.0])
        else:
            tangent = pts[i + k] - pts[i - k]

        norm = np.linalg.norm(tangent)
        if norm > 0:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0])

        # Perpendicular (rotate tangent 90°): normal = (-ty, tx)
        normal_x = -tangent[1]
        normal_y = tangent[0]

        d_gauss = _measure_diameter_at_point(
            prob_map, pts[i, 0], pts[i, 1], normal_x, normal_y, max_radius,
        )

        # EDT cross-check: prevent Gaussian overestimates
        ix, iy = int(round(pts[i, 0])), int(round(pts[i, 1]))
        if 0 <= iy < h and 0 <= ix < w:
            d_edt = float(edt[iy, ix]) * 2.0
        else:
            d_edt = 0.0

        # If Gaussian gives a wildly larger value than EDT, cap it.
        # EDT measures inscribed circle (lower bound), Gaussian measures
        # perpendicular width (can be larger for oblique views).
        # A ratio > 2.0 indicates a measurement error.
        if d_gauss > 0 and d_edt > 0 and d_gauss > 2.0 * d_edt:
            d = d_edt
        elif d_gauss > 0:
            d = d_gauss
        else:
            d = d_edt

        diameters.append(d)

    return diameters
