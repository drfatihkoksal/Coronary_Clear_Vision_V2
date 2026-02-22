"""QFR pipeline orchestration service."""
import logging
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter1d
from app.core.stereo_reconstructor import reconstruct_3d_centerline, compute_angular_separation, ProjectionParams, build_camera_matrix
from app.core.qfr_calculator import calculate_qfr
from app.core.vessel_mesher import generate_vessel_mesh
from app.core.centerline_extractor import measure_diameters_at_points
from app.infra.persistence.session_store import Session

logger = logging.getLogger(__name__)


@dataclass
class QFRProjection:
    """State for one QFR projection."""
    frames: list | None = None
    num_frames: int = 0
    image_width: int = 0
    image_height: int = 0
    pixel_spacing: float = 0.3
    angle_deg: float = 0.0
    secondary_angle_deg: float = 0.0
    sod: float = 1000.0
    sid: float = 1000.0
    frame_rate: float = 15.0
    centerline: list = field(default_factory=list)
    diameters_px: list = field(default_factory=list)
    diameters_mm: list = field(default_factory=list)
    timi_start: int | None = None
    timi_end: int | None = None
    mask: np.ndarray | None = None
    probability_map: np.ndarray | None = None


@dataclass
class QFRSession:
    projection1: QFRProjection | None = None
    projection2: QFRProjection | None = None
    result_3d: dict | None = None
    mesh: dict | None = None
    qfr_result: dict | None = None


@dataclass
class MatchedDiameters:
    """Diameter measurements from both views and their combination."""
    combined: np.ndarray      # Harmonic mean of both views (mm)
    view1_mm: np.ndarray      # View 1 diameters (mm), 0 where invalid
    view2_mm: np.ndarray      # View 2 diameters (mm), 0 where invalid


class QFRService:
    @staticmethod
    def _arc_length_fractions(cl: np.ndarray) -> np.ndarray:
        """Compute normalized arc-length fractions [0..1] for a 2D centerline."""
        diffs = np.diff(cl, axis=0)
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        cumlen = np.concatenate([[0.0], np.cumsum(lengths)])
        total = cumlen[-1]
        if total < 1e-10:
            return np.linspace(0, 1, len(cl))
        return cumlen / total

    @staticmethod
    def _interpolate_2d_points(
        centerline_xy: list, t_fractions: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Interpolate 2D centerline positions at given arc-length fractions."""
        cl = np.array(centerline_xy, dtype=np.float64)
        n = len(cl)
        t_uniform = np.linspace(0, 1, n)
        xs = np.interp(t_fractions, t_uniform, cl[:, 0])
        ys = np.interp(t_fractions, t_uniform, cl[:, 1])
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    @staticmethod
    def _interpolate_gaps(d_mm: np.ndarray) -> np.ndarray | None:
        """Interpolate zero-gaps in a diameter profile.

        Returns None if fewer than 50% of measurements are valid.
        """
        valid = d_mm > 0
        if valid.sum() < len(d_mm) * 0.5:
            return None
        d = d_mm.copy()
        if not valid.all():
            valid_idx = np.where(valid)[0]
            invalid_idx = np.where(~valid)[0]
            d[invalid_idx] = np.interp(invalid_idx, valid_idx, d[valid_idx])
        return d

    @staticmethod
    def _measure_matched_diameters(
        p1: 'QFRProjection',
        p2: 'QFRProjection',
        t1_fractions: np.ndarray,
        t2_fractions: np.ndarray,
        depth_correction_1: np.ndarray | float = 1.0,
        depth_correction_2: np.ndarray | float = 1.0,
    ) -> MatchedDiameters:
        """Measure diameters at epipolar-matched 2D points from both masks.

        Uses perpendicular Gaussian fitting on the segmentation masks for
        sub-pixel accuracy.

        Combination strategy: harmonic mean d_eq = 2·d1·d2 / (d1 + d2).
        For Poiseuille flow, pressure drop scales as 1/r⁴. The harmonic
        mean gives the resistance-equivalent diameter for an elliptical
        cross-section — more conservative than the geometric mean (Tu et al.)
        when views disagree, which is critical when the MLD appears at
        different vessel positions in the two views.

        Args:
            depth_correction_1: Per-point Z_vessel/SOD array for view 1, or
                scalar fallback. Corrects isocenter pixel spacing to actual
                vessel depth at each measurement point.
            depth_correction_2: Same for view 2.

        Falls back to interpolated EDT diameters if masks are unavailable.
        """
        has_mask1 = p1.mask is not None
        has_mask2 = p2.mask is not None

        if has_mask1 and has_mask2:
            # Re-measure from both masks at matched 2D positions.
            # Use arc-length parameterization (not uniform index) so that
            # t_fractions from stereo reconstruction map to the correct
            # physical positions on each 2D centerline.
            cl1 = np.array(p1.centerline, dtype=np.float64)
            cl2 = np.array(p2.centerline, dtype=np.float64)
            arc1 = QFRService._arc_length_fractions(cl1)
            arc2 = QFRService._arc_length_fractions(cl2)

            pts1_x = np.interp(t1_fractions, arc1, cl1[:, 0])
            pts1_y = np.interp(t1_fractions, arc1, cl1[:, 1])
            pts1 = [(float(x), float(y)) for x, y in zip(pts1_x, pts1_y)]

            pts2_x = np.interp(t2_fractions, arc2, cl2[:, 0])
            pts2_y = np.interp(t2_fractions, arc2, cl2[:, 1])
            pts2 = [(float(x), float(y)) for x, y in zip(pts2_x, pts2_y)]

            d1_px = measure_diameters_at_points(pts1, p1.mask, probability_map=p1.probability_map)
            d2_px = measure_diameters_at_points(pts2, p2.mask, probability_map=p2.probability_map)

            # Convert px → mm using per-point depth-corrected pixel spacing.
            # isocenter_ps × (Z_vessel_i / SOD) gives the true pixel spacing
            # at each point's actual depth from the X-ray source.
            d1_mm = np.array(d1_px, dtype=np.float64) * p1.pixel_spacing * depth_correction_1
            d2_mm = np.array(d2_px, dtype=np.float64) * p2.pixel_spacing * depth_correction_2

            # Harmonic mean: d_eq = 2·d1·d2 / (d1 + d2)
            # More conservative than geometric mean when views disagree.
            # For Poiseuille flow (ΔP ∝ 1/r⁴), harmonic mean gives the
            # resistance-equivalent diameter of an elliptical cross-section.
            d1_valid = d1_mm > 0
            d2_valid = d2_mm > 0
            both_valid = d1_valid & d2_valid

            diameters = np.full(len(d1_mm), np.nan)
            diameters[both_valid] = (
                2.0 * d1_mm[both_valid] * d2_mm[both_valid]
                / (d1_mm[both_valid] + d2_mm[both_valid])
            )
            diameters[d1_valid & ~d2_valid] = d1_mm[d1_valid & ~d2_valid]
            diameters[~d1_valid & d2_valid] = d2_mm[~d1_valid & d2_valid]

            # Fill remaining NaN by interpolation from neighbors
            nan_mask = np.isnan(diameters)
            if nan_mask.any() and not nan_mask.all():
                valid_idx = np.where(~nan_mask)[0]
                invalid_idx = np.where(nan_mask)[0]
                diameters[invalid_idx] = np.interp(
                    invalid_idx, valid_idx, diameters[valid_idx],
                )
            elif nan_mask.all():
                diameters = np.zeros(len(t1_fractions))

            # Comprehensive diagnostics for debugging real-data QFR issues
            d1_min = float(np.min(d1_mm[d1_valid])) if d1_valid.any() else 0
            d2_min = float(np.min(d2_mm[d2_valid])) if d2_valid.any() else 0
            d1_min_idx = int(np.argmin(d1_mm)) if d1_valid.any() else -1
            d2_min_idx = int(np.argmin(d2_mm)) if d2_valid.any() else -1
            combined_min = float(np.nanmin(diameters))
            combined_min_idx = int(np.nanargmin(diameters))

            # Per-view reference (upper quartile of proximal 20%)
            n_fifth = max(1, len(d1_mm) // 5)
            d1_ref = float(np.mean(np.sort(d1_mm[:n_fifth])[-max(1, n_fifth//2):])) if d1_valid[:n_fifth].any() else 0
            d2_ref = float(np.mean(np.sort(d2_mm[:n_fifth])[-max(1, n_fifth//2):])) if d2_valid[:n_fifth].any() else 0

            d1_ds = (1 - d1_min / d1_ref) * 100 if d1_ref > 0 else 0
            d2_ds = (1 - d2_min / d2_ref) * 100 if d2_ref > 0 else 0
            combined_ref = float(np.nanmean(np.sort(diameters[:n_fifth])[-max(1, n_fifth//2):]))
            combined_ds = (1 - combined_min / combined_ref) * 100 if combined_ref > 0 else 0

            # Check for large per-view disagreement at the stenosis
            mld_region = slice(max(0, combined_min_idx - 5), min(len(d1_mm), combined_min_idx + 6))
            d1_at_mld = float(np.min(d1_mm[mld_region])) if d1_valid[mld_region].any() else 0
            d2_at_mld = float(np.min(d2_mm[mld_region])) if d2_valid[mld_region].any() else 0
            view_ratio = max(d1_at_mld, d2_at_mld) / min(d1_at_mld, d2_at_mld) if min(d1_at_mld, d2_at_mld) > 0 else 0

            logger.info(
                "Diameters: n=%d combined_MLD=%.2fmm@%d combined_ref=%.2fmm DS=%.1f%%",
                len(diameters), combined_min, combined_min_idx, combined_ref, combined_ds,
            )
            logger.info(
                "  View1: MLD=%.2fmm@%d ref=%.2fmm DS=%.1f%% | "
                "View2: MLD=%.2fmm@%d ref=%.2fmm DS=%.1f%%",
                d1_min, d1_min_idx, d1_ref, d1_ds,
                d2_min, d2_min_idx, d2_ref, d2_ds,
            )
            if view_ratio > 1.5:
                logger.warning(
                    "  Large view disagreement at MLD: d1=%.2fmm d2=%.2fmm ratio=%.1fx "
                    "(eccentric stenosis or segmentation mismatch)",
                    d1_at_mld, d2_at_mld, view_ratio,
                )
            logger.info(
                "  Pixel spacings: p1=%.4fmm p2=%.4fmm | "
                "d1_px range=[%.1f, %.1f] d2_px range=[%.1f, %.1f]",
                p1.pixel_spacing, p2.pixel_spacing,
                float(np.min(d1_px)) if len(d1_px) > 0 else 0,
                float(np.max(d1_px)) if len(d1_px) > 0 else 0,
                float(np.min(d2_px)) if len(d2_px) > 0 else 0,
                float(np.max(d2_px)) if len(d2_px) > 0 else 0,
            )
            return MatchedDiameters(
                combined=diameters,
                view1_mm=d1_mm,
                view2_mm=d2_mm,
            )

        # Fallback: interpolate pre-computed EDT diameters using arc-length.
        # Use median (scalar) depth correction here since EDT diameters are at
        # 2D positions and per-point depth correction arrays are at 3D positions.
        dc1_scalar = float(np.median(depth_correction_1)) if isinstance(depth_correction_1, np.ndarray) else float(depth_correction_1)
        dc2_scalar = float(np.median(depth_correction_2)) if isinstance(depth_correction_2, np.ndarray) else float(depth_correction_2)
        d1 = np.array(p1.diameters_mm, dtype=np.float64) * dc1_scalar
        d2 = np.array(p2.diameters_mm, dtype=np.float64) * dc2_scalar
        cl1 = np.array(p1.centerline, dtype=np.float64)
        cl2 = np.array(p2.centerline, dtype=np.float64)
        arc1 = QFRService._arc_length_fractions(cl1) if len(cl1) == len(d1) else np.linspace(0, 1, len(d1))
        arc2 = QFRService._arc_length_fractions(cl2) if len(cl2) == len(d2) else np.linspace(0, 1, len(d2))
        d1_aligned = np.interp(t1_fractions, arc1, d1)
        d2_aligned = np.interp(t2_fractions, arc2, d2)

        d1_valid = d1_aligned > 0
        d2_valid = d2_aligned > 0
        both_valid = d1_valid & d2_valid

        diameters = np.full(len(d1_aligned), np.nan)
        diameters[both_valid] = (
            2.0 * d1_aligned[both_valid] * d2_aligned[both_valid]
            / (d1_aligned[both_valid] + d2_aligned[both_valid])
        )
        diameters[d1_valid & ~d2_valid] = d1_aligned[d1_valid & ~d2_valid]
        diameters[~d1_valid & d2_valid] = d2_aligned[~d1_valid & d2_valid]
        diameters = np.nan_to_num(diameters, nan=0.0)
        logger.warning("Using fallback EDT-interpolated diameters (masks not available)")
        return MatchedDiameters(
            combined=diameters,
            view1_mm=np.nan_to_num(d1_aligned, nan=0.0),
            view2_mm=np.nan_to_num(d2_aligned, nan=0.0),
        )

    def reconstruct_and_calculate(
        self,
        session: Session,
        mode: str = "fQFR",
        kt: float = 1.52,
    ) -> dict:
        """Run full QFR pipeline: reconstruct 3D + calculate QFR."""
        qfr_session = session.qfr_session
        if qfr_session is None or qfr_session.projection1 is None or qfr_session.projection2 is None:
            raise ValueError("Both projections must be loaded")

        p1, p2 = qfr_session.projection1, qfr_session.projection2

        if not p1.centerline or not p2.centerline:
            raise ValueError("Both projections must be segmented with centerlines")

        # Check angular separation
        angular_sep = compute_angular_separation(p1.angle_deg, p2.angle_deg)

        # 3D reconstruction
        params1 = ProjectionParams(
            primary_angle=p1.angle_deg,
            secondary_angle=p1.secondary_angle_deg,
            sid=p1.sid,
            sod=p1.sod,
            pixel_spacing=p1.pixel_spacing,
            image_size=(p1.image_width, p1.image_height),
        )
        params2 = ProjectionParams(
            primary_angle=p2.angle_deg,
            secondary_angle=p2.secondary_angle_deg,
            sid=p2.sid,
            sod=p2.sod,
            pixel_spacing=p2.pixel_spacing,
            image_size=(p2.image_width, p2.image_height),
        )

        recon = reconstruct_3d_centerline(p1.centerline, p2.centerline, params1, params2)

        if len(recon.points_3d) < 2:
            raise ValueError("3D reconstruction failed - insufficient points")

        # Convert to numpy
        centerline_3d = recon.points_3d
        centerline_3d_arr = np.asarray(centerline_3d, dtype=np.float64)
        n_3d = len(centerline_3d_arr)

        # ---- Per-point depth correction ----
        # The isocenter pixel spacing (ps_iso = detector_ps × SOD/SID) assumes
        # the vessel is at the isocenter. In cardiac angiography, coronary
        # arteries are typically 5-15cm anterior to the isocenter.
        # After stereo reconstruction, we KNOW the actual 3D depth at each point.
        # True diameter_i = pixel_diam_i × ps_iso × (Z_vessel_i / SOD).
        #
        # Previously a single median Z was used. Now we compute smoothed
        # per-point Z to capture proximal→distal magnification differences
        # (up to several cm depth variation along a wrapping coronary).
        K1, R1, t1_cam, P1 = build_camera_matrix(params1)
        K2, R2, t2_cam, P2 = build_camera_matrix(params2)

        Rt1 = np.hstack([R1, t1_cam])  # 3×4
        Rt2 = np.hstack([R2, t2_cam])
        pts_hom = np.column_stack([centerline_3d_arr, np.ones(n_3d)])  # N×4
        Z1_all = (Rt1 @ pts_hom.T)[2, :]  # depth in camera 1 frame
        Z2_all = (Rt2 @ pts_hom.T)[2, :]  # depth in camera 2 frame

        # Smoothed per-point depth correction.
        # Raw per-point Z from DLT is noisy (~1mm/pixel mismatch), so
        # Gaussian smoothing removes jitter while preserving the real
        # proximal→distal depth gradient (which drives magnification change).
        # Sigma scales with point count to give consistent physical smoothing.
        z_smooth_sigma = max(5, n_3d // 10)

        # Replace non-positive Z with median of positive values (robust anchor)
        Z1_pos = Z1_all[Z1_all > 0]
        Z2_pos = Z2_all[Z2_all > 0]
        median_Z1 = float(np.median(Z1_pos)) if len(Z1_pos) > 0 else p1.sod
        median_Z2 = float(np.median(Z2_pos)) if len(Z2_pos) > 0 else p2.sod

        Z1_clean = np.where(Z1_all > 0, Z1_all, median_Z1)
        Z2_clean = np.where(Z2_all > 0, Z2_all, median_Z2)

        Z1_smooth = gaussian_filter1d(Z1_clean, sigma=z_smooth_sigma)
        Z2_smooth = gaussian_filter1d(Z2_clean, sigma=z_smooth_sigma)

        depth_corr_1 = Z1_smooth / p1.sod  # N-length array
        depth_corr_2 = Z2_smooth / p2.sod  # N-length array

        # Scalar median for vessel length (smoothed per-point is for diameters)
        median_depth_corr_1 = median_Z1 / p1.sod
        median_depth_corr_2 = median_Z2 / p2.sod

        logger.info(
            "Depth correction (per-point): view1 Z_range=[%.1f, %.1f]mm median=%.1fmm SOD=%.1fmm | "
            "view2 Z_range=[%.1f, %.1f]mm median=%.1fmm SOD=%.1fmm",
            float(Z1_smooth.min()), float(Z1_smooth.max()), median_Z1, p1.sod,
            float(Z2_smooth.min()), float(Z2_smooth.max()), median_Z2, p2.sod,
        )

        # Vessel length from 2D centerlines (more robust than noisy 3D depth).
        # With typical 25-40° angular separation, parallax-based depth is noisy
        # (~1mm error per pixel mismatch), inflating the 3D path length.
        # Using max of 2D lengths picks the less foreshortened projection.
        # Median depth correction for length (scalar is appropriate here since
        # length integrates over the full vessel).
        cl1_arr = np.array(p1.centerline, dtype=np.float64)
        cl2_arr = np.array(p2.centerline, dtype=np.float64)
        len_2d_1 = float(np.sum(np.sqrt(np.sum(np.diff(cl1_arr, axis=0) ** 2, axis=1)))) * p1.pixel_spacing * median_depth_corr_1
        len_2d_2 = float(np.sum(np.sqrt(np.sum(np.diff(cl2_arr, axis=0) ** 2, axis=1)))) * p2.pixel_spacing * median_depth_corr_2
        vessel_length_2d = max(len_2d_1, len_2d_2)

        # Re-measure diameters at epipolar-matched 2D points from masks.
        # This ensures each 3D point gets diameters measured at the SAME
        # physical vessel location in both views, using perpendicular
        # cross-section Gaussian fitting (sub-pixel accurate).
        matched = self._measure_matched_diameters(
            p1, p2, recon.t1_fractions, recon.t2_fractions,
            depth_correction_1=depth_corr_1,
            depth_correction_2=depth_corr_2,
        )
        diameters_3d = matched.combined

        # TIMI frame count — average from both projections if available
        timi_count = None
        if mode in ("cQFR", "aQFR"):
            timi_counts = []
            if p1.timi_start is not None and p1.timi_end is not None:
                timi_counts.append(abs(p1.timi_end - p1.timi_start))
            if p2.timi_start is not None and p2.timi_end is not None:
                timi_counts.append(abs(p2.timi_end - p2.timi_start))
            if timi_counts:
                timi_count = int(round(sum(timi_counts) / len(timi_counts)))

        # Average frame rate from both projections
        frame_rate = (p1.frame_rate + p2.frame_rate) / 2.0

        # Shared QFR kwargs for combined and per-view calculations
        qfr_kwargs = dict(
            kt=kt, mode=mode, timi_frame_count=timi_count,
            frame_rate=frame_rate, vessel_length_override_mm=vessel_length_2d,
        )

        # Calculate combined QFR (harmonic mean diameters)
        qfr_result = calculate_qfr(centerline_3d_arr, diameters_3d, **qfr_kwargs)

        # Per-view QFR: compute QFR from each view's diameters independently
        # (circular cross-section assumption). This catches stenoses that are
        # diluted by the two-view combination when MLD positions don't align.
        per_view_qfr = {}
        for label, d_raw in [("view1", matched.view1_mm), ("view2", matched.view2_mm)]:
            d_filled = self._interpolate_gaps(d_raw)
            if d_filled is not None:
                # Log pre-QFR diagnostics to trace smoothing effects
                raw_mld = float(np.min(d_filled[d_filled > 0])) if (d_filled > 0).any() else 0
                raw_mld_idx = int(np.argmin(d_filled)) if len(d_filled) > 0 else -1
                n_zeros = int(np.sum(d_raw <= 0))
                pv_result = calculate_qfr(centerline_3d_arr, d_filled, **qfr_kwargs)
                logger.info(
                    "Per-view %s: raw_MLD=%.2fmm@%d → smoothed_MLD=%.2fmm "
                    "(zeros_filled=%d/%d) QFR=%.3f",
                    label, raw_mld, raw_mld_idx, pv_result["mld_mm"],
                    n_zeros, len(d_raw), pv_result["qfr"],
                )
                per_view_qfr[label] = {
                    "qfr": pv_result["qfr"],
                    "mld_mm": pv_result["mld_mm"],
                    "reference_diameter_mm": pv_result["reference_diameter_mm"],
                }

        # Primary QFR: minimum of combined and per-view (conservative estimate)
        all_qfr_values = [qfr_result["qfr"]]
        for pv in per_view_qfr.values():
            if pv["qfr"] > 0:
                all_qfr_values.append(pv["qfr"])
        primary_qfr = min(all_qfr_values)

        if primary_qfr < qfr_result["qfr"]:
            logger.warning(
                "Per-view QFR lower than combined: combined=%.3f %s → using %.3f",
                qfr_result["qfr"],
                {k: v["qfr"] for k, v in per_view_qfr.items()},
                primary_qfr,
            )

        # Inject per-view data and primary QFR into result
        qfr_result["qfr_combined"] = qfr_result["qfr"]
        qfr_result["qfr"] = round(primary_qfr, 3)
        qfr_result["qfr_per_view"] = per_view_qfr

        # Vessel length from result
        vessel_length = qfr_result.get("vessel_length_mm", 0.0)

        # 3D length for comparison (usually inflated by depth noise)
        raw_3d_length = float(np.sum(
            np.sqrt(np.sum(np.diff(centerline_3d_arr, axis=0) ** 2, axis=1))
        ))

        logger.info(
            "QFR=%s (combined=%.3f v1=%s v2=%s) mode=%s vessel=%.1fmm(2D) "
            "raw3D=%.1fmm n_pts=%d angular_sep=%.1f°",
            qfr_result["qfr"], qfr_result["qfr_combined"],
            per_view_qfr.get("view1", {}).get("qfr", "N/A"),
            per_view_qfr.get("view2", {}).get("qfr", "N/A"),
            mode, vessel_length, raw_3d_length, n_3d, angular_sep,
        )

        # Generate mesh
        mesh = generate_vessel_mesh(
            centerline_3d, diameters_3d.tolist(), qfr_result.get("pressure_profile"),
        )

        # Store in session (include per-view diameters and vessel length for recalculate)
        qfr_session.result_3d = {
            "centerline_3d": centerline_3d,
            "diameters_3d": diameters_3d.tolist(),
            "view1_diameters_mm": matched.view1_mm.tolist(),
            "view2_diameters_mm": matched.view2_mm.tolist(),
            "angular_separation": angular_sep,
            "vessel_length_2d_mm": vessel_length_2d,
        }
        qfr_session.mesh = mesh
        qfr_session.qfr_result = qfr_result

        return {
            "qfr": qfr_result,
            "mesh": mesh,
            "angular_separation": angular_sep,
            "reconstruction": {"num_points": n_3d, "vessel_length_mm": vessel_length},
        }

    def recalculate_qfr(
        self,
        session: Session,
        stenosis_indices: list[int],
        mode: str = "fQFR",
        kt: float = 1.52,
    ) -> dict:
        """Recalculate QFR with user-specified stenosis positions.

        Uses cached 3D data from a previous reconstruction — no re-reconstruction needed.
        """
        qfr_session = session.qfr_session
        if qfr_session is None or qfr_session.result_3d is None:
            raise ValueError("No 3D reconstruction available. Run reconstruct first.")

        p1, p2 = qfr_session.projection1, qfr_session.projection2
        if p1 is None or p2 is None:
            raise ValueError("Both projections must be loaded")

        centerline_3d = np.array(qfr_session.result_3d["centerline_3d"], dtype=np.float64)
        diameters_3d = np.array(qfr_session.result_3d["diameters_3d"], dtype=np.float64)
        n_3d = len(centerline_3d)

        # Validate stenosis indices
        for idx in stenosis_indices:
            if idx < 0 or idx >= len(diameters_3d):
                raise ValueError(f"Stenosis index {idx} out of range (0-{len(diameters_3d)-1})")

        # Use cached depth-corrected vessel length from reconstruction
        vessel_length_2d = qfr_session.result_3d.get("vessel_length_2d_mm")
        if vessel_length_2d is None:
            # Fallback for sessions reconstructed before this field was added
            cl1_arr = np.array(p1.centerline, dtype=np.float64)
            cl2_arr = np.array(p2.centerline, dtype=np.float64)
            len_2d_1 = float(np.sum(np.sqrt(np.sum(np.diff(cl1_arr, axis=0) ** 2, axis=1)))) * p1.pixel_spacing
            len_2d_2 = float(np.sum(np.sqrt(np.sum(np.diff(cl2_arr, axis=0) ** 2, axis=1)))) * p2.pixel_spacing
            vessel_length_2d = max(len_2d_1, len_2d_2)

        # TIMI frame count
        timi_count = None
        if mode in ("cQFR", "aQFR"):
            timi_counts = []
            if p1.timi_start is not None and p1.timi_end is not None:
                timi_counts.append(abs(p1.timi_end - p1.timi_start))
            if p2.timi_start is not None and p2.timi_end is not None:
                timi_counts.append(abs(p2.timi_end - p2.timi_start))
            if timi_counts:
                timi_count = int(round(sum(timi_counts) / len(timi_counts)))

        frame_rate = (p1.frame_rate + p2.frame_rate) / 2.0

        qfr_kwargs = dict(
            kt=kt, mode=mode, timi_frame_count=timi_count,
            frame_rate=frame_rate, vessel_length_override_mm=vessel_length_2d,
            stenosis_indices_override=stenosis_indices,
        )
        qfr_result = calculate_qfr(centerline_3d, diameters_3d, **qfr_kwargs)

        # Per-view QFR from cached per-view diameters
        per_view_qfr = {}
        for label, key in [("view1", "view1_diameters_mm"), ("view2", "view2_diameters_mm")]:
            raw = qfr_session.result_3d.get(key)
            if raw is not None:
                d_raw = np.array(raw, dtype=np.float64)
                d_filled = self._interpolate_gaps(d_raw)
                if d_filled is not None:
                    pv = calculate_qfr(centerline_3d, d_filled, **qfr_kwargs)
                    per_view_qfr[label] = {
                        "qfr": pv["qfr"],
                        "mld_mm": pv["mld_mm"],
                        "reference_diameter_mm": pv["reference_diameter_mm"],
                    }

        all_qfr_values = [qfr_result["qfr"]]
        for pv in per_view_qfr.values():
            if pv["qfr"] > 0:
                all_qfr_values.append(pv["qfr"])
        primary_qfr = min(all_qfr_values)

        qfr_result["qfr_combined"] = qfr_result["qfr"]
        qfr_result["qfr"] = round(primary_qfr, 3)
        qfr_result["qfr_per_view"] = per_view_qfr

        angular_sep = qfr_session.result_3d.get("angular_separation", 0.0)
        vessel_length = qfr_result.get("vessel_length_mm", 0.0)

        logger.info(
            "QFR recalculated=%s (combined=%.3f) mode=%s vessel=%.1fmm override_indices=%s",
            qfr_result["qfr"], qfr_result["qfr_combined"], mode, vessel_length,
            stenosis_indices,
        )

        # Regenerate mesh with updated pressure profile
        mesh = generate_vessel_mesh(
            centerline_3d.tolist(), diameters_3d.tolist(),
            qfr_result.get("pressure_profile"),
        )

        # Update session
        qfr_session.mesh = mesh
        qfr_session.qfr_result = qfr_result

        return {
            "qfr": qfr_result,
            "mesh": mesh,
            "angular_separation": angular_sep,
            "reconstruction": {"num_points": n_3d, "vessel_length_mm": vessel_length},
        }
