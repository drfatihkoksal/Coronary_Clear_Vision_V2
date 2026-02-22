import logging
import numpy as np
from app.core.centerline_extractor import extract_centerline, compute_perpendicular_diameters
from app.core.qca_engine import compute_qca_measurements
from app.core.calibration import calibrate_from_catheter, calibrate_manual, calibrate_from_mask
from app.infra.persistence.session_store import Session
from app.models.domain import PixelSpacing, QCAMetrics
from app.models.enums import CalibrationSource

logger = logging.getLogger(__name__)


class AnalysisService:
    def compute_qca(
        self,
        session: Session,
        frame_index: int,
        method: str = "gaussian",
        num_points: int | None = None,
    ) -> dict:
        """Compute QCA measurements for a segmented frame."""
        seg_data = session.segmentations.get(frame_index)
        if seg_data is None:
            raise ValueError(f"No segmentation for frame {frame_index}")

        mask = seg_data["mask"]
        centerline = seg_data.get("centerline", [])
        probability_map = seg_data.get("probability_map")

        if not centerline:
            raise ValueError("No centerline extracted for this frame")

        # Get pixel spacing
        ps = self._get_pixel_spacing(session)

        result = compute_qca_measurements(
            mask=mask,
            centerline=centerline,
            pixel_spacing_mm=ps,
            method=method,
            num_points=num_points,
            probability_map=probability_map,
        )

        # Cache in session
        session.qca_measurements[frame_index] = result

        return result

    def calibrate_catheter_from_segmentation(
        self,
        session: Session,
        frame_index: int,
        catheter_size_fr: int,
    ) -> dict:
        """Calibrate using catheter segmentation mask.

        Extracts centerline from the segmentation mask, computes the
        mean perpendicular diameter in pixels, and uses that as the
        catheter diameter for calibration.
        """
        seg_data = session.segmentations.get(frame_index)
        if seg_data is None:
            raise ValueError(f"No segmentation for frame {frame_index}")

        mask = seg_data["mask"]

        # Extract centerline and diameter profile from the catheter mask
        centerline = extract_centerline(mask, method="skeleton")
        if len(centerline) < 2:
            raise ValueError("Could not extract centerline from catheter segmentation")

        diameters_px = compute_perpendicular_diameters(centerline, mask)
        if not diameters_px:
            raise ValueError("Could not compute diameter profile from catheter mask")

        # Use median (robust to endpoints) as the catheter diameter in pixels
        mean_diameter_px = float(np.median(diameters_px))
        if mean_diameter_px <= 0:
            raise ValueError("Computed catheter diameter is zero")

        logger.info(
            "Catheter calibration: %d diameters, median=%.2f px, Fr=%d",
            len(diameters_px), mean_diameter_px, catheter_size_fr,
        )

        return self.calibrate_catheter(session, mean_diameter_px, catheter_size_fr)

    def calibrate_catheter(
        self,
        session: Session,
        catheter_diameter_px: float,
        catheter_size_fr: int,
    ) -> dict:
        """Calibrate using catheter measurement."""
        spacing = calibrate_from_catheter(catheter_diameter_px, catheter_size_fr)
        ps = PixelSpacing(
            row_spacing=spacing,
            col_spacing=spacing,
            source=CalibrationSource.CATHETER,
            confidence=0.9,
        )
        session.calibration = ps
        if session.study:
            session.study.pixel_spacing = ps
        return {"pixel_spacing_mm": spacing, "source": "catheter"}

    def calibrate_manual(
        self,
        session: Session,
        known_distance_mm: float,
        measured_distance_px: float,
    ) -> dict:
        spacing = calibrate_manual(known_distance_mm, measured_distance_px)
        ps = PixelSpacing(
            row_spacing=spacing,
            col_spacing=spacing,
            source=CalibrationSource.MANUAL,
            confidence=0.7,
        )
        session.calibration = ps
        if session.study:
            session.study.pixel_spacing = ps
        return {"pixel_spacing_mm": spacing, "source": "manual"}

    def _get_pixel_spacing(self, session: Session) -> float:
        """Get pixel spacing in mm/pixel from session."""
        if session.calibration:
            return session.calibration.row_spacing
        if session.study and session.study.pixel_spacing:
            return session.study.pixel_spacing.row_spacing
        return 0.1  # Default fallback
