"""Pixel-to-mm calibration methods."""
import numpy as np

# Standard catheter sizes (French to mm diameter)
CATHETER_SIZES_FR = {
    4: 1.333,
    5: 1.667,
    6: 2.000,
    7: 2.333,
    8: 2.667,
}


def calibrate_from_catheter(
    catheter_diameter_px: float,
    catheter_size_fr: int,
) -> float:
    """Compute pixel spacing from known catheter size.

    Args:
        catheter_diameter_px: Measured catheter diameter in pixels
        catheter_size_fr: Catheter size in French (4-8)

    Returns:
        Pixel spacing in mm/pixel
    """
    if catheter_size_fr not in CATHETER_SIZES_FR:
        raise ValueError(f"Unknown catheter size: {catheter_size_fr}Fr. Valid: {list(CATHETER_SIZES_FR.keys())}")
    if catheter_diameter_px <= 0:
        raise ValueError("Catheter diameter must be positive")

    catheter_mm = CATHETER_SIZES_FR[catheter_size_fr]
    return catheter_mm / catheter_diameter_px


def calibrate_manual(known_distance_mm: float, measured_distance_px: float) -> float:
    """Compute pixel spacing from manually measured distance."""
    if known_distance_mm <= 0 or measured_distance_px <= 0:
        raise ValueError("Both distances must be positive")
    return known_distance_mm / measured_distance_px


def calibrate_from_mask(
    mask: np.ndarray,
    known_diameter_mm: float,
    centerline_point: tuple[float, float] | None = None,
) -> float:
    """Estimate pixel spacing from mask width at a known diameter location."""
    if known_diameter_mm <= 0:
        raise ValueError("Known diameter must be positive")

    binary = (mask > 127).astype(bool)
    if not binary.any():
        raise ValueError("Empty mask")

    if centerline_point:
        # Measure width at specific point
        y = int(round(centerline_point[1]))
        row = binary[y, :] if 0 <= y < mask.shape[0] else binary[mask.shape[0] // 2, :]
    else:
        # Use row with maximum width
        row_widths = binary.sum(axis=1)
        y = np.argmax(row_widths)
        row = binary[y, :]

    # Measure width in pixels
    cols = np.where(row)[0]
    if len(cols) < 2:
        raise ValueError("Could not measure mask width")

    width_px = cols[-1] - cols[0]
    return known_diameter_mm / width_px if width_px > 0 else 0.0
