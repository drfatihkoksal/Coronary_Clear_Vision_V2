import numpy as np
import pytest
from app.core.calibration import calibrate_from_catheter, calibrate_manual, calibrate_from_mask


def test_catheter_calibration_6fr():
    """6Fr catheter = 2.0mm diameter."""
    spacing = calibrate_from_catheter(catheter_diameter_px=20.0, catheter_size_fr=6)
    assert abs(spacing - 0.1) < 0.001  # 2.0mm / 20px = 0.1 mm/px


def test_catheter_calibration_5fr():
    """5Fr catheter = 1.667mm diameter."""
    spacing = calibrate_from_catheter(catheter_diameter_px=10.0, catheter_size_fr=5)
    assert abs(spacing - 0.1667) < 0.001


def test_catheter_calibration_invalid():
    with pytest.raises(ValueError):
        calibrate_from_catheter(10.0, catheter_size_fr=3)


def test_catheter_calibration_zero_diameter():
    with pytest.raises(ValueError):
        calibrate_from_catheter(0.0, catheter_size_fr=6)


def test_manual_calibration():
    spacing = calibrate_manual(known_distance_mm=10.0, measured_distance_px=50.0)
    assert abs(spacing - 0.2) < 0.001


def test_manual_calibration_zero():
    with pytest.raises(ValueError):
        calibrate_manual(0.0, 50.0)


def test_manual_calibration_zero_px():
    with pytest.raises(ValueError):
        calibrate_manual(10.0, 0.0)


def test_mask_calibration():
    mask = np.zeros((64, 128), dtype=np.uint8)
    mask[28:36, 40:60] = 255  # 20px wide
    spacing = calibrate_from_mask(mask, known_diameter_mm=2.0)
    assert abs(spacing - 0.1) < 0.01  # 2.0mm / 20px ~ 0.1


def test_mask_calibration_empty():
    with pytest.raises(ValueError):
        calibrate_from_mask(np.zeros((64, 64), dtype=np.uint8), 2.0)
