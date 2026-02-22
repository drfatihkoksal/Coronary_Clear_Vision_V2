import pytest
import numpy as np
from app.core.rws_calculator import calculate_rws
from app.models.enums import OutlierMethod


def test_uniform_vessel_low_rws():
    """Vessel with constant diameter should have ~0% RWS."""
    profiles = {}
    for i in range(10):
        profiles[i] = [3.0] * 50  # 3mm constant
    result = calculate_rws(profiles, 0, 9, OutlierMethod.NONE)
    assert result["mld_rws_pct"] < 1.0  # Near zero


def test_pulsating_vessel():
    """Vessel that changes from 3mm to 2.5mm should show ~16.7% RWS."""
    profiles = {}
    for i in range(10):
        d = 3.0 - 0.5 * (i / 9)  # Goes from 3.0 to 2.5mm
        profiles[i] = [d] * 50
    result = calculate_rws(profiles, 0, 9, OutlierMethod.NONE)
    # (3.0 - 2.5) / 3.0 * 100 = 16.7%
    assert 15 < result["mld_rws_pct"] < 20


def test_interpretation_normal():
    profiles = {0: [3.0] * 50, 1: [2.9] * 50}
    result = calculate_rws(profiles, 0, 1, OutlierMethod.NONE)
    assert result["interpretation"] == "normal"


def test_interpretation_high_risk():
    profiles = {0: [3.0] * 50, 1: [2.0] * 50}
    result = calculate_rws(profiles, 0, 1, OutlierMethod.NONE)
    assert result["interpretation"] in ("vulnerable", "high_risk")


def test_with_hampel_filter():
    profiles = {}
    for i in range(10):
        profiles[i] = [3.0] * 50
    profiles[5] = [10.0] * 50  # Outlier frame
    result = calculate_rws(profiles, 0, 9, OutlierMethod.HAMPEL)
    assert result["outliers_removed"] > 0


def test_insufficient_frames():
    with pytest.raises(ValueError):
        calculate_rws({0: [3.0] * 50}, 0, 5, OutlierMethod.NONE)
