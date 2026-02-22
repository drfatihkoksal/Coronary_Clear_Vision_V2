import numpy as np
from app.core.mask_ops import apply_brush, apply_flood_fill, apply_morphological_op


def test_brush_paints():
    mask = np.zeros((64, 64), dtype=np.uint8)
    result = apply_brush(mask, [(32, 32)], radius=5, value=255)
    assert result[32, 32] == 255
    assert result.sum() > 0


def test_brush_erases():
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    result = apply_brush(mask, [(32, 32)], radius=5, value=0)
    assert result[32, 32] == 0


def test_flood_fill():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255  # Square region
    # Fill from inside the square
    result = apply_flood_fill(mask, (30, 30), value=0)
    # The square should be erased (flood filled with 0)
    assert result[30, 30] == 0


def test_morphological_dilate():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:34, 30:34] = 255  # Small square
    result = apply_morphological_op(mask, "dilate", kernel_size=3)
    assert result.sum() > mask.sum()


def test_morphological_erode():
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    result = apply_morphological_op(mask, "erode", kernel_size=3)
    assert result.sum() < mask.sum()
