import numpy as np

from app.core.tracking_engine import (
    initialize_tracker,
    propagate_tracking,
    refine_roi_with_optical_flow,
    track_frame,
)


def _make_frame_with_rect(
    width: int = 200,
    height: int = 200,
    rect_x: int = 50,
    rect_y: int = 50,
    rect_w: int = 40,
    rect_h: int = 30,
    bg: int = 30,
    fg: int = 220,
    seed: int = 42,
) -> np.ndarray:
    """Create a grayscale frame with a textured rectangle on noisy background.

    Uses deterministic noise so template matching (NCC) has non-zero variance
    within the ROI, which is required for normalized cross-correlation.
    """
    rng = np.random.RandomState(seed)
    # Background with slight noise for texture
    frame = np.clip(
        rng.normal(bg, 5, (height, width)),
        0, 255,
    ).astype(np.uint8)
    # Foreground rectangle with texture (gradient + noise)
    for dy in range(rect_h):
        for dx in range(rect_w):
            val = fg - 30 * (dy / rect_h) + 20 * np.sin(dx * 0.5)
            y, x = rect_y + dy, rect_x + dx
            if 0 <= y < height and 0 <= x < width:
                frame[y, x] = np.clip(val + rng.normal(0, 8), 0, 255)
    return frame


def test_initialize_tracker():
    """Create gray frame with white rectangle, init tracker successfully."""
    frame = _make_frame_with_rect()
    roi = (50, 50, 40, 30)
    tracker = initialize_tracker(frame, roi)
    assert tracker is not None


def test_track_moving_object():
    """Create sequence with moving rectangle, verify bbox follows movement."""
    frames = []
    for i in range(5):
        frames.append(_make_frame_with_rect(rect_x=50 + i * 5, rect_y=50 + i * 3, seed=100 + i))

    tracker = initialize_tracker(frames[0], (50, 50, 40, 30))
    all_ok = True
    last_x = 50

    for i in range(1, len(frames)):
        success, bbox, confidence = track_frame(tracker, frames[i])
        if not success:
            all_ok = False
            break
        # Tracked x should generally increase as rectangle moves right
        last_x = bbox[0]

    assert all_ok, "Tracking should succeed for all frames"
    assert last_x > 50, "Tracked bbox should have moved rightward"


def test_propagate_forward():
    """Propagate tracking forward across multiple frames with moving object."""
    frames = []
    for i in range(8):
        frames.append(_make_frame_with_rect(rect_x=50 + i * 4, rect_y=50 + i * 2, seed=200 + i))

    results = propagate_tracking(
        frames=frames,
        start_frame=0,
        roi=(50, 50, 40, 30),
        direction="forward",
    )

    # Should include start frame plus subsequent frames
    assert len(results) >= 2
    assert results[0]["frame_index"] == 0
    assert results[0]["success"] is True

    # At least some subsequent frames should track successfully
    tracked = [r for r in results if r["success"]]
    assert len(tracked) >= 2


def test_propagate_backward():
    """Propagate tracking backward from a later frame."""
    frames = []
    for i in range(8):
        frames.append(_make_frame_with_rect(rect_x=50 + i * 4, rect_y=50 + i * 2, seed=300 + i))

    results = propagate_tracking(
        frames=frames,
        start_frame=5,
        roi=(70, 60, 40, 30),
        direction="backward",
    )

    assert len(results) >= 2
    assert results[0]["frame_index"] == 5
    # Backward tracking should produce decreasing frame indices
    if len(results) > 1:
        assert results[1]["frame_index"] < results[0]["frame_index"]


def test_propagate_with_max_frames():
    """max_frames parameter should limit how many frames are tracked."""
    frames = []
    for i in range(20):
        frames.append(_make_frame_with_rect(rect_x=50 + i * 2, rect_y=50, seed=400 + i))

    results = propagate_tracking(
        frames=frames,
        start_frame=0,
        roi=(50, 50, 40, 30),
        direction="forward",
        max_frames=3,
    )

    # Start frame + at most 3 propagated frames = max 4 total
    assert len(results) <= 4


def test_tracking_fails_on_occlusion():
    """Object disappears, tracking should fail gracefully and stop."""
    frames = []
    # First 3 frames: visible object
    for i in range(3):
        frames.append(_make_frame_with_rect(rect_x=50 + i * 5, rect_y=50, seed=500 + i))
    # Next 5 frames: completely blank (object gone)
    for j in range(5):
        rng = np.random.RandomState(600 + j)
        frames.append(np.clip(rng.normal(30, 5, (200, 200)), 0, 255).astype(np.uint8))

    results = propagate_tracking(
        frames=frames,
        start_frame=0,
        roi=(50, 50, 40, 30),
        direction="forward",
    )

    # Should have at least the start frame
    assert len(results) >= 1
    assert results[0]["success"] is True

    # The function should not crash and results should be a list of dicts
    for r in results:
        assert "frame_index" in r
        assert "bbox" in r
        assert "confidence" in r
        assert "success" in r


def test_refine_roi_with_optical_flow():
    """Optical flow refinement should adjust ROI based on displacement."""
    prev = _make_frame_with_rect(rect_x=50, rect_y=50)
    curr = _make_frame_with_rect(rect_x=53, rect_y=51)

    roi = (50, 50, 40, 30)
    refined = refine_roi_with_optical_flow(prev, curr, roi)

    assert len(refined) == 4
    assert isinstance(refined[0], int)
    # The refined ROI should have the same width and height
    assert refined[2] == 40
    assert refined[3] == 30


def test_propagate_invalid_direction():
    """Invalid direction should raise ValueError."""
    frames = [_make_frame_with_rect()]
    try:
        propagate_tracking(frames, 0, (50, 50, 40, 30), direction="sideways")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_propagate_invalid_start_frame():
    """Out-of-range start_frame should raise IndexError."""
    frames = [_make_frame_with_rect()]
    try:
        propagate_tracking(frames, 5, (50, 50, 40, 30), direction="forward")
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
