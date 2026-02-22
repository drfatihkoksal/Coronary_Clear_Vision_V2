"""Vessel tracking across angiographic frames using template matching.

Uses normalized cross-correlation (NCC) for ROI-based vessel tracking
with Farneback optical flow refinement for sub-pixel accuracy.

Template matching is ideal for fixed-size ROI tracking with small
frame-to-frame displacements (coronary angiograms).
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# NCC confidence below which tracking is considered lost
_MIN_CONFIDENCE = 0.25

# How many pixels beyond ROI to search in the next frame
_SEARCH_MARGIN = 50


def _ensure_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert frame to grayscale if needed."""
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is 3-channel."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _track_template(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    roi: tuple[int, int, int, int],
    search_margin: int = _SEARCH_MARGIN,
) -> tuple[bool, tuple[int, int, int, int], float]:
    """Track ROI from prev_frame to curr_frame using template matching.

    Extracts the ROI region from prev_frame as a template and searches
    for the best match in an expanded region of curr_frame using
    normalized cross-correlation (TM_CCOEFF_NORMED).

    Args:
        prev_frame: Previous frame (grayscale or BGR).
        curr_frame: Current frame (grayscale or BGR).
        roi: Bounding box (x, y, w, h) in prev_frame.
        search_margin: Pixels to expand search region beyond ROI.

    Returns:
        (success, new_bbox, confidence) where confidence is the NCC
        peak value in [0, 1].
    """
    prev_gray = _ensure_grayscale(prev_frame)
    curr_gray = _ensure_grayscale(curr_frame)

    x, y, w, h = roi
    img_h, img_w = prev_gray.shape[:2]

    # Extract template from previous frame (clamped to bounds)
    tx1, ty1 = max(0, x), max(0, y)
    tx2, ty2 = min(img_w, x + w), min(img_h, y + h)

    if tx2 <= tx1 or ty2 <= ty1:
        return False, (0, 0, 0, 0), 0.0

    template = prev_gray[ty1:ty2, tx1:tx2]

    # Search region: expand around expected position
    sx1 = max(0, tx1 - search_margin)
    sy1 = max(0, ty1 - search_margin)
    sx2 = min(img_w, tx2 + search_margin)
    sy2 = min(img_h, ty2 + search_margin)

    search_region = curr_gray[sy1:sy2, sx1:sx2]

    # Template must fit inside search region
    if template.shape[0] > search_region.shape[0] or template.shape[1] > search_region.shape[1]:
        return False, roi, 0.0

    # Normalized cross-correlation
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Convert match location back to image coordinates
    new_x = sx1 + max_loc[0]
    new_y = sy1 + max_loc[1]

    # Clamp to image bounds
    new_x = max(0, min(new_x, img_w - w))
    new_y = max(0, min(new_y, img_h - h))

    confidence = float(max(0.0, max_val))
    success = confidence >= _MIN_CONFIDENCE

    return success, (new_x, new_y, w, h), confidence


class TemplateTracker:
    """Stateful wrapper around template matching for OpenCV Tracker-like API."""

    def __init__(self) -> None:
        self._prev_frame: np.ndarray | None = None
        self._roi: tuple[int, int, int, int] | None = None

    def init(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> None:
        self._prev_frame = _ensure_grayscale(frame)
        self._roi = roi

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int], float]:
        if self._prev_frame is None or self._roi is None:
            return False, (0, 0, 0, 0), 0.0

        curr_gray = _ensure_grayscale(frame)
        success, bbox, confidence = _track_template(
            self._prev_frame, curr_gray, self._roi,
        )

        if success:
            # Update state for next frame
            self._prev_frame = curr_gray
            self._roi = bbox

        return success, bbox, confidence


def initialize_tracker(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
) -> TemplateTracker:
    """Initialize template-matching tracker on a frame with given ROI.

    Args:
        frame: Grayscale or BGR frame (HxW or HxWx3, uint8).
        roi: Bounding box as (x, y, width, height).

    Returns:
        Initialized TemplateTracker object.
    """
    tracker = TemplateTracker()
    tracker.init(_ensure_bgr(frame), roi)
    return tracker


def track_frame(
    tracker: TemplateTracker,
    frame: np.ndarray,
) -> tuple[bool, tuple[int, int, int, int], float]:
    """Track object in next frame.

    Args:
        tracker: Previously initialized TemplateTracker.
        frame: Next video frame (grayscale or BGR).

    Returns:
        Tuple of (success, bbox, confidence) where bbox is (x, y, w, h)
        and confidence is in [0, 1].
    """
    bgr = _ensure_bgr(frame)
    return tracker.update(bgr)


def propagate_tracking(
    frames: list[np.ndarray],
    start_frame: int,
    roi: tuple[int, int, int, int],
    direction: str = "forward",
    max_frames: int | None = None,
) -> list[dict]:
    """Propagate tracking across multiple frames using template matching.

    Args:
        frames: List of video frames (grayscale or BGR).
        start_frame: Index of the frame where ROI is defined.
        roi: Initial bounding box (x, y, w, h).
        direction: "forward" or "backward".
        max_frames: Maximum number of frames to track (None = all).

    Returns:
        List of dicts with keys: frame_index, bbox, confidence, success.
        Stops if tracking fails (confidence too low).
    """
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
    if start_frame < 0 or start_frame >= len(frames):
        raise IndexError(f"start_frame {start_frame} out of range [0, {len(frames)})")

    results: list[dict] = []

    # Include the start frame itself
    results.append({
        "frame_index": start_frame,
        "bbox": roi,
        "confidence": 1.0,
        "success": True,
    })

    # Build the frame indices to iterate
    if direction == "forward":
        indices = range(start_frame + 1, len(frames))
    else:
        indices = range(start_frame - 1, -1, -1)

    if max_frames is not None:
        indices = list(indices)[:max_frames]

    tracker = TemplateTracker()
    tracker.init(frames[start_frame], roi)

    for idx in indices:
        success, bbox, confidence = tracker.update(frames[idx])

        if not success or confidence < _MIN_CONFIDENCE:
            results.append({
                "frame_index": idx,
                "bbox": bbox,
                "confidence": confidence,
                "success": False,
            })
            break

        results.append({
            "frame_index": idx,
            "bbox": bbox,
            "confidence": confidence,
            "success": True,
        })

    return results


def refine_roi_with_optical_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Refine tracked ROI using Farneback optical flow for sub-pixel correction.

    Computes dense optical flow within the ROI region and adjusts the
    bounding box by the median displacement vector.

    Args:
        prev_frame: Previous frame (grayscale, HxW uint8).
        curr_frame: Current frame (grayscale, HxW uint8).
        roi: Current bounding box (x, y, w, h).

    Returns:
        Refined bounding box (x, y, w, h).
    """
    prev_gray = _ensure_grayscale(prev_frame)
    curr_gray = _ensure_grayscale(curr_frame)

    x, y, w, h = roi
    img_h, img_w = prev_gray.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    if x2 <= x1 or y2 <= y1:
        return roi

    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]

    flow = cv2.calcOpticalFlowFarneback(
        prev_roi,
        curr_roi,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    dx = float(np.median(flow[..., 0]))
    dy = float(np.median(flow[..., 1]))

    new_x = int(round(x + dx))
    new_y = int(round(y + dy))

    # Clamp to image bounds
    new_x = max(0, min(new_x, img_w - w))
    new_y = max(0, min(new_y, img_h - h))

    return (new_x, new_y, w, h)
