"""Mask editing operations: brush, eraser, smart brush, flood fill, morphological ops."""
import numpy as np
import cv2


def apply_brush(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    radius: int = 5,
    value: int = 255,
) -> np.ndarray:
    """Apply brush stroke to mask.

    Args:
        mask: Binary mask (HxW uint8)
        points: List of (x, y) brush positions
        radius: Brush radius in pixels
        value: 255 for paint, 0 for erase

    Returns:
        Modified mask copy
    """
    result = mask.copy()
    for x, y in points:
        cv2.circle(result, (int(x), int(y)), radius, int(value), -1)
    return result


def apply_smart_brush(
    mask: np.ndarray,
    image: np.ndarray,
    points: list[tuple[int, int]],
    radius: int = 5,
    tolerance: int = 30,
    value: int = 255,
) -> np.ndarray:
    """Smart brush that only paints on pixels similar to the seed point intensity.

    Args:
        mask: Binary mask (HxW uint8)
        image: Grayscale image for intensity comparison
        points: Brush positions
        radius: Brush radius
        tolerance: Intensity tolerance for similarity
        value: Paint value

    Returns:
        Modified mask copy
    """
    result = mask.copy()

    for x, y in points:
        x, y = int(x), int(y)
        if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
            continue

        ref_intensity = int(image[y, x])

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx**2 + dy**2 > radius**2:
                    continue
                py, px = y + dy, x + dx
                if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                    if abs(int(image[py, px]) - ref_intensity) <= tolerance:
                        result[py, px] = value

    return result


def apply_flood_fill(
    mask: np.ndarray,
    seed_point: tuple[int, int],
    value: int = 255,
    tolerance: int = 0,
) -> np.ndarray:
    """Flood fill from seed point on mask.

    Args:
        mask: Binary mask
        seed_point: (x, y) seed
        value: Fill value
        tolerance: Currently unused for binary masks

    Returns:
        Modified mask copy
    """
    result = mask.copy()
    x, y = int(seed_point[0]), int(seed_point[1])

    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return result

    h, w = mask.shape
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(result, fill_mask, (x, y), int(value))

    return result


def apply_morphological_op(
    mask: np.ndarray,
    operation: str,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operation to mask.

    Args:
        mask: Binary mask
        operation: "dilate", "erode", "open", "close"
        kernel_size: Structuring element size
        iterations: Number of iterations

    Returns:
        Modified mask copy
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if operation == "dilate":
        return cv2.dilate(mask, kernel, iterations=iterations)
    elif operation == "erode":
        return cv2.erode(mask, kernel, iterations=iterations)
    elif operation == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")
