"""
AngioPy Vessel Segmentation Engine

Seed-guided deep learning segmentation for coronary arteries.
Based on: https://gitlab.com/epfl-center-for-imaging/angiopy/angiopy-segmentation
Paper: https://doi.org/10.1016/j.ijcard.2024.132598

Architecture:
- U-Net with InceptionResNetV2 encoder (ImageNet pretrained)
- Input: 3-channel RGB (512x512)
  * Channel 0: Grayscale angiography image
  * Channel 1: Start/End seed points (first/last points as 4x4 squares)
  * Channel 2: Middle seed points (intermediate points as 4x4 squares)
- Output: Binary segmentation mask (2 classes: background, artery)

Requirements:
- 2-10 seed points along vessel
- Points ordered from proximal to distal
- Automatically resizes to 512x512 for inference
"""

import logging
import os
from pathlib import Path

import numpy as np

from app.infra.ml_engines.base import BaseSegmentationEngine
from app.models.enums import SegmentationEngine

logger = logging.getLogger(__name__)

# Lazy imports for optional heavy dependencies
TORCH_AVAILABLE = False
SMP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import segmentation_models_pytorch as smp  # noqa: F401

    SMP_AVAILABLE = True
except ImportError:
    pass

# Model configuration constants
_INPUT_SIZE = 512
_N_CLASSES = 2
_ENCODER_NAME = "inceptionresnetv2"
_ENCODER_WEIGHTS = "imagenet"
_WEIGHT_FILENAME = "modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth"


class AngioPyEngine(BaseSegmentationEngine):
    """AngioPy (U-Net + InceptionResNetV2) seed-guided segmentation engine.

    When torch + segmentation_models_pytorch are available, uses real model
    inference.  Otherwise falls back to simple threshold segmentation.
    """

    def __init__(self, model_dir: Path | None = None, *, device: str = "cpu"):
        self._model_dir = model_dir
        self._device_name = device
        self._model = None
        self._available = False
        self._model_loaded = False
        self._check_availability()

    # -- BaseSegmentationEngine interface ------------------------------------

    @property
    def engine_type(self) -> SegmentationEngine:
        return SegmentationEngine.ANGIOPY

    @property
    def is_available(self) -> bool:
        return self._available

    def segment(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Segment vessel using seed-point guidance. Returns (mask, confidence)."""
        if not self._available:
            raise RuntimeError("AngioPy model not available")

        # Without seed points, fall back to threshold
        if not seed_points or len(seed_points) < 2:
            logger.debug(
                "AngioPy requires >= 2 seed points, got %d, using threshold fallback",
                len(seed_points) if seed_points else 0,
            )
            return self._threshold_fallback(image, roi, seed_points)

        # Without torch/smp, also fall back
        if not (TORCH_AVAILABLE and SMP_AVAILABLE):
            logger.debug("torch/smp not available, using threshold fallback")
            return self._threshold_fallback(image, roi, seed_points)

        # Lazy-load model
        if not self._model_loaded:
            self.load_model()

        if self._model is None:
            return self._threshold_fallback(image, roi, seed_points)

        try:
            return self._predict(image, seed_points)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM during AngioPy inference, clearing cache")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            logger.error("AngioPy inference failed: %s", e)
            return self._threshold_fallback(image, roi, seed_points)

    def load_model(self) -> None:
        """Lazy-load the AngioPy U-Net model."""
        if self._model_loaded:
            return

        if not (TORCH_AVAILABLE and SMP_AVAILABLE):
            logger.warning("torch/smp not installed -- AngioPy will use threshold fallback")
            self._model_loaded = True
            return

        weight_path = self._resolve_weight_path()
        if weight_path is None or not weight_path.exists():
            logger.warning("AngioPy weights not found at %s", weight_path)
            self._model_loaded = True
            return

        try:
            import torch
            import torch.nn as nn
            import segmentation_models_pytorch as smp

            logger.info("Loading AngioPy model from %s", weight_path)

            model = smp.Unet(
                encoder_name=_ENCODER_NAME,
                encoder_weights=_ENCODER_WEIGHTS,
                in_channels=3,
                classes=_N_CLASSES,
            )
            model = nn.DataParallel(model)

            device = self._get_torch_device()
            model.to(device=device)

            checkpoint = torch.load(str(weight_path), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint)
            model.eval()

            self._model = model
            self._torch_device = device
            self._model_loaded = True
            logger.info("AngioPy model loaded on %s", device)

        except Exception as e:
            logger.error("Failed to load AngioPy model: %s", e)
            self._model = None
            self._model_loaded = True  # prevent repeated attempts

    def unload_model(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._model_loaded = False
        if TORCH_AVAILABLE:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- Real inference -------------------------------------------------------

    def _predict(
        self, image: np.ndarray, seed_points: list[tuple[int, int]]
    ) -> tuple[np.ndarray, float]:
        """Run full AngioPy inference pipeline."""
        import torch

        # Handle seed point count edge cases (same as v1.1)
        pts = list(seed_points)
        if len(pts) == 2:
            pts.append(pts[-1])  # duplicate last for better results
        if len(pts) > 10:
            pts = pts[:10]

        original_shape = image.shape[:2]

        # Prepare 3-channel input
        rgb_input, scale_x, scale_y = self._prepare_input(image, pts)

        # Normalise to [0, 1], transpose to (C, H, W), add batch dim
        img_array = rgb_input.astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # (3, 512, 512)
        tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, 3, 512, 512)
        tensor = tensor.to(device=self._torch_device)

        with torch.no_grad():
            output = self._model(tensor)

        # Post-process
        probs = torch.softmax(output, dim=1)
        artery_prob = probs[0, 1].cpu().numpy()  # (512, 512)
        binary_mask = (artery_prob > 0.5).astype(np.uint8)

        # Resize back to original dimensions
        if binary_mask.shape != original_shape:
            try:
                import cv2

                binary_mask = cv2.resize(
                    binary_mask,
                    (original_shape[1], original_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            except ImportError:
                from scipy.ndimage import zoom

                zy = original_shape[0] / binary_mask.shape[0]
                zx = original_shape[1] / binary_mask.shape[1]
                binary_mask = (zoom(binary_mask.astype(np.float32), (zy, zx), order=0) > 0.5).astype(np.uint8)

        # Scale mask values to 0/255 to match v2 convention
        mask_255 = binary_mask * 255

        # Confidence heuristic
        coverage = np.sum(binary_mask > 0) / binary_mask.size
        if 0.005 <= coverage <= 0.15:
            confidence = 0.9
        elif 0.001 <= coverage <= 0.25:
            confidence = 0.75
        else:
            confidence = 0.5

        return mask_255, confidence

    def _prepare_input(
        self, image: np.ndarray, seed_points: list[tuple[int, int]]
    ) -> tuple[np.ndarray, float, float]:
        """Prepare 3-channel (512x512) input with seed point encoding.

        Returns (rgb_array, scale_x, scale_y).
        """
        import scipy.ndimage

        # Ensure grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                try:
                    import cv2

                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                except ImportError:
                    gray = np.mean(image, axis=2).astype(image.dtype)
            else:
                gray = image[:, :, 0]
        else:
            gray = image

        h, w = gray.shape[:2]

        # Resize to 512x512
        if gray.shape != (_INPUT_SIZE, _INPUT_SIZE):
            scale_y = _INPUT_SIZE / h
            scale_x = _INPUT_SIZE / w
            gray = scipy.ndimage.zoom(gray.astype(np.float64), (scale_y, scale_x), order=1).astype(gray.dtype)
        else:
            scale_x, scale_y = 1.0, 1.0

        # Normalise to uint8
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Create RGB (512, 512, 3)
        rgb = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        rgb[:, :, 0] = gray

        # Scale seed points
        scaled = [(x * scale_x, y * scale_y) for x, y in seed_points]

        if len(scaled) >= 2:
            start = scaled[0]
            end = scaled[-1]
            # Channel 1: start + end as 4x4 white squares
            for px, py in [start, end]:
                ix, iy = int(round(px)), int(round(py))
                y_lo = max(0, iy - 2)
                y_hi = min(_INPUT_SIZE, iy + 2)
                x_lo = max(0, ix - 2)
                x_hi = min(_INPUT_SIZE, ix + 2)
                rgb[y_lo:y_hi, x_lo:x_hi, 1] = 255

            # Channel 2: middle points
            if len(scaled) > 2:
                for px, py in scaled[1:-1]:
                    ix, iy = int(round(px)), int(round(py))
                    y_lo = max(0, iy - 2)
                    y_hi = min(_INPUT_SIZE, iy + 2)
                    x_lo = max(0, ix - 2)
                    x_hi = min(_INPUT_SIZE, ix + 2)
                    rgb[y_lo:y_hi, x_lo:x_hi, 2] = 255

        return rgb, scale_x, scale_y

    # -- Helpers ---------------------------------------------------------------

    def _check_availability(self) -> None:
        """Check whether model weight file exists on disk."""
        if self._model_dir is None:
            self._available = False
            return
        weight_path = self._resolve_weight_path()
        self._available = weight_path is not None and weight_path.exists()

    def _resolve_weight_path(self) -> Path | None:
        if self._model_dir is not None:
            path = self._model_dir / "angiopy" / _WEIGHT_FILENAME
            if path.exists():
                return path
            # Also check for any .pth file in the angiopy dir
            angiopy_dir = self._model_dir / "angiopy"
            if angiopy_dir.exists():
                for f in angiopy_dir.iterdir():
                    if f.suffix == ".pth":
                        return f

        # Fallback: derive from __file__
        backend_dir = Path(__file__).resolve().parent.parent.parent.parent
        path = backend_dir / "models" / "angiopy" / _WEIGHT_FILENAME
        if path.exists():
            return path

        return None

    def _get_torch_device(self):
        import torch

        if self._device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self._device_name == "mps" and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return torch.device("mps")
        if self._device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    # -- Threshold fallback ----------------------------------------------------

    def _threshold_fallback(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Threshold fallback with optional seed-based region growing."""
        threshold = np.mean(image) * 0.7
        mask = (image < threshold).astype(np.uint8) * 255

        if roi is not None:
            x, y, w, h = roi
            roi_mask = np.zeros_like(mask)
            roi_mask[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
            mask = roi_mask

        return mask, 0.3
