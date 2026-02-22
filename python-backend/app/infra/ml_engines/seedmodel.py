"""
SeedModel Vessel Segmentation Engine

Seed point-guided nnU-Net segmentation for coronary arteries.
Trained on 2-channel input: grayscale image + Gaussian seed heatmap.

Architecture:
- nnU-Net 2D (8.46M params), base_channels=32, num_stages=5
- Input: 2-channel 512x512
  * Channel 0: Normalized grayscale image [0, 1]
  * Channel 1: Gaussian seed heatmap (sigma=10, peak=1.0) [0, 1]
- Output: Binary segmentation mask (sigmoid > 0.5)

Requirements:
- Minimum 1 seed point on the vessel
- Checkpoint with embedded config: ckpt["config"]
"""

import logging
from pathlib import Path

import numpy as np

from app.infra.ml_engines.base import BaseSegmentationEngine
from app.models.enums import SegmentationEngine

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    pass

_INPUT_SIZE = 512
_DEFAULT_SIGMA = 10.0


class SeedModelEngine(BaseSegmentationEngine):
    """SeedModel (nnU-Net + Gaussian seed heatmap) segmentation engine."""

    def __init__(self, model_dir: Path | None = None, *, device: str = "cpu"):
        self._model_dir = model_dir
        self._device_name = device
        self._model = None
        self._available = False
        self._model_loaded = False
        self._sigma = _DEFAULT_SIGMA
        self._check_availability()

    # -- BaseSegmentationEngine interface ------------------------------------

    @property
    def engine_type(self) -> SegmentationEngine:
        return SegmentationEngine.SEEDMODEL

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
            raise RuntimeError("SeedModel not available")

        # Without seed points, fall back to threshold
        if not seed_points or len(seed_points) < 1:
            logger.debug(
                "SeedModel requires >= 1 seed point, got %d, using threshold fallback",
                len(seed_points) if seed_points else 0,
            )
            return self._threshold_fallback(image, roi, seed_points)

        if not TORCH_AVAILABLE:
            logger.debug("torch not available, using threshold fallback")
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
                logger.warning("CUDA OOM during SeedModel inference, clearing cache")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            logger.error("SeedModel inference failed: %s", e)
            return self._threshold_fallback(image, roi, seed_points)

    def load_model(self) -> None:
        """Lazy-load the SeedModel from checkpoint."""
        if self._model_loaded:
            return

        if not TORCH_AVAILABLE:
            logger.warning("torch not installed -- SeedModel will use threshold fallback")
            self._model_loaded = True
            return

        weight_path = self._resolve_weight_path()
        if weight_path is None or not weight_path.exists():
            logger.warning("SeedModel weights not found at %s", weight_path)
            self._model_loaded = True
            return

        try:
            import torch

            from app.infra.ml_engines._seedmodel_arch import SeedSegmentor

            logger.info("Loading SeedModel from %s", weight_path)

            device = self._get_torch_device()
            checkpoint = torch.load(str(weight_path), map_location=device, weights_only=False)

            # Extract config from checkpoint
            config = checkpoint.get("config", {})
            model_cfg = config.get("model", {})
            seed_cfg = config.get("seed", {})
            self._sigma = seed_cfg.get("sigma_heatmap", _DEFAULT_SIGMA)

            # Build model from config
            nnunet_cfg = model_cfg.get("nnunet", {})
            in_channels = model_cfg.get("in_channels", 2)

            model = SeedSegmentor(
                in_channels=in_channels,
                nnunet_cfg=nnunet_cfg,
            )

            # Load state dict with _orig_mod. prefix stripping
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            cleaned = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                cleaned[new_key] = v

            # strict=False: checkpoint may have deep_supervision heads not needed for inference
            model.load_state_dict(cleaned, strict=False)
            model.to(device)
            model.eval()

            self._model = model
            self._torch_device = device
            self._model_loaded = True
            logger.info(
                "SeedModel loaded on %s (sigma=%.1f)", device, self._sigma
            )

        except Exception as e:
            logger.error("Failed to load SeedModel: %s", e)
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
        """Run full SeedModel inference pipeline."""
        import torch

        original_shape = image.shape[:2]

        # Prepare 2-channel input
        image_norm, heatmap, scale_x, scale_y = self._prepare_input(image, seed_points)

        # To tensors: (1, 1, 512, 512)
        image_t = torch.from_numpy(image_norm[None, None]).float().to(self._torch_device)
        seed_t = torch.from_numpy(heatmap[None, None]).float().to(self._torch_device)

        with torch.no_grad():
            logits = self._model(image_t, seed_t)

        # Post-process: sigmoid -> threshold
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
        binary_mask = (prob > 0.5).astype(np.uint8)

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

        # Scale mask values to 0/255 (codebase convention)
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
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Prepare normalized grayscale (512x512) + Gaussian heatmap (512x512).

        Returns (image_norm, heatmap, scale_x, scale_y).
        """
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
            try:
                import cv2

                gray = cv2.resize(
                    gray, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR
                )
            except ImportError:
                from scipy.ndimage import zoom

                gray = zoom(gray.astype(np.float64), (scale_y, scale_x), order=1).astype(gray.dtype)
        else:
            scale_x, scale_y = 1.0, 1.0

        # Normalize to [0, 1] float32
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

        # Build Gaussian heatmap from seed points
        heatmap = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
        yy, xx = np.mgrid[0:_INPUT_SIZE, 0:_INPUT_SIZE].astype(np.float32)

        for sx, sy in seed_points:
            # Scale seed to 512 coordinate space
            cx = sx * scale_x
            cy = sy * scale_y
            dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
            gauss = np.exp(-dist_sq / (2.0 * self._sigma ** 2))
            np.maximum(heatmap, gauss, out=heatmap)

        return gray, heatmap, scale_x, scale_y

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
            seedmodel_dir = self._model_dir / "seedmodel"
            if seedmodel_dir.exists():
                for f in seedmodel_dir.iterdir():
                    if f.suffix == ".pth":
                        return f

        # Fallback: derive from __file__
        backend_dir = Path(__file__).resolve().parent.parent.parent.parent
        path = backend_dir / "models" / "seedmodel"
        if path.exists():
            for f in path.iterdir():
                if f.suffix == ".pth":
                    return f

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
        """Threshold fallback when seeds are missing or model unavailable."""
        threshold = np.mean(image) * 0.7
        mask = (image < threshold).astype(np.uint8) * 255

        if roi is not None:
            x, y, w, h = roi
            roi_mask = np.zeros_like(mask)
            roi_mask[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
            mask = roi_mask

        return mask, 0.3
