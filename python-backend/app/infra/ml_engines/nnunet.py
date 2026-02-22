"""
nnU-Net Segmentation Engine

Adapter for nnU-Net v2 model for coronary artery segmentation.
Supports dual-channel (image + Gaussian spatial attention) and single-channel modes.

Architecture:
- Input: 1-channel (grayscale) or 2-channel (grayscale + Gaussian spatial map)
- Output: Binary segmentation mask
- Post-processing: CenterComponentKeeper (bifurcation suppression)

Datasets:
- Dataset503: 160x160, dual-channel, sigma=26 (default, Dice=0.9605)
- Dataset505: 192x192, dual-channel, sigma=40 (wide ROI)
- Dataset507: 192x192, dual-channel, sigma=40 (merged)
- Dataset600: 512x512, full-frame (no ROI crop)

References:
- Isensee et al., "nnU-Net", Nature Methods 2021
- Shit et al., "clDice", CVPR 2021
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.infra.ml_engines.base import BaseSegmentationEngine
from app.models.enums import SegmentationEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIGS: dict[int, dict[str, Any]] = {
    503: {
        "name": "Dataset503_Coronary150GaussianDualChannel",
        "trainer": "nnUNetTrainer",
        "dual_channel": True,
        "sigma": 26,
        "size": 160,
    },
    505: {
        "name": "Dataset505_CoronaryROICrop",
        "trainer": "nnUNetTrainer",
        "dual_channel": True,
        "sigma": 40,
        "size": 192,
    },
    507: {
        "name": "Dataset507_CoronaryMerged",
        "trainer": "nnUNetTrainer",
        "dual_channel": True,
        "sigma": 40,
        "size": 192,
    },
    600: {
        "name": "Dataset600_CoronaryROI",
        "trainer": "nnUNetTrainer",
        "dual_channel": False,
        "sigma": None,
        "size": 512,
    },
}

# Variant -> dataset id mapping
VARIANT_DATASET: dict[str, int] = {
    "roi": 503,
    "wide": 505,
    "fullframe": 600,
}


# ---------------------------------------------------------------------------
# Post-processor
# ---------------------------------------------------------------------------
class CenterComponentKeeper:
    """Keep only the connected component closest to the image centre.

    Removes bifurcations and side branches that the model may predict.
    """

    def __init__(self, center_tolerance_radius: int = 10):
        self.center_tolerance_radius = center_tolerance_radius

    def process(self, mask: np.ndarray) -> np.ndarray:
        try:
            from scipy import ndimage

            labeled, num_features = ndimage.label(mask)
            if num_features <= 1:
                return mask

            h, w = mask.shape
            center_y, center_x = h // 2, w // 2

            best_component = 0
            min_distance = float("inf")

            for label_id in range(1, num_features + 1):
                component = labeled == label_id
                coords = np.column_stack(np.where(component))
                if len(coords) == 0:
                    continue
                distances = np.sqrt(
                    (coords[:, 0] - center_y) ** 2 + (coords[:, 1] - center_x) ** 2
                )
                closest_distance = float(np.min(distances))
                if closest_distance < min_distance:
                    min_distance = closest_distance
                    best_component = label_id

            if best_component > 0:
                return (labeled == best_component).astype(np.uint8)
            return mask

        except ImportError:
            logger.warning("scipy not available for connected component filtering")
            return mask
        except Exception as e:
            logger.warning("Center component filtering failed: %s", e)
            return mask


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class NNUNetEngine(BaseSegmentationEngine):
    """nnU-Net v2 segmentation engine. Supports roi, wide, and fullframe variants.

    When nnunetv2 + torch are available, uses real model inference.
    Otherwise falls back to simple threshold segmentation.
    """

    DEFAULT_CONFIGURATION = "2d"
    ALL_FOLDS = (0, 1, 2, 3, 4)

    def __init__(
        self,
        model_dir: Path | None = None,
        variant: str = "roi",
        *,
        use_ensemble: bool = True,
        enable_bifurcation_suppression: bool = True,
        device: str = "cpu",
    ):
        self._model_dir = model_dir
        self._variant = variant
        self._device_name = device
        self._use_ensemble = use_ensemble
        self._enable_bifurcation_suppression = enable_bifurcation_suppression

        self._predictor = None
        self._available = False
        self._model_loaded = False

        # Resolve dataset config from variant
        self._dataset_id = VARIANT_DATASET.get(variant, 503)
        self._ds_config = DATASET_CONFIGS.get(self._dataset_id, DATASET_CONFIGS[503])
        self._use_dual_channel: bool = self._ds_config.get("dual_channel", False)
        self._sigma: int | None = self._ds_config.get("sigma")
        self._target_size: int = self._ds_config.get("size", 160)
        self._trainer: str = self._ds_config.get("trainer", "nnUNetTrainer")

        # Post-processor
        self._post_processor = (
            CenterComponentKeeper(center_tolerance_radius=10)
            if enable_bifurcation_suppression
            else None
        )

        self._check_availability()
        logger.info(
            "NNUNetEngine(%s) init: dataset=%d, dual=%s, sigma=%s, size=%d, available=%s",
            variant,
            self._dataset_id,
            self._use_dual_channel,
            self._sigma,
            self._target_size,
            self._available,
        )

    # -- BaseSegmentationEngine interface ------------------------------------

    @property
    def engine_type(self) -> SegmentationEngine:
        variants = {
            "roi": SegmentationEngine.NNUNET,
            "wide": SegmentationEngine.NNUNET_WIDE,
            "fullframe": SegmentationEngine.NNUNET_FULLFRAME,
        }
        return variants.get(self._variant, SegmentationEngine.NNUNET)

    @property
    def is_available(self) -> bool:
        return self._available

    def segment(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
        seed_points: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, float, None]:
        """Segment vessel in image. Returns (mask, confidence, None).

        nnU-Net predictor returns binary masks; probability maps are not
        exposed through its high-level API. Returns None for prob_map.
        """
        if not self._available:
            raise RuntimeError(f"nnU-Net {self._variant} model not available")

        # Lazy-load the model on first inference
        if not self._model_loaded:
            self.load_model()

        # If the model failed to load (nnunetv2 not installed), use threshold
        if self._predictor is None:
            return self._threshold_fallback(image, roi)

        try:
            if roi is not None:
                return self._predict_with_roi(image, roi)
            else:
                return self._predict_direct(image)
        except RuntimeError as e:
            # Handle CUDA OOM
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM during nnU-Net inference, clearing cache")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            logger.error("nnU-Net inference failed: %s", e)
            return self._threshold_fallback(image, roi)

    def load_model(self) -> None:
        """Lazy-load the nnU-Net predictor."""
        if self._model_loaded:
            return

        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError:
            logger.warning(
                "nnunetv2 not installed -- NNUNetEngine will use threshold fallback"
            )
            self._model_loaded = True  # prevent repeated attempts
            return

        model_folder = self._resolve_model_folder()
        if model_folder is None:
            logger.warning("nnU-Net model folder not found, using threshold fallback")
            self._model_loaded = True
            return

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,  # TTA disabled for speed
            perform_everything_on_device=True,
            device=self._get_torch_device(),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        folds = self._find_available_folds(model_folder)
        checkpoint_name = self._find_checkpoint(model_folder, folds)
        logger.info("Loading nnU-Net folds=%s checkpoint=%s from %s", folds, checkpoint_name, model_folder)

        try:
            self._predictor.initialize_from_trained_model_folder(
                model_folder,
                use_folds=tuple(folds),
                checkpoint_name=checkpoint_name,
            )
            self._model_loaded = True
            logger.info("nnU-Net %s model loaded", self._variant)
        except Exception as e:
            logger.error("Failed to load nnU-Net model: %s", e)
            self._predictor = None
            self._model_loaded = True  # prevent repeated attempts

    def unload_model(self) -> None:
        """Release model from memory."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        self._model_loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- Real inference -------------------------------------------------------

    def _predict_direct(self, image: np.ndarray) -> tuple[np.ndarray, float, None]:
        """Predict without explicit ROI.

        For roi/wide variants the image is center-cropped to target_size
        (the model was trained on small crops, not full frames).
        For fullframe variant the full image is passed through.
        """
        original_shape = image.shape[:2]
        target_size = self._target_size

        # To grayscale float [0, 1]
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.float32)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        needs_crop = self._variant != "fullframe" and (
            image.shape[0] > target_size or image.shape[1] > target_size
        )

        if needs_crop:
            # Center-crop to target_size (same as reference project)
            center_x = image.shape[1] // 2
            center_y = image.shape[0] // 2
            half = target_size // 2
            x1 = max(0, center_x - half)
            y1 = max(0, center_y - half)
            x2 = min(image.shape[1], x1 + target_size)
            y2 = min(image.shape[0], y1 + target_size)
            if x2 - x1 < target_size:
                x1 = max(0, x2 - target_size)
            if y2 - y1 < target_size:
                y1 = max(0, y2 - target_size)
            crop = image[y1:y2, x1:x2]
        else:
            crop = image
            x1, y1 = 0, 0

        # Prepare channel input
        if self._use_dual_channel:
            spatial_map = self._generate_gaussian_map(crop.shape)
            preprocessed = np.stack([crop, spatial_map], axis=0)
        else:
            preprocessed = crop[np.newaxis, :, :]

        input_array = preprocessed[:, np.newaxis, :, :].astype(np.float32)

        properties = {
            "spacing": [999.0, 1.0, 1.0],
            "shape_after_cropping_and_before_resampling": input_array.shape[1:],
        }
        result = self._predictor.predict_single_npy_array(
            input_array, properties, None, None, save_or_return_probabilities=False,
        )
        mask = self._parse_prediction(result)

        if self._post_processor is not None:
            mask = self._post_processor.process(mask)

        # Paste back into full-size mask if we cropped
        if needs_crop:
            full_mask = np.zeros(original_shape, dtype=np.uint8)
            mh, mw = mask.shape
            paste_y2 = min(y1 + mh, original_shape[0])
            paste_x2 = min(x1 + mw, original_shape[1])
            ha = paste_y2 - y1
            wa = paste_x2 - x1
            full_mask[y1:paste_y2, x1:paste_x2] = mask[:ha, :wa]
            mask = full_mask

        # Scale to 0/255 convention (codebase-wide standard)
        mask = mask * 255

        confidence = self._calculate_confidence(mask)
        return mask, confidence, None

    def _predict_with_roi(
        self, image: np.ndarray, roi: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float, None]:
        """ROI-based prediction: crop -> inference -> restore to original coordinates."""
        original_shape = image.shape[:2]
        target_size = self._target_size

        # To grayscale float [0, 1]
        if len(image.shape) == 3:
            image = np.mean(image, axis=2).astype(np.float32)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Crop centred on ROI
        x, y, w, h = roi
        center_x = x + w // 2
        center_y = y + h // 2
        half = target_size // 2

        x1 = max(0, center_x - half)
        y1 = max(0, center_y - half)
        x2 = min(image.shape[1], x1 + target_size)
        y2 = min(image.shape[0], y1 + target_size)
        if x2 - x1 < target_size:
            x1 = max(0, x2 - target_size)
        if y2 - y1 < target_size:
            y1 = max(0, y2 - target_size)

        crop = image[y1:y2, x1:x2]

        # Prepare channel input
        if self._use_dual_channel:
            spatial_map = self._generate_gaussian_map(crop.shape)
            preprocessed = np.stack([crop, spatial_map], axis=0)
        else:
            preprocessed = crop[np.newaxis, :, :]

        input_array = preprocessed[:, np.newaxis, :, :].astype(np.float32)
        properties = {
            "spacing": [999.0, 1.0, 1.0],
            "shape_after_cropping_and_before_resampling": input_array.shape[1:],
        }

        result = self._predictor.predict_single_npy_array(
            input_array, properties, None, None, save_or_return_probabilities=False,
        )
        mask = self._parse_prediction(result)

        if self._post_processor is not None:
            mask = self._post_processor.process(mask)

        # Paste back into full-size mask
        full_mask = np.zeros(original_shape, dtype=np.uint8)
        mh, mw = mask.shape
        paste_y2 = min(y1 + mh, original_shape[0])
        paste_x2 = min(x1 + mw, original_shape[1])
        ha = paste_y2 - y1
        wa = paste_x2 - x1
        full_mask[y1:paste_y2, x1:paste_x2] = mask[:ha, :wa]

        # Scale to 0/255 convention (codebase-wide standard)
        full_mask = full_mask * 255

        confidence = self._calculate_confidence(full_mask)
        return full_mask, confidence, None

    # -- Preprocessing / postprocessing helpers --------------------------------

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Return (C, H, W) float32 tensor."""
        if len(image.shape) == 3:
            image = (
                np.mean(image, axis=2) if image.shape[2] == 3 else image[:, :, 0]
            )
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        if self._use_dual_channel:
            spatial_map = self._generate_gaussian_map(image.shape)
            return np.stack([image, spatial_map], axis=0)
        return image[np.newaxis, :, :]

    def _parse_prediction(self, result: Any) -> np.ndarray:
        """Extract binary mask from nnU-Net prediction output.

        Returns mask with values 0/1 (scaled to 0/255 after post-processing).
        """
        prediction = result[0] if isinstance(result, tuple) else result

        # Remove Z dim if present
        if len(prediction.shape) == 4:
            prediction = prediction[:, 0, :, :]

        if len(prediction.shape) == 3 and prediction.shape[0] > 1:
            mask = (prediction[1] > 0.5).astype(np.uint8)
        else:
            prob = prediction[0] if len(prediction.shape) == 3 else prediction
            mask = (prob > 0.5).astype(np.uint8)

        return mask

    def _generate_gaussian_map(self, shape: tuple[int, int]) -> np.ndarray:
        """Generate centred Gaussian spatial attention map normalised to [0, 1]."""
        if self._sigma is None or self._sigma <= 0:
            return np.ones(shape, dtype=np.float32)

        h, w = shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * self._sigma ** 2))

        g_min, g_max = gaussian.min(), gaussian.max()
        if g_max - g_min > 0:
            gaussian = (gaussian - g_min) / (g_max - g_min)
        return gaussian.astype(np.float32)

    def _calculate_confidence(self, mask: np.ndarray) -> float:
        """Heuristic confidence based on segmentation coverage."""
        total = mask.size
        segmented = np.sum(mask > 0)
        coverage = segmented / total

        if coverage < 0.001 or coverage > 0.3:
            return 0.5
        if 0.01 <= coverage <= 0.15:
            return 0.9
        if 0.005 <= coverage <= 0.2:
            return 0.8
        return 0.7

    # -- Model resolution helpers ----------------------------------------------

    def _check_availability(self) -> None:
        """Check whether model files actually exist on disk."""
        if self._model_dir is None:
            self._available = False
            return

        nnunet_dir = self._model_dir / "nnunet"
        if not nnunet_dir.exists():
            self._available = False
            return

        # Check that the dataset folder and at least one fold with a checkpoint exist
        ds_name = self._ds_config.get("name", "")
        trainer_folder = f"{self._trainer}__nnUNetPlans__{self.DEFAULT_CONFIGURATION}"
        model_folder = nnunet_dir / ds_name / trainer_folder

        if not model_folder.exists():
            self._available = False
            return

        # At least one fold must have a checkpoint
        for fold in self.ALL_FOLDS:
            fold_dir = model_folder / f"fold_{fold}"
            if fold_dir.exists():
                for ckpt in ("checkpoint_best.pth", "checkpoint_latest.pth", "checkpoint_final.pth"):
                    if (fold_dir / ckpt).exists():
                        self._available = True
                        return

        self._available = False

    def _resolve_model_folder(self) -> str | None:
        """Return the absolute model folder path or None if not found."""
        if self._model_dir is not None:
            base = self._model_dir / "nnunet"
        elif os.environ.get("nnUNet_results"):
            base = Path(os.environ["nnUNet_results"])
        elif os.environ.get("MODEL_PATH"):
            base = Path(os.environ["MODEL_PATH"]) / "nnunet"
        else:
            # Derive from __file__ -> python-backend/models/nnunet
            backend_dir = Path(__file__).resolve().parent.parent.parent.parent
            base = backend_dir / "models" / "nnunet"

        ds_name = self._ds_config.get("name", "")
        trainer_folder = f"{self._trainer}__nnUNetPlans__{self.DEFAULT_CONFIGURATION}"
        model_folder = base / ds_name / trainer_folder

        if model_folder.exists():
            return str(model_folder)

        if base.exists():
            return str(base)

        return None

    def _find_available_folds(self, model_folder: str) -> list[int]:
        if not self._use_ensemble:
            return [0]
        available: list[int] = []
        for fold in self.ALL_FOLDS:
            if os.path.exists(os.path.join(model_folder, f"fold_{fold}")):
                available.append(fold)
        return available or [0]

    def _find_checkpoint(self, model_folder: str, folds: list[int]) -> str:
        for fold in folds:
            fold_dir = os.path.join(model_folder, f"fold_{fold}")
            for name in ("checkpoint_best.pth", "checkpoint_latest.pth", "checkpoint_final.pth"):
                if os.path.exists(os.path.join(fold_dir, name)):
                    return name
        return "checkpoint_best.pth"

    def _get_torch_device(self):
        try:
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
        except ImportError:
            return "cpu"

    # -- Threshold fallback (used when torch / nnunetv2 not installed) ---------

    def _threshold_fallback(
        self, image: np.ndarray, roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, float, None]:
        """Simple threshold-based segmentation as fallback."""
        if roi is not None:
            x, y, w, h = roi
            region = image[y : y + h, x : x + w]
        else:
            region = image

        threshold = np.mean(region) * 0.7
        mask = np.zeros_like(image, dtype=np.uint8)

        if roi is not None:
            x, y, w, h = roi
            mask[y : y + h, x : x + w] = (region < threshold).astype(np.uint8) * 255
        else:
            mask = (image < threshold).astype(np.uint8) * 255

        return mask, 0.3, None
