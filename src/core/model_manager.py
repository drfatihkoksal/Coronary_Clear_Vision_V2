"""
Singleton Model Manager for AngioPy segmentation model.
Ensures the model is loaded only once and shared across the application.
Implements lazy loading to improve startup time.
"""

import threading
import logging
from typing import Optional, Callable, Dict, Any
from ..analysis.angiopy_segmentation import AngioPySegmentation
from ..config.app_config import get_config

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class to manage AngioPy model instance."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config = get_config()
        self._segmentation_model: Optional[AngioPySegmentation] = None
        self._model_lock = threading.Lock()
        self._initialized = True
        self._model_loaded = False
        self._loading_thread: Optional[threading.Thread] = None
        self._load_on_demand = self._config.performance.lazy_load_models
        self._model_cache: Dict[str, Any] = {}
        self._preload_requested = False

    def get_segmentation_model(self, auto_download: bool = True) -> AngioPySegmentation:
        """
        Get the AngioPy segmentation model instance with lazy loading.
        Creates it only when actually needed if lazy loading is enabled.

        Args:
            auto_download: Whether to automatically download the model if not found

        Returns:
            AngioPySegmentation instance
        """
        if self._segmentation_model is None:
            with self._model_lock:
                if self._segmentation_model is None:
                    # Log lazy loading
                    if self._load_on_demand:
                        logger.info("Lazy loading AngioPy model on first use...")
                    else:
                        logger.info("Loading AngioPy model...")

                    # Create model instance
                    self._segmentation_model = AngioPySegmentation(auto_download=auto_download)
                    self._model_loaded = True
                    logger.info("AngioPy model loaded successfully")

        # Wait for preloading to complete if in progress
        # But don't try to join if we're the loading thread itself
        if self._loading_thread and self._loading_thread.is_alive():
            current_thread = threading.current_thread()
            if current_thread != self._loading_thread:
                logger.debug("Waiting for model preloading to complete...")
                self._loading_thread.join()

        return self._segmentation_model

    def reset_model(self):
        """Reset the model instance. Useful for testing or reloading."""
        with self._model_lock:
            self._segmentation_model = None
            self._model_loaded = False

    def preload_model_async(
        self,
        auto_download: bool = True,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ):
        """
        Preload the model asynchronously in background.
        Only preloads if lazy loading is disabled or explicitly requested.
        """
        # Skip preloading if lazy loading is enabled and not explicitly requested
        if self._load_on_demand and not self._preload_requested:
            logger.debug("Lazy loading enabled - skipping model preload")
            return

        def _load_model():
            try:
                # Create model instance
                model = self.get_segmentation_model(auto_download)

                # Check if model needs to be downloaded
                if model and hasattr(model, "model_downloader"):
                    cached_path = model.model_downloader.get_model_path()
                    if not cached_path:
                        if progress_callback:
                            progress_callback("Downloading AngioPy model...", 0)

                        # Download model with progress
                        def download_progress(current, total):
                            if total > 0:
                                percent = int((current / total) * 100)
                                if progress_callback:
                                    progress_callback(f"Downloading model: {percent}%", percent)

                        downloaded_path = model.model_downloader.download_model(
                            progress_callback=download_progress
                        )
                        if downloaded_path:
                            if progress_callback:
                                progress_callback("Model downloaded successfully", 100)

                # Force model loading
                if model:
                    if progress_callback:
                        progress_callback("Loading model into memory...", 95)

                    # Ensure model is loaded
                    if hasattr(model, "load_model"):
                        if not model.model_loaded:
                            # Get the model path
                            if hasattr(model, "model_downloader"):
                                model_path = model.model_downloader.get_model_path()
                                if model_path:
                                    model.load_model(model_path)

                    # Do a dummy forward pass to ensure model is fully loaded
                    if hasattr(model, "model") and model.model is not None:
                        try:
                            import torch

                            dummy_input = torch.zeros(1, 3, 512, 512).to(model.device)
                            with torch.no_grad():
                                _ = model.model(dummy_input)
                        except:
                            pass

                    self._model_loaded = True
                    if progress_callback:
                        progress_callback("Model ready", 100)

            except Exception as e:
                import logging
                import traceback

                logging.error(f"Error preloading model: {e}")
                traceback.print_exc()
                if progress_callback:
                    progress_callback(f"Error loading model: {str(e)}", -1)

        if not self._model_loaded and (
            not self._loading_thread or not self._loading_thread.is_alive()
        ):
            self._loading_thread = threading.Thread(target=_load_model, daemon=True)
            self._loading_thread.start()

    def is_model_loaded(self) -> bool:
        """Check if model is fully loaded and ready."""
        return self._model_loaded and self._segmentation_model is not None

    def request_preload(self):
        """
        Request model preloading even if lazy loading is enabled.
        Useful for preloading during idle time.
        """
        self._preload_requested = True
        self.preload_model_async()

    def unload_model(self):
        """
        Unload the model from memory to free resources.
        Model will be reloaded on next use if lazy loading is enabled.
        """
        if not self._load_on_demand:
            logger.warning("Unloading model while lazy loading is disabled")

        with self._model_lock:
            if self._segmentation_model is not None:
                # Clear PyTorch cache if available
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                self._segmentation_model = None
                self._model_loaded = False
                self._model_cache.clear()
                logger.info("Model unloaded from memory")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get estimated memory usage of loaded models.

        Returns:
            Dict with memory usage in MB
        """
        usage = {"model_loaded": self.is_model_loaded(), "estimated_mb": 0.0}

        if self._segmentation_model is not None:
            # Rough estimate: 200-500 MB for typical segmentation model
            usage["estimated_mb"] = 350.0  # Conservative estimate

            try:
                import torch

                if torch.cuda.is_available():
                    usage["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                    usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            except ImportError:
                pass

        return usage

    @classmethod
    def instance(cls):
        """Get the singleton instance."""
        return cls()
