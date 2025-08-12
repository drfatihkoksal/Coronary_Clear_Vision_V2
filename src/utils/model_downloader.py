"""
Model downloader for AngioPy weights
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable
import hashlib
import urllib.request

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handle downloading and caching of AngioPy model weights"""

    MODEL_URL = "doi:10.5281/zenodo.13848135/modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth"
    MODEL_HASH = "md5:bf893ef57adaf39cfee33b25c7c1d87b"
    MODEL_FILENAME = "angiopy_model_weights.pth"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model downloader

        Args:
            cache_dir: Directory to cache model weights (default: ~/.cache/angiopy)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/angiopy")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.cache_dir / self.MODEL_FILENAME

    def download_model(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[str]:
        """
        Download model weights if not cached

        Args:
            progress_callback: Callback function(current_bytes, total_bytes)

        Returns:
            Path to model file or None if failed
        """
        # Check if already cached
        if self.model_path.exists():
            logger.info(f"Model already cached at: {self.model_path}")
            if self._verify_hash():
                return str(self.model_path)
            else:
                logger.warning("Cached model hash mismatch, re-downloading...")
                self.model_path.unlink()

        try:
            import pooch

            logger.info("Downloading AngioPy model weights from Zenodo...")
            logger.info(f"URL: {self.MODEL_URL}")
            logger.info(f"Destination: {self.model_path}")

            # Use pooch.retrieve directly for DOI URLs
            try:
                file_path = pooch.retrieve(
                    url=self.MODEL_URL,
                    known_hash=self.MODEL_HASH,
                    fname=self.MODEL_FILENAME,
                    path=self.cache_dir,
                    progressbar=True
                )

                logger.info(f"✓ Model downloaded successfully to: {file_path}")
                return file_path

            except Exception as e:
                logger.error(f"Pooch download failed: {e}")
                # Try with direct URL
                return self._download_with_urllib(progress_callback)

        except ImportError:
            logger.error("pooch library not installed. Install with: pip install pooch")

            # Try alternative download method
            return self._download_with_urllib(progress_callback)

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None

    def _download_with_urllib(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[str]:
        """Fallback download method using urllib"""
        try:

            # Convert DOI to direct URL
            # Note: This is a simplified approach - real DOI resolution would be more complex
            direct_url = "https://zenodo.org/record/13848135/files/modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth"

            logger.info(f"Attempting direct download from: {direct_url}")

            def download_progress(block_num, block_size, total_size):
                if progress_callback:
                    progress_callback(block_num * block_size, total_size)

            urllib.request.urlretrieve(
                direct_url,
                str(self.model_path),
                reporthook=download_progress
            )

            if self._verify_hash():
                logger.info("✓ Model downloaded and verified successfully")
                return str(self.model_path)
            else:
                logger.error("Downloaded model hash mismatch")
                self.model_path.unlink()
                return None

        except Exception as e:
            logger.error(f"Fallback download failed: {e}")
            return None

    def _verify_hash(self) -> bool:
        """Verify model file hash"""
        if not self.model_path.exists():
            return False

        # Extract expected hash
        hash_type, expected_hash = self.MODEL_HASH.split(":")

        if hash_type != "md5":
            logger.warning(f"Unsupported hash type: {hash_type}")
            return True  # Assume OK if we can't verify

        # Calculate actual hash
        md5_hash = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)

        actual_hash = md5_hash.hexdigest()

        if actual_hash == expected_hash:
            logger.info("✓ Model hash verified")
            return True
        else:
            logger.error(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
            return False

    def get_model_path(self) -> Optional[str]:
        """Get path to cached model if available"""
        if self.model_path.exists() and self._verify_hash():
            return str(self.model_path)
        return None