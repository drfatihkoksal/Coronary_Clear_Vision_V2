"""
AngioPy Segmentation Module - Clean Implementation
Single-frame vessel segmentation with user guidance
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Callable
import cv2
from pathlib import Path
import logging
import scipy.ndimage
from skimage.morphology import skeletonize
try:
    from fil_finder import FilFinder2D
    import astropy.units as u
    FIL_FINDER_AVAILABLE = True
except ImportError:
    FIL_FINDER_AVAILABLE = False

from ..utils.model_downloader import ModelDownloader

logger = logging.getLogger(__name__)

class AngioPySegmentation:
    """Clean implementation of AngioPy vessel segmentation"""

    def __init__(self, model_path: Optional[str] = None, auto_download: bool = True):
        """
        Initialize segmentation module

        Args:
            model_path: Path to model weights (optional)
            auto_download: Whether to auto-download model if not found
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        self.model_downloading = False
        self.auto_download = auto_download
        self.force_angiopy = True  # Always True - no fallback
        self.model_downloader = ModelDownloader()

        # Model configuration
        self.input_size = 512
        self.num_classes = 2

        logger.info(f"AngioPy Segmentation initialized on {self.device}")
        logger.info("AngioPy mode enabled - no fallback")

        # Try to load model
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Check cache but don't load at init
            cached_path = self.model_downloader.get_model_path()
            if cached_path:
                logger.info(f"Found cached model at: {cached_path} (will load on first use)")

    def load_model(self, model_path: str) -> bool:
        """
        Load pre-trained model

        Args:
            model_path: Path to model file

        Returns:
            Success status
        """
        try:
            logger.info(f"Loading model from {model_path}")

            # Import segmentation models
            try:
                import segmentation_models_pytorch as smp
            except ImportError:
                logger.error("segmentation_models_pytorch not installed")
                return False

            # Create model architecture
            self.model = smp.Unet(
                encoder_name="inceptionresnetv2",
                encoder_weights=None,
                in_channels=3,
                classes=self.num_classes,
                activation=None
            )

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract state dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            else:
                state_dict = checkpoint

            # Handle DataParallel prefix
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '') if key.startswith('module.') else key
                cleaned_state_dict[new_key] = value

            # Load weights
            self.model.load_state_dict(cleaned_state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            logger.info("✓ Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            return False

    def segment_vessel(self,
                      image: np.ndarray,
                      user_points: List[Tuple[int, int]],
                      progress_callback: Optional[Callable[[str, int], None]] = None,
                      **kwargs) -> Dict:
        """
        Perform vessel segmentation

        Args:
            image: Input image (grayscale or RGB)
            user_points: List of (x, y) click points from user
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with segmentation results
        """
        # Ensure user points are integers to avoid numpy indexing errors
        if user_points:
            user_points = [(int(float(p[0])), int(float(p[1]))) for p in user_points]
            
        result = {
            'success': False,
            'mask': None,
            'probability': None,
            'centerline': None,
            'boundaries': None,
            'method': 'none'
        }

        try:
            logger.info(f"segment_vessel called with {len(user_points)} points")
            logger.info(f"Model loaded: {self.model_loaded}")

            if progress_callback:
                progress_callback("Starting segmentation...", 10)

            # Always ensure model is loaded before proceeding
            if not self.model_loaded and not self.model_downloading:
                self.model_downloading = True
                try:
                    if progress_callback:
                        progress_callback("Checking AngioPy model...", 5)

                    # First check if model is cached
                    cached_path = self.model_downloader.get_model_path()
                    if cached_path:
                        logger.info(f"Found cached model, loading...")
                        if progress_callback:
                            progress_callback("Loading cached AngioPy model...", 50)
                        success = self.load_model(cached_path)
                        if success:
                            if progress_callback:
                                progress_callback("AngioPy model ready!", 95)
                        else:
                            # Cache corrupted, try download
                            cached_path = None

                    if not cached_path and not self.model_loaded:
                        if progress_callback:
                            progress_callback("AngioPy model not found. Downloading from Zenodo...", 10)

                        # Try to download model
                        def download_progress(current, total):
                            if progress_callback and total > 0:
                                percentage = min(90, int((current / total) * 85))
                                mb_current = current // (1024 * 1024)
                                mb_total = total // (1024 * 1024)
                                progress_callback(f"Downloading AngioPy model... {mb_current}MB/{mb_total}MB", percentage)

                        model_path = self.model_downloader.download_model(download_progress)
                        if model_path:
                            if progress_callback:
                                progress_callback("Loading AngioPy model...", 90)
                            success = self.load_model(model_path)
                            if success:
                                if progress_callback:
                                    progress_callback("AngioPy model ready!", 95)
                            else:
                                logger.error("Failed to load downloaded model")
                                if progress_callback:
                                    progress_callback("Failed to load AngioPy model", 100)
                                result['error'] = 'Failed to load AngioPy model'
                                return result
                        else:
                            logger.error("Failed to download model")
                            if progress_callback:
                                progress_callback("Failed to download AngioPy model", 100)
                            result['error'] = 'Failed to download AngioPy model. Please check your internet connection.'
                            return result
                finally:
                    self.model_downloading = False

            # Check if we can use AI model
            if self.model_loaded and self.model is not None:
                logger.info("Using AngioPy AI model")
                result = self._ai_segmentation(image, user_points, progress_callback, **kwargs)
                result['method'] = 'angiopy'
            else:
                # No fallback - AngioPy model is required
                logger.error("AngioPy model not available")
                result['success'] = False
                result['error'] = 'AngioPy model not loaded. Please wait for the model to download or check your internet connection.'
                result['method'] = 'none'
                if progress_callback:
                    progress_callback("AngioPy model required for segmentation", 100)

            if progress_callback:
                progress_callback("Segmentation complete", 100)

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            result['error'] = str(e)

        return result

    def _ai_segmentation(self,
                        image: np.ndarray,
                        user_points: List[Tuple[int, int]],
                        progress_callback: Optional[Callable[[str, int], None]] = None,
                        **kwargs) -> Dict:
        """
        AI-based segmentation using AngioPy model
        """
        # Ensure user points are integers
        if user_points:
            user_points = [(int(float(p[0])), int(float(p[1]))) for p in user_points]
            
        if progress_callback:
            progress_callback("Preprocessing image...", 20)

        # Preprocess image with user points
        input_tensor, original_shape = self._preprocess_for_model(image, user_points)

        if progress_callback:
            progress_callback("Running AI model...", 40)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            logger.info(f"Model output shape: {output.shape}, min: {output.min():.3f}, max: {output.max():.3f}")

            # Get vessel probability
            if output.shape[1] == 2:
                probability = torch.softmax(output, dim=1)[0, 1]
            else:
                probability = torch.sigmoid(output[0, 0])

            probability = probability.cpu().numpy()
            logger.info(f"Probability stats - min: {probability.min():.3f}, max: {probability.max():.3f}, mean: {probability.mean():.3f}")

        if progress_callback:
            progress_callback("Post-processing results...", 60)

        # Resize to original shape
        if probability.shape != original_shape:
            probability = cv2.resize(probability, (original_shape[1], original_shape[0]))

        # Create binary mask for connected components analysis
        # Dynamic threshold based on probability distribution
        # Use Otsu's method or percentile-based threshold
        prob_mean = np.mean(probability)
        prob_std = np.std(probability)
        
        # Adaptive threshold: mean + 0.7 * std, bounded between 0.5 and 0.75
        # Balanced threshold for better vessel coverage while maintaining cleanliness
        threshold = np.clip(prob_mean + 0.7 * prob_std, 0.5, 0.75)
        
        # Alternative: use percentile-based threshold  
        # Only use percentile if the initial threshold is too low
        if threshold < 0.5 and np.sum(probability > 0.1) > 100:  # If we have enough non-zero pixels
            percentile_threshold = np.percentile(probability[probability > 0.1], 50)  # Balanced percentile
            # Use maximum to ensure minimum threshold
            threshold = max(threshold, percentile_threshold)
        
        # Final safety check - threshold should never be above 0.75
        threshold = min(threshold, 0.75)
        
        binary_mask = (probability > threshold).astype(np.uint8)

        logger.info(f"AI segmentation: probability range [{probability.min():.3f}, {probability.max():.3f}]")
        logger.info(f"AI segmentation: Using threshold {threshold}")
        logger.info(f"AI segmentation: mask pixels before post-process: {np.sum(binary_mask)}")
        logger.info(f"AI segmentation: pixels above threshold: {np.sum(probability > threshold)}")

        # Find connected vessel that passes through user points
        if user_points and len(user_points) >= 1:
            # Find the largest connected component that contains or is near user points
            _, labels = cv2.connectedComponents(binary_mask)

            # Find which component(s) contain user points
            valid_labels = set()
            for pt in user_points:
                x, y = pt
                # Check in a small neighborhood around the point
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        check_y = y + dy
                        check_x = x + dx
                        if 0 <= check_y < labels.shape[0] and 0 <= check_x < labels.shape[1]:
                            label = labels[check_y, check_x]
                            if label > 0:  # 0 is background
                                valid_labels.add(label)

            # Create new mask with only the selected vessel
            if valid_labels:
                # Create component mask
                component_mask = np.zeros_like(binary_mask, dtype=np.float32)
                for label in valid_labels:
                    component_mask[labels == label] = 1
                
                # Apply to probability map to preserve sub-pixel information
                probability = probability * component_mask
                binary_mask = (probability > threshold).astype(np.uint8)
            else:
                logger.warning("No vessel found near user points")

        # Post-process with bounds limiting (use binary mask for morphological operations)
        binary_mask = self._postprocess_mask(binary_mask, user_points, original_shape, **kwargs)
        
        # Apply post-processing to probability map as well
        probability = probability * (binary_mask > 0).astype(np.float32)

        # Create final mask WITHOUT adding reference points (2025 approach)
        final_mask = binary_mask.copy()
        # REMOVE reference point artifacts for clean centerline extraction
        if user_points and len(user_points) > 0:
            final_mask = self._remove_reference_point_artifacts(final_mask, user_points)
            logger.info(f"Cleaned reference point artifacts from {len(user_points)} points")

        logger.info(f"AI segmentation: mask pixels after post-process: {np.sum(final_mask)}")

        if progress_callback:
            progress_callback("Extracting vessel features...", 80)

        # REMOVED: AngioPy centerline extraction
        # Centerline will be generated by QCA using tracked points instead
        logger.info("Skipping AngioPy centerline extraction - will use tracked points in QCA")
        centerline = None
        boundaries = []
            
        # Resize mask to original shape
        if final_mask.shape != original_shape:
            # Use INTER_LINEAR for smoother edges, then threshold
            resized_mask_float = cv2.resize(final_mask.astype(np.float32), 
                                           (original_shape[1], original_shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
            # Apply threshold to maintain binary nature
            resized_mask = (resized_mask_float > 127).astype(np.uint8) * 255
            
            resized_probability = cv2.resize(probability, (original_shape[1], original_shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
            logger.info(f"Resized mask from {final_mask.shape} to {original_shape} with smooth interpolation")
        else:
            resized_mask = final_mask
            resized_probability = probability

        # Calculate vessel wall thickness using 512x512 mask
        thickness_stats = self.calculate_vessel_wall_thickness(final_mask)

        # Extract proximal and distal points from user points
        proximal_point = user_points[0] if user_points else None
        distal_point = user_points[-1] if user_points else None
        
        # NO CENTERLINE - will be generated by QCA from tracked points
        scaled_centerline = None
        
        # Debug logging
        logger.info(f"Final mask shape: {resized_mask.shape}")
        logger.info(f"Original shape: {original_shape}")
        logger.info(f"Centerline shape: None (will use tracked points in QCA)")
        
        return {
            'success': True,
            'mask': resized_mask,
            'probability': resized_probability,
            'centerline': scaled_centerline,  # Use scaled centerline
            'boundaries': boundaries,
            'thickness_stats': thickness_stats,
            'proximal_point': proximal_point,
            'distal_point': distal_point
        }


    def _preprocess_for_model(self, image: np.ndarray, user_points: List[Tuple[int, int]] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for model input following AngioPy's EXACT approach
        """
        original_shape = image.shape[:2]

        # Resize to 512x512 first (like AngioPy)
        image = cv2.resize(image, (512, 512))

        # Convert to RGB if grayscale (AngioPy uses RGB)
        if len(image.shape) == 2:
            # Create RGB image with grayscale in all channels
            rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = image
            rgb_image[:, :, 1] = image
            rgb_image[:, :, 2] = image
            image = rgb_image
        elif image.shape[2] == 1:
            rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = image[:, :, 0]
            rgb_image[:, :, 1] = image[:, :, 0]
            rgb_image[:, :, 2] = image[:, :, 0]
            image = rgb_image

        # Clear last two channels (green and blue) - EXACTLY like AngioPy
        image[:, :, 1] = 0  # Green channel
        image[:, :, 2] = 0  # Blue channel

        # Add user points to image channels following AngioPy's EXACT approach
        if user_points and len(user_points) >= 1:
            # Scale points to 512x512
            scale_x = 512.0 / original_shape[1]
            scale_y = 512.0 / original_shape[0]

            # Use MINIMAL reference points (2025 single seed approach)
            # Only use first and last points as guidance to reduce artifacts
            guidance_points = [user_points[0], user_points[-1]]
            for point in guidance_points:
                x = int(point[0] * scale_x)
                y = int(point[1] * scale_y)
                # Use smaller 2x2 square to reduce artifacts (was 4x4)
                y_start, y_end = max(0, y-1), min(512, y+1)
                x_start, x_end = max(0, x-1), min(512, x+1)
                image[y_start:y_end, x_start:x_end, 1] = 128  # Lighter intensity (was 255)

        # Convert to PIL Image and back (like AngioPy)
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image.astype(np.uint8))

        # Use AngioPy's preprocess method
        img_array = np.array(pil_image)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)

        # HWC to CHW
        img_trans = img_array.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(img_trans.astype(np.float32)).unsqueeze(0)

        logger.info(f"Preprocessing complete - Tensor shape: {tensor.shape}, min: {tensor.min():.3f}, max: {tensor.max():.3f}")
        logger.info(f"User points scaled to 512x512: {[(int(p[0]*512/original_shape[1]), int(p[1]*512/original_shape[0])) for p in user_points] if user_points else 'None'}")

        return tensor.to(self.device), original_shape



    def _postprocess_mask(self, mask: np.ndarray, user_points: List[Tuple[int, int]] = None,
                         original_shape: Tuple[int, int] = None, **kwargs) -> np.ndarray:
        """
        Clean up segmentation mask and limit to user-defined bounds
        """
        # Remove reference point special marking - treat all points equally
        # (Reference point coloring removed as requested)
        
        # Convert to binary for component analysis
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

        if num_labels > 1:
            # Keep only large components
            # Dynamic minimum area based on image size
            image_area = mask.shape[0] * mask.shape[1]
            min_area = max(100, int(image_area * 0.0001))  # 0.01% of image area, min 100 pixels
            cleaned_mask = np.zeros_like(mask)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_area:
                    component_mask = labels == i
                    cleaned_mask[component_mask] = 1

            mask = cleaned_mask
        else:
            mask = binary_mask

        # Erosion removed - use full mask for more accurate diameter measurements
        # The diameter measurement algorithm now handles mask boundaries correctly
        
        # Convert to standard binary mask
        final_mask = mask.astype(np.uint8) * 255  # Vessel areas = 255
        
        mask = final_mask

        # Limit segmentation to only between most proximal and distal points
        if user_points and len(user_points) >= 2 and original_shape:
            # Use only first and last points for boundary limitation
            proximal_distal_points = [user_points[0], user_points[-1]]
            
            # Check if light mask limiting is requested
            use_light_mask_limiting = kwargs.get('use_light_mask_limiting', False)
            
            if use_light_mask_limiting:
                mask = self._limit_mask_to_bounds_light(mask, proximal_distal_points, original_shape)
                logger.info(f"Applied LIGHT mask limiting between proximal {user_points[0]} and distal {user_points[-1]} points")
            else:
                mask = self._limit_mask_to_bounds(mask, proximal_distal_points, original_shape)
                logger.info(f"Applied MODERATE mask limiting between proximal {user_points[0]} and distal {user_points[-1]} points")

        return mask

    def _limit_mask_to_bounds(self, mask: np.ndarray, user_points: List[Tuple[int, int]],
                              original_shape: Tuple[int, int]) -> np.ndarray:
        """
        MODERATELY limit segmentation to the vessel segment between proximal and distal points.
        Uses balanced geometric constraints with reasonable tolerances.

        Args:
            mask: Binary segmentation mask (512x512)
            user_points: User-defined points in original image coordinates
            original_shape: Original image shape (H, W)

        Returns:
            Moderately trimmed mask containing vessel between reference points
        """
        try:
            # Scale points to 512x512
            scale_x = 512.0 / original_shape[1]
            scale_y = 512.0 / original_shape[0]

            # Get proximal and distal points (first and last)
            proximal = user_points[0]
            distal = user_points[-1]

            # Scale to mask coordinates
            prox_x = int(proximal[0] * scale_x)
            prox_y = int(proximal[1] * scale_y)
            dist_x = int(distal[0] * scale_x)
            dist_y = int(distal[1] * scale_y)

            logger.info(f"MODERATE limiting with reasonable constraints at proximal ({prox_x}, {prox_y}) and distal ({dist_x}, {dist_y})")

            # Find the best vessel component first
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            
            best_component = None
            best_score = 0
            search_radius = 25  # Reduced search radius for tighter proximal limiting (was 35)
            
            for i in range(1, num_labels):
                component_mask = (labels == i).astype(np.uint8)
                
                # Check coverage near both points
                prox_region = component_mask[max(0, prox_y-search_radius):min(512, prox_y+search_radius),
                                           max(0, prox_x-search_radius):min(512, prox_x+search_radius)]
                dist_region = component_mask[max(0, dist_y-search_radius):min(512, dist_y+search_radius),
                                           max(0, dist_x-search_radius):min(512, dist_x+search_radius)]
                
                prox_coverage = np.sum(prox_region)
                dist_coverage = np.sum(dist_region)
                
                if prox_coverage > 0 and dist_coverage > 0:
                    score = prox_coverage + dist_coverage + stats[i, cv2.CC_STAT_AREA] * 0.001
                    if score > best_score:
                        best_component = i
                        best_score = score
            
            if best_component is None:
                logger.warning("No valid component found, using original mask")
                return mask
            
            # Get the main vessel component
            vessel_mask = (labels == best_component).astype(np.uint8)
            
            # Create MODERATE perpendicular cutting with generous tolerances
            # Calculate vessel direction vector
            dx = dist_x - prox_x
            dy = dist_y - prox_y
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 5:  # Apply perpendicular cuts even for closer points
                # Normalize direction vector
                dir_x = dx / length
                dir_y = dy / length
                
                # Perpendicular vectors for cutting planes
                perp_x = -dir_y  # Rotate 90 degrees
                perp_y = dir_x
                
                # Create cutting masks
                cut_mask = np.ones_like(vessel_mask)
                
                # MODERATE tolerances - generous but still protective
                proximal_tolerance = 20  # 20 pixel tolerance at proximal
                distal_tolerance = 25    # 25 pixel tolerance at distal  
                
                # Cut everything BEFORE proximal point (more proximal) - but with generous tolerance
                for y in range(512):
                    for x in range(512):
                        # Vector from proximal to current pixel
                        px_dx = x - prox_x
                        px_dy = y - prox_y
                        
                        # Project onto vessel direction
                        projection = px_dx * dir_x + px_dy * dir_y
                        
                        # If projection is negative, pixel is before proximal point
                        if projection < -proximal_tolerance:
                            cut_mask[y, x] = 0
                
                # Cut everything AFTER distal point (more distal) - with generous tolerance
                for y in range(512):
                    for x in range(512):
                        # Vector from proximal to current pixel
                        px_dx = x - prox_x
                        px_dy = y - prox_y
                        
                        # Project onto vessel direction
                        projection = px_dx * dir_x + px_dy * dir_y
                        
                        # If projection is beyond vessel length, pixel is after distal point
                        if projection > length + distal_tolerance:
                            cut_mask[y, x] = 0
                
                # Apply cuts to vessel mask
                trimmed_mask = vessel_mask * cut_mask
                
                # Moderate perpendicular distance constraint
                # Remove pixels too far from the proximal-distal line
                max_perpendicular_distance = 60  # Generous perpendicular distance
                
                for y in range(512):
                    for x in range(512):
                        if trimmed_mask[y, x] > 0:
                            # Vector from proximal to current pixel
                            px_dx = x - prox_x
                            px_dy = y - prox_y
                            
                            # Calculate perpendicular distance to line
                            if length > 0:
                                # Calculate perpendicular component
                                perp_dist = abs(px_dx * perp_x + px_dy * perp_y)
                                
                                # Remove if too far from line
                                if perp_dist > max_perpendicular_distance:
                                    trimmed_mask[y, x] = 0
                
            else:
                # Points are very close, use generous elliptical regions
                logger.warning("Reference points very close, using generous elliptical constraint")
                # Create elliptical mask that encompasses both points
                center_x = (prox_x + dist_x) // 2
                center_y = (prox_y + dist_y) // 2
                
                # Semi-major axis along the line between points
                semi_major = max(40, int(length * 1.0))  # More generous
                semi_minor = 35  # Wider perpendicular width
                
                elliptical_mask = np.zeros_like(vessel_mask)
                
                # Calculate ellipse orientation
                if length > 0:
                    angle = np.arctan2(dy, dx)
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    
                    for y in range(512):
                        for x in range(512):
                            # Translate to ellipse center
                            dx_centered = x - center_x
                            dy_centered = y - center_y
                            
                            # Rotate to ellipse axes
                            x_rot = dx_centered * cos_angle + dy_centered * sin_angle
                            y_rot = -dx_centered * sin_angle + dy_centered * cos_angle
                            
                            # Check if inside ellipse
                            if (x_rot**2 / semi_major**2 + y_rot**2 / semi_minor**2) <= 1:
                                elliptical_mask[y, x] = 1
                else:
                    # Fallback to larger circle
                    y_coords, x_coords = np.mgrid[0:512, 0:512]
                    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                    elliptical_mask[distances <= 45] = 1
                
                trimmed_mask = vessel_mask * elliptical_mask
            
            # Very lenient safety check
            if np.sum(trimmed_mask) < 100:  # Much higher threshold
                logger.warning("Moderate cutting removed too much, using generous fallback")
                # Fallback to generous rectangular constraint
                margin = 50  # Generous margin
                min_x = max(0, min(prox_x, dist_x) - margin)
                max_x = min(512, max(prox_x, dist_x) + margin)
                min_y = max(0, min(prox_y, dist_y) - margin)
                max_y = min(512, max(prox_y, dist_y) + margin)
                
                fallback_mask = np.zeros_like(vessel_mask)
                fallback_mask[min_y:max_y, min_x:max_x] = 1
                trimmed_mask = vessel_mask * fallback_mask
            
            logger.info(f"MODERATE trimming: {np.sum(mask)} -> {np.sum(trimmed_mask)} pixels")
            return trimmed_mask

        except Exception as e:
            logger.error(f"Error in moderate mask limiting: {e}")
            return mask

    def _limit_mask_to_bounds_light(self, mask: np.ndarray, user_points: List[Tuple[int, int]],
                                   original_shape: Tuple[int, int]) -> np.ndarray:
        """
        LIGHTLY limit segmentation to vessel segment between proximal and distal points.
        Uses very generous tolerances and minimal constraints for complex vessel geometries.

        Args:
            mask: Binary segmentation mask (512x512)
            user_points: User-defined points in original image coordinates
            original_shape: Original image shape (H, W)

        Returns:
            Lightly trimmed mask preserving most vessel segments
        """
        try:
            # Scale points to 512x512
            scale_x = 512.0 / original_shape[1]
            scale_y = 512.0 / original_shape[0]

            # Get proximal and distal points (first and last)
            proximal = user_points[0]
            distal = user_points[-1]

            # Scale to mask coordinates
            prox_x = int(proximal[0] * scale_x)
            prox_y = int(proximal[1] * scale_y)
            dist_x = int(distal[0] * scale_x)
            dist_y = int(distal[1] * scale_y)

            logger.info(f"LIGHT limiting with very generous constraints at proximal ({prox_x}, {prox_y}) and distal ({dist_x}, {dist_y})")

            # Find the best vessel component first
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            
            best_component = None
            best_score = 0
            search_radius = 35  # Reduced generous search radius for tighter limiting (was 45)
            
            for i in range(1, num_labels):
                component_mask = (labels == i).astype(np.uint8)
                
                # Check coverage near both points
                prox_region = component_mask[max(0, prox_y-search_radius):min(512, prox_y+search_radius),
                                           max(0, prox_x-search_radius):min(512, prox_x+search_radius)]
                dist_region = component_mask[max(0, dist_y-search_radius):min(512, dist_y+search_radius),
                                           max(0, dist_x-search_radius):min(512, dist_x+search_radius)]
                
                prox_coverage = np.sum(prox_region)
                dist_coverage = np.sum(dist_region)
                
                if prox_coverage > 0 and dist_coverage > 0:
                    score = prox_coverage + dist_coverage + stats[i, cv2.CC_STAT_AREA] * 0.001
                    if score > best_score:
                        best_component = i
                        best_score = score
            
            if best_component is None:
                logger.warning("No valid component found, using original mask")
                return mask
            
            # Get the main vessel component
            vessel_mask = (labels == best_component).astype(np.uint8)
            
            # Create LIGHT perpendicular cutting with very generous tolerances
            # Calculate vessel direction vector
            dx = dist_x - prox_x
            dy = dist_y - prox_y
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 3:  # Apply even for very close points
                # Normalize direction vector
                dir_x = dx / length
                dir_y = dy / length
                
                # Perpendicular vectors for cutting planes
                perp_x = -dir_y  # Rotate 90 degrees
                perp_y = dir_x
                
                # Create cutting masks
                cut_mask = np.ones_like(vessel_mask)
                
                # LIGHT tolerances - very generous and permissive
                proximal_tolerance = 35  # 35 pixel tolerance at proximal
                distal_tolerance = 40    # 40 pixel tolerance at distal  
                
                # Cut everything BEFORE proximal point - with very generous tolerance
                for y in range(512):
                    for x in range(512):
                        # Vector from proximal to current pixel
                        px_dx = x - prox_x
                        px_dy = y - prox_y
                        
                        # Project onto vessel direction
                        projection = px_dx * dir_x + px_dy * dir_y
                        
                        # If projection is negative, pixel is before proximal point
                        if projection < -proximal_tolerance:
                            cut_mask[y, x] = 0
                
                # Cut everything AFTER distal point - with very generous tolerance
                for y in range(512):
                    for x in range(512):
                        # Vector from proximal to current pixel
                        px_dx = x - prox_x
                        px_dy = y - prox_y
                        
                        # Project onto vessel direction
                        projection = px_dx * dir_x + px_dy * dir_y
                        
                        # If projection is beyond vessel length, pixel is after distal point
                        if projection > length + distal_tolerance:
                            cut_mask[y, x] = 0
                
                # Apply cuts to vessel mask
                trimmed_mask = vessel_mask * cut_mask
                
                # Very permissive perpendicular distance constraint
                # Remove pixels only if they are VERY far from the proximal-distal line
                max_perpendicular_distance = 80  # Very generous perpendicular distance
                
                for y in range(512):
                    for x in range(512):
                        if trimmed_mask[y, x] > 0:
                            # Vector from proximal to current pixel
                            px_dx = x - prox_x
                            px_dy = y - prox_y
                            
                            # Calculate perpendicular distance to line
                            if length > 0:
                                # Calculate perpendicular component
                                perp_dist = abs(px_dx * perp_x + px_dy * perp_y)
                                
                                # Remove only if VERY far from line
                                if perp_dist > max_perpendicular_distance:
                                    trimmed_mask[y, x] = 0
                
            else:
                # Points are very close, use very generous elliptical regions
                logger.warning("Reference points very close, using very generous elliptical constraint")
                # Create very generous elliptical mask
                center_x = (prox_x + dist_x) // 2
                center_y = (prox_y + dist_y) // 2
                
                # Semi-major axis along the line between points
                semi_major = max(60, int(length * 1.5))  # Very generous
                semi_minor = 50  # Very wide perpendicular width
                
                elliptical_mask = np.zeros_like(vessel_mask)
                
                # Calculate ellipse orientation
                if length > 0:
                    angle = np.arctan2(dy, dx)
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    
                    for y in range(512):
                        for x in range(512):
                            # Translate to ellipse center
                            dx_centered = x - center_x
                            dy_centered = y - center_y
                            
                            # Rotate to ellipse axes
                            x_rot = dx_centered * cos_angle + dy_centered * sin_angle
                            y_rot = -dx_centered * sin_angle + dy_centered * cos_angle
                            
                            # Check if inside ellipse
                            if (x_rot**2 / semi_major**2 + y_rot**2 / semi_minor**2) <= 1:
                                elliptical_mask[y, x] = 1
                else:
                    # Fallback to very large circle
                    y_coords, x_coords = np.mgrid[0:512, 0:512]
                    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                    elliptical_mask[distances <= 65] = 1
                
                trimmed_mask = vessel_mask * elliptical_mask
            
            # Very permissive safety check
            if np.sum(trimmed_mask) < 200:  # Very high threshold
                logger.warning("Light cutting removed too much, using very generous fallback")
                # Fallback to very generous rectangular constraint
                margin = 70  # Very generous margin
                min_x = max(0, min(prox_x, dist_x) - margin)
                max_x = min(512, max(prox_x, dist_x) + margin)
                min_y = max(0, min(prox_y, dist_y) - margin)
                max_y = min(512, max(prox_y, dist_y) + margin)
                
                fallback_mask = np.zeros_like(vessel_mask)
                fallback_mask[min_y:max_y, min_x:max_x] = 1
                trimmed_mask = vessel_mask * fallback_mask
            
            logger.info(f"LIGHT trimming: {np.sum(mask)} -> {np.sum(trimmed_mask)} pixels")
            return trimmed_mask

        except Exception as e:
            logger.error(f"Error in light mask limiting: {e}")
            return mask


    def _extract_features(self, mask: np.ndarray) -> Tuple[Optional[List], List]:
        """
        Extract centerline and boundaries from mask
        """
        try:
            # Convert to binary for contour detection (but preserve multi-level mask)
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Find contours (boundaries)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boundaries = []
            for contour in contours:
                if len(contour) > 5:
                    boundary = [(p[0][0], p[0][1]) for p in contour]
                    boundaries.append(boundary)

            # Direct skeletonization of the entire mask for complete centerline
            if mask.any():
                from skimage.morphology import skeletonize
                
                # Debug: Log mask statistics
                logger.info(f"Mask shape: {mask.shape}, non-zero pixels: {np.sum(mask > 0)}")
                logger.info(f"Mask bounds - y: [{np.where(mask)[0].min()}, {np.where(mask)[0].max()}], x: [{np.where(mask)[1].min()}, {np.where(mask)[1].max()}]")
                
                # Skeletonize the entire mask to get full vessel centerline
                skeleton = skeletonize(mask > 0)
                
                # Get all skeleton points
                centerline_points = np.column_stack(np.where(skeleton))
                
                logger.info(f"Skeleton extraction found {len(centerline_points)} points")

                if len(centerline_points) > 0:
                    # Order the points to form a continuous path
                    # Use reference-aware ordering to ensure consistent direction
                    # Graph-based ordering with shortest path can create straight lines
                    ordered_points = self._order_centerline_points_with_reference(centerline_points, user_points)
                    
                    # Keep in (y, x) format as QCA expects
                    centerline = np.array(ordered_points)
                    logger.info(f"Ordered centerline has {len(centerline)} points in (y,x) format")
                    
                    # Debug: Log sample points to check if it's a straight line
                    if len(centerline) > 10:
                        logger.info(f"Centerline sample - First 5: {centerline[:5].tolist()}")
                        logger.info(f"Centerline sample - Last 5: {centerline[-5:].tolist()}")
                        logger.info(f"Centerline sample - Middle: {centerline[len(centerline)//2-2:len(centerline)//2+3].tolist()}")
                else:
                    centerline = None
            else:
                centerline = None

            return centerline, boundaries

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None, []

    def _extract_features_with_angiopy(self, mask: np.ndarray, probability: np.ndarray) -> Tuple[Optional[np.ndarray], List]:
        """
        Extract centerline using AngioPy's skeletonization method and refine to sub-pixel accuracy.
        """
        try:
            # Get boundaries first
            _, boundaries = self._extract_features(mask)
            
            # Skeletonize the mask to get an initial centerline
            skeleton = skeletonize(mask > 0)
            initial_skel_points = np.sum(skeleton > 0)
            logger.info(f"Initial skeleton has {initial_skel_points} points")
            
            # Choose skeleton processing method based on kwargs
            use_curvature_resistant = kwargs.get('use_curvature_resistant_centerline', False)
            
            if use_curvature_resistant:
                # Use curvature-resistant centerline extraction
                logger.info("Using curvature-resistant centerline extraction")
                skeleton = self._curvature_resistant_skeleton_processing(skeleton, mask)
            elif FIL_FINDER_AVAILABLE:
                # Use FilFinder to prune the skeleton (original method)
                logger.info("Using FilFinder for skeleton processing")
                try:
                    fil = FilFinder2D(skeleton.astype('uint8'), 
                                    distance=250 * u.pc, 
                                    mask=skeleton, 
                                    beamwidth=10.0*u.pix)
                    fil.preprocess_image(flatten_percent=85)
                    fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
                    fil.medskel(verbose=False)
                    fil.analyze_skeletons(branch_thresh=400 * u.pix, 
                                        skel_thresh=10 * u.pix, 
                                        prune_criteria='length')
                    skeleton = fil.skeleton.astype('uint8') * 255
                    logger.info(f"FilFinder processed skeleton has {np.sum(skeleton > 0)} points")
                except Exception as e:
                    logger.warning(f"FilFinder2D processing failed: {e}, using raw skeleton")
            else:
                logger.info("Using raw skeleton (no FilFinder or curvature-resistant processing)")
            
            skel_points = np.column_stack(np.where(skeleton > 0))
            
            if len(skel_points) < 3:
                logger.warning(f"Too few skeleton points ({len(skel_points)}), skipping refinement.")
                return self._extract_features(mask) # Fallback to simple extraction
            
            # Order the skeleton points to form a continuous path with reference awareness
            ordered_points = self._order_centerline_points_with_reference(skel_points, user_points)
            integer_centerline = np.array(ordered_points)
            logger.info(f"Ordered integer centerline has {len(integer_centerline)} points.")

            # *** NEW STEP: Refine the centerline to sub-pixel accuracy ***
            logger.info("Refining centerline to sub-pixel accuracy using probability map.")
            subpixel_centerline = self._refine_centerline_to_subpixel(integer_centerline, probability)

            if subpixel_centerline is not None and len(subpixel_centerline) > 0:
                logger.info(f"Successfully refined centerline to {len(subpixel_centerline)} sub-pixel points.")
                return subpixel_centerline, boundaries
            else:
                logger.warning("Sub-pixel refinement failed. Returning integer-based centerline.")
                return integer_centerline, boundaries
                
        except Exception as e:
            logger.error(f"AngioPy feature extraction failed: {e}")
            # Fallback to simple extraction
            return self._extract_features(mask)
    
    def _order_skeleton_points(self, skeleton: np.ndarray, start_point: np.ndarray) -> np.ndarray:
        """
        Order skeleton points starting from a given point
        """
        try:
            # Get all skeleton points
            skel_points = set(map(tuple, np.column_stack(np.where(skeleton > 0))))
            
            # Initialize ordered list
            ordered = [tuple(start_point)]
            current = tuple(start_point)
            skel_points.remove(current)
            
            # Traverse skeleton
            while skel_points:
                # Find nearest unvisited neighbor
                min_dist = float('inf')
                next_point = None
                
                for point in skel_points:
                    dist = np.sqrt((point[0] - current[0])**2 + (point[1] - current[1])**2)
                    if dist <= np.sqrt(2) + 0.1:  # Adjacent or diagonal
                        next_point = point
                        break
                    elif dist < min_dist:
                        min_dist = dist
                        next_point = point
                
                if next_point is None or min_dist > 5:  # No nearby point found
                    break
                    
                ordered.append(next_point)
                skel_points.remove(next_point)
                current = next_point
            
            return np.array(ordered)
            
        except Exception as e:
            logger.error(f"Failed to order skeleton points: {e}")
            return None
    
    def _refine_centerline_with_distance_transform(self, centerline: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refine centerline to stay in the center of the vessel using distance transform
        """
        try:
            # Calculate distance transform
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            
            # For each point in centerline, adjust to local maximum of distance transform
            refined_points = []
            window_size = 7  # Search window radius
            
            for point in centerline:
                x, y = int(point[0]), int(point[1])
                
                # Skip if outside mask bounds
                if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                    refined_points.append(point)
                    continue
                
                # Define search window
                y_min = max(0, y - window_size)
                y_max = min(mask.shape[0], y + window_size + 1)
                x_min = max(0, x - window_size)
                x_max = min(mask.shape[1], x + window_size + 1)
                
                # Get local distance transform
                local_dist = dist[y_min:y_max, x_min:x_max]
                
                if local_dist.size > 0 and np.max(local_dist) > 0:
                    # Find local maximum (center of vessel)
                    local_max_idx = np.unravel_index(np.argmax(local_dist), local_dist.shape)
                    new_y = y_min + local_max_idx[0]
                    new_x = x_min + local_max_idx[1]
                    
                    # Check if new position is significantly better (more centered)
                    current_dist = dist[y, x] if mask[y, x] > 0 else 0
                    new_dist = dist[new_y, new_x]
                    
                    # Only move if new position is more centered
                    if new_dist > current_dist * 1.1:  # 10% improvement threshold
                        refined_points.append([new_x, new_y])
                    else:
                        refined_points.append(point)
                else:
                    refined_points.append(point)
            
            refined_centerline = np.array(refined_points)
            
            # Skip smoothing to preserve stenosis details - smoothing can miss narrow stenosis
            # Original smoothing code disabled to maintain vessel anatomy accuracy
            # if len(refined_centerline) > 5:
            #     from scipy.ndimage import gaussian_filter1d
            #     refined_centerline[:, 0] = gaussian_filter1d(refined_centerline[:, 0], sigma=1.5)
            #     refined_centerline[:, 1] = gaussian_filter1d(refined_centerline[:, 1], sigma=1.5)
            
            return refined_centerline
            
        except Exception as e:
            logger.warning(f"Failed to refine centerline with distance transform: {e}")
            return centerline
    
    def _order_centerline_points_with_reference(self, points: np.ndarray, user_points: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        Order centerline points with reference point awareness for consistent direction.
        This ensures the centerline is always ordered from proximal to distal.
        
        Args:
            points: Centerline points to order (N x 2) in (y, x) format
            user_points: User reference points (proximal first, distal last) in (x, y) format
            
        Returns:
            Ordered centerline points ensuring proximal->distal direction
        """
        if len(points) < 2:
            return points
            
        # First, use the simple ordering algorithm
        ordered_points = self._order_centerline_points_simple(points)
        
        # If we have reference points, ensure correct orientation
        if user_points and len(user_points) >= 2:
            # Convert user points to (y, x) format to match centerline
            proximal_ref = np.array([user_points[0][1], user_points[0][0]])  # (y, x)
            distal_ref = np.array([user_points[-1][1], user_points[-1][0]])  # (y, x)
            
            # Check if the ordered points start closer to proximal than distal
            start_point = ordered_points[0]
            end_point = ordered_points[-1]
            
            start_to_proximal = np.linalg.norm(start_point - proximal_ref)
            start_to_distal = np.linalg.norm(start_point - distal_ref)
            end_to_proximal = np.linalg.norm(end_point - proximal_ref) 
            end_to_distal = np.linalg.norm(end_point - distal_ref)
            
            # If start is closer to distal or end is closer to proximal, reverse the order
            if start_to_distal < start_to_proximal or end_to_proximal < end_to_distal:
                ordered_points = ordered_points[::-1]
                logger.debug("Reversed centerline ordering to match proximal->distal reference direction")
            else:
                logger.debug("Centerline ordering matches proximal->distal reference direction")
        
        return ordered_points

    def _order_centerline_points_simple(self, points: np.ndarray) -> np.ndarray:
        """
        Improved ordering of centerline points using endpoint detection
        """
        if len(points) < 2:
            return points
            
        # Find endpoints (points with only one neighbor within threshold)
        threshold = 2.0  # pixels
        neighbor_counts = []
        
        for i, point in enumerate(points):
            distances = np.linalg.norm(points - point, axis=1)
            neighbors = np.sum((distances > 0) & (distances <= threshold))
            neighbor_counts.append(neighbors)
        
        neighbor_counts = np.array(neighbor_counts)
        
        # Find potential endpoints (points with 1 neighbor)
        endpoints = np.where(neighbor_counts == 1)[0]
        
        if len(endpoints) >= 2:
            # Start from the first endpoint
            start_idx = endpoints[0]
        else:
            # If no clear endpoints, start from a point with minimum neighbors
            start_idx = np.argmin(neighbor_counts)
        
        # Order points starting from endpoint
        ordered = [points[start_idx]]
        used = {start_idx}
        
        while len(ordered) < len(points):
            last_point = ordered[-1]
            
            # Find nearest unused point
            min_dist = float('inf')
            next_idx = None
            
            for i, point in enumerate(points):
                if i not in used:
                    dist = np.linalg.norm(point - last_point)
                    if dist < min_dist:
                        min_dist = dist
                        next_idx = i
            
            if next_idx is None or min_dist > 10:  # Gap too large
                # Look for another component
                for i in range(len(points)):
                    if i not in used:
                        ordered.append(points[i])
                        used.add(i)
                        break
            else:
                ordered.append(points[next_idx])
                used.add(next_idx)
            
        return np.array(ordered)
    
    def _order_centerline_points_graph(self, points: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Graph-based approach can create straight lines
        Always use simple ordering instead
        """
        logger.debug("Graph-based ordering called but redirecting to simple ordering")
        return self._order_centerline_points_simple(points)

    def _refine_centerline_to_subpixel(self, centerline: np.ndarray, probability_map: np.ndarray) -> np.ndarray:
        """
        Refines an integer-based centerline to sub-pixel accuracy by analyzing the probability map.

        Args:
            centerline: The initial integer-based centerline (N, 2) in (y, x) format.
            probability_map: The grayscale probability map from the AI model.

        Returns:
            A new centerline (N, 2) with sub-pixel coordinates.
        """
        if probability_map is None or len(probability_map.shape) < 2:
            logger.warning("Probability map is invalid. Skipping sub-pixel refinement.")
            return centerline

        refined_centerline = []
        h, w = probability_map.shape
        
        # Smooth the probability map slightly to reduce noise for gradient calculation
        smoothed_prob_map = cv2.GaussianBlur(probability_map, (5, 5), 0)

        for i in range(len(centerline)):
            p_int = centerline[i]
            
            # Get perpendicular direction
            perpendicular = self._calculate_perpendicular_vector(centerline, i)
            if perpendicular is None:
                refined_centerline.append(p_int)
                continue

            # Sample points along the perpendicular line
            search_range = 5  # pixels to search on each side
            n_samples = 20  # number of samples
            distances = np.linspace(-search_range, search_range, n_samples)
            
            sample_points_y = p_int[0] + distances * perpendicular[0]
            sample_points_x = p_int[1] + distances * perpendicular[1]

            # Ensure sample points are within bounds
            valid_indices = (sample_points_y >= 0) & (sample_points_y < h -1) & \
                            (sample_points_x >= 0) & (sample_points_x < w -1)
            
            if not np.any(valid_indices):
                refined_centerline.append(p_int)
                continue

            sample_points_y = sample_points_y[valid_indices]
            sample_points_x = sample_points_x[valid_indices]
            
            # Get probability values at sample points using bilinear interpolation
            prob_values = scipy.ndimage.map_coordinates(
                smoothed_prob_map, 
                [sample_points_y, sample_points_x], 
                order=1, 
                mode='constant', 
                cval=0.0
            )

            if len(prob_values) < 3:
                refined_centerline.append(p_int)
                continue

            # Find the peak of the probability profile (the "ridge")
            try:
                # Fit a quadratic to the peak to find sub-pixel maximum
                max_idx = np.argmax(prob_values)
                if 0 < max_idx < len(prob_values) - 1:
                    y_points = prob_values[max_idx-1 : max_idx+2]
                    x_points = distances[valid_indices][max_idx-1 : max_idx+2]
                    
                    # Quadratic fit: y = ax^2 + bx + c
                    coeffs = np.polyfit(x_points, y_points, 2)
                    
                    # Peak of parabola is at -b / 2a
                    if abs(coeffs[0]) > 1e-6: # avoid division by zero
                        offset = -coeffs[1] / (2 * coeffs[0])
                        
                        # New sub-pixel point
                        new_p = p_int + offset * perpendicular
                        refined_centerline.append(new_p)
                    else:
                        refined_centerline.append(p_int) # No curvature, stick to integer point
                else:
                    refined_centerline.append(p_int) # Max is at the edge, no refinement
            except np.linalg.LinAlgError:
                refined_centerline.append(p_int) # Could not fit polynomial

        refined_centerline = np.array(refined_centerline)
        
        # Apply 2025 research-based endpoint-aware smoothing
        if len(refined_centerline) > 10:
            refined_centerline = self._apply_modern_centerline_smoothing(refined_centerline)

        logger.info(f"Refined centerline to sub-pixel accuracy. Original points: {len(centerline)}, Refined points: {len(refined_centerline)}")
        return refined_centerline

    def _apply_modern_centerline_smoothing(self, centerline: np.ndarray) -> np.ndarray:
        """
        2025 research-based centerline smoothing:
        - SIRE: Scale-invariant endpoint preservation
        - Key point detection for tortuosity handling
        - Graph-based local neighborhood smoothing
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter1d
        
        if len(centerline) < 15:
            return centerline
        
        smoothed = centerline.copy()
        n_points = len(centerline)
        
        # 1. ENDPOINT PRESERVATION (2025 SIRE approach)
        endpoint_preserve = min(6, n_points // 5)  # Adaptive endpoint preservation
        
        # 2. TORTUOSITY-AWARE SMOOTHING (Key point detection approach)
        curvatures = self._compute_local_curvature_fast(centerline)
        
        # 3. ADAPTIVE KERNEL REGRESSION (2025 research)
        for i in range(endpoint_preserve, n_points - endpoint_preserve):
            curvature = curvatures[i]
            
            # Dynamic window size based on local geometry
            if curvature > 0.7:  # High curvature - preserve shape
                continue  # No smoothing for high curvature regions
            elif curvature > 0.3:  # Medium curvature - light smoothing
                window = 2
                sigma = 0.2
            else:  # Low curvature - more smoothing
                window = 3
                sigma = 0.4
            
            # Local neighborhood smoothing
            start_idx = max(endpoint_preserve, i - window)
            end_idx = min(n_points - endpoint_preserve, i + window + 1)
            
            if end_idx > start_idx + 2:
                # Weighted average with Gaussian weights
                local_points = centerline[start_idx:end_idx]
                center_in_local = i - start_idx
                
                # Gaussian kernel weights
                distances = np.arange(len(local_points)) - center_in_local
                weights = np.exp(-0.5 * (distances / sigma) ** 2)
                weights = weights / np.sum(weights)
                
                # Apply weighted smoothing
                smoothed[i] = np.sum(local_points * weights[:, np.newaxis], axis=0)
        
        logger.info(f"Applied 2025 modern centerline smoothing: preserved {endpoint_preserve} endpoints")
        return smoothed
    
    def _compute_local_curvature_fast(self, centerline: np.ndarray) -> np.ndarray:
        """
        Fast local curvature computation for tortuosity detection
        Based on 2025 key point detection algorithms
        """
        n_points = len(centerline)
        curvatures = np.zeros(n_points)
        
        for i in range(1, n_points - 1):
            # Vectors to previous and next points
            v1 = centerline[i] - centerline[i-1]
            v2 = centerline[i+1] - centerline[i]
            
            # Normalize vectors
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            
            if n1 > 1e-6 and n2 > 1e-6:
                v1_norm = v1 / n1
                v2_norm = v2 / n2
                
                # Curvature from angle change
                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                curvatures[i] = 1 - (dot_product + 1) / 2  # Normalize to [0,1]
        
        # Handle endpoints
        curvatures[0] = curvatures[1] if n_points > 1 else 0
        curvatures[-1] = curvatures[-2] if n_points > 1 else 0
        
        return curvatures

    def _remove_reference_point_artifacts(self, mask: np.ndarray, user_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Remove reference point artifacts that can cause centerline distortion
        Based on 2025 research: single seed point approach + post-segmentation cleaning
        """
        cleaned_mask = mask.copy()
        
        # 1. Remove isolated pixels around reference points
        for point in user_points:
            x, y = point
            # Remove small neighborhood around reference point
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        # Check if this pixel is isolated (not part of main vessel)
                        if self._is_isolated_pixel(mask, nx, ny):
                            cleaned_mask[ny, nx] = 0
        
        # 2. Apply morphological operations to smooth boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Apply light erosion to reduce vessel width slightly
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned_mask = cv2.erode(cleaned_mask, erode_kernel, iterations=1)
        
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Keep only the largest connected component
        num_labels, labels = cv2.connectedComponents(cleaned_mask)
        if num_labels > 1:
            # Find largest component (excluding background)
            component_sizes = np.bincount(labels.ravel())[1:]  # Skip background
            if len(component_sizes) > 0:
                largest_component = np.argmax(component_sizes) + 1
                cleaned_mask = (labels == largest_component).astype(np.uint8) * 255
        
        logger.info(f"Reference point artifact removal: {np.sum(mask)} -> {np.sum(cleaned_mask)} pixels")
        return cleaned_mask
    
    def _is_isolated_pixel(self, mask: np.ndarray, x: int, y: int) -> bool:
        """
        Check if a pixel is isolated (likely an artifact)
        """
        if mask[y, x] == 0:
            return False
        
        # Count connected neighbors in 8-connectivity
        connected_neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if mask[ny, nx] > 0:
                        connected_neighbors += 1
        
        # If pixel has less than 2 connected neighbors, it's likely isolated
        return connected_neighbors < 2

    def _calculate_perpendicular_vector(self, centerline: np.ndarray, index: int) -> Optional[np.ndarray]:
        """
        Calculates the perpendicular vector at a given index on the centerline.
        """
        n_points = len(centerline)
        if n_points < 2:
            return None

        # Determine tangent
        if index == 0:
            # Forward difference for the first point
            tangent = centerline[1] - centerline[0]
        elif index == n_points - 1:
            # Backward difference for the last point
            tangent = centerline[index] - centerline[index - 1]
        else:
            # Central difference for inner points
            tangent = centerline[index + 1] - centerline[index - 1]

        # Convert tangent to float before normalization
        tangent = tangent.astype(np.float64)

        # Normalize tangent vector
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            # If tangent is zero, try a wider window for inner points
            if 1 < index < n_points - 2:
                tangent = (centerline[index + 2] - centerline[index - 2]).astype(np.float64)
                norm = np.linalg.norm(tangent)
            
            # If still no norm, we cannot proceed
            if norm < 1e-6:
                return None

        tangent /= norm

        # Perpendicular is a 90-degree rotation: (x, y) -> (-y, x)
        # Centerline is (y, x), so tangent is (dy, dx)
        # Perpendicular is (-dx, dy)
        perpendicular = np.array([-tangent[1], tangent[0]])
        return perpendicular

    def calculate_vessel_wall_thickness(self, mask: np.ndarray) -> Dict:
        """
        Calculate vessel wall thickness statistics
        """
        try:
            # Calculate distance transform
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

            # Get non-zero distances (inside vessel)
            vessel_distances = dist[mask > 0]

            if len(vessel_distances) > 0:
                thickness_stats = {
                    'mean_thickness': float(np.mean(vessel_distances)),
                    'max_thickness': float(np.max(vessel_distances)),
                    'min_thickness': float(np.min(vessel_distances)),
                    'std_thickness': float(np.std(vessel_distances)),
                    'median_thickness': float(np.median(vessel_distances))
                }

                logger.info(f"Vessel wall thickness - Mean: {thickness_stats['mean_thickness']:.2f}px, "
                          f"Max: {thickness_stats['max_thickness']:.2f}px, "
                          f"Min: {thickness_stats['min_thickness']:.2f}px")
            else:
                thickness_stats = {
                    'mean_thickness': 0.0,
                    'max_thickness': 0.0,
                    'min_thickness': 0.0,
                    'std_thickness': 0.0,
                    'median_thickness': 0.0
                }

            return thickness_stats

        except Exception as e:
            logger.error(f"Wall thickness calculation failed: {e}")
            return {
                'mean_thickness': 0.0,
                'max_thickness': 0.0,
                'min_thickness': 0.0,
                'std_thickness': 0.0,
                'median_thickness': 0.0
            }

    def _limit_mask_to_user_region(self, mask: np.ndarray, user_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Limit mask to region between ONLY the most proximal and distal points
        This prevents segmentation from extending beyond the first and last reference points,
        ignoring any middle reference points for boundary calculation.
        """
        if len(user_points) < 2:
            return mask

        original_sum = np.sum(mask)

        # Get proximal and distal points
        proximal = np.array(user_points[0], dtype=np.float32)
        distal = np.array(user_points[-1], dtype=np.float32)

        # Calculate the vessel direction vector
        direction = distal - proximal
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0:
            direction = direction / direction_norm

            # Create perpendicular vector
            perpendicular = np.array([-direction[1], direction[0]])

            # Create mask to limit region
            h, w = mask.shape
            limit_mask = np.zeros_like(mask)

            # Create a tighter band along the vessel path for proximal limiting
            # Dynamic radius based on image size - TIGHTER than moderate
            img_diagonal = np.sqrt(h**2 + w**2)
            base_radius = int(img_diagonal * 0.025)  # 2.5% of diagonal (reduced from 5%)
            
            # Use only proximal and distal points for region definition
            proximal_point = user_points[0]
            distal_point = user_points[-1]
            
            # Create circular regions for proximal and distal points only - TIGHTER
            radius = int(base_radius * 1.2)  # Reduced endpoint radius (was 1.5)
            cv2.circle(limit_mask, (int(proximal_point[0]), int(proximal_point[1])), radius, 1, -1)
            cv2.circle(limit_mask, (int(distal_point[0]), int(distal_point[1])), radius, 1, -1)

            # Connect proximal and distal with a thick line (ignore middle points)
            pt1 = (int(proximal_point[0]), int(proximal_point[1]))
            pt2 = (int(distal_point[0]), int(distal_point[1]))
            cv2.line(limit_mask, pt1, pt2, 1, thickness=base_radius)

            # Apply limit mask
            mask = mask & limit_mask

            # Now cut off extensions beyond endpoints
            # Only if we still have a good amount of pixels
            if np.sum(mask) > 100:
                # Proximal cutoff - TIGHTER limiting
                cutoff_perpendicular = perpendicular * 150  # Tighter cutoff (reduced from 200)
                prox_center = proximal - direction * 10  # Move cutoff line less (reduced from 20)

                # Create polygon for proximal cutoff
                prox_poly = np.array([
                    prox_center - cutoff_perpendicular,
                    prox_center + cutoff_perpendicular,
                    prox_center + cutoff_perpendicular - direction * 1000,
                    prox_center - cutoff_perpendicular - direction * 1000
                ], np.int32)

                # Remove proximal extension
                cv2.fillPoly(mask, [prox_poly], 0)

                # Distal cutoff - TIGHTER limiting
                dist_center = distal + direction * 10  # Move cutoff line less (reduced from 20)

                # Create polygon for distal cutoff
                dist_poly = np.array([
                    dist_center - cutoff_perpendicular,
                    dist_center + cutoff_perpendicular,
                    dist_center + cutoff_perpendicular + direction * 1000,
                    dist_center - cutoff_perpendicular + direction * 1000
                ], np.int32)

                # Remove distal extension
                cv2.fillPoly(mask, [dist_poly], 0)

            final_sum = np.sum(mask)
            logger.info(f"Limited mask to user region. Pixels: {original_sum} -> {final_sum}")

        return mask

    def _enforce_centerline_direction(self, centerline: np.ndarray, user_points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Enforce consistent centerline direction to always go from proximal to distal reference point.
        This is CRITICAL for ensuring consistent QCA analysis across frames.
        
        Args:
            centerline: Centerline points (N x 2) in (y, x) format
            user_points: User reference points, first is proximal, last is distal
            
        Returns:
            Centerline with correct orientation (proximal -> distal)
        """
        if len(centerline) < 2 or len(user_points) < 2:
            return centerline
            
        try:
            # Get proximal and distal reference points
            proximal_ref = np.array([user_points[0][1], user_points[0][0]])  # Convert to (y, x)
            distal_ref = np.array([user_points[-1][1], user_points[-1][0]])  # Convert to (y, x)
            
            # Calculate distances from centerline endpoints to reference points
            start_point = centerline[0]
            end_point = centerline[-1]
            
            # Distance from start of centerline to proximal reference
            start_to_proximal = np.linalg.norm(start_point - proximal_ref)
            start_to_distal = np.linalg.norm(start_point - distal_ref)
            
            # Distance from end of centerline to proximal reference  
            end_to_proximal = np.linalg.norm(end_point - proximal_ref)
            end_to_distal = np.linalg.norm(end_point - distal_ref)
            
            logger.debug(f"Centerline orientation check:")
            logger.debug(f"Start->Proximal: {start_to_proximal:.2f}, Start->Distal: {start_to_distal:.2f}")
            logger.debug(f"End->Proximal: {end_to_proximal:.2f}, End->Distal: {end_to_distal:.2f}")
            
            # Check if centerline is oriented correctly (start closer to proximal, end closer to distal)
            correct_orientation = (start_to_proximal < start_to_distal) and (end_to_distal < end_to_proximal)
            
            if not correct_orientation:
                # Reverse the centerline to correct orientation
                centerline = centerline[::-1]
                logger.info("Reversed centerline to ensure proximal->distal orientation")
                
                # Verify the correction
                new_start = centerline[0]
                new_end = centerline[-1]
                new_start_to_proximal = np.linalg.norm(new_start - proximal_ref)
                new_end_to_distal = np.linalg.norm(new_end - distal_ref)
                logger.debug(f"After reversal - Start->Proximal: {new_start_to_proximal:.2f}, End->Distal: {new_end_to_distal:.2f}")
            else:
                logger.debug("Centerline orientation already correct")
            
            return centerline
            
        except Exception as e:
            logger.warning(f"Failed to enforce centerline direction: {e}")
            return centerline

    def _interpolate_centerline(self, centerline: np.ndarray, target_points: int = 200) -> np.ndarray:
        """
        Interpolate centerline to have exactly target_points.
        This ensures consistency with QCA analysis that expects 200 points.
        
        Args:
            centerline: Original centerline points (N x 2) 
            target_points: Target number of points (default 200)
            
        Returns:
            Interpolated centerline with target_points points
        """
        if len(centerline) < 2:
            logger.warning("Centerline too short for interpolation")
            return centerline
            
        if len(centerline) == target_points:
            return centerline
            
        try:
            from scipy.interpolate import interp1d
            
            # Calculate cumulative distances along centerline
            distances = np.zeros(len(centerline))
            for i in range(1, len(centerline)):
                distances[i] = distances[i-1] + np.linalg.norm(centerline[i] - centerline[i-1])
            
            # Normalize distances to [0, 1]
            if distances[-1] > 0:
                distances = distances / distances[-1]
            else:
                distances = np.linspace(0, 1, len(centerline))
            
            # Create interpolation functions for x and y coordinates
            # Use cubic for longer centerlines, linear for shorter ones
            kind = 'cubic' if len(centerline) >= 4 else 'linear'
            interp_y = interp1d(distances, centerline[:, 0], kind=kind, 
                               bounds_error=False, fill_value='extrapolate')
            interp_x = interp1d(distances, centerline[:, 1], kind=kind,
                               bounds_error=False, fill_value='extrapolate')
            
            # Generate new parameter values for target_points
            new_distances = np.linspace(0, 1, target_points)
            
            # Interpolate coordinates
            new_y = interp_y(new_distances)
            new_x = interp_x(new_distances)
            
            # Combine into new centerline
            interpolated_centerline = np.column_stack((new_y, new_x))
            
            logger.info(f"Interpolated centerline from {len(centerline)} to {len(interpolated_centerline)} points")
            return interpolated_centerline
            
        except Exception as e:
            logger.error(f"Centerline interpolation failed: {e}")
            return centerline

    def _curvature_resistant_skeleton_processing(self, skeleton: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Curvature-resistant centerline extraction that minimizes endpoint artifacts.
        This method is more robust to reference point artifacts than FilFinder.
        
        Args:
            skeleton: Initial skeleton from skeletonization
            mask: Original vessel mask
            
        Returns:
            Processed skeleton with reduced endpoint curvature artifacts
        """
        try:
            # Step 1: Distance-based skeleton refinement
            # Use distance transform to guide skeleton to vessel center
            dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            
            # Step 2: Remove small branches and spurious endpoints
            refined_skeleton = self._remove_small_branches(skeleton, min_length=10)
            
            # Step 3: Smooth skeleton using distance transform guidance
            smooth_skeleton = self._distance_guided_smoothing(refined_skeleton, dist_transform)
            
            # Step 4: Remove endpoint artifacts using vessel geometry
            clean_skeleton = self._remove_endpoint_artifacts(smooth_skeleton, mask, dist_transform)
            
            # Step 5: Ensure connectivity
            final_skeleton = self._ensure_skeleton_connectivity(clean_skeleton)
            
            logger.info(f"Curvature-resistant processing: {np.sum(skeleton > 0)} -> {np.sum(final_skeleton > 0)} points")
            return final_skeleton
            
        except Exception as e:
            logger.error(f"Curvature-resistant processing failed: {e}")
            return skeleton

    def _remove_small_branches(self, skeleton: np.ndarray, min_length: int = 10) -> np.ndarray:
        """Remove small branches that often appear at endpoints due to artifacts"""
        try:
            from skimage.morphology import skeletonize
            from scipy import ndimage
            
            # Find branch points (points with more than 2 neighbors)
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
            branch_points = (neighbor_count > 2) & (skeleton > 0)
            
            # Find endpoints (points with only 1 neighbor)
            end_points = (neighbor_count == 1) & (skeleton > 0)
            
            # Label connected components
            labeled_skeleton, num_features = ndimage.label(skeleton > 0)
            
            # For each branch, check if it's short and remove if so
            refined_skeleton = skeleton.copy()
            
            # Get all branch point locations
            branch_coords = np.column_stack(np.where(branch_points))
            
            for branch_y, branch_x in branch_coords:
                # Find all paths from this branch point to endpoints
                # Remove paths shorter than min_length
                self._prune_short_paths_from_branch(refined_skeleton, (branch_y, branch_x), min_length)
            
            return refined_skeleton
            
        except Exception as e:
            logger.error(f"Branch removal failed: {e}")
            return skeleton

    def _prune_short_paths_from_branch(self, skeleton: np.ndarray, branch_point: tuple, min_length: int):
        """Prune short paths emanating from a branch point"""
        try:
            by, bx = branch_point
            
            # Get 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                        
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx] > 0:
                            # Trace path from this neighbor
                            path_length = self._trace_path_length(skeleton, (ny, nx), (by, bx))
                            if path_length < min_length and path_length > 0:
                                # Remove this short path
                                self._remove_path(skeleton, (ny, nx), (by, bx))
                                
        except Exception as e:
            logger.error(f"Path pruning failed: {e}")

    def _trace_path_length(self, skeleton: np.ndarray, start_point: tuple, avoid_point: tuple) -> int:
        """Trace path length from start point avoiding a specific point"""
        try:
            visited = set()
            stack = [start_point]
            length = 0
            
            while stack and length < 50:  # Limit to prevent infinite loops
                current = stack.pop()
                if current in visited or current == avoid_point:
                    continue
                    
                visited.add(current)
                length += 1
                
                cy, cx = current
                # Check 8-connected neighbors
                neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                            
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                            skeleton[ny, nx] > 0 and (ny, nx) not in visited and (ny, nx) != avoid_point):
                            neighbors += 1
                            stack.append((ny, nx))
                
                # If this is an endpoint (only 1 neighbor), stop tracing
                if neighbors <= 1:
                    break
                    
            return length
            
        except Exception as e:
            logger.error(f"Path tracing failed: {e}")
            return 0

    def _remove_path(self, skeleton: np.ndarray, start_point: tuple, avoid_point: tuple):
        """Remove a path from start point avoiding a specific point"""
        try:
            visited = set()
            stack = [start_point]
            
            while stack:
                current = stack.pop()
                if current in visited or current == avoid_point:
                    continue
                    
                visited.add(current)
                cy, cx = current
                skeleton[cy, cx] = 0  # Remove this point
                
                # Continue to neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                            
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                            skeleton[ny, nx] > 0 and (ny, nx) not in visited and (ny, nx) != avoid_point):
                            
                            # Count neighbors of this point
                            neighbor_count = 0
                            for ddy in [-1, 0, 1]:
                                for ddx in [-1, 0, 1]:
                                    if ddy == 0 and ddx == 0:
                                        continue
                                    nny, nnx = ny + ddy, nx + ddx
                                    if (0 <= nny < skeleton.shape[0] and 0 <= nnx < skeleton.shape[1] and
                                        skeleton[nny, nnx] > 0):
                                        neighbor_count += 1
                            
                            # Only continue on path if it's not a branch point
                            if neighbor_count <= 2:
                                stack.append((ny, nx))
                                
        except Exception as e:
            logger.error(f"Path removal failed: {e}")

    def _distance_guided_smoothing(self, skeleton: np.ndarray, dist_transform: np.ndarray) -> np.ndarray:
        """Smooth skeleton using distance transform to stay in vessel center"""
        try:
            smoothed_skeleton = skeleton.copy()
            
            # Get skeleton points
            skel_points = np.column_stack(np.where(skeleton > 0))
            
            for y, x in skel_points:
                # Look at 5x5 neighborhood
                best_y, best_x = y, x
                best_distance = dist_transform[y, x]
                
                # Find point with maximum distance (closest to vessel center)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]):
                            if dist_transform[ny, nx] > best_distance:
                                best_distance = dist_transform[ny, nx]
                                best_y, best_x = ny, nx
                
                # Move skeleton point to better location if significantly better
                if best_distance > dist_transform[y, x] * 1.2:  # 20% improvement threshold
                    smoothed_skeleton[y, x] = 0
                    smoothed_skeleton[best_y, best_x] = 255
                    
            return smoothed_skeleton
            
        except Exception as e:
            logger.error(f"Distance-guided smoothing failed: {e}")
            return skeleton

    def _remove_endpoint_artifacts(self, skeleton: np.ndarray, mask: np.ndarray, dist_transform: np.ndarray) -> np.ndarray:
        """Remove artifacts at endpoints that cause curvature"""
        try:
            from scipy import ndimage
            
            # Find endpoints
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, 1] = 0
            neighbor_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
            endpoints = (neighbor_count == 1) & (skeleton > 0)
            
            cleaned_skeleton = skeleton.copy()
            endpoint_coords = np.column_stack(np.where(endpoints))
            
            for ey, ex in endpoint_coords:
                # Trace back from endpoint and check for curvature artifacts
                path = self._trace_endpoint_path(skeleton, (ey, ex), max_length=15)
                
                if len(path) >= 8:  # Need sufficient points to detect curvature
                    # Check if endpoint region has high curvature (artifact)
                    if self._detect_endpoint_artifact(path, dist_transform):
                        # Remove the artifactual curved part (first few points)
                        removal_count = min(5, len(path) // 3)
                        for i in range(removal_count):
                            if i < len(path):
                                py, px = path[i]
                                cleaned_skeleton[py, px] = 0
                        
                        logger.info(f"Removed endpoint artifact at ({ey}, {ex}), removed {removal_count} points")
                        
            return cleaned_skeleton
            
        except Exception as e:
            logger.error(f"Endpoint artifact removal failed: {e}")
            return skeleton

    def _trace_endpoint_path(self, skeleton: np.ndarray, start_point: tuple, max_length: int = 15) -> list:
        """Trace path from endpoint for curvature analysis"""
        path = [start_point]
        current = start_point
        visited = {start_point}
        
        for _ in range(max_length - 1):
            cy, cx = current
            next_point = None
            
            # Find next unvisited neighbor
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] > 0 and (ny, nx) not in visited):
                        next_point = (ny, nx)
                        break
                        
                if next_point:
                    break
            
            if not next_point:
                break
                
            path.append(next_point)
            visited.add(next_point)
            current = next_point
            
        return path

    def _detect_endpoint_artifact(self, path: list, dist_transform: np.ndarray) -> bool:
        """Detect if endpoint path contains artifacts (high curvature + low distance)"""
        if len(path) < 6:
            return False
            
        try:
            # Calculate curvature at each point in the path
            curvatures = []
            distances = []
            
            for i in range(1, len(path) - 1):
                p1 = np.array(path[i-1])
                p2 = np.array(path[i])
                p3 = np.array(path[i+1])
                
                # Calculate angle change (curvature indicator)
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    
                    # Calculate angle between vectors (0 = straight, 1 = curved)
                    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                    curvature = 1 - (dot_product + 1) / 2
                    curvatures.append(curvature)
                    
                    # Get distance transform value at this point
                    py, px = path[i]
                    distances.append(dist_transform[py, px])
            
            if not curvatures or not distances:
                return False
                
            # Check if early part of path has high curvature and low distance
            early_curvature = np.mean(curvatures[:3]) if len(curvatures) >= 3 else 0
            early_distance = np.mean(distances[:3]) if len(distances) >= 3 else 0
            late_distance = np.mean(distances[-3:]) if len(distances) >= 3 else 0
            
            # Artifact indicators:
            # 1. High curvature at start (> 0.4)
            # 2. Low distance compared to later parts (< 0.7 ratio)
            has_high_curvature = early_curvature > 0.4
            has_low_distance = late_distance > 0 and early_distance / late_distance < 0.7
            
            return has_high_curvature and has_low_distance
            
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            return False

    def _ensure_skeleton_connectivity(self, skeleton: np.ndarray) -> np.ndarray:
        """Ensure skeleton maintains connectivity after processing"""
        try:
            from scipy import ndimage
            
            # Find connected components
            labeled, num_features = ndimage.label(skeleton > 0)
            
            if num_features <= 1:
                return skeleton  # Already connected or empty
                
            # Keep the largest component
            component_sizes = []
            for i in range(1, num_features + 1):
                size = np.sum(labeled == i)
                component_sizes.append((size, i))
                
            # Sort by size, keep largest
            component_sizes.sort(reverse=True)
            largest_component_label = component_sizes[0][1]
            
            # Create result with only largest component
            result = np.zeros_like(skeleton)
            result[labeled == largest_component_label] = 255
            
            logger.info(f"Connectivity: kept largest component ({component_sizes[0][0]} points) out of {num_features} components")
            return result
            
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            return skeleton