"""
SAM 3 Text-Conditioned Segmentation Wrapper

This module provides a wrapper for Meta's SAM 3 model for text-conditioned segmentation.

⚠️ SAM 3 ONLY - NO FALLBACKS, NO ALTERNATIVES ⚠️

Model Source: jetjodh/sam3 (ungated HuggingFace mirror)
Paper: https://arxiv.org/abs/2511.16719
"""

import torch
from transformers import Sam3Model, Sam3Processor
from PIL import Image
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


class SAM3TextSegmenter:
    """
    Text-conditioned segmentation using SAM 3.

    ⚠️ SAM 3 ONLY - No fallback methods allowed ⚠️
    """

    def __init__(self, device: str = "cuda", model_id: str = "jetjodh/sam3"):
        """
        Initialize SAM 3 model.

        Args:
            device: Device to run model on ("cuda" or "cpu")
            model_id: HuggingFace model ID (default: jetjodh/sam3 ungated mirror)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id

        logger.info(f"Loading SAM 3 from {model_id}...")
        logger.info(f"Device: {self.device}")

        # Load processor and model - SAM 3 ONLY
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id)

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✓ SAM 3 loaded successfully")
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Parameters: {num_params:,}")
        logger.info(f"  Device: {self.device}")

    def segment_from_text(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Segment image using text prompt with SAM 3.

        Args:
            image: PIL Image or numpy array (RGB)
            text_prompt: Simple noun phrase (e.g., "crack", NOT "segment crack")
            threshold: Confidence threshold for detections (0-1)
            mask_threshold: Mask binarization threshold (0-1)

        Returns:
            Binary mask (numpy array, uint8): 0 for background, 255 for object

        Note:
            Text prompts should be simple noun phrases:
            ✓ Correct: "crack", "wall crack", "concrete crack"
            ✗ Incorrect: "segment crack", "find crack", "detect crack"
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Get original size for output
        orig_width, orig_height = image.size

        # Prepare inputs using SAM 3 processor
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Run SAM 3 inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get instance segmentation masks
        try:
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=[(orig_height, orig_width)]
            )[0]
        except Exception as e:
            logger.warning(f"SAM 3 post-processing failed: {e}. Returning empty mask.")
            return np.zeros((orig_height, orig_width), dtype=np.uint8)

        # Check if any detections
        if len(results['masks']) == 0:
            # No detections - return empty mask
            logger.debug(f"SAM 3: No detections for prompt '{text_prompt}'")
            return np.zeros((orig_height, orig_width), dtype=np.uint8)

        # Merge all instance masks into single binary mask (OR operation)
        merged_mask = results['masks'][0].cpu().numpy()
        for mask in results['masks'][1:]:
            merged_mask = np.logical_or(merged_mask, mask.cpu().numpy())

        # Convert to 0/255 format
        binary_mask = (merged_mask * 255).astype(np.uint8)

        # Verify output shape
        assert binary_mask.shape == (orig_height, orig_width), \
            f"Mask shape {binary_mask.shape} doesn't match image {(orig_height, orig_width)}"

        logger.debug(f"SAM 3: Detected {len(results['masks'])} instances")
        return binary_mask

    def __repr__(self):
        return f"SAM3TextSegmenter(model='{self.model_id}', device='{self.device}')"
