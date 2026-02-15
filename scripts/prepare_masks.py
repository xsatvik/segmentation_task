"""
Script to convert YOLO polygon annotations to binary masks for validation set.

This processes only the validation set (806 images) to enable baseline evaluation.
Training set conversion can be done later if fine-tuning is needed.
"""

import sys
from pathlib import Path
import logging
import argparse
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import (
    convert_yolo_labels_to_masks,
    visualize_mask_overlay
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO annotations to binary masks for validation set'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='cracks_data/valid',
        help='Path to validation data directory'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        default=0,
        help='YOLO class ID to convert (default: 0 for crack)'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        default=5,
        help='Number of random samples to visualize (default: 5)'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    masks_dir = data_dir / 'masks'

    # Verify input directories exist
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    logger.info(f"Processing validation set from: {data_dir}")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Labels: {labels_dir}")
    logger.info(f"Output masks: {masks_dir}")

    # Convert labels to masks
    logger.info("Converting YOLO polygons to binary masks...")
    num_masks = convert_yolo_labels_to_masks(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=masks_dir,
        class_id=args.class_id
    )

    logger.info(f"Successfully created {num_masks} binary masks")

    # Verify masks
    mask_files = list(masks_dir.glob('*.png'))
    logger.info(f"Verification: {len(mask_files)} mask files found in {masks_dir}")

    # Create visualizations for random samples
    if args.visualize > 0 and mask_files:
        logger.info(f"Creating {args.visualize} sample visualizations...")
        vis_dir = masks_dir.parent / 'mask_visualizations'
        vis_dir.mkdir(exist_ok=True)

        # Select random samples
        num_samples = min(args.visualize, len(mask_files))
        sample_masks = random.sample(mask_files, num_samples)

        for i, mask_path in enumerate(sample_masks, 1):
            # Find corresponding image
            img_name = mask_path.stem
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                candidate = images_dir / f"{img_name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            # Create overlay visualization
            output_path = vis_dir / f"sample_{i:02d}_{mask_path.stem}.jpg"
            try:
                visualize_mask_overlay(image_path, mask_path, output_path)
                logger.info(f"Created visualization: {output_path.name}")
            except Exception as e:
                logger.error(f"Error creating visualization: {e}")

        logger.info(f"Visualizations saved to: {vis_dir}")

    logger.info("Mask preparation complete!")
    logger.info(f"Next step: Run baseline inference with SAM 3")


if __name__ == '__main__':
    main()
