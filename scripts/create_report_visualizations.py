#!/usr/bin/env python3
"""
Create visualization images for the report showing:
- Original image
- Ground truth mask
- Predicted mask overlaid on original
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json

def create_visualization(image_path, gt_mask_path, pred_mask_path, output_path, title=""):
    """Create a 3-panel visualization."""
    # Load images
    image = Image.open(image_path).convert('RGB')
    gt_mask = Image.open(gt_mask_path).convert('L')
    pred_mask = Image.open(pred_mask_path).convert('L')

    # Convert to numpy
    image_np = np.array(image)
    gt_mask_np = np.array(gt_mask)
    pred_mask_np = np.array(pred_mask)

    # Create overlay (red overlay on original image)
    overlay = image_np.copy()
    pred_binary = pred_mask_np > 127
    overlay[pred_binary] = overlay[pred_binary] * 0.5 + np.array([255, 0, 0]) * 0.5

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Ground truth mask
    axes[1].imshow(gt_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Prediction overlay
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Predicted Segmentation', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    # Paths
    images_dir = Path('cracks_data/valid/images')
    gt_masks_dir = Path('cracks_data/valid/masks')
    pred_masks_dir = Path('results/sam3_lora/val_predictions_epoch_6')
    output_dir = Path('results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-image metrics to select good examples
    metrics_file = Path('results/epoch_comparison/epoch_6_per_image_metrics.csv')

    # If metrics file doesn't exist, compute IoU for each image
    if not metrics_file.exists():
        print("Computing metrics to select best examples...")
        import pandas as pd

        pred_files = sorted(list(pred_masks_dir.glob('*.png')))
        gt_files = sorted(list(gt_masks_dir.glob('*.png')))

        metrics = []
        for i, (pred_path, gt_path) in enumerate(zip(pred_files, gt_files)):
            pred = np.array(Image.open(pred_path).convert('L'))
            gt = np.array(Image.open(gt_path).convert('L'))

            pred_binary = (pred > 127).astype(np.uint8)
            gt_binary = (gt > 127).astype(np.uint8)

            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            iou = intersection / (union + 1e-6)

            metrics.append({
                'index': i,
                'iou': iou,
                'pred_file': pred_path.name,
                'gt_file': gt_path.name
            })

        df = pd.DataFrame(metrics)
    else:
        import pandas as pd
        df = pd.read_csv(metrics_file)

    # Select examples based on IoU ranges
    examples = []

    # Best performance (IoU > 0.85)
    best = df[df['iou'] > 0.85].nlargest(2, 'iou')
    if len(best) > 0:
        for _, row in best.iterrows():
            examples.append(('best', row['index'], row['iou']))

    # Good performance (IoU 0.70-0.85)
    good = df[(df['iou'] >= 0.70) & (df['iou'] <= 0.85)].sample(min(2, len(df[(df['iou'] >= 0.70) & (df['iou'] <= 0.85)])))
    if len(good) > 0:
        for _, row in good.iterrows():
            examples.append(('good', row['index'], row['iou']))

    # Challenging cases (IoU 0.40-0.60)
    challenging = df[(df['iou'] >= 0.40) & (df['iou'] <= 0.60)].sample(min(2, len(df[(df['iou'] >= 0.40) & (df['iou'] <= 0.60)])))
    if len(challenging) > 0:
        for _, row in challenging.iterrows():
            examples.append(('challenging', row['index'], row['iou']))

    print(f"\nCreating {len(examples)} visualization examples...")

    # Get file lists
    image_files = sorted(list(images_dir.glob('*.jpg'))) + sorted(list(images_dir.glob('*.png')))
    gt_files = sorted(list(gt_masks_dir.glob('*.png')))
    pred_files = sorted(list(pred_masks_dir.glob('*.png')))

    # Create visualizations
    for category, idx, iou in examples:
        if idx >= len(image_files):
            continue

        image_path = image_files[idx]
        gt_path = gt_files[idx]
        pred_path = pred_files[idx]

        output_path = output_dir / f'{category}_example_iou_{iou:.3f}.png'
        title = f'{category.capitalize()} Performance - IoU: {iou:.3f}'

        create_visualization(image_path, gt_path, pred_path, output_path, title)

    print(f"\n✓ Created {len(examples)} visualizations in {output_dir}")
    print(f"\nVisualization files:")
    for viz_file in sorted(output_dir.glob('*.png')):
        print(f"  - {viz_file.name}")


if __name__ == '__main__':
    main()
