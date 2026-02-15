#!/usr/bin/env python3
"""
Compute metrics for all saved epoch predictions and create comparison visualizations.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import wandb


def compute_metrics(pred_mask, gt_mask):
    """Compute comprehensive segmentation metrics."""
    # Binarize masks
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)

    # Flatten
    pred_flat = pred_binary.flatten()
    gt_flat = gt_binary.flatten()

    # Compute intersection and union
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()

    # Compute metrics
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred_flat.sum() + gt_flat.sum() + 1e-6)

    # Precision and Recall
    tp = intersection
    fp = (pred_flat & ~gt_flat).sum()
    fn = (gt_flat & ~pred_flat).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall)
    }


def evaluate_epoch(pred_dir, gt_dir, epoch_num):
    """Evaluate predictions for a single epoch."""
    print(f"\nEvaluating Epoch {epoch_num}...")

    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    pred_files = sorted(list(pred_dir.glob('*.png')))
    gt_files = sorted(list(gt_dir.glob('*.png')))

    if len(pred_files) == 0:
        print(f"  Warning: No predictions found in {pred_dir}")
        return None

    all_metrics = []

    for pred_path in tqdm(pred_files, desc=f"  Epoch {epoch_num}", leave=False):
        # Load prediction
        pred_mask = np.array(Image.open(pred_path).convert('L'))

        # Find matching GT (by index)
        idx = pred_files.index(pred_path)
        if idx >= len(gt_files):
            continue

        gt_path = gt_files[idx]
        gt_mask = np.array(Image.open(gt_path).convert('L'))

        # Compute metrics
        metrics = compute_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)

    # Compute aggregate metrics
    df = pd.DataFrame(all_metrics)

    results = {
        'epoch': epoch_num,
        'num_images': len(df),
        'mIoU': df['iou'].mean(),
        'mean_dice': df['dice'].mean(),
        'mean_precision': df['precision'].mean(),
        'mean_recall': df['recall'].mean(),
        'std_iou': df['iou'].std(),
        'std_dice': df['dice'].std(),
        'median_iou': df['iou'].median(),
        'median_dice': df['dice'].median()
    }

    print(f"  ✓ Epoch {epoch_num}: mIoU={results['mIoU']:.4f}, Dice={results['mean_dice']:.4f}")

    return results, df


def main():
    parser = argparse.ArgumentParser(description='Compare metrics across all epochs')
    parser.add_argument('--predictions_base', type=str, default='results/sam3_lora',
                        help='Base directory containing epoch predictions')
    parser.add_argument('--gt_dir', type=str, default='cracks_data/valid/masks',
                        help='Ground truth masks directory')
    parser.add_argument('--output_dir', type=str, default='results/epoch_comparison',
                        help='Output directory for comparison results')
    parser.add_argument('--num_epochs', type=int, default=6,
                        help='Number of epochs to evaluate')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='sam3-crack-segmentation',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default='epoch_comparison',
                        help='W&B run name')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": "SAM3-LoRA",
                "dataset": "crack_validation",
                "num_epochs": args.num_epochs
            }
        )
        print(f"W&B initialized: {wandb.run.name}")

    print("="*70)
    print("Comparing Metrics Across All Epochs")
    print("="*70)

    # Evaluate each epoch
    all_results = []
    epoch_dfs = {}

    base_dir = Path(args.predictions_base)
    gt_dir = Path(args.gt_dir)

    for epoch in range(1, args.num_epochs + 1):
        pred_dir = base_dir / f'val_predictions_epoch_{epoch}'

        if not pred_dir.exists():
            print(f"\nWarning: Predictions not found for epoch {epoch} at {pred_dir}")
            continue

        results, df = evaluate_epoch(pred_dir, gt_dir, epoch)
        if results:
            all_results.append(results)
            epoch_dfs[epoch] = df

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results)

    # Add baseline (zero-shot)
    baseline = {
        'epoch': 0,
        'num_images': 806,
        'mIoU': 0.3855,
        'mean_dice': 0.5170,
        'mean_precision': 0.7361,
        'mean_recall': 0.4716,
        'std_iou': 0.0,
        'std_dice': 0.0,
        'median_iou': 0.0,
        'median_dice': 0.0
    }
    comparison_df = pd.concat([pd.DataFrame([baseline]), comparison_df], ignore_index=True)

    # Print comparison table
    print("\n" + "="*70)
    print("EPOCH COMPARISON")
    print("="*70)
    print(comparison_df[['epoch', 'mIoU', 'mean_dice', 'mean_precision', 'mean_recall']].to_string(index=False))
    print("="*70)

    # Calculate improvements
    print("\nImprovement from Baseline:")
    for _, row in comparison_df[comparison_df['epoch'] > 0].iterrows():
        epoch = int(row['epoch'])
        miou_improvement = (row['mIoU'] - baseline['mIoU']) / baseline['mIoU'] * 100
        dice_improvement = (row['mean_dice'] - baseline['mean_dice']) / baseline['mean_dice'] * 100
        print(f"  Epoch {epoch}: mIoU +{miou_improvement:.1f}%, Dice +{dice_improvement:.1f}%")

    # Save comparison results
    comparison_file = output_dir / 'epoch_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Saved epoch comparison to: {comparison_file}")

    # Save JSON summary
    summary_file = output_dir / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved summary to: {summary_file}")

    # Log to wandb
    if args.use_wandb:
        # Log comparison table
        wandb_table = wandb.Table(dataframe=comparison_df)
        wandb.log({"epoch_comparison_table": wandb_table})

        # Log metrics over epochs
        for _, row in comparison_df.iterrows():
            if row['epoch'] > 0:  # Skip baseline for line plots
                wandb.log({
                    'epoch': int(row['epoch']),
                    'eval/mIoU': row['mIoU'],
                    'eval/dice': row['mean_dice'],
                    'eval/precision': row['mean_precision'],
                    'eval/recall': row['mean_recall']
                }, step=int(row['epoch']))

        # Create improvement chart
        improvement_data = []
        for _, row in comparison_df[comparison_df['epoch'] > 0].iterrows():
            improvement_data.append([
                int(row['epoch']),
                (row['mIoU'] - baseline['mIoU']) * 100,  # Absolute improvement in percentage points
                (row['mean_dice'] - baseline['mean_dice']) * 100
            ])

        improvement_table = wandb.Table(
            columns=["Epoch", "mIoU Improvement (pp)", "Dice Improvement (pp)"],
            data=improvement_data
        )
        wandb.log({"improvement_from_baseline": improvement_table})

        # Log distribution comparisons for best and worst epochs
        if len(epoch_dfs) >= 2:
            epochs_to_compare = [1, len(epoch_dfs)]  # First and last
            for epoch_num in epochs_to_compare:
                if epoch_num in epoch_dfs:
                    df = epoch_dfs[epoch_num]
                    wandb.log({
                        f"epoch_{epoch_num}/iou_distribution": wandb.Histogram(df['iou'].values),
                        f"epoch_{epoch_num}/dice_distribution": wandb.Histogram(df['dice'].values)
                    })

        # Create summary metrics
        best_epoch = comparison_df.loc[comparison_df['mIoU'].idxmax()]
        wandb.run.summary['best_epoch'] = int(best_epoch['epoch'])
        wandb.run.summary['best_mIoU'] = best_epoch['mIoU']
        wandb.run.summary['best_dice'] = best_epoch['mean_dice']
        wandb.run.summary['improvement_from_baseline'] = (best_epoch['mIoU'] - baseline['mIoU']) / baseline['mIoU'] * 100

    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)

    # Finish wandb
    if args.use_wandb:
        wandb.finish()
        print("W&B run finished")


if __name__ == '__main__':
    main()
