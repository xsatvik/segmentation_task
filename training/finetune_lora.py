#!/usr/bin/env python3
"""
SAM 3 LoRA Fine-Tuning Script

Fine-tune SAM 3 with LoRA (Low-Rank Adaptation) on crack segmentation dataset.
Uses parameter-efficient fine-tuning to adapt the model to crack detection.
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import Sam3Model, Sam3Processor
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrackSegmentationDataset(Dataset):
    """Dataset for crack segmentation with text prompts."""

    def __init__(self, images_dir, masks_dir, processor, text_prompt="crack"):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.processor = processor
        self.text_prompt = text_prompt

        # Get list of images
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) +
                                  list(self.images_dir.glob('*.png')))

        logger.info(f"Found {len(self.image_files)} images in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Load ground truth mask
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Binary mask (0 or 1)

        # Process inputs
        inputs = self.processor(
            images=image,
            text=self.text_prompt,
            return_tensors="pt"
        )

        # Remove batch dimension
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.squeeze(0)

        # Add ground truth mask
        inputs['ground_truth_mask'] = torch.from_numpy(mask)
        inputs['image_path'] = str(img_path)

        return inputs


def compute_dice_loss(pred_masks, gt_masks, smooth=1.0):
    """
    Compute Dice loss for segmentation.

    Args:
        pred_masks: Predicted masks (B, H, W) - logits
        gt_masks: Ground truth masks (B, H, W) - binary
        smooth: Smoothing factor

    Returns:
        Dice loss (scalar)
    """
    # Apply sigmoid to predictions
    pred_masks = torch.sigmoid(pred_masks)

    # Flatten
    pred_flat = pred_masks.reshape(pred_masks.size(0), -1)
    gt_flat = gt_masks.reshape(gt_masks.size(0), -1)

    # Compute Dice coefficient
    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Return Dice loss (1 - Dice)
    return 1.0 - dice.mean()


def compute_iou(pred_masks, gt_masks, threshold=0.5):
    """Compute IoU metric."""
    pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

    intersection = (pred_binary * gt_masks).sum()
    union = (pred_binary + gt_masks).clamp(0, 1).sum()

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def setup_lora_model(model, rank=8, alpha=16, target_modules=None):
    """
    Apply LoRA to SAM 3 model.

    Args:
        model: Base SAM 3 model
        rank: LoRA rank (controls number of trainable parameters)
        alpha: LoRA alpha (scaling factor)
        target_modules: Layers to apply LoRA to

    Returns:
        PEFT model with LoRA adapters
    """
    if target_modules is None:
        # Apply LoRA to vision encoder attention layers
        # Use simple module names - PEFT will match all layers with these names
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=None  # Custom task
    )

    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    logger.info(f"LoRA Configuration:")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Alpha: {alpha}")
    logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"  Total params: {total_params:,}")

    return peft_model


def train_epoch(model, dataloader, optimizer, device, epoch, scaler=None, grad_accum_steps=1):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        gt_masks = batch['ground_truth_mask'].to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get semantic segmentation mask (single aggregated mask for binary segmentation)
            # SAM 3 outputs semantic_seg with shape [B, 1, H, W]
            pred_masks = outputs.semantic_seg if hasattr(outputs, 'semantic_seg') else outputs.pred_masks[:, 0:1]

            # Resize predictions to match GT if needed
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks = nn.functional.interpolate(
                    pred_masks,
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Compute loss (normalize by accumulation steps)
            loss = compute_dice_loss(pred_masks.squeeze(1), gt_masks) / grad_accum_steps

        # Backward pass with gradient accumulation
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Metrics
        batch_iou = compute_iou(pred_masks.squeeze(1), gt_masks)
        total_loss += loss.item() * grad_accum_steps  # Denormalize for logging
        total_iou += batch_iou

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * grad_accum_steps:.4f}',
            'iou': f'{batch_iou:.4f}'
        })

        # Clear GPU cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def validate(model, dataloader, device, epoch, scaler=None):
    """Validate the model with mixed precision."""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")
        for batch in pbar:
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gt_masks = batch['ground_truth_mask'].to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Get semantic segmentation mask (single aggregated mask for binary segmentation)
                pred_masks = outputs.semantic_seg if hasattr(outputs, 'semantic_seg') else outputs.pred_masks[:, 0:1]

                # Resize if needed
                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = nn.functional.interpolate(
                        pred_masks,
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Compute metrics
                loss = compute_dice_loss(pred_masks.squeeze(1), gt_masks)
                batch_iou = compute_iou(pred_masks.squeeze(1), gt_masks)

            total_loss += loss.item()
            total_iou += batch_iou

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })

    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou


def save_validation_predictions(model, dataloader, device, output_dir, epoch, scaler=None):
    """Save validation predictions to disk."""
    model.eval()
    predictions_dir = output_dir / f'val_predictions_epoch_{epoch}'
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving validation predictions to {predictions_dir}...")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Saving Val Preds]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Get semantic segmentation mask
                pred_masks = outputs.semantic_seg if hasattr(outputs, 'semantic_seg') else outputs.pred_masks[:, 0:1]

                # Resize to original size if needed
                gt_masks = batch['ground_truth_mask']
                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = nn.functional.interpolate(
                        pred_masks,
                        size=gt_masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

            # Save each prediction in the batch
            for i in range(pred_masks.shape[0]):
                # Convert to binary mask (0-255)
                pred_mask = (pred_masks[i, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255

                # Get image filename from batch if available, otherwise use index
                img_idx = batch_idx * dataloader.batch_size + i
                pred_path = predictions_dir / f'pred_{img_idx:05d}.png'

                # Save prediction
                Image.fromarray(pred_mask).save(pred_path)

    logger.info(f"✓ Saved {len(dataloader.dataset)} validation predictions")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune SAM 3 with LoRA')
    parser.add_argument('--train_dir', type=str, default='cracks_data/train',
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='cracks_data/valid',
                        help='Validation data directory')
    parser.add_argument('--output_dir', type=str, default='results/sam3_lora',
                        help='Output directory for checkpoints')
    parser.add_argument('--model_id', type=str, default='jetjodh/sam3',
                        help='SAM 3 model ID')
    parser.add_argument('--text_prompt', type=str, default='crack',
                        help='Text prompt for segmentation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='sam3-crack-segmentation',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (fp16)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    logger.info("="*70)
    logger.info("SAM 3 LoRA Fine-Tuning")
    logger.info("="*70)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Text prompt: '{args.text_prompt}'")
    logger.info(f"Training dir: {args.train_dir}")
    logger.info(f"Validation dir: {args.val_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"Device: {args.device}")
    logger.info(f"W&B logging: {args.use_wandb}")
    logger.info("="*70)

    # Initialize wandb
    if args.use_wandb:
        run_name = args.wandb_run_name or f"sam3_lora_r{args.lora_rank}_lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model_id,
                "text_prompt": args.text_prompt,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "device": args.device,
                "seed": args.seed
            }
        )
        logger.info(f"W&B initialized: {wandb.run.name}")

    # Load processor and model
    logger.info("Loading SAM 3 model and processor...")
    processor = Sam3Processor.from_pretrained(args.model_id)
    base_model = Sam3Model.from_pretrained(args.model_id)

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = setup_lora_model(base_model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(args.device)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        if hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
        elif hasattr(model.base_model, 'enable_gradient_checkpointing'):
            model.base_model.enable_gradient_checkpointing()
        else:
            logger.warning("Gradient checkpointing not available for this model")

    # Create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        logger.info("Mixed precision training enabled (fp16)")
    if args.grad_accum_steps > 1:
        logger.info(f"Gradient accumulation steps: {args.grad_accum_steps}")
        logger.info(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CrackSegmentationDataset(
        images_dir=Path(args.train_dir) / 'images',
        masks_dir=Path(args.train_dir) / 'masks',
        processor=processor,
        text_prompt=args.text_prompt
    )

    val_dataset = CrackSegmentationDataset(
        images_dir=Path(args.val_dir) / 'images',
        masks_dir=Path(args.val_dir) / 'masks',
        processor=processor,
        text_prompt=args.text_prompt
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_iou = 0.0
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, optimizer, args.device, epoch,
            scaler=scaler, grad_accum_steps=args.grad_accum_steps
        )

        # Validate
        val_loss, val_iou = validate(model, val_loader, args.device, epoch, scaler=scaler)

        # Save validation predictions
        save_validation_predictions(model, val_loader, args.device, output_dir, epoch, scaler=scaler)

        # Update scheduler
        scheduler.step()

        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/iou': train_iou,
                'val/loss': val_loss,
                'val/iou': val_iou,
                'learning_rate': scheduler.get_last_lr()[0]
            }, step=epoch)

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = output_dir / 'best_model.pt'
            model.save_pretrained(output_dir / 'best_lora_adapters')
            logger.info(f"✓ Saved best model (IoU: {val_iou:.4f})")

        # Save checkpoint every epoch
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
        model.save_pretrained(output_dir / f'lora_adapters_epoch_{epoch}')
        logger.info(f"✓ Saved checkpoint at epoch {epoch}")

    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    logger.info("="*70)
    logger.info("Training Complete!")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best validation IoU: {best_val_iou:.4f}")
    logger.info(f"LoRA adapters saved to: {output_dir / 'best_lora_adapters'}")
    logger.info("="*70)

    # Finish wandb run
    if args.use_wandb:
        wandb.log({
            'best_val_iou': best_val_iou,
            'total_training_time_hours': total_time / 3600
        })
        wandb.finish()
        logger.info("W&B run finished")


if __name__ == '__main__':
    main()
