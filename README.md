# Text-Conditioned Segmentation - Technical Report

## Goal
Train a text-conditioned segmentation model using **SAM 3** fine-tuned with **LoRA** for crack detection from natural language prompt: "crack"

---

## Approach

**Model**: SAM 3 (840M parameters) + LoRA (3.88M trainable parameters, 0.46% of total)
- Fine-tuning method: Low-Rank Adaptation
- Loss: Dice Loss
- Optimizer: AdamW (lr=1e-4)
- Training: 6 epochs, batch size 1, FP16 mixed precision
- Hardware: NVIDIA RTX 3080 Laptop (16GB VRAM)

---

## Dataset

**Source**: Roboflow Cracks Dataset (5,369 images)
- Training: 4,563 images (85%)
- Validation: 806 images (15%)
- Format: YOLO polygons → binary masks (0/255)

---

## Results

### Performance Progression

| Epoch | mIoU | Dice | Precision | Recall | mIoU Improvement |
|-------|------|------|-----------|--------|------------------|
| **Baseline (Zero-shot)** | 38.55% | 51.70% | 73.61% | 47.16% | - |
| **1** | 55.72% | 69.81% | 72.84% | 77.39% | +17.17 pp |
| **2** | 58.39% | 71.94% | 71.45% | 81.86% | +19.84 pp |
| **3** | 60.31% | 73.63% | 77.11% | 77.91% | +21.76 pp |
| **4** | 61.54% | 74.66% | 78.33% | 78.05% | +22.99 pp |
| **5** | 62.63% | 75.56% | 79.56% | 77.92% | +24.08 pp |
| **6 (Best)** ✅ | **63.99%** | **76.63%** | **77.04%** | **82.15%** | **+25.44 pp** |

### Key Achievements
- **Final mIoU: 63.99%** (+25.44 points from baseline)
- **Final Dice: 76.63%** (+24.93 points from baseline)
- **66% relative improvement** over zero-shot baseline
- Balanced precision (77%) and recall (82%)
- Consistent improvement across all epochs

---

## Visual Examples

### Best Performance (IoU > 0.90)

![Best Example 1](visualizations/best_example_iou_0.927.png)
*Excellent segmentation - IoU: 0.927*

![Best Example 2](visualizations/best_example_iou_0.922.png)
*Excellent segmentation - IoU: 0.922*

### Good Performance (IoU 0.70-0.80)

![Good Example 1](visualizations/good_example_iou_0.800.png)
*Good segmentation - IoU: 0.800*

![Good Example 2](visualizations/good_example_iou_0.722.png)
*Good segmentation - IoU: 0.722*

### Challenging Cases (IoU 0.55-0.60)

![Challenging Example 1](visualizations/challenging_example_iou_0.582.png)
*Challenging case - IoU: 0.582*

![Challenging Example 2](visualizations/challenging_example_iou_0.569.png)
*Challenging case - IoU: 0.569*

---

## Runtime & Footprint

### Model Size
| Component | Size |
|-----------|------|
| **Base SAM 3 Model** | 3.2 GB (840M parameters) |
| **LoRA Adapters Only** | ~15 MB (3.88M parameters) |
| **Total Fine-tuned Model** | 3.2 GB | 14 GB training |
| **Deployment Advantage** | Only 15 MB adapters needed if base cached |

### Efficiency Summary
- **Parameter Efficiency**: 99.54% frozen, 0.46% trainable
- **Training Cost**: 13 GPU-hours (rtx 3080 laptop)
- **Memory Efficient**: FP16 mixed precision
- **Deployable**: Lightweight LoRA adapters (15 MB)

---

## Implementation Details

### Code Structure
```
wrapper
├── training/
│   └── finetune_lora.py             # LoRA fine-tuning 
├── scripts/
│   └── prepare_masks.py             # binary mask 
```

### Key Features
- Text-conditioned segmentation with natural language prompts
- Efficient LoRA fine-tuning (only 0.46% parameters trained)
- Comprehensive W&B logging
- Validation predictions saved per epoch

### Technologies Used
- **Framework**: PyTorch 2.5.1, CUDA 12.1
- **Model Library**: HuggingFace Transformers
- **LoRA**: PEFT (Parameter-Efficient Fine-Tuning)
- **Logging**: Weights & Biases (W&B)

---


### Tracked Metrics
1. Train/validation loss and IoU (per epoch)
2. Learning rate scheduling
3. Metric progression across epochs
4. Distribution histograms (IoU, Dice)
5. Sample predictions (best/worst cases)

---
