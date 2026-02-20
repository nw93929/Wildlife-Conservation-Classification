# Wildlife Conservation Image Classification

## Overview

A deep learning pipeline for classifying wildlife species from camera trap images, built for the [DrivenData Conser-vision Competition](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/). The system identifies 8 species categories from images captured in Tai National Park, Cote d'Ivoire.

Camera traps generate massive volumes of images that overwhelm manual review. This project automates species classification to help conservationists monitor wildlife populations more efficiently.

## Species Classes

| Class | Description |
|-------|-------------|
| antelope_duiker | Duiker antelope species |
| bird | Various bird species |
| blank | No animal detected |
| civet_genet | Civet or genet |
| hog | Wild hog species |
| leopard | Leopard |
| monkey_prosimian | Monkeys and prosimians |
| rodent | Rodent species |

## Key Features

- **5-Fold Cross-Validation** with `StratifiedGroupKFold` — splits by camera trap site to prevent data leakage from shared backgrounds
- **Two-Stage Inference Pipeline** — binary blank/animal classifier gates predictions before species classifier; thresholds jointly optimized via grid search on OOF predictions
- **Binary Classifier** — separate ConvNeXt model trained to distinguish blank vs animal, reducing false positives from low-confidence species predictions
- **Weighted Ensemble** — 5 models combined with weights proportional to validation performance (1/val_loss)
- **Out-of-Fold Evaluation** — unbiased accuracy metrics using predictions from models that never saw that data
- **Temperature Scaling** — post-hoc calibration via scalar T learned on OOF logits (minimizes NLL); applied to both species and binary models before test inference
- **Test-Time Augmentation (TTA)** — 4 deterministic augmentations per image (original, h-flip, v-flip, h+v flip); soft-voted before ensemble weighting
- **Two-Phase Training** — frozen backbone for epochs 1–4, then fine-tuning at epoch 5 with differential learning rates (10x lower for backbone)
- **Gradient Accumulation** — 2 steps over batch size 8 for effective batch size of 16
- **Class-Weighted Loss** — inverse-frequency weighting so rare species like `hog` get proportionally more gradient signal
- **Label Smoothing** (0.1) — prevents overconfident predictions, improves log loss
- **Data Augmentation** — random crop, flips, rotation, color jitter, random erasing
- **Weights & Biases** — full training run tracking with per-fold metrics, confusion matrix, and sweep results
- **Resume Support** — progress saved to JSON after each fold; interrupted runs resume automatically

## Architecture

### Species Classifier (8 classes)

| Component | Details |
|-----------|---------|
| Backbone | ConvNeXt Base (pretrained on ImageNet-22k, fine-tuned on ImageNet-1k) |
| Classifier Head | Dropout(0.4) -> Linear(1024, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 8) |
| Drop Path Rate | 0.2 |
| Input Resolution | 512 x 512 |
| Loss Function | CrossEntropyLoss with class weights + label smoothing (0.1) |
| Optimizer | AdamW (head lr=3e-4, backbone lr=3e-5, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Precision | Mixed Precision (AMP) with FP16 |

### Binary Classifier (blank vs animal)

| Component | Details |
|-----------|---------|
| Backbone | ConvNeXt Base (same pretrained weights) |
| Classifier Head | Dropout(0.4) -> Linear(1024, 128) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(128, 2) |
| Training | Same schedule, phases, and hyperparameters as species classifier |

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4070 (12GB VRAM) |
| CUDA | 12.1 |
| TF32 | Enabled for Tensor Core acceleration |
| cuDNN | Benchmark mode enabled |
| Batch Size | 8 x 2 accumulation steps = 16 effective |

## Project Structure

```
.
├── benchmark.ipynb                          # Main training and evaluation notebook
├── train_features.csv                       # Image IDs, filepaths, and site IDs
├── train_labels.csv                         # One-hot encoded species labels
├── test_features.csv                        # Test set image IDs and filepaths
├── submission_format.csv                    # Expected submission format
├── train_features/                          # Training images directory
├── test_features/                           # Test images directory
├── training_progress.json                   # Species model fold checkpointing (auto-generated)
├── binary_training_progress.json            # Binary model fold checkpointing (auto-generated)
├── convnext_base_bs16_..._fold{1-5}.pt      # Saved species fold model weights
├── binary_convnext_base_bs16_..._fold{1-5}.pt  # Saved binary fold model weights
└── submission_df.csv                        # Competition submission file
```

## Training Pipeline

1. **Data Loading** — read CSVs, explore class distribution
2. **K-Fold Split** — 5-fold stratified group split by camera site (zero site overlap between train/val)
3. **Species Model — Per-Fold Training**
   - Epochs 1–4: Backbone frozen, only classifier head trains
   - Epoch 5+: Full model fine-tuning with differential learning rates
   - Early stopping (patience=3) saves best checkpoint per fold
   - All metrics logged to W&B per fold
4. **Binary Model — Per-Fold Training** — same schedule, blank vs animal labels
5. **Ensemble Assembly** — load all 5 fold models (species + binary), compute weights from validation loss
6. **OOF Evaluation** — each sample predicted by the fold that never trained on it; raw logits saved for calibration
7. **Temperature Scaling** — learn scalar T per model (species + binary) by minimizing NLL on OOF logits; calibrated probs replace OOF outputs for all downstream steps
8. **Threshold Optimization** — joint grid search over binary threshold and confidence threshold to maximize weighted F1
9. **Test Inference** — two-stage with TTA: for each image, 4 augmented views × 5 fold models averaged per stage; temperature applied before softmax; soft probability blending for submission
10. **Submission** — export calibrated probabilities to CSV

## Results (Updated Every Submission)

Evaluated using Out-of-Fold predictions across the full 16,488 training samples.
| Accuracy | Precision | Recall | F1 Score | Log Loss |
|----------|-----------|--------|----------|----------|
| 0.7643 | 0.7789 | 0.7643 | 0.7669 | 0.7683 |

## Dataset

- **Source**: [DrivenData Conser-vision Competition](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/)
- **Training images**: 16,488
- **Test images**: 4,464
- **Camera trap sites**: 120+
- **Image format**: JPEG (variable resolution, resized to 512x512)

## Dependencies

- Python 3.x
- PyTorch + CUDA 12.1
- timm (PyTorch Image Models)
- torchvision
- scikit-learn
- scipy
- pandas
- matplotlib
- PIL / Pillow
- tqdm
- wandb
- python-dotenv
