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
- **Weighted Ensemble** — 5 models combined with weights proportional to validation performance (1/val_loss)
- **Out-of-Fold Evaluation** — unbiased accuracy metrics using predictions from models that never saw that data
- **Two-Phase Training** — frozen backbone for initial epochs, then fine-tuning with differential learning rates (10x lower for backbone)
- **Class-Weighted Loss** — inverse-frequency weighting so rare species like `hog` get proportionally more gradient signal
- **Label Smoothing** (0.1) — prevents overconfident predictions, improves log loss
- **Data Augmentation** — random crop, flips, rotation, color jitter, random erasing

## Architecture

| Component | Details |
|-----------|---------|
| Backbone | ConvNeXt Base (pretrained on ImageNet-22k, fine-tuned on ImageNet-1k) |
| Classifier Head | Dropout(0.4) -> Linear(1024, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 8) |
| Input Resolution | 512 x 512 |
| Loss Function | CrossEntropyLoss with class weights + label smoothing |
| Optimizer | AdamW (head lr=3e-4, backbone lr=3e-5, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Precision | Mixed Precision (AMP) with FP16 |

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4070 (12GB VRAM) |
| CUDA | 12.1 |
| TF32 | Enabled for Tensor Core acceleration |
| cuDNN | Benchmark mode enabled |
| Batch Size | 16 |

## Project Structure

```
.
├── benchmark.ipynb          # Main training and evaluation notebook
├── train_features.csv       # Image IDs, filepaths, and site IDs
├── train_labels.csv         # One-hot encoded species labels
├── test_features.csv        # Test set image IDs and filepaths
├── submission_format.csv    # Expected submission format
├── train_features/          # Training images directory
├── test_features/           # Test images directory
├── best_model_fold{1-5}.pt  # Saved fold model weights
└── submission_df.csv        # Competition submission file
```

## Training Pipeline

1. **Data Loading** — read CSVs, explore class distribution
2. **K-Fold Split** — 5-fold stratified group split by camera site (zero site overlap between train/val)
3. **Per-Fold Training**
   - Epochs 1-2: Backbone frozen, only classifier head trains
   - Epoch 3+: Full model fine-tuning with differential learning rates
   - Early stopping (patience=3) saves best checkpoint per fold
4. **Ensemble Assembly** — load all 5 fold models, compute weights from validation loss
5. **Inference** — weighted average of softmax predictions across all fold models
6. **Submission** — export predictions to CSV

## Results (Updated Every Submission)

Evaluated using Out-of-Fold predictions across the full 16,488 training samples.
| Accuracy | Precision | Recall | F1 Score | Log Loss |
|----------|-----------|--------|----------|----------|
| 0.7015 | 0.7033 | 0.7015 | 0.6980 | 1.4424 |

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
- pandas
- matplotlib
- PIL / Pillow
- tqdm
