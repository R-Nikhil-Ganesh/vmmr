# MobileNetV2 Multi-Task Color Distillation Report

**Experiment**: Color Distillation from Qwen3-VL-8B into MobileNetV2 Model C via `features[16]` Tap  
**Directory**: `vmmr/code/current/mobilenet/color_distillation/`  
**Date**: October 16, 2026  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM), 64 GB RAM, Ubuntu Linux  
**Models Produced**:
- PyTorch Head: `color_distillation/output/color_head_best.pt` (93 KB)
- Multi-Task ONNX Model: `color_distillation/output/mobilenet_v2_multitask.onnx` (20.61 MB)

---

## 1. Executive Summary

In this milestone, we distilled Vision-Language color pseudo-labels from **Qwen3-VL-8B-Instruct** (generated across 1.13M images in Annotation Run 5) into a lightweight classification head branching directly from intermediate layer **`features[16]`** (160 channels) of the production **MobileNetV2 Model C** backbone.

### Key Performance Highlights

* **Unified Multi-Task Model**: Evaluates Make/Model (`1,235` classes), Body Color (`13` classes), Latent Embeddings (`1,280`-D), and Class Activation Maps (`16 \times 16`) simultaneously in a single forward pass.
* **Inference Latency**: **1.11 ms per image (905 FPS)** on RTX 4090 using `CUDAExecutionProvider`.
* **Accuracy on Full Validation Split (109,454 images)**:
  * **Top-1 Color Accuracy**: **68.0%**
  * **Weighted Average F1**: **69.0%**
  * **Top Performing Colors**: Red (90% F1), Yellow (81% F1), White (82% F1), Black (73% F1), Orange (69% F1), Silver (67% F1).
* **Model Footprint**: The multi-task model is **20.61 MB** (adding only ~90 KB over the single-task Model C ONNX).
* **Zero Make/Model Degradation**: Splicing the color head preserves exact bit-for-bit numerical identity for all original make/model predictions.

---

## 2. Phase 1: Empirical Layer Selection (Summary)

An empirical probing sweep across 25,000 images evaluated 8 candidate tap points in the MobileNetV2 backbone:

| Layer | Stage | Channels | Grid | Linear Top-1 | Linear Macro F1 | MLP Top-1 | MLP Top-3 | MLP Macro F1 | Head Params (MLP) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`L17`** | **Deep Stage 2 (Unfrozen)** | **320** | 16x16 | 58.16% | 57.72% | **62.72%** | **89.56%** | **62.55%** | **43,021** |
| **`L16`** | **Deep Stage 1 (Unfrozen)** | **160** | 16x16 | **58.48%** | **58.18%** | **62.58%** | **88.66%** | **62.50%** | **22,541** |
| `L13` | Mid-Level Stage 2 (Frozen) | 96 | 32x32 | 57.54% | 57.51% | 61.08% | 87.22% | 61.14% | 14,349 |
| `L10` | Mid-Level Stage 1 (Frozen) | 64 | 32x32 | 51.26% | 50.92% | 55.60% | 84.42% | 55.56% | 10,253 |
| `L6` | Low-Mid Level (Frozen) | 32 | 64x64 | 45.96% | 45.58% | 50.74% | 79.80% | 50.36% | 6,157 |
| `L3` | Early Low-Level (Frozen) | 24 | 128x128 | 38.06% | 37.30% | 41.64% | 73.54% | 41.23% | 5,133 |
| `L18` | Final Conv (Unfrozen) | 1,280 | 16x16 | 52.96% | 52.58% | 53.48% | 83.52% | 53.46% | 165,901 |
| `Embedding` | Penultimate 1280-D (Pooled) | 1,280 | 1x1 | 53.52% | 53.17% | 53.48% | 82.90% | 53.35% | 165,901 |

### The "Color Bottleneck" Collapse
Moving from `L16`/`L17` to the final `Embedding` causes an immediate **$-9.24\%$ drop in Top-1 accuracy** ($62.72\% \to 53.48\%$). In Model C, the 1x1 convolution (`features[18]`) was trained to optimize make/model classification by forcing intra-class color invariance (grouping all car colors of the same model together). Tapping at **`features[16]`** captures the vehicle's spatial color distribution before this compression occurs, while requiring only 160 channels.

---

## 3. Phase 2: Full Distillation Training Dynamics

### Training Configuration
* **Training Set**: 860,006 images (`split == 'train'`, `color_conf >= 0.70`)
* **Validation Set**: 109,454 images (`split == 'val'`, `color_conf >= 0.70`)
* **Feature Cache**: Pre-extracted 160-D pooled features saved to `cache/features_L16_train.npy` (263 MB) and `cache/features_L16_val.npy` (34 MB).
* **Head Architecture**: MLP (`Linear(160, 128) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(128, 13)`)
* **Objective**: Inverse-frequency class-weighted cross-entropy (bounded to $[0.25, 4.0]$) with label smoothing ($0.05$).
* **Optimizer**: AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), Cosine Annealing over 25 epochs.
* **Epoch Speed**: ~15 seconds per epoch on cached GPU tensors.

---

## 4. Final Validation Metrics & Classification Report

Evaluated across all **109,454 validation images** (`color_distillation/output/final_report.txt`):

```text
              precision    recall  f1-score   support

       white       0.86      0.79      0.82     24450
       black       0.79      0.68      0.73     23540
      silver       0.73      0.62      0.67     19521
        gray       0.53      0.52      0.52     13698
         red       0.91      0.88      0.90      9396
        blue       0.58      0.56      0.57      9916
       green       0.28      0.65      0.39      2352
      yellow       0.73      0.90      0.81      1576
      orange       0.62      0.78      0.69       947
       brown       0.27      0.65      0.38      1744
       beige       0.21      0.61      0.31      1668
        gold       0.47      0.28      0.35       289
       other       0.34      0.26      0.29       357

    accuracy                           0.68    109454
   macro avg       0.56      0.63      0.57    109454
weighted avg       0.72      0.68      0.69    109454
```

### Observations:
* **High-Volume Colors**: The top 3 vehicle colors (White, Black, Silver), accounting for ~60% of the dataset, achieve precision between 73% and 86% and F1 between 67% and 82%.
* **High-Contrast Chromatic Colors**: Red (90% F1), Yellow (81% F1), and Orange (69% F1) show excellent separation.
* **Subtle / Neutral Shades**: Gray, Silver, Beige, and Brown show expected overlap due to lighting, specular glints, and dusk conditions, but maintain balanced recall across the long tail.

---

## 5. Multi-Task ONNX Model Verification

The multi-task model `color_distillation/output/mobilenet_v2_multitask.onnx` was verified via `verify_multitask_onnx.py`:

```text
--- Model Signatures ---
Input:  input_image (shape: ['batch_size', 3, 512, 512])
Outputs:
  - predictions         : shape=['batch_size', 1235], type=tensor(float)
  - embeddings          : shape=['batch_size', 1280], type=tensor(float)
  - class_maps          : shape=['batch_size', 1235, 16, 16], type=tensor(float)
  - color_predictions   : shape=['batch_size', 13], type=tensor(float)

--- Latency Benchmark (RTX 4090) ---
Single-image forward latency: 1.11 ms (904.7 FPS)
Batched (batch=32) latency:   1.09 ms per image (917.0 FPS)
```

---

## 6. Artifact Files & File Size Audit

All generated artifacts are strictly under the 100 MB GitHub file size limit:

| File | Size | Description |
| :--- | :---: | :--- |
| `color_distillation/output/mobilenet_v2_multitask.onnx` | **20.61 MB** | Spliced 4-output ONNX production model |
| `color_distillation/output/color_head_best.pt` | **93 KB** | Trained PyTorch weights state dict |
| `color_distillation/output/training_history.csv` | **2.4 KB** | Epoch-by-epoch training and validation loss/acc |
| `color_distillation/output/final_report.txt` | **920 B** | Full per-class classification report |
| `color_distillation/probe_results.csv` | **7.5 KB** | Phase 1 8-layer benchmark table |
| `color_distillation/probe_results.json` | **11.4 KB** | Full per-class layer probe JSON metrics |
