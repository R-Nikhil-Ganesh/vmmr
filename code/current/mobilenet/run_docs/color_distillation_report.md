# MobileNetV2 Multi-Task Color Distillation Report

**Experiment**: Color Distillation from Qwen3-VL-8B into MobileNetV2 Model C via `features[16]` Tap  
**Directory**: `vmmr/code/current/mobilenet/color_distillation/`  
**Date**: October 2026 (Updated with Phase 3 E-08 Reference Head)  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM), 64 GB RAM, Ubuntu Linux  
**Models Produced**:
- ⭐ **Reference Multi-Task ONNX Model**: `color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx` (**22.4 MB**, DVC-tracked)
  - PyTorch Head: `color_distillation/output_mlp512x2/color_head_best.pt` (**1.4 MB**, Git-tracked)
  - Configuration: Deep MLP-512×2, unweighted cross-entropy (`--no-class-weights`), label smoothing 0.0
- **Historical Baseline ONNX Model**: `color_distillation/output/mobilenet_v2_multitask.onnx` (20.61 MB, DVC-tracked)
  - PyTorch Head: `color_distillation/output/color_head_best.pt` (93 KB, 128-unit weighted MLP)

---

## 1. Executive Summary

In this milestone, we distilled Vision-Language color pseudo-labels from **Qwen3-VL-8B-Instruct** (generated across 1.13M images in Annotation Run 5) into a lightweight classification head branching directly from intermediate layer **`features[16]`** (160 channels) of the production **MobileNetV2 Model C** backbone.

### Key Performance Highlights

* **Unified Multi-Task Model**: Evaluates Make/Model (`1,235` classes), Body Color (`13` classes), Latent Embeddings (`1,280`-D), and Class Activation Maps (`16 \times 16`) simultaneously in a single forward pass.
* **Inference Latency**: **1.11 ms per image (~900 FPS)** on RTX 4090 using `CUDAExecutionProvider` for both the baseline and the Deep MLP-512×2 reference head (<0.04 ms isolated compute difference).
* **Reference Head Accuracy (Deep MLP-512×2, `output_mlp512x2`)**:
  * **Held-Out Test Top-1 (`color_conf >= 0.70`, 110,217 images)**: **`75.53%`** (Macro F1: **`62.89%`**, Top-3: **`95.81%`**)
  * **All Held-Out Test Rows (114,201 images, 100% coverage)**: **`74.31%`** (Top-3: **`95.40%`**)
  * **Independent CompCars Human Ground-Truth (2,867 test images)**: **`77.43%`** (recovering **98.7%** of the Qwen3-VL-8B teacher model's `78.44%` score with zero human labels used during training).
  * **Rare-Color Precision Recovery (`E-08`)**: Removing inverse-frequency class weights eliminated false-positive explosions: `green` precision rose from `28% -> 66%`, `brown` from `27% -> 61%`, and `beige` from `21% -> 52%` (Macro precision average: **`71.0%`** vs `56.0%`).
* **Historical Baseline (Weighted MLP-128, `output`)**:
  * Scored `68.0%` validation accuracy on high-confidence images with severe rare-color precision collapse due to class weighting.
* **Model Footprint**: The reference multi-task model is **22.4 MB** (adding ~1.4 MB over the base Model C ONNX).
* **Zero Make/Model Degradation (Verified)**: Splicing the color head preserves exact bit-for-bit numerical identity (`max |diff| = 0.000e+00` across 256 test images) for all original make/model outputs (`predictions`, `embeddings`, `class_maps`).

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

Evaluated across the **109,454 validation images with `color_conf >= 0.70`** (out of 114,252 total validation rows; `color_distillation/output/final_report.txt`):

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

## 4. Phase 3 Architecture & Loss Ablations (`E-07` & `E-08`)

To address the precision collapse on rare colors caused by inverse-frequency class weights (`E-08`) and establish an unbiased evaluation on held-out test data (`E-07`), we evaluated two improved architectures against the original baseline on the 160-D `features[16]` tap:

1. **Baseline (Shipped)**: Single-hidden-layer MLP (128 units), inverse-frequency class weights $[0.25, 4.0]$, label smoothing $0.05$.
2. **Ablation 1 (Unweighted MLP-128)**: Same 128-unit MLP, `--no-class-weights`, label smoothing $0.0$.
3. **Ablation 2 (Unweighted Deep MLP-512×2)**: 2-layer `512 -> 512 -> 13` MLP with BatchNorm1d + ReLU + Dropout(0.2) per hidden layer, `--no-class-weights`, label smoothing $0.0$.

### Multi-Split Performance Comparison

| Model Architecture & Loss | Val Top-1 (`conf>=0.70`) | Val Top-3 (`conf>=0.70`) | Val Macro F1 | Test Top-1 (`conf>=0.70`) | Test Top-3 (`conf>=0.70`) | Test Macro F1 | Test Top-1 (All Rows) | CompCars Human GT Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Weighted MLP-128)** | `68.00%` | `93.81%` | `55.66%` | *N/A* | *N/A* | *N/A* | *N/A* | `71.08%` |
| **Ablation 1 (Unweighted MLP-128)** | `73.46%` | `95.25%` | `59.05%` | `73.23%` | `95.05%` | `58.05%` | `72.02%` | `74.89%` |
| **Ablation 2 (Deep MLP-512×2)** | **`75.85%`** | **`96.17%`** | **`63.96%`** | **`75.53%`** | **`95.81%`** | **`62.89%`** | **`74.31%`** | **`77.43%`** |
| *Reference: Qwen3-VL-8B Teacher* | *—* | *—* | *—* | *—* | *—* | *—* | *—* | *`78.44%`* |

### Per-Class Precision & F1 Recovery on Rare Classes

Removing class weights directly resolved the precision collapse on rare colors:

| Color Class | Baseline Precision | Ablation 1 Precision | Ablation 2 Precision | Baseline F1 | Ablation 1 F1 | Ablation 2 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`green`** | `28.0%` | `67.0%` | **`66.0%`** | `39.0%` | `45.0%` | **`51.0%`** |
| **`brown`** | `27.0%` | `57.0%` | **`61.0%`** | `38.0%` | `41.0%` | **`49.0%`** |
| **`beige`** | `21.0%` | `50.0%` | **`52.0%`** | `31.0%` | `30.0%` | **`39.0%`** |
| **`gold`** | `47.0%` | `56.0%` | **`62.0%`** | `35.0%` | `32.0%` | **`43.0%`** |
| **`white`** | `86.0%` | `82.0%` | **`84.0%`** | `82.0%` | `85.0%` | **`87.0%`** |
| **`black`** | `79.0%` | `72.0%` | **`75.0%`** | `73.0%` | `78.0%` | **`80.0%`** |
| **`silver`** | `73.0%` | `70.0%` | **`74.0%`** | `67.0%` | `70.0%` | **`73.0%`** |
| **`red`** | `91.0%` | `89.0%` | **`90.0%`** | `90.0%` | `92.0%` | **`93.0%`** |
| **`Macro Avg`** | `56.0%` | `69.0%` | **`71.0%`** | `57.0%` | `59.0%` | **`64.0%`** |

### Key Conclusions from Phase 3:
1. **Removing Inverse-Frequency Class Weights is Crucial**: Class weights boosted recall on rare classes by creating a flood of false-positives, severely lowering precision to 21–28%. Unweighted training boosted macro precision from **56% $\to$ 71%** and lifted Top-1 accuracy by **+5.46 pp** on the identical 128-unit MLP.
2. **Deep MLP-512×2 Reaches Near-Teacher Quality**: Expanding the capacity from 128 units to two 512-unit layers with batch normalization boosted validation accuracy to **75.85%** (+7.85 pp) and Top-3 accuracy to **96.17%**, with **63.96%** Macro F1.
3. **Human Ground-Truth Validation Confirms Genuine Generalization**: When evaluated against the independent, human-labeled CompCars CCTV test set (`2,867` images), the Deep MLP-512×2 head achieves **77.43%** aligned accuracy—recovering **98.7%** of the Qwen3-VL-8B teacher model's score (`78.44%`) with zero additional latency.
4. **Generalization to Noisy Pseudo-Labels**: Even across all 114,201 held-out test images including rows with `color_conf < 0.70`, the Deep MLP-512×2 head achieves **74.31%** Top-1 accuracy and **95.40%** Top-3 accuracy.

---

## 5. Multi-Task ONNX Model Verification

Both the baseline and the **reference `output_mlp512x2`** multi-task models were verified via `verify_multitask_onnx.py`:

```text
--- Model Signatures ---
Input:  input_image (shape: ['batch_size', 3, 512, 512])
Outputs:
  - predictions         : shape=['batch_size', 1235], type=tensor(float)
  - embeddings          : shape=['batch_size', 1280], type=tensor(float)
  - class_maps          : shape=['batch_size', 1235, 16, 16], type=tensor(float)
  - color_predictions   : shape=['batch_size', 13], type=tensor(float)

--- Latency Benchmark on RTX 4090 (CUDAExecutionProvider) ---
Reference Model (Deep MLP-512x2, output_mlp512x2):
  Single-image forward latency: 1.11 ms (902.8 FPS)
  Batched (batch=32) latency:   1.28 ms per image (779.0 FPS)
Baseline Model (MLP-128, output):
  Single-image forward latency: 1.13 ms (881.6 FPS)
  Batched (batch=32) latency:   1.27 ms per image (784.9 FPS)

--- Numerical Equivalence vs Base Model C ONNX (256 images) ---
Checked 256 test images:
  predictions max |diff| : 0.000e+00  (top-1 argmax agreement: 256/256 = 100.00%)
  embeddings  max |diff| : 0.000e+00
  class_maps  max |diff| : 0.000e+00

--- Numerical Equivalence: PyTorch Color Head vs Fused ONNX Weights ---
Checked 1,024 cached validation feature vectors:
  color_predictions max |diff| (unfused BN vs fused ONNX GEMMs): 6.840e-06
  color_predictions top-1 argmax agreement: 1024/1024 (100.00%)
```

---

## 6. Artifact Files & Storage Audit

Large binaries (`mobilenet_v2_multitask.onnx` and `cache/`) are tracked via **DVC** on Google Drive, while lightweight checkpoints and reports are tracked directly in Git:

| File | Size | Tracking | Description |
| :--- | :---: | :---: | :--- |
| ⭐ `color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx` | **22.40 MB** | **DVC** (`.onnx.dvc`) | **Reference Production Multi-Task ONNX Model** |
| `color_distillation/output_mlp512x2/color_head_best.pt` | **1.41 MB** | Git | Reference Deep MLP-512×2 PyTorch weights |
| `color_distillation/output_mlp512x2/eval_metrics.json` | **400 B** | Git | Held-out test (all & conf>=0.70) evaluation metrics |
| `color_distillation/output_mlp512x2/final_report.txt` | **1.8 KB** | Git | Full per-class classification report (val & test) |
| `color_distillation/output_unweighted_mlp128/mobilenet_v2_multitask.onnx` | **21.00 MB** | **DVC** (`.onnx.dvc`) | Ablation 1 unweighted 128-unit ONNX model |
| `color_distillation/output/mobilenet_v2_multitask.onnx` | **20.61 MB** | **DVC** (`.onnx.dvc`) | Historical baseline 128-unit weighted ONNX model |
| `color_distillation/output/color_head_best.pt` | **93 KB** | Git | Historical baseline PyTorch weights state dict |
| `color_distillation/cache/` | **303.25 MB** | **DVC** (`cache.dvc`) | Cached `features_L16_{train,val}.npy` & `labels_{train,val}.npy` |
| `color_distillation/probe_results.csv` | **7.5 KB** | Git | Phase 1 8-layer benchmark table |
| `color_distillation/probe_results.json` | **11.4 KB** | Git | Full per-class layer probe JSON metrics |
