# MobileNetV2 Color Distillation: Architecture, Empirical Layer Selection, and Multi-Task ONNX Export

**Module**: `code/current/mobilenet/color_distillation/`  
**Teacher Model**: Vision-Language pseudo-labels from Qwen3-VL-8B (`annotation_runs/run5/annotations.csv`, 1,126,737 images)  
**Student Model**: MobileNetV2 Model C (`output_merged_run4/models/mobilenet_v2_best.onnx`)  
**Deployment Target**: Unified Dual-Head ONNX Model (`output/mobilenet_v2_multitask.onnx`)  
**Date**: October 2026  

---

## 1. Executive Summary & Objective

In **Annotation Run 5**, the zero-shot Vision-Language model **Qwen3-VL-8B-Instruct** generated high-confidence vehicle color annotations across **1.13 million images** spanning the PlatesMania and External datasets. While Qwen achieves high visual fidelity, its 8-billion parameter size, 6 GB VRAM memory requirement (in 4-bit NF4), and ~37 img/s throughput make it unsuitable for high-throughput edge deployment.

The objective of this module is to **distill Qwen's color recognition capabilities directly into MobileNetV2 Model C**, enabling:
1. **Simultaneous Make/Model (1,235 classes) and Body Color (13 classes) recognition** in a single forward pass.
2. **Sub-2 millisecond inference** per frame on GPU and sub-15 ms on edge CPU.
3. **Zero memory overhead**: Adding less than 90 KB (0.4%) to the 22 MB MobileNetV2 model.
4. **Preserved Make/Model accuracy**: 100% bit-for-bit identity preserved on all existing Model C outputs (`predictions`, `embeddings`, `class_maps`).

---

## 2. Theoretical Motivation: The "Color vs. Identity" Dilemma

In convolutional neural networks, vehicle make/model recognition and vehicle body color classification have **fundamentally conflicting representational goals**:

* **Make/Model Classification (Semantic Identity)**: Must be **color-invariant**. A red Honda Civic, a blue Honda Civic, and a black Honda Civic share identical sheet metal, grille contours, and lamp clusters. Higher layers in the backbone are actively trained to discard low-level color variations to group all trims into the same class logit.
* **Color Classification (Photometric Property)**: Must be **identity-invariant**. A red Ferrari, a red pickup truck, and a red minivan must all activate the exact same "Red" feature, while ignoring make-specific body styling, badges, and wheels.
* **Spatial Selectivity (Vehicle vs. Background)**: Raw early filters (e.g. Layer 0–3) lack spatial receptive fields and semantic vehicle boundaries. Tapping too early risks confusing the car's paint with asphalt (black/gray), road markings (white/yellow), or foliage (green).

```
[Input: 3 x 512 x 512]
         │
         ▼
MobileNetV2 features[0:3]    (24-32 ch, 128-256px) ──> ⚠️ High background noise (road, sky)
         │
         ▼
MobileNetV2 features[4:13]   (32-96 ch, 32-64px)   ──> Mid-level boundaries form
         │
         ▼
MobileNetV2 features[14:17]  (160-320 ch, 16px)    ──> ⭐ SWEET SPOT (Vehicle body masked, color preserved)
         │
         ▼
MobileNetV2 features[18]     (1280 ch, 16px)       ──> ❌ 1x1 Conv discards color to enforce make invariance
         │
         ▼
Global Average Pooling       (1280-d Embedding)    ──> ❌ Color bottleneck collapse (-9.2% accuracy)
         │
         ▼
Linear(1280, 1235)           (Make/Model Logits)
```

---

## 3. Phase 1: Empirical Layer Probing Benchmark

To scientifically verify where color information resides in the Model C backbone, an empirical layer sweep was conducted across **8 candidate layers** using **25,000 images** (20,000 train, 5,000 validation, stratified across 13 colors and both domains) directly using `layer_probe.py`:

### Benchmark Results Table

| Layer | Backbone Stage | Channels | Spatial Grid | Linear Top-1 | Linear Top-3 | Linear Macro F1 | MLP Top-1 | MLP Top-3 | MLP Macro F1 | Head Params (MLP) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L17** | **Deep Stage 2 (Unfrozen)** | **320** | 16x16 | 58.16% | 86.36% | 57.72% | **62.72%** | **89.56%** | **62.55%** | **43,021** |
| **L16** | **Deep Stage 1 (Unfrozen)** | **160** | 16x16 | **58.48%** | 85.74% | **58.18%** | **62.58%** | **88.66%** | **62.50%** | **22,541** |
| L13 | Mid-Level Stage 2 (Frozen) | 96 | 32x32 | 57.54% | 84.88% | 57.51% | 61.08% | 87.22% | 61.14% | 14,349 |
| L10 | Mid-Level Stage 1 (Frozen) | 64 | 32x32 | 51.26% | 80.48% | 50.92% | 55.60% | 84.42% | 55.56% | 10,253 |
| L6 | Low-Mid Level (Frozen) | 32 | 64x64 | 45.96% | 75.78% | 45.58% | 50.74% | 79.80% | 50.36% | 6,157 |
| L3 | Early Low-Level (Frozen) | 24 | 128x128 | 38.06% | 69.66% | 37.30% | 41.64% | 73.54% | 41.23% | 5,133 |
| L18 | Final Conv (Unfrozen) | 1,280 | 16x16 | 52.96% | 82.22% | 52.58% | 53.48% | 83.52% | 53.46% | 165,901 |
| Embedding | Penultimate 1280-D (Pooled) | 1,280 | 1x1 | 53.52% | 82.52% | 53.17% | 53.48% | 82.90% | 53.35% | 165,901 |

### Key Discoveries:
1. **The "Color Bottleneck" Collapse in Layer 18 / Embedding**:
   * Moving from **`L17` $\to$ `Embedding`** causes an immediate **$-9.24\%$ collapse in Top-1 accuracy** and **$-9.20\%$ in Macro F1** ($62.55\% \to 53.35\%$).
   * `features[18]` (the $1\times1$ convolution projecting $320 \to 1280$ channels) acts as a strict bottleneck that actively compresses away photometric color variance to optimize make/model classification.
2. **`features[16]` is the Optimal Production Tap Point**:
   * Both `L16` ($160$ channels) and `L17` ($320$ channels) hit peak accuracy (~$62.6\%$ Top-1, ~$89\%$ Top-3).
   * `L16` achieves identical accuracy to `L17` while requiring **half the channel width (160 vs 320)**, resulting in an ultra-compact MLP head of just **22,541 parameters** (~90 KB).
3. **Consistent Per-Class Superiority**:
   Every single color class achieves a significantly higher F1 score at `L16`/`L17` than at the final embedding:
   * **Blue**: $53.7\%$ vs $38.6\%$ ($+15.1\%$)
   * **White**: $74.1\%$ vs $63.2\%$ ($+10.9\%$)
   * **Green**: $54.9\%$ vs $44.5\%$ ($+10.4\%$)
   * **Silver**: $56.3\%$ vs $46.9\%$ ($+9.4\%$)
   * **Gray**: $46.8\%$ vs $37.0\%$ ($+9.8\%$)
   * **Gold**: $60.1\%$ vs $52.7\%$ ($+7.4\%$)

---

## 4. Head Architecture & Mathematical Formulation

### Why an MLP Head Outperforms a Pure Linear Head
In Phase 1, the **MLP Head** achieved a **+4.10% Top-1 accuracy** and **+4.32% Macro F1 gain** over a single Linear layer ($62.58\%$ vs $58.48\%$).

```
Pooled Activation: x in R^160
         │
         ▼
Linear(160 -> 128)  ──> First Dense Projection
         │
         ▼
BatchNorm1d(128)    ──> Adaptive Lighting Normalization
         │
         ▼
ReLU                ──> Non-linear Activation
         │
         ▼
Dropout(p=0.2)      ──> Feature Regularization (Training only)
         │
         ▼
Linear(128 -> 13)   ──> 13 Automotive Color Logits
```

#### Physical / Mathematical Reasons:
1. **Non-linear Conjunctions for Metallic & Subtle Finishes**:
   CNN feature channels do not represent isolated RGB pigments; they encode edge gradients, reflectance, and specular highlights.
   * **Silver vs. Gray vs. White**: All three have zero chromatic saturation. The distinction lies in specular glints vs. matte diffuse surfaces. A linear boundary cannot separate specular glints on a gray background; the MLP's hidden layer learns conjunctions ($x_{\text{specular}} > \tau_1 \text{ AND } x_{\text{diffuse}} > \tau_2$).
   * **Brown vs. Orange**: Brown is low-luminance dark orange.
   * **Beige vs. Gold**: Requires non-linear thresholds on yellow saturation and metallic shine.
2. **Zero Runtime Overhead via BatchNorm Folding**:
   During ONNX export, the BatchNorm parameters ($\gamma, \beta, \mu, \sigma^2$) are **mathematically folded** into the first dense layer's weights:
   $$W_{\text{fused}} = W_1 \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}, \quad b_{\text{fused}} = (b_1 - \mu) \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   In production ONNX execution, the head runs as two standard GEMM operations with **$< 0.05\text{ ms}$ latency** on GPU.

---

## 5. Implementation & Training Pipeline

### A. Feature Caching Strategy (`cache/`)
Processing ~1M images repeatedly from disk across multiple training epochs would create an unnecessary I/O bottleneck. 
* The script filters `annotations.csv` to high-confidence Qwen pseudo-labels (`color_conf >= 0.70`, retaining ~95.8% of rows), extracts the $160$-dimensional pooled features from `features[16]` in a single pass using the ONNX GPU runtime, and saves them as memory-mapped `float16` NumPy arrays (tracked in DVC via `cache.dvc`):
  * `cache/features_L16_train.npy` (860,006 vectors $\times$ 160 channels $\times$ 2 bytes = **262.45 MB**)
  * `cache/features_L16_val.npy` (109,454 vectors $\times$ 160 channels $\times$ 2 bytes = **33.40 MB**)
* **Result**: Once cached, a 25-epoch training run over 860k training samples executes in **under 2 minutes** on an RTX 4090.

### B. Loss Function & Class Balancing
Vehicle color distributions exhibit significant class imbalance (White, Black, and Silver represent ~60.9% of the fleet, while Gold, Beige, and Orange represent <2% each).
* To prevent majority collapse, the training uses **bounded inverse-frequency class weights**:
  $$w_c = \text{clip}\left(\frac{N}{13 \cdot N_c}, 0.25, 4.0\right)$$
* Combined with **Label Smoothing ($0.05$)** to prevent overconfidence on ambiguous lighting edge cases.

---

## 6. Spliced Multi-Task ONNX Export (`mobilenet_v2_multitask.onnx`)

Rather than retraining MobileNetV2 from scratch (which risks degrading the 98.5% Make / 93.0% Model accuracy), the trained color head is **spliced directly into the existing ONNX computational graph** using `onnx.helper`:

```
Input: input_image (B, 3, 512, 512)
                 │
                 ▼
        MobileNetV2 features[0:16]
                 │
                 ├────────────────────────────────────────┐ (features[16] tap)
                 │                                        │
                 ▼                                        ▼
        features[17:18]                          GlobalAveragePool
                 │                                        │
                 ▼                                        ▼
        GlobalAveragePool                         Flatten & Linear Norm
                 │                                        │
                 ▼                                        ▼
        Make/Model Classifier                     Color MLP Head (fused)
                 │                                        │
                 ▼                                        ▼
    ┌─────────────────────────┐              ┌─────────────────────────┐
    │ predictions: (B, 1235)  │              │ color_predictions:      │
    │ embeddings:  (B, 1280)  │              │              (B, 13)    │
    │ class_maps:  (B, 1235,  │              └─────────────────────────┘
    │                  16, 16)│
    └─────────────────────────┘
```

### Graph Integrity & Numerical Verification (`verify_multitask_onnx.py`):
* The original graph nodes for `predictions`, `embeddings`, and `class_maps` are **100% unaltered**. Verified across 256 test images against `output_merged_run4/models/mobilenet_v2_best.onnx`: **`max |diff| = 0.000e+00`** across all three outputs (`256/256 = 100.00%` top-1 agreement).
* The fused ONNX color branch matches the unfused PyTorch `color_head_best.pt` (`BatchNorm1d`) across 1,024 cached validation vectors with **`max |diff| = 5.722e-06`** (`float32` rounding) and **`1024/1024 = 100.00%`** top-1 argmax agreement.
* The color branch reads from `/features/features.16/Add_output_0` in parallel without affecting the downstream computation of `features.17` and `features.18`.
* All 4 outputs run in a single, fused engine pass with zero duplicate FLOPs.

---

## 7. Directory Layout & CLI Usage

### Directory Structure
```text
code/current/mobilenet/color_distillation/
├── train_color_head.py             # Full distillation trainer and ONNX splicer
├── layer_probe.py                  # Phase 1 multi-layer empirical probing script
├── verify_multitask_onnx.py        # Signature, latency & numerical equivalence verifier
├── probe_results.csv               # Empirical benchmark metrics table across 8 layers
├── probe_results.json              # Full per-class classification metrics
├── color_distillation_approach.md  # Architectural and empirical documentation (this file)
│
├── cache/                          # (DVC-tracked via cache.dvc) Cached float16 feature tensors (color_conf >= 0.70)
│   ├── features_L16_train.npy      # 860,006 x 160 float16 (262.45 MB)
│   ├── labels_train.npy            # 860,006 int64 labels (6.56 MB)
│   ├── features_L16_val.npy        # 109,454 x 160 float16 (33.40 MB)
│   └── labels_val.npy              # 109,454 int64 labels (0.84 MB)
│
└── output/                         # Trained model artifacts & logs
    ├── mobilenet_v2_multitask.onnx # (DVC-tracked via .onnx.dvc) Unified 4-output ONNX model (20.61 MB)
    ├── color_head_best.pt          # (Git-tracked) PyTorch checkpoint (93 KB)
    ├── training_history.csv        # (Git-tracked) Epoch training and validation logs
    └── final_report.txt            # (Git-tracked) Per-class precision, recall, and F1 report
```

### Quick Start Commands

```bash
cd vmmr/code/current/mobilenet

# 1. Full training in tmux session 'color_distill' (extracts features on first run, then trains)
bash bash_scripts/run_train_color_distillation.sh

# 2. Re-train with Linear head instead of MLP
bash bash_scripts/run_train_color_distillation.sh --head-type linear

# 3. Instant re-training on cached features (skips extraction, finishes in <2 min)
bash bash_scripts/run_train_color_distillation.sh --skip-extract --epochs 30 --lr 5e-4
```

