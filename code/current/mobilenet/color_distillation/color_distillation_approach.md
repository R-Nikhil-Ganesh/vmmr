# MobileNetV2 Color Distillation: Architecture, Empirical Layer Selection, and Multi-Task ONNX Export

**Module**: `code/current/mobilenet/color_distillation/`  
**Teacher Model**: Vision-Language pseudo-labels from Qwen3-VL-8B (`annotation_runs/run5/annotations.csv`, 1,126,737 images)  
**Student Model**: MobileNetV2 Model C (`merged_dataset/output_merged_run4/models/mobilenet_v2_best.onnx`)  
**Deployment Target (Reference)**: Unified Dual-Head ONNX Model (`output_mlp512x2/mobilenet_v2_multitask.onnx`, Deep MLP-512×2)  
**Historical Baseline**: `output/mobilenet_v2_multitask.onnx` (128-unit weighted MLP)  
**Date**: October 2026  

---

## 1. Executive Summary & Objective

In **Annotation Run 5**, the zero-shot Vision-Language model **Qwen3-VL-8B-Instruct** generated high-confidence vehicle color annotations across **1.13 million images** spanning the PlatesMania and External datasets. While Qwen achieves high visual fidelity, its 8-billion parameter size, 6 GB VRAM memory requirement (in 4-bit NF4), and ~37 img/s throughput make it unsuitable for high-throughput edge deployment.

The objective of this module is to **distill Qwen's color recognition capabilities directly into MobileNetV2 Model C**, enabling:
1. **Simultaneous Make/Model (1,235 classes) and Body Color (13 classes) recognition** in a single forward pass.
2. **Sub-2 millisecond inference** per frame on GPU (~900 FPS) and sub-30 ms on edge CPU.
3. **Reference Architecture**: The production-adopted reference head is the **Unweighted Deep MLP-512×2** (`output_mlp512x2/`), which achieves **75.53% held-out test Top-1** accuracy (Macro F1 62.89%) and **77.43% on independent CompCars human ground truth** (recovering 98.7% of the Qwen teacher's 78.44%), adding only 39 microseconds (<0.04 ms) isolated compute over the 128-unit baseline.
4. **Zero memory overhead**: Adding only ~1.4 MB (6%) to the 22 MB MobileNetV2 ONNX graph.
5. **Preserved Make/Model accuracy**: 100% bit-for-bit identity preserved on all existing Model C outputs (`predictions`, `embeddings`, `class_maps`).

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

### Evolution from MLP-128 Baseline to Reference Deep MLP-512×2

In Phase 1, an initial **128-unit MLP Head** achieved a **+4.10% Top-1 accuracy** and **+4.32% Macro F1 gain** over a single Linear layer ($62.58\%$ vs $58.48\%$). However, evaluating on held-out test data (`E-07`) and against CompCars human ground truth (`E-09`) revealed that expanding capacity to a **2-layer Deep MLP-512×2** provided a massive accuracy leap while keeping latency completely negligible.

```
[REFERENCE PRODUCTION HEAD: Deep MLP-512×2]

Pooled Activation: x in R^160 (features[16] tap)
         │
         ▼
Linear(160 -> 512)     ──> First Expansion Projection
         │
         ▼
BatchNorm1d(512)       ──> Adaptive Lighting Normalization
         │
         ▼
ReLU + Dropout(p=0.2)  ──> Non-linear Activation & Regularization
         │
         ▼
Linear(512 -> 512)     ──> Second Hidden Representation
         │
         ▼
BatchNorm1d(512)       ──> Intermediate Normalization
         │
         ▼
ReLU + Dropout(p=0.2)  ──> Non-linear Activation
         │
         ▼
Linear(512 -> 13)      ──> 13 Automotive Color Logits
```

#### Physical & Mathematical Rationale:
1. **Non-linear Conjunctions for Metallic & Subtle Finishes**:
   CNN feature channels do not represent isolated RGB pigments; they encode edge gradients, reflectance, and specular highlights.
   * **Silver vs. Gray vs. White**: All three have zero chromatic saturation. The distinction lies in specular glints vs. matte diffuse surfaces. A linear boundary cannot separate specular glints on a gray background; the MLP's hidden layers learn conjunctions ($x_{\text{specular}} > \tau_1 \text{ AND } x_{\text{diffuse}} > \tau_2$).
   * **Brown vs. Orange**: Brown is low-luminance dark orange.
   * **Beige vs. Gold**: Requires non-linear thresholds on yellow saturation and metallic shine.
2. **Double Non-Linearity Disentangles Lighting Variations**:
   The two stacked 512-unit layers allow the network to first map raw photometric features into an illumination-normalized color manifold (layer 1) before computing decision hyperplanes between subtly overlapping vehicle paint formulations (layer 2).
3. **Zero Runtime Overhead via BatchNorm Folding**:
   During ONNX export, both BatchNorm layers ($\gamma_i, \beta_i, \mu_i, \sigma_i^2$) are **mathematically folded** into the respective preceding dense layers' weights:
   $$W_{\text{fused}} = W_i \cdot \frac{\gamma_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad b_{\text{fused}} = (b_i - \mu_i) \cdot \frac{\gamma_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_i$$
   In production ONNX execution, the entire 512×2 head runs as three fused GEMM + ReLU operations adding only **39 microseconds (0.039 ms)** of isolated compute, with **zero change to the 1.11 ms end-to-end forward pass latency**.

---

## 5. Implementation & Training Pipeline

### A. Feature Caching Strategy (`cache/`)
Processing ~1M images repeatedly from disk across multiple training epochs would create an unnecessary I/O bottleneck. 
* The script filters `annotations.csv` to high-confidence Qwen pseudo-labels (`color_conf >= 0.70`, retaining ~95.8% of rows), extracts the $160$-dimensional pooled features from `features[16]` in a single pass using the ONNX GPU runtime, and saves them as memory-mapped `float16` NumPy arrays (tracked in DVC via `cache.dvc`):
  * `cache/features_L16_train.npy` (860,006 vectors $\times$ 160 channels $\times$ 2 bytes = **262.45 MB**)
  * `cache/features_L16_val.npy` (109,454 vectors $\times$ 160 channels $\times$ 2 bytes = **33.40 MB**)
  * `cache/features_L16_test_all.npy` (114,201 vectors $\times$ 160 channels $\times$ 2 bytes = **34.86 MB**)
* **Result**: Once cached, a 25-epoch training run over 860k training samples executes in **under 2 minutes** on an RTX 4090.

### B. Loss Function & Resolution of Rare-Color Collapse (`E-08`)
Vehicle color distributions exhibit natural class imbalance (White, Black, and Silver represent ~60.9% of the fleet, while Gold, Beige, and Orange represent <2% each).
* **The Baseline Flaw**: The original baseline used inverse-frequency class weights $[0.25, 4.0]$. This boosted minority recall by flooding predictions with false-positives, causing rare-class precision to collapse to unacceptable levels (`green` 28%, `brown` 27%, `beige` 21%).
* **The Production Resolution**: In Phase 3 ablations (`E-08`), removing class weights (`--no-class-weights`) and setting `label_smoothing=0.0` eliminated false-positive explosions, dramatically lifting precision across all minority colors (`green` $28\% \to 66\%$, `brown` $27\% \to 61\%$, `beige` $21\% \to 52\%$) and boosting macro precision average from **$56\% \to 71\%$**. This unweighted scheme is the adopted production standard.

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
* The original graph nodes for `predictions`, `embeddings`, and `class_maps` are **100% unaltered**. Verified across 256 test images against `run4/output_merged_run4/models/mobilenet_v2_best.onnx`: **`max |diff| = 0.000e+00`** across all three outputs (`256/256 = 100.00%` top-1 agreement).
* The fused ONNX color branch matches the unfused PyTorch `color_head_best.pt` (`BatchNorm1d`) across 1,024 cached validation vectors with **`max |diff| = 5.722e-06`** (`float32` rounding) and **`1024/1024 = 100.00%`** top-1 argmax agreement.
* The color branch reads from `/features/features.16/Add_output_0` in parallel without affecting the downstream computation of `features.17` and `features.18`.
* All 4 outputs run in a single, fused engine pass with zero duplicate FLOPs.

---

## 7. Directory Layout & CLI Usage

### Directory Structure
```text
code/current/mobilenet/color_distillation/
├── train_color_head.py                 # Full distillation trainer and ONNX splicer
├── layer_probe.py                      # Phase 1 multi-layer empirical probing script
├── verify_multitask_onnx.py            # Signature, latency & numerical equivalence verifier
├── probe_results.csv                   # Empirical benchmark metrics table across 8 layers
├── probe_results.json                  # Full per-class classification metrics
├── color_distillation_approach.md      # Architectural and empirical documentation (this file)
│
├── cache/                              # (DVC-tracked via cache.dvc) Cached float16 feature tensors
│   ├── features_L16_train.npy          # 860,006 x 160 float16 (262.45 MB)
│   ├── labels_train.npy                # 860,006 int64 labels (6.56 MB)
│   ├── features_L16_val.npy            # 109,454 x 160 float16 (33.40 MB)
│   ├── labels_val.npy                  # 109,454 int64 labels (0.84 MB)
│   ├── features_L16_test_all.npy       # 114,201 x 160 float16 (34.86 MB)
│   ├── labels_test_all.npy             # 114,201 int64 labels (0.87 MB)
│   └── confs_test_all.npy              # 114,201 float32 confidences (0.44 MB)
│
├── output_mlp512x2/                    # [REFERENCE PRODUCTION HEAD] Deep MLP-512x2
│   ├── mobilenet_v2_multitask.onnx     # (DVC-tracked via .onnx.dvc) Reference multi-task ONNX (22.4 MB)
│   ├── color_head_best.pt              # (Git-tracked) PyTorch checkpoint (1.4 MB)
│   ├── training_history.csv            # (Git-tracked) Epoch training and validation logs
│   ├── eval_metrics.json               # (Git-tracked) Unbiased held-out test and validation metrics
│   └── final_report.txt                # (Git-tracked) Per-class precision, recall, and F1 report
│
├── output_unweighted_mlp128/           # [ABLATION 1] Unweighted 128-unit MLP
│   └── mobilenet_v2_multitask.onnx     # (DVC-tracked via .onnx.dvc) 21.0 MB
│
└── output/                             # [HISTORICAL BASELINE] Weighted 128-unit MLP
    ├── mobilenet_v2_multitask.onnx     # (DVC-tracked via .onnx.dvc) 20.61 MB
    └── color_head_best.pt              # (Git-tracked) 93 KB
```

### Quick Start Commands

```bash
cd vmmr/code/current/mobilenet

# 1. Train Reference Deep MLP-512x2 head (Unweighted, LS 0.0)
"${PYTHON_BIN:-python3}" -u color_distillation/train_color_head.py \
  --head-type mlp_512x2 \
  --no-class-weights \
  --label-smoothing 0.0 \
  --batch-size 2048 \
  --skip-extract \
  --out-dir color_distillation/output_mlp512x2

# 2. Re-train baseline 128-unit head
"${PYTHON_BIN:-python3}" -u color_distillation/train_color_head.py \
  --head-type mlp \
  --out-dir color_distillation/output

# 3. Verify reference multi-task ONNX model signature, latency & accuracy
python3 color_distillation/verify_multitask_onnx.py \
  --onnx-path color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx \
  --pt-head-path color_distillation/output_mlp512x2/color_head_best.pt
```

