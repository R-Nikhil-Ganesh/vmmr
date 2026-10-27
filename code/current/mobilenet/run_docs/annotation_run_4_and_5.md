# MobileNetV2 Annotation Pipeline Report: Run 4 & Run 5

**System**: Fine-Grained Vehicle Recognition (MobileNetV2 Model C) + Vision-Language Color Tagging (Qwen3-VL-8B)  
**Directory**: `vmmr/code/current/mobilenet/annotation/`  
**Artifact Outputs**:  
- Run 4: `annotation_runs/run4/` (Test split baseline)  
- Run 5: `annotation_runs/run5/` (Full dataset: train, val, and test)  
**Date**: October 14–16, 2026  
**Hardware**: Single NVIDIA GeForce RTX 4090 (24 GB VRAM), 64 GB System RAM, Ubuntu Linux  
**Environment**: PyTorch 2.14+cu121, ONNX Runtime GPU (CUDAExecutionProvider), Hugging Face Transformers (`bitsandbytes` 4-bit NF4)

---

## 1. Executive Summary

The annotation subsystem generates multi-task vehicle annotations across:
1. **Make & Model Classification**: Evaluated via the merged Model C MobileNetV2 ONNX engine (`1,235` classes, `35` automotive makes) with feature embeddings (`1,280`-dim vector per image).
2. **Body Color Pseudo-Labeling**: Tagged with zero-shot Vision-Language model **Qwen3-VL-8B-Instruct** across a controlled vocabulary of 13 primary automotive paint colors.
3. **Interactive 2D/3D Data Exploration**: Dimensionality reduction (UMAP) and embedded HTML visualization (`viewer.html`) with interactive filtering, split toggles, confidence thresholds, and base64 vehicle thumbnails.

### Run Comparison Matrix

| Attribute | Annotation Run 4 | Annotation Run 5 |
| :--- | :--- | :--- |
| **Scope** | Test split only (unseen baseline) | **Full dataset** (`train`, `val`, `test`) |
| **Total Images Annotated** | 114,201 images | **1,126,737 images** |
| **PlatesMania Coverage** | 109,331 images | 1,093,408 images |
| **External Coverage** | 4,870 images | 33,329 images |
| **Make Accuracy** | 98.15% (Test) | **98.15% (Test)** *(98.50% across all splits incl. 79.7% train)* |
| **Model Accuracy** | 91.53% (Test) | **91.53% (Test)** *(93.03% across all splits incl. 79.7% train)* |
| **Mean Color Confidence** | 0.9726 | 0.9674 |
| **Flagged for Review (`needs_review`)** | 12,982 (11.37%) | 117,829 (10.46%) |
| **Qwen3-VL Precision** | bfloat16 *(reconstructed; flags unlogged)* | **4-bit NF4 (`bitsandbytes`)** |
| **Qwen Attention** | Eager / Standard SDPA *(reconstructed)* | **Native SDPA (FlashAttention / Cutlass)** |
| **Qwen Image Resolution** | `max_side=448` *(reconstructed)* | `max_side=336` (fast inference) |
| **Peak VRAM Allocated** | ~16.5 GB | **~5.96 GB** (allocating ~7.1 GB during batch) |
| **Throughput (Color Stage)** | ~23.5 img/s | **~37.0–39.1 img/s** (+66% speedup) |
| **Cache Reuse Strategy** | Cold start | **`--cache-from run4`** (instant reuse of 114k test set) |
| **Embeddings Tensor** | (114,201, 1280) float16 (279 MB) | **(1,126,737, 1280) float16 (2.7 GB)** |
| **CSV Footprint** | 34 MB `annotations.csv` | 340 MB `annotations.csv` |
| **HTML Viewer** | 16 MB (3,000 thumbnails) | 76 MB (15,000 thumbnails + split toggles) |
| **Total Overnight Duration** | ~1.5 hours | **~7 hours 36 minutes** (1.01M uncached images) |

---

## 2. Pipeline Architecture & Execution Stages

The pipeline is implemented in `code/current/mobilenet/annotation/annotate.py` and runs sequentially in 4 stages:

```
[Dataset Manifests (PM + Ext)]
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: MobileNetV2 ONNX GPU Inference                │
│ - Batch size: 64, CUDAExecutionProvider               │
│ - Outputs: class logits, make marginalization,        │
│   top-3 predictions, 1280-dim feature embeddings       │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: UMAP 2D Embedding Projection                  │
│ - Subsamples up to 30,000 points (--coords-max 30000)  │
│ - Populates (x, y) for landmark sample (rest are NaN)  │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Qwen3-VL-8B Color Pseudo-Labeling             │
│ - 4-bit NF4 quantized backbone with SDPA               │
│ - Prompt: "What is the main body color...?"            │
│ - Batch size: 16 (load_crop with domain top/bot crop)  │
│ - Incremental disk flush every 500 images              │
└───────────────────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Interactive HTML Viewer (build_viewer.py)     │
│ - Multi-threaded thumbnail embedding (16 workers)     │
│ - 2D scatter with colors, confidence, & split filters  │
└────────────────────────────────────────────────────────┘
```

### Stage 1: MobileNetV2 ONNX Feature Extraction
* Evaluates images using `output_merged_run4/models/mobilenet_v2_best.onnx`.
* Extracts the penultimate 1,280-dimensional feature embedding before the classification head.
* Predicts the 1,235 model classes and sums probabilities over parent makes to obtain exact `pred_make` and `make_conf`.
* Flags predictions below confidence thresholds (`--review-conf 0.70`, `--review-color-conf 0.50`) with `needs_review=True`.

### Stage 2: 2D Manifold Projection (UMAP)
* Subsamples up to `--coords-max 30000` feature embeddings (`15,000` per domain) and fits PCA(50) followed by UMAP(2) (`n_neighbors=15, min_dist=0.1, metric="cosine"`).
* Populates normalized 2D coordinates `(x, y)` **only for that landmark subset** (14,870 rows = 13.0% in Run 4; 30,000 rows = 2.7% in Run 5) for interactive rendering in `viewer.html`. All remaining rows (`87.0%` in Run 4; `97.3%` in Run 5) have `x = NaN, y = NaN`.

### Stage 3: Vision-Language Color Tagging (Qwen3-VL-8B)
* Applies the same per-domain cropping used by the classifier (`crop_like_classifier`: 15% top banner crop for PlatesMania; 5% bottom watermark crop for External, since `bbox_* = -1` on 100% of rows in both domains).
* Uses **`Qwen/Qwen3-VL-8B-Instruct`** (chosen over the 4B variant for higher zero-shot color fidelity under challenging street lighting while using only ~6 GB VRAM in 4-bit NF4).
* Prompts Qwen3-VL-8B with a single-token constrained classification over the 13 colors defined in `annotation/color_heads.py` (`COLORS`):  
  `white`, `black`, `silver`, `gray`, `red`, `blue`, `green`, `yellow`, `orange`, `brown`, `beige`, `gold`, `other`.
* Calculates exact softmax probabilities over the first generated token logits to output the primary `color` and `color_conf`.
* Flushes predictions incrementally to `color_cache.csv` every 500 images to prevent progress loss.

### Stage 4: Standalone Embedded Viewer (`build_viewer.py`)
* Generates a self-contained, offline-compatible HTML dashboard powered by Plotly.
* Embeds 15,000 base64 webp thumbnails using multi-threaded image workers (`ThreadPoolExecutor(max_workers=16)`).
* Includes interactive color coding by **Color**, **Make**, **Review Status**, and **Split** (`train` = Blue, `val` = Green, `test` = Amber).

---

## 3. Run 5 Optimizations & Breakthroughs

### A. Instant Cache Reuse (`--cache-from run4`)
* Avoided redundant computation by importing all 114,201 test set predictions, embeddings, and Qwen color labels directly from `run4`.
* Verified identity match: zero recalculation was needed for the test split, reducing the overnight workload from 1.13M to 1.01M images.

### B. 4-bit NF4 Quantization (`bitsandbytes`)
* Loaded Qwen3-VL-8B in 4-bit NormalFloat4 with double quantization (`bnb_4bit_compute_dtype=torch.bfloat16`).
* Reduced static VRAM footprint from **16.5 GB down to 5.96 GB**, completely eliminating GPU Out-Of-Memory (OOM) risks.

### C. Scaled Dot-Product Attention (SDPA)
* Configured `attn_implementation="sdpa"`, engaging PyTorch's native C++/CUDA FlashAttention and Cutlass kernels on the Ada Lovelace RTX 4090 architecture.

### D. Vision Token Resolution Scaling (`max_side=336`)
* Downscaled maximum image input dimension from 448px to 336px.
* Reduced vision patch tokens from 256 to 144 per crop (-44% sequence length).
* **Validation**: An empirical agreement test on 100 images revealed **100% agreement (0 disagreements)** between 336px and 448px, confirming that vehicle body color is a macro-spatial feature unaffected by the lower token count.

### E. Memory-Efficient Pipeline Hardening
* Preallocated numpy arrays (`float16`) to eliminate dynamic memory fragmentation across 1.13M rows.
* Streamlined PlatesMania manifest parsing (`pm_table` sets `bbox_* = -1` directly, using the 15% top banner crop without per-row bbox lookup overhead).
* Parallelized thumbnail generation across 16 CPU cores, shrinking HTML compilation time from minutes to 6 seconds.

---

## 4. Performance & Accuracy Breakdown

### Split-Wise Breakdown (Run 5)

| Dataset Split | Image Count | % of Dataset | Make Accuracy | Model Accuracy | Review Flagged | Mean Color Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** *(in-sample)* | 898,284 | 79.72% | 98.60% | 93.39% | 10.21% | 0.9668 |
| **Val** | 114,252 | 10.14% | 98.10% | 91.64% | 11.46% | 0.9673 |
| **Test** *(generalization)* | 114,201 | 10.14% | **98.15%** | **91.53%** | 11.37% | 0.9726 |
| **Combined** *(79.7% train)* | **1,126,737** | **100.0%** | 98.50% | 93.03% | 10.46% | 0.9674 |

> **Key Observation**: Only the **Test** row (`98.15%` Make / `91.53%` Model) represents held-out generalization accuracy (`Train` and `Combined` include the `79.72%` training split that Model C was trained on). Validation model accuracy (`91.64%`) and Test model accuracy (`91.53%`) are virtually identical ($\Delta = 0.11\%$), confirming that checkpoint selection on validation loss introduced no validation selection bias.

### Domain-Wise Breakdown (Run 5)

| Domain | All Splits (incl. ~80% Train) | All-Splits Make / Model Acc | Test Split Count | **Test Split Make / Model Acc** | Review Flagged (All) | Mean Color Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PlatesMania** | 1,093,408 | 98.48% / 92.92% | 109,331 | **98.15% / 91.46%** | 10.61% | 0.9675 |
| **External (Run 4/5 35-Make)** | 33,329 | 99.34% / 96.62% | 4,870 | **98.19% / 93.00%** | 5.57% | 0.9652 |

---

## 5. Vehicle Color Distribution Analysis

Comparing the color distributions between Run 4 (test set only, 114,201 images) and Run 5 (entire dataset, 1,126,737 images) reveals exceptional consistency:

| Color | Run 4 Count | Run 4 Share (%) | Run 5 Count | Run 5 Share (%) | Delta (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **White** | 24,974 | 21.87% | 246,019 | 21.83% | -0.04% |
| **Black** | 23,751 | 20.80% | 235,749 | 20.92% | +0.12% |
| **Silver** | 21,063 | 18.44% | 204,335 | 18.14% | -0.30% |
| **Gray** | 14,620 | 12.80% | 149,414 | 13.26% | +0.46% |
| **Blue** | 10,519 | 9.21% | 101,789 | 9.03% | -0.18% |
| **Red** | 9,726 | 8.52% | 93,342 | 8.28% | -0.24% |
| **Green** | 2,798 | 2.45% | 25,310 | 2.25% | -0.20% |
| **Brown** | 1,668 | 1.46% | 18,946 | 1.68% | +0.22% |
| **Beige** | 1,623 | 1.42% | 18,746 | 1.66% | +0.24% |
| **Yellow** | 1,632 | 1.43% | 15,523 | 1.38% | -0.05% |
| **Orange** | 906 | 0.79% | 10,043 | 0.89% | +0.10% |
| **Gold** | 457 | 0.40% | 3,578 | 0.32% | -0.08% |
| **Other / Ambiguous** | 464 | 0.41% | 3,943 | 0.35% | -0.06% |

The top 3 neutral vehicle colors (White, Black, Silver) account for **60.9%** of the entire fleet, matching standard automotive industry color production statistics worldwide.

---

## 6. Directory Artifacts & File Schemas

All artifacts are persisted in `code/current/mobilenet/annotation_runs/run5/`:

### 1. `annotations.csv` (340 MB, 1,126,738 rows)
Primary database containing per-image annotations:
* `image_id`: Unique integer row index (0 to 1,126,736).
* `domain`: Image dataset source (`platesmania` or `external`).
* `filename` & `path`: Full filesystem path to the source image.
* `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: Cropping bounding box coordinates (`-1` on 100% of rows in Run 4 and Run 5; domain top/bottom crops are applied instead).
* `pred_model`, `pred_model_conf`: MobileNetV2 top-1 predicted model name and softmax probability.
* `top3`: Formatted string of top-3 model predictions and scores.
* `pred_make`, `make_conf`: Marginalized parent make prediction and aggregated probability.
* `gt_class`, `gt_make`: Ground truth class and parent make from dataset manifest.
* `correct_model`, `correct_make`: Boolean evaluation flags.
* `split`: Split assignment (`train`, `val`, `test`).
* `x`, `y`: UMAP 2D manifold projection coordinates (populated for up to `--coords-max 30000` landmark rows: `14,870` rows in Run 4, `30,000` rows in Run 5; `NaN` for all remaining rows).
* `color`, `color_conf`: Qwen3-VL-8B assigned color category and softmax confidence.
* `needs_review`: Boolean flag indicating confidence below threshold.

### 2. `embeddings.npy` (2.7 GB, `shape=(1126737, 1280)`, `dtype=float16`)
Normalized penultimate MobileNetV2 feature embeddings. Supports similarity search and nearest-neighbor clustering.

### 3. `color_cache.csv` (108 MB, 1,126,738 rows)
Resumable key-value store (`image_id,color,color_conf`) mapping each image directly to its vision-language color label.

### 4. `viewer.html` (76 MB)
Self-contained visualization dashboard with 15,000 embedded base64 WebP vehicle thumbnails, interactive zoom/pan, metadata inspection on hover/click, and dynamic filtering.

---

## 7. Next Steps: MobileNet Color Head Distillation *(Completed — See `color_distillation_report.md`)*

With pseudo-labels available across **1.13M images**, color distillation into MobileNetV2 was implemented in [`color_distillation/`](../color_distillation/README.md) and documented in [`color_distillation_report.md`](color_distillation_report.md):

1. **Phase 1: Multi-Layer Probing (`layer_probe.py` — Completed)**:
   * Empirical probing revealed that a `Linear(1280, 13)` head on the penultimate `embeddings.npy` achieved only **53.52%** accuracy because `features[18]` compresses away color variance for make/model invariance.
   * Tapping intermediate block **`features[16]`** (`160` channels) before the final bottleneck recovered **+9.1% accuracy** (`62.58%` on the 25k probe, **68.0%** on the full validation split with a `160 -> 128 -> 13` MLP head).
2. **Phase 2: Frozen-Backbone Graph Splicing (Completed — Preserves Exact Make/Model Identity)**:
   * Rather than unfreezing the backbone (which would alter make/model weights), the trained MLP head (with BatchNorm folded) was spliced directly onto `/features/features.16/Add_output_0`.
3. **Phase 3: Multi-Task ONNX Export (`mobilenet_v2_multitask.onnx` — Completed)**:
   * Exported a 4-output ONNX model (`predictions`, `embeddings`, `class_maps`, `color_predictions`) with **0.000e+00 max absolute difference** on all original make/model outputs.

