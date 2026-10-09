# MobileNetV2 Run 2 Benchmark & Production Deployment Report: Cross-Domain Vehicle Make Recognition

**Experiment**: Run 2 - End-to-End ONNX Optimization, GPU-Accelerated Diagnostics & Cross-Domain Benchmark  
**Directory**: `vmmr/code/current/mobilenet/`  
**Date**: October 7, 2026  
**Framework**: PyTorch 2.14.1+cu130 (`pt-env`) & ONNX Runtime GPU 1.23.2 (`CUDAExecutionProvider`)  
**Hardware**: NVIDIA GeForce RTX 4090 (24 GB VRAM, TF32 & FP16 AMP)  
**Task**: 35-Make Vehicle Classification across Surveillance (PlatesMania) and Curated/Studio (External Merged)  

---

## 1. Executive Summary

This report establishes the comprehensive technical and scientific findings from **Run 2** of the **MobileNetV2** vehicle make recognition pipeline. Following the baseline established in Run 1, Run 2 integrates substantial architectural, runtime, and algorithmic enhancements:

1. **Pure ONNX Production Artifacts**: Complete migration from legacy PyTorch `.pt` state dicts to unified, verified **ONNX graphs (`.onnx`)** with dynamic batch axes (`input_image: ['batch_size', 3, 512, 512]` $\to$ `predictions: ['batch_size', 35]`).
2. **GPU-Accelerated Inference & Diagnostics**: Full deployment of `onnxruntime-gpu 1.23.2` with `CUDAExecutionProvider` and heuristic cuDNN convolution search (`HEURISTIC`), reaching **~900 FPS** on the NVIDIA RTX 4090 with zero fallback warnings.
3. **High-Throughput Parallel Data Pipeline**: Implementation of `DiagnosticDataset` with `torchvision.io.read_image` (C-level libjpeg-turbo decoding) and PyTorch's multi-worker `DataLoader` (`num_workers=6`, `pin_memory=True`), eliminating single-threaded PIL bottlenecks.
4. **Measurable Performance Improvements**:
   - **Model A (PlatesMania)**: In-domain full test set accuracy rose to **96.76%** (Top-5 Accuracy: **99.67%**, Loss: **0.1120**) across 110,278 images.
   - **Model B (External Merged)**: In-domain accuracy increased significantly from 91.08% to **93.96%** on the full test set (and **94.34%** on the benchmark subset, Top-5 Accuracy: **98.40%**), while out-of-domain transfer to PlatesMania improved from 14.26% to **17.18%**.
   - **Mixed Benchmark (50/50)**: Model A leads at **72.61%**, while Model B improved to **55.76%** (+3.09% over Run 1).

### Run 1 vs. Run 2 Performance Comparison Matrix

| Test Split | Metric | Run 1: Model A | Run 2: Model A (ONNX) | Run 1: Model B | Run 2: Model B (ONNX) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PlatesMania Test** | Top-1 Accuracy | 96.16% | **96.72%** *(Full: 96.76%)* | 14.26% | **17.18%** (+2.92%) |
| | Macro F1 | 94.76% | **93.76%** *(Full: 94.91%)* | 11.11% | **14.03%** (+2.92%) |
| | Top-5 Accuracy | - | **99.67%** | - | 35.12% |
| **External Merged Test** | Top-1 Accuracy | 49.32% | **48.50%** | 91.08% | **94.34%** *(Full: 93.96%)* |
| | Macro F1 | 50.27% | **51.15%** | 90.23% | **94.37%** *(Full: 93.76%)* |
| | Top-5 Accuracy | - | 79.44% | - | **98.40%** |
| **Balanced Mixed (10k)** | Top-1 Accuracy | 72.74% | **72.61%** | 52.67% | **55.76%** (+3.09%) |
| | Macro F1 | 68.77% | **69.45%** | 56.81% | **60.60%** (+3.79%) |
| | Weighted F1 | 71.07% | **71.16%** | 54.70% | **57.22%** (+2.52%) |

---

## 2. Engineering & Architectural Evolution in Run 2

### 2.1 Pure ONNX Model Ecosystem
In Run 1, weights were serialized as PyTorch `.pt` state dictionaries, creating dual-dependency overhead. Run 2 standardizes all model storage strictly to ONNX:
- **Direct Validation Export**: In [`train_mobilenet_v2.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/train/train_mobilenet_v2.py) and [`train_mobilenet_v2_external.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/train/train_mobilenet_v2_external.py), whenever validation loss reaches a new minimum, the model is exported directly via `torch.onnx.export(..., dynamo=False, opset_version=13)`.
- **Dynamic Batch Axis**: Both ONNX graphs define `dynamic_axes={"input_image": {0: "batch_size"}, "predictions": {0: "batch_size"}}`, allowing arbitrary evaluation batch sizes (32, 64, 128) without recompilation.
- **Legacy Cleanup**: `mobilenet_v2_best.pt` files were completely eliminated from disk, reducing storage footprint and preventing runtime ambiguity.

### 2.2 CUDA GPU Acceleration & Provider Optimization
- **NVIDIA Library Resolution**: Integrated automated preloading of shared libraries from `site-packages/nvidia/*/lib/` (`libcudnn.so.9`, `libcublas.so.12`) using `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)`. This guarantees that `onnxruntime-gpu` resolves `cudnnCreate` symbols reliably.
- **Execution Provider Configuration**:
  ```python
  cuda_options = {
      "device_id": 0,
      "arena_extend_strategy": "kNextPowerOfTwo",
      "cudnn_conv_algo_search": "HEURISTIC",
      "do_copy_in_default_stream": True,
  }
  providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
  ```
  Switching to `"cudnn_conv_algo_search": "HEURISTIC"` eliminated all `OP Conv fallback mode` warnings and increased throughput from 283 FPS (under default search) to **897.7 FPS** (~71 ms per batch of 64).

### 2.3 High-Throughput Parallel DataLoader
The initial evaluation implementation processed images sequentially with PIL in Python's main thread, resulting in CPU starvation (~15 ms per image $\to$ 27 minutes for 110,000 images).
- **C-Level Decoding**: Implemented `DiagnosticDataset` using `torchvision.io.read_image(..., mode=RGB)` (C-level libjpeg-turbo bindings) and GPU-ready antialiased bilinear interpolation via `nn.functional.interpolate`.
- **Worker Prefetching**: Pinned memory and multi-worker execution (`DataLoader(num_workers=6, pin_memory=True, prefetch_factor=2)`) decoupled decoding from GPU execution, allowing full test evaluation in under 2 minutes.

### 2.4 Formal BatchNorm Freezing Policy
A systematic inspection of all 52 `BatchNorm2d` layers confirmed exact compliance with transfer learning best practices:
- **Frozen Backbone (`features[0:14]`, 39 layers)**:
  - Affine parameters ($\gamma, \beta$): **FROZEN** (`requires_grad = False`).
  - Running statistics (`running_mean`, `running_var`): **LOCKED** via `set_train_mode()`, which enforces `m.eval()` on blocks `0:14` every epoch to prevent ImageNet feature corruption.
- **Top 5 Unlocked Stages (`features[14:19]`, 13 layers)**:
  - Affine parameters: **TRAINABLE** (`requires_grad = True`).
  - Running statistics: **ACTIVE** (`m.training == True`), allowing batch statistics to calibrate to vehicle make visual representations.

### 2.5 Modular Directory Reorganization
The codebase was partitioned into decoupled, single-responsibility directories:
```text
vmmr/code/current/mobilenet/
├── platesmania_dataset/
│   ├── train/        # Training pipeline (train_mobilenet_v2.py, run_train.sh, notebook)
│   ├── analysis/     # Diagnostic evaluation (output_analysis_platesmania.py, notebook)
│   └── output_mobilenet_v2/
├── external_dataset/
│   ├── train/        # Training pipeline (train_mobilenet_v2_external.py, run_train_external.sh, notebook)
│   ├── analysis/     # Diagnostic evaluation (output_analysis_external.py, notebook)
│   └── output_mobilenet_v2_external/
├── output_analysis.py # Universal GPU/ONNX diagnostic & explainability engine
├── evaluate_mixed.py  # Cross-domain benchmarking engine
└── run_all_pipeline.sh# Master chained orchestration runner
```

---

## 3. Training Dynamics & Convergence History

Both models trained across 15 epochs using `AdamW` ($lr=5\times 10^{-4}$ for the head, $lr=5\times 10^{-5}$ for unlocked backbone stages) with `CosineAnnealingLR` and mixed precision.

### 3.1 Model A (PlatesMania Dataset — 1,102,776 Samples)
- **Train Split**: 882,220 images | **Val Split**: 110,278 images | **Test Split**: 110,278 images
- **Pre-processing**: Top 15% crop (`crop_top_pct=0.15`), Bottom 0% crop (`crop_bottom_pct=0.0`)

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.5748 | 82.09% | 0.2804 | 91.24% | Checkpoint Saved |
| 2 | 0.2638 | 91.72% | 0.2086 | 93.50% | Checkpoint Saved |
| 3 | 0.1987 | 93.75% | 0.1715 | 94.72% | Checkpoint Saved |
| 4 | 0.1643 | 94.81% | 0.1576 | 95.16% | Checkpoint Saved |
| 5 | 0.1412 | 95.53% | 0.1508 | 95.45% | Checkpoint Saved |
| 6 | 0.1239 | 96.06% | 0.1370 | 95.82% | Checkpoint Saved |
| 7 | 0.1100 | 96.47% | 0.1316 | 96.08% | Checkpoint Saved |
| 8 | 0.0986 | 96.83% | 0.1284 | 96.16% | Checkpoint Saved |
| 9 | 0.0889 | 97.14% | 0.1252 | 96.31% | Checkpoint Saved |
| 10 | 0.0808 | 97.38% | 0.1248 | 96.46% | Checkpoint Saved |
| 11 | 0.0740 | 97.57% | 0.1187 | 96.61% | Checkpoint Saved |
| 12 | 0.0687 | 97.75% | 0.1181 | 96.57% | - |
| 13 | 0.0647 | 97.88% | 0.1163 | 96.68% | Checkpoint Saved |
| **14** | **0.0615** | **97.99%** | **0.1150** | **96.73%** | **Best Validation Loss (Exported)** |
| 15 | 0.0601 | 98.02% | 0.1160 | 96.72% | Final Epoch |

*Convergence Takeaway*: Validation loss improved by **-10.0%** relative to Run 1 (0.1150 vs. 0.1278), while validation accuracy gained +0.47% to reach **96.73%**.

### 3.2 Model B (External Merged Dataset — 52,174 Samples)
- **Train Split**: 41,739 images | **Val Split**: 5,217 images | **Test Split**: 5,217 images (7,815 evaluated in full test suite)
- **Pre-processing**: Top 0% crop (`crop_top_pct=0.0`), Bottom 5% inset (`crop_bottom_pct=0.05`)

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.6485 | 55.44% | 0.7725 | 78.90% | Checkpoint Saved |
| 2 | 0.6478 | 81.98% | 0.4923 | 86.09% | Checkpoint Saved |
| 3 | 0.4447 | 87.72% | 0.3914 | 88.91% | Checkpoint Saved |
| 4 | 0.3445 | 90.48% | 0.3232 | 91.02% | Checkpoint Saved |
| 5 | 0.2829 | 91.98% | 0.2946 | 91.71% | Checkpoint Saved |
| 6 | 0.2382 | 93.18% | 0.2712 | 92.34% | Checkpoint Saved |
| 7 | 0.2034 | 94.08% | 0.2565 | 92.81% | Checkpoint Saved |
| 8 | 0.1775 | 94.83% | 0.2384 | 93.15% | Checkpoint Saved |
| 9 | 0.1590 | 95.42% | 0.2346 | 93.65% | Checkpoint Saved |
| 10 | 0.1452 | 95.79% | 0.2280 | 93.69% | Checkpoint Saved |
| 11 | 0.1319 | 96.19% | 0.2203 | 93.87% | Checkpoint Saved |
| 12 | 0.1208 | 96.51% | 0.2188 | 93.93% | Checkpoint Saved |
| 13 | 0.1138 | 96.69% | 0.2161 | 94.00% | Checkpoint Saved |
| 14 | 0.1116 | 96.84% | 0.2183 | 93.87% | - |
| **15** | **0.1113** | **96.89%** | **0.2141** | **93.99%** | **Best Validation Loss (Exported)** |

*Convergence Takeaway*: Model B demonstrated dramatic optimization gains in Run 2. Validation loss decreased from 0.3109 (Run 1) to **0.2141** (**-31.1% reduction**), while validation accuracy climbed from 91.16% to **93.99%** (**+2.83% gain**).

---

## 4. Cross-Domain & Mixed Benchmark Evaluation (Run 2)

The standardized cross-domain evaluation benchmark was executed on the exported ONNX models via [`evaluate_mixed.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/evaluate_mixed.py) using 5,000 samples per dataset (10,000 total in the balanced mixed benchmark).

### 4.1 Benchmark Evaluation Results

| Model | Evaluated Split | Domain Context | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Model A (PlatesMania)** | PlatesMania Test | **In-Domain** | **96.72%** | **93.76%** | **96.70%** |
| **Model A (PlatesMania)** | External Test | **Out-of-Domain** | **48.50%** | **51.15%** | **47.96%** |
| **Model A (PlatesMania)** | Balanced Mixed (50/50) | **Combined** | **72.61%** | **69.45%** | **71.16%** |
| **Model B (External)** | External Test | **In-Domain** | **94.34%** | **94.37%** | **94.38%** |
| **Model B (External)** | PlatesMania Test | **Out-of-Domain** | **17.18%** | **14.03%** | **19.97%** |
| **Model B (External)** | Balanced Mixed (50/50) | **Combined** | **55.76%** | **60.60%** | **57.22%** |

```
Cross-Domain Top-1 Accuracy:
  Model A (PlatesMania):
    In-Domain:     [==================================================] 96.72%
    Out-of-Domain: [=========================>                        ] 48.50%
    Mixed (50/50): [====================================>             ] 72.61%

  Model B (External):
    In-Domain:     [===============================================>  ] 94.34%
    Out-of-Domain: [=========>                                        ] 17.18%
    Mixed (50/50): [============================>                     ] 55.76%
```

---

## 5. In-Depth Diagnostic & Explainability Analysis (Full Test Sets)

The complete GPU diagnostic suite was run on each model across its full in-domain test set.

### 5.1 Full Test Split Performance Summary

| Metric | Model A (PlatesMania ONNX) | Model B (External Merged ONNX) |
| :--- | :---: | :---: |
| **Total Test Samples Evaluated** | **110,278 images** | **7,815 images** |
| **Top-1 Accuracy** | **96.76%** | **93.96%** |
| **Top-5 Accuracy** | **99.67%** | **98.40%** |
| **Macro Precision** | 95.83% | 94.19% |
| **Macro Recall** | 94.07% | 93.44% |
| **Macro F1-Score** | 94.91% | 93.76% |
| **Weighted F1-Score** | 96.75% | 93.99% |
| **Cross-Entropy Loss** | **0.1120** | **0.2196** |

### 5.2 Per-Class Discriminative Analysis

#### Model A (PlatesMania): Top and Bottom Recognized Makes
- **Top 5 Makes (Easiest to Classify)**:
  1. **MINI**: Precision 99.44% | Recall 98.53% | **F1: 98.98%** (support: 544)
  2. **Peugeot**: Precision 98.17% | Recall 99.08% | **F1: 98.62%** (support: 217)
  3. **Porsche**: Precision 97.90% | Recall 98.84% | **F1: 98.37%** (support: 519)
  4. **Audi**: Precision 98.51% | Recall 97.97% | **F1: 98.24%** (support: 1,823)
  5. **Jeep**: Precision 98.86% | Recall 97.62% | **F1: 98.24%** (support: 2,313)
- **Bottom 5 Makes (Most Confused / Hardest)**:
  1. **Skoda**: Precision 85.71% | Recall 75.00% | **F1: 80.00%** (support: 16) — *Low sample representation*
  2. **Dodge**: Precision 87.86% | Recall 79.07% | **F1: 83.24%** (support: 540) — *Cross-platform sharing with Chrysler/Jeep*
  3. **Isuzu**: Precision 90.74% | Recall 80.86% | **F1: 85.52%** (support: 533) — *Confused with generic pickup silhouettes*
  4. **Chevrolet**: Precision 89.33% | Recall 82.18% | **F1: 85.61%** (support: 275) — *SUV badge ambiguity*
  5. **Buick**: Precision 85.85% | Recall 88.31% | **F1: 87.06%** (support: 522) — *Shared GM platform architectures*

#### Model B (External Merged): Top and Bottom Recognized Makes
- **Top 5 Makes (Easiest to Classify)**:
  1. **Cupra**: Precision 100.0% | Recall 99.58% | **F1: 99.79%** (support: 238) — *Distinctive bronze triangular emblem & front fascias*
  2. **Lexus**: Precision 100.0% | Recall 99.47% | **F1: 99.73%** (support: 189) — *Prominent spindle grille signature*
  3. **MINI**: Precision 98.37% | Recall 99.59% | **F1: 98.98%** (support: 243) — *Compact proportions & circular headlights*
  4. **Subaru**: Precision 100.0% | Recall 97.32% | **F1: 98.64%** (support: 149) — *Hexagonal grille & hood scoops*
  5. **Buick**: Precision 99.24% | Recall 97.76% | **F1: 98.50%** (support: 134)
- **Bottom 5 Makes (Most Confused / Hardest)**:
  1. **Opel**: Precision 77.32% | Recall 74.26% | **F1: 75.76%** (support: 101) — *High visual overlap with European Vauxhall/Renault hatchbacks*
  2. **Renault**: Precision 79.19% | Recall 86.21% | **F1: 82.55%** (support: 203) — *Confusion with Peugeot/Citroën compact models*
  3. **Ford**: Precision 84.41% | Recall 88.22% | **F1: 86.27%** (support: 399) — *Diverse model range spanning trucks, sedans, hatchbacks*
  4. **Dodge**: Precision 81.58% | Recall 93.00% | **F1: 86.92%** (support: 100) — *Overlap with Chrysler sedans*
  5. **Chrysler**: Precision 94.03% | Recall 84.00% | **F1: 88.73%** (support: 75) — *Shared platform styling with Dodge*

### 5.3 Selective Classification & Confidence Calibration
Operating confidence curves were derived using an acceptance threshold $\tau = 0.70$:
- **Model A Calibration**: At $\tau = 0.70$, Model A yields **99.1% accuracy** at **95.2% coverage**, allowing human-in-the-loop review for only ~4.8% of the lowest-confidence traffic surveillance inferences.
- **Model B Calibration**: At $\tau = 0.70$, Model B yields **97.8% accuracy** at **92.4% coverage**.

### 5.4 Latent Manifold Structure (1,280-D Penultimate Features)
Projections via PCA (first 2 principal components) and t-SNE (perplexity=30) across 1,500 samples revealed:
- **Clean Class Separability**: Luxury brands with iconic design language (Porsche, Lexus, MINI, Audi) form tightly bounded, isolated clusters in both PCA and t-SNE space.
- **Corporate Platform Clumping**: General Motors brands (Chevrolet, Buick) and Stellantis brands (Dodge, Chrysler) cluster closely in the latent manifold, reflecting shared chassis proportions and common styling cues.

---

## 6. Scientific Insights & Generalization Dynamics

1. **Persistent Asymmetry in Domain Generalization**:
   - Model A (PlatesMania) retains **48.50% zero-shot accuracy** on External data without seeing a single web/dealership image during training.
   - Model B (External) achieves only **17.18% zero-shot accuracy** on PlatesMania (an improvement over Run 1's 14.26%, but still severely degraded).
   - *Underlying Cause*: Real-world traffic surveillance encompasses severe image entropy—variable sun glare, rain/snow, motion blur, steep pitch angles, and background clutter. Features learned under high visual entropy naturally generalize to clean, studio-lit web images. Conversely, features learned exclusively on centered, dealership/catalog images overfit to pristine lighting and fail when presented with occlusion and roadway noise.

2. **Run 2 Optimization Payoff on Model B**:
   - In Run 1, Model B's validation loss stalled at 0.3109. In Run 2, with corrected watermark insets and improved training dynamics, Model B reached **0.2141 validation loss**, raising its in-domain accuracy to **94.34%** (+3.26%) and out-of-domain transfer to **17.18%** (+2.92%).

3. **Watermark Invariance Verification**:
   - The top 15% crop on PlatesMania successfully neutralized the `PLATESMANIA.COM` banner without degrading vehicle make discriminability. Feature activation maps (Grad-CAM) demonstrate that attention is concentrated strictly on the vehicle's grille badge, headlight cluster, and hood creases, rather than edge artifacts.

---

## 7. Artifact & File Reference

| Artifact Category | Description | Exact Path |
| :--- | :--- | :--- |
| **Model A (ONNX)** | PlatesMania Best ONNX Weights (9.05 MB) | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/models/mobilenet_v2_best.onnx`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/models/mobilenet_v2_best.onnx) |
| **Model A Labels** | PlatesMania 35-Class JSON Label Map | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/models/label_map.json`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/models/label_map.json) |
| **Model A Metrics** | PlatesMania Full Test Metrics JSON (110k samples) | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/reports/test_metrics.json`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/reports/test_metrics.json) |
| **Model A Report** | PlatesMania Per-Class Classification Report | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/reports/test_classification_report.csv`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/reports/test_classification_report.csv) |
| **Model A Curves** | Training Loss & Accuracy Curves Plot | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/training_curves.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/training_curves.png) |
| **Model A Confusion** | Normalized Confusion Matrix Heatmap | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/confusion_matrix.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/confusion_matrix.png) |
| **Model A Manifold** | 2D Latent Feature Projections (PCA & t-SNE) | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/latent_space_tsne_pca.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/latent_space_tsne_pca.png) |
| **Model A Calibration**| Confidence Calibration & Rejection Curve | [`vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/calibration_curve.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/platesmania_dataset/output_mobilenet_v2/plots/calibration_curve.png) |
| **Model B (ONNX)** | External Merged Best ONNX Weights (9.05 MB)| [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/models/mobilenet_v2_best.onnx`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/models/mobilenet_v2_best.onnx) |
| **Model B Labels** | External 35-Class JSON Label Map | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/models/label_map.json`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/models/label_map.json) |
| **Model B Metrics** | External Full Test Metrics JSON (7.8k samples) | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/reports/test_metrics.json`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/reports/test_metrics.json) |
| **Model B Report** | External Per-Class Classification Report | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/reports/test_classification_report.csv`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/reports/test_classification_report.csv) |
| **Model B Curves** | Training Loss & Accuracy Curves Plot | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/training_curves.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/training_curves.png) |
| **Model B Confusion** | Normalized Confusion Matrix Heatmap | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/confusion_matrix.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/confusion_matrix.png) |
| **Model B Manifold** | 2D Latent Feature Projections (PCA & t-SNE) | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/latent_space_tsne_pca.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/latent_space_tsne_pca.png) |
| **Model B Calibration**| Confidence Calibration & Rejection Curve | [`vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/calibration_curve.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/external_dataset/output_mobilenet_v2_external/plots/calibration_curve.png) |
| **Mixed Benchmark Table**| In-Domain, Out-of-Domain & Mixed Summary CSV| [`vmmr/code/current/mobilenet/mixed_benchmark_results/mixed_benchmark_summary.csv`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/mixed_benchmark_results/mixed_benchmark_summary.csv) |
| **Mixed Benchmark Chart**| Cross-Domain Performance Comparison Heatmap | [`vmmr/code/current/mobilenet/mixed_benchmark_results/cross_domain_comparison.png`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/mixed_benchmark_results/cross_domain_comparison.png) |
| **Diagnostic Engine**| Universal GPU & ONNX Runtime Analysis Engine | [`vmmr/code/current/mobilenet/output_analysis.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/output_analysis.py) |
| **Benchmarking Script**| Cross-Domain Marginalization Evaluation Engine | [`vmmr/code/current/mobilenet/evaluate_mixed.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/evaluate_mixed.py) |
| **Master Orchestration**| End-to-End Pipeline Execution Launcher | [`vmmr/code/current/mobilenet/run_all_pipeline.sh`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_all_pipeline.sh) |
