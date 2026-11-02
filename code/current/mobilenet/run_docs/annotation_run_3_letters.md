# MobileNetV2 Annotation Pipeline Report: Annotation Run 3 (Letters Prompt & bfloat16)

**System**: Multi-Task Vehicle Recognition (MobileNetV2 Model C) + Calibrated Vision-Language Color Tagging (Qwen3-VL-8B)  
**Directory**: `vmmr/code/current/mobilenet/annotation/`  
**Artifact Outputs**:  
- Champion Dataset Run: `annotation_runs/run3_letters/` (1,161,103 images, full bfloat16 letters prompt)  
- Benchmark Suite: `annotation_runs/teacher_eval/` (13 prompt & quantization ablations on human ground truth)  
**DVC Pointer**: `code/current/mobilenet/annotation_runs/run3_letters.dvc`  
**Date**: October 29–31, 2026  
**Hardware**: Single NVIDIA GeForce RTX 4090 (24 GB VRAM), 64 GB System RAM, Ubuntu Linux  
**Environment**: PyTorch 2.14+cu130, ONNX Runtime GPU (CUDAExecutionProvider), Hugging Face Transformers  

---

## 1. Executive Summary

**Annotation Run 3 (`run3_letters`)** represents the major upgrade to our pseudo-labeling pipeline, addressing quantization noise and prompt-bias discovered during extensive ground-truth validation against CompCars CCTV.

### Key Milestones in Run 3
1. **Full bfloat16 Inference (`--no-4bit`)**: Abandoned 4-bit NF4 quantization in favor of native `bfloat16`, recovering subtle chromatic gradients and eliminating color boundary artifacts.
2. **Multiple-Choice Letter Prompting (`--qwen-scoring letters`)**: Replaced raw color token scoring with an explicit lettered choice prompt (`A: white`, `B: black`, ... `M: other`). This achieved **`82.16%`** grouped accuracy on CompCars CCTV ground truth (+3.46% over Run 2's 78.70%).
3. **Comprehensive Dataset Expansion**: Scaled from 1.126M to **1,161,103 images**, doubling external dataset coverage (from 33,329 to **67,695** images).
4. **Full 13-Class Probability Distribution Caching**: Generated `color_cache.csv` (400 MB) storing complete softmax probability vectors (`p_white` ... `p_other`), enabling continuous label-smoothing and temperature-scaling for downstream distillation.

---

## 2. Chronological Run Comparison Matrix

| Attribute | Annotation Run 1 | Annotation Run 2 | Annotation Run 3 (`run3_letters`) ⭐ |
| :--- | :--- | :--- | :--- |
| **Scope** | Test split only (114k) | Full dataset (1.126M) | **Full dataset (1.161M images)** |
| **Total Images Annotated** | 114,201 images | 1,126,737 images | **1,161,103 images** (+34,366 images) |
| **PlatesMania Coverage** | 109,331 images | 1,093,408 images | **1,093,408 images** |
| **External Coverage** | 4,870 images | 33,329 images | **67,695 images** (+103% expansion) |
| **Qwen Precision** | bfloat16 | 4-bit NF4 (`bitsandbytes`) | **Full bfloat16 (`--no-4bit`)** |
| **Prompt / Scoring Mode** | First-token string match | First-token string match | **Multiple-Choice Letters (`A-M`)** |
| **CompCars CCTV Teacher Acc** | ~78.7% grouped | 78.70% grouped | **`82.16%` grouped (`72.58%` strict)** |
| **White Recall (Ground Truth)** | ~74.9% | 74.90% | **`83.47%`** (+8.57%) |
| **Red Recall (Ground Truth)** | ~79.2% | 79.23% | **`90.34%`** (+11.11%) |
| **Black Recall (Ground Truth)** | ~92.1% | 92.10% | **`94.37%`** (+2.27%) |
| **Mean Color Confidence** | 0.9726 | 0.9674 | **`0.9745`** (Highest calibration) |
| **Flagged for Review (`needs_review`)**| 12,982 (11.37%) | 117,829 (10.46%) | **134,678 (11.60%)** |
| **Color Cache Details** | Top-1 class only | Top-1 class + conf | **Full 13-class softmax vector (400 MB)** |
| **Embeddings Tensor** | (114k, 1280) float16 (279 MB)| (1.12M, 1280) float16 (2.7 GB) | **(1.16M, 1280) float16 (2.8 GB)** |
| **DVC Remote Artifact** | `run1.dvc` | `run2.dvc` | **`run3_letters.dvc`** |

---

## 3. Teacher Calibration Benchmark (`teacher_eval`)

Before launching the full 1.16M-image re-annotation, we conducted a rigorous prompt engineering and quantization ablation suite across 2,859 held-out human-annotated CompCars CCTV images using [`annotation/eval_color_teacher.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/annotation/eval_color_teacher.py).

### Benchmark Results Table

| Experiment Tag | Precision | Scoring Mode | Max Side | Center Crop | TTA | Grouped Acc | Strict Acc | White Rec | Red Rec | Black Rec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `s0_original` (Run 2) | 4-bit NF4 | First Token | 336 | 0.00 | None | 78.70% | 70.23% | 74.90% | 79.23% | 92.10% |
| `s1_bf16` | bfloat16 | First Token | 336 | 0.00 | None | 80.62% | 72.58% | 79.48% | 84.54% | 92.79% |
| `s2_variants` | bfloat16 | Variants | 336 | 0.00 | None | 80.41% | 72.02% | 78.88% | 83.57% | 92.60% |
| **`s2_letters` (Selected)** ⭐ | **bfloat16** | **Letters** | **336** | **0.00** | **None** | **`82.16%`** | **`72.58%`** | **`83.47%`** | **`90.34%`** | **`94.37%`** |
| `s3_side672` | bfloat16 | Variants | 672 | 0.00 | None | 80.55% | 71.74% | 76.29% | 84.54% | 92.00% |
| `s3_side1024` | bfloat16 | Variants | 1024 | 0.00 | None | 80.62% | 71.70% | 76.10% | 84.54% | 92.20% |
| `s3_center15` | bfloat16 | Variants | 336 | 0.15 | None | 79.64% | 71.28% | 78.69% | 82.13% | 93.39% |
| `s4_defined` | bfloat16 | Variants | 336 | 0.00 | None | 76.01% | 72.44% | 57.57% | 82.13% | 94.87% |
| `s5_flip` | bfloat16 | Variants | 336 | 0.00 | Flip | 80.38% | 72.16% | 78.88% | 84.06% | 93.09% |
| `s6_letters_672` | bfloat16 | Letters | 672 | 0.00 | None | 82.65% | 72.23% | 83.07% | 89.86% | 93.58% |

### Key Benchmark Discoveries

1. **Quantization Penalty**: Moving from 4-bit NF4 to native `bfloat16` immediately improved accuracy by **+1.92%** (78.70% $\to$ 80.62%), with white recall rising +4.58% and red recall rising +5.31%.
2. **Multiple-Choice Letter Calibration**: The multiple-choice format (`A: white`, `B: black`...) provided a further **+1.54%** accuracy boost (reaching **82.16%** grouped accuracy). It solved Qwen's token-length bias where longer color words ("champagne", "silver") suffered lower softmax mass than single-syllable colors ("red", "black").
3. **Resolution & Crop Saturation**: Scaling `max_side` to 672 or 1024 halved inference throughput (32 img/s $\to$ 16 img/s) without improving strict accuracy (71.7% vs. 72.6%). Cropping 15% center border degraded recall (79.64%) by clipping vehicle extremities.
4. **Independent Test Split Validation**: On the unseen held-out test split (2,867 CompCars images), `s2_letters` verified at **`81.34%`** grouped accuracy and **`72.24%`** strict accuracy.

---

## 4. Run 3 Dataset Breakdown & Distribution

### Split & Domain Distribution (1,161,103 Total Images)

```
Split      | PlatesMania | External (CompCars + Others) | Total Images | Make Accuracy | Model Accuracy
-----------+-------------+------------------------------+--------------+---------------+---------------
train      |     874,726 |                       47,789 |      922,515 |        96.61% |         91.01%
val        |     109,341 |                        9,973 |      119,314 |        94.89% |         87.87%
test       |     109,341 |                        9,933 |      119,274 |        94.93% |         87.74%
-----------+-------------+------------------------------+--------------+---------------+---------------
Total      |   1,093,408 |                       67,695 |    1,161,103 |        96.26% |         90.35%
```

### Full-Fleet Predicted Color Distribution

Across the 1,161,103 images in `run3_letters`, the predicted color distribution accurately reflects in-the-wild automotive market share:

| Color Class | Total Images | Fleet Percentage | Cumulative Share | Market Category |
| :--- | :---: | :---: | :---: | :---: |
| **White** | 259,423 | **`22.34%`** | 22.34% | Neutral |
| **Black** | 252,317 | **`21.73%`** | 44.07% | Neutral |
| **Silver** | 195,170 | **`16.81%`** | 60.88% | Neutral |
| **Gray** | 150,756 | **`12.98%`** | **`73.86%`** | Neutral |
| **Red** | 104,398 | **`8.99%`** | 82.85% | Chromatic |
| **Blue** | 102,618 | **`8.84%`** | 91.69% | Chromatic |
| **Green** | 28,610 | **`2.46%`** | 94.15% | Chromatic |
| **Beige** | 20,432 | **`1.76%`** | 95.91% | Neutral / Earth |
| **Yellow** | 16,755 | **`1.44%`** | 97.35% | Chromatic |
| **Brown** | 12,500 | **`1.08%`** | 98.43% | Earth |
| **Orange** | 11,325 | **`0.98%`** | 99.41% | Chromatic |
| **Gold** | 4,137 | **`0.36%`** | 99.77% | Earth / Metallic |
| **Other** | 2,662 | **`0.23%`** | 100.00% | Non-Standard |

---

## 5. Artifacts, Caches & DVC Tracking

The full dataset artifacts generated by Run 3 are stored on disk and version-tracked via DVC:

```
vmmr/code/current/mobilenet/annotation_runs/run3_letters/
├── annotations.csv      (314 MB, 1,161,104 rows: make/model preds, color, conf, review flags)
├── color_cache.csv      (400 MB, 13-class softmax probability distributions per image)
├── embeddings.npy       (2.8 GB, 1,161,103 × 1280 float16 latent feature vectors)
├── viewer.html          (74 MB, standalone interactive UMAP visualization with 15k points)
├── plotly.min.js        (4.4 MB, offline visualization dependency)
└── run_config.json      (2.3 KB, reproducible runtime configuration & hardware specs)
```

### DVC Tracking Status
- **DVC Pointer**: [`code/current/mobilenet/annotation_runs/run3_letters.dvc`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/annotation_runs/run3_letters.dvc)
- **Git Commit**: [`bd2a790`](https://github.com/R-Nikhil-Ganesh/vmmr/commit/bd2a790) on branch `main`.
- **Remote Storage**: Synchronized to Google Drive remote `gdrive-storage` (`gdrive://1jmsSPZKa0o1LpOW8mgZFg1kghPBolerc`).
