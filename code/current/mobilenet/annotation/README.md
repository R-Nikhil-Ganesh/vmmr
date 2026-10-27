# MobileNet Vehicle Annotation & Visualization Suite

This module provides an automated, multi-stage pipeline for generating fine-grained vehicle annotations (make/model classification, feature embeddings, 2D manifold projection, and vision-language color pseudo-labeling) alongside an interactive, self-contained HTML viewer.

---

## 1. Module Architecture

```text
code/current/mobilenet/annotation/
├── annotate.py         # Primary multi-stage pipeline coordinator
├── color_heads.py      # Color classification interfaces (Qwen3-VL-8B, 4-bit NF4, SDPA)
├── build_viewer.py     # Standalone Plotly HTML dashboard builder with embedded thumbnails
├── common.py           # Shared data loaders, bounding box parsers, and crop preprocessors
└── plotly.min.js       # Offline JavaScript library embedded into viewer.html
```

---

## 2. Pipeline Execution Stages

Running `annotate.py` executes 4 distinct stages:

1. **Stage 1: MobileNetV2 ONNX Inference**
   - Batched forward pass on GPU (`CUDAExecutionProvider`).
   - Generates make/model predictions (`1,235` classes) and marginalized parent make predictions (`35` makes).
   - Extracts penultimate `1,280`-dimensional feature embeddings saved as `embeddings.npy`.
   - Flags low-confidence predictions with `needs_review=True`.

2. **Stage 2: 2D Manifold Projection (UMAP / t-SNE)**
   - Subsamples up to `--coords-max 30000` feature embeddings (`15,000` per domain) and fits PCA(50) + UMAP(2) using cosine distance.
   - Populates 2D coordinates `(x, y)` for that landmark subset for `viewer.html` clustering visualization (remaining rows have `x, y = NaN`).

3. **Stage 3: Vision-Language Color Pseudo-Labeling (Qwen3-VL-8B)**
   - Zero-shot body color tagging over 13 primary automotive paint colors.
   - Optimized with `bitsandbytes` 4-bit NF4 quantization, PyTorch native SDPA, and `max_side=336` resolution scaling.
   - Writes incremental updates to `color_cache.csv` every 500 images to support seamless pause and resume.

4. **Stage 4: Standalone Interactive HTML Viewer**
   - Embeds up to 15,000 high-resolution WebP thumbnails using multi-threaded image workers (`ThreadPoolExecutor`).
   - Offers interactive filters by vehicle color, parent make, prediction confidence, needs-review flags, and dataset split.

---

## 3. CLI Usage & Flags

### Recommended Launch via Bash Runner

The easiest way to execute annotations is using the tmux-wrapped runner:

```bash
cd vmmr/code/current/mobilenet

# Run annotation across the full dataset (all splits) with automatic run4 cache reuse
RUN_NAME=run5 bash bash_scripts/run_annotate.sh --split all

# Smoke test on 200 samples without Qwen
bash bash_scripts/run_annotate.sh --max-samples 200 --skip-color
```

### Direct Python Execution

```bash
python annotation/annotate.py \
    --model-path merged_dataset/output_merged_run4/models/mobilenet_v2_best.onnx \
    --label-map merged_dataset/output_merged_run4/models/label_map.json \
    --output-dir annotation_runs/run5 \
    --cache-from annotation_runs/run4 \
    --split all \
    --color-batch-size 16 \
    --qwen-max-side 336
```

### Key CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--split {test,all,train,val}` | `test` | Which dataset splits to process (`all` processes train, val, and test). |
| `--cache-from DIR` | `None` | Path to a previous run (e.g. `run4`) to reuse existing predictions, embeddings, and color labels. |
| `--domain {both,pm,ext}` | `both` | Which dataset domains to include (`pm` = PlatesMania, `ext` = External). |
| `--max-samples N` | `None` | Cap maximum samples per domain (useful for quick ablations and smoke tests). |
| `--skip-color` | `False` | Skip vision-language color pseudo-labeling. |
| `--color-head` | `qwen3vl` | Color labeling backend (`qwen3vl` or `clip`). |
| `--qwen-model` | `Qwen/Qwen3-VL-8B-Instruct` | Hugging Face model ID or local directory path. |
| `--color-batch-size N` | `16` | Batch size for Qwen inference. |
| `--qwen-max-side N` | `336` | Maximum image side resolution for Qwen (reduces vision token count). |
| `--no-4bit` | `False` | Disable 4-bit NF4 quantization and run in bfloat16. |
| `--coords-max N` | `30000` | Maximum number of rows (`N/2` per domain) projected into 2D UMAP coordinates `(x, y)`. |
| `--max-points N` | `15000` | Number of sample points with embedded base64 thumbnails in `viewer.html`. |
| `--review-conf F` | `0.70` | Model confidence threshold below which row is marked `needs_review`. |
| `--review-color-conf F` | `0.50` | Color confidence threshold below which row is marked `needs_review`. |
| `--no-viewer` | `False` | Skip generating the HTML viewer. |

---

## 4. Rebuilding or Customizing the Viewer

To rebuild the viewer independently from an existing `annotations.csv` without re-running inference:

```bash
python annotation/build_viewer.py \
    --annotations annotation_runs/run5/annotations.csv \
    --out annotation_runs/run5/viewer.html \
    --max-points 20000 \
    --workers 16
```

---

## 5. Artifact Outputs

Each annotation run writes the following artifacts inside `annotation_runs/<RUN_NAME>/`:

* `annotations.csv`: Complete per-image predictions, confidence scores, bounding boxes, splits, UMAP coordinates, and review flags.
* `embeddings.npy`: `(N, 1280)` float16 penultimate feature embeddings extracted from MobileNetV2.
* `color_cache.csv`: Resumable key-value cache mapping image indices to assigned colors and probabilities.
* `viewer.html`: Standalone, interactive 2D visualization dashboard with embedded image thumbnails.

