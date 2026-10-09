# MobileNetV2 Vehicle Recognition (PyTorch & ONNX)

This directory contains the mobile-first PyTorch training pipelines, diagnostic evaluation suites, and cross-domain mixed benchmark runners for fine-grained vehicle recognition using **MobileNetV2**.

---

## Directory Overview

```text
vmmr/code/current/mobilenet/
├── platesmania_dataset/                  # PlatesMania Dataset Pipeline (Model A)
│   ├── train/                            # Training subsystem
│   │   ├── train_mobilenet_v2.py         # PyTorch training pipeline (AMP, CosineAnnealing, ONNX export)
│   │   ├── train_mobilenet_v2.ipynb      # Interactive training notebook
│   │   └── run_train.sh                  # Headless training launcher
│   ├── analysis/                         # Evaluation & explainability subsystem
│   │   ├── output_analysis_platesmania.py    # Diagnostic evaluation adapter
│   │   └── output_analysis_platesmania.ipynb # Full diagnostic evaluation notebook
│   └── output_mobilenet_v2/              # Model artifacts (.onnx), metrics, plots, logs
│
├── external_dataset/                     # External Merged Dataset Pipeline (Model B)
│   ├── train/                            # Training subsystem
│   │   ├── train_mobilenet_v2_external.py    # PyTorch training pipeline (35 Makes, 52,174 images)
│   │   ├── train_mobilenet_v2_external.ipynb # Interactive training notebook
│   │   └── run_train_external.sh             # Headless training launcher
│   ├── analysis/                         # Evaluation & explainability subsystem
│   │   ├── output_analysis_external.py       # Diagnostic evaluation adapter
│   │   └── output_analysis_external.ipynb    # Full diagnostic evaluation notebook
│   └── output_mobilenet_v2_external/     # Model artifacts (.onnx), metrics, plots, logs
│
├── output_analysis.py                    # Universal PyTorch & ONNX diagnostic & explainability engine
├── evaluate_mixed.py                     # Cross-domain & mixed benchmark evaluation engine
├── run_all_pipeline.sh                   # Master chained runner for both models + eval
├── run_eval_mixed.sh                     # Mixed benchmark evaluation launcher
├── run_1.md                              # Comprehensive benchmark report for Run 1
├── run_2.md                              # Comprehensive benchmark & production report for Run 2
└── README.md                             # Documentation & user guide
```

---

## 1. Quick Start: End-to-End Pipeline

Run the full pipeline (Model A training $\to$ Model B training $\to$ Mixed Benchmark evaluation $\to$ **Auto Git Push**):

```bash
cd vmmr/code/current/mobilenet

# One-click detached tmux launcher (auto pushes to GitHub upon completion)
bash start_pipeline_tmux.sh

# To view / monitor progress:
tmux attach -t mobilenet_pipeline

# Or direct foreground execution
bash run_all_pipeline.sh
```

---

## 2. PlatesMania Dataset Pipeline (`platesmania_dataset/`)

Trained on surveillance & natural street-level vehicle images (1,093,501 images across **1,235 fine-grained Make/Model classes** spanning 35 makes):
- **Manifest**: `/home/researchadmin/Econ/models/dataset_manifests/dataset_1235models_splits.csv`
- **Label Map**: `/home/researchadmin/Econ/models/dataset_manifests/label_map_1235models.json` (1,235 classes)
- **Watermark Countermeasure**: On-the-fly cropping of top 15% banner (`PLATESMANIA.COM`) only (`crop_top_pct=0.15`, `crop_bottom_pct=0.0`).
- **Head Strategy**: Single flat fine-grained head (`nn.Linear(1280, 1235)`) with top 5 backbone layers unlocked.
- **Auto Git Push**: Automatically commits and pushes `.onnx` models, training logs, and metrics to GitHub when done.

```bash
cd vmmr/code/current/mobilenet/platesmania_dataset/train

# One-click detached tmux launcher (auto pushes to GitHub upon completion)
bash start_train_tmux.sh

# Monitor session:
tmux attach -t train_pm

# Or run directly in foreground
bash run_train.sh

# Run diagnostic suite
cd ../analysis
python output_analysis_platesmania.py
```

---

## 3. External Merged Dataset Pipeline (`external_dataset/`)

Trained on 41,880 curated vehicle images mapped directly into the **1,235 Make/Model taxonomy** across 268 active models and 35 automotive makes (BoxCars116k, Stanford Cars, CompCars CCTV):
- **Splits**: `splits_1235models/` (`train.csv`: 29,554, `val.csv`: 6,163, `test.csv`: 6,163)
- **Label Map**: `splits_1235models/label_map.json` (1,235 classes, shared taxonomy)
- **Bounding Boxes**: On-the-fly cropping of vehicle bounding boxes for Stanford Cars.
- **Head Strategy**: `nn.Linear(1280, 1235)` with top 5 backbone layers unlocked.
- **Auto Git Push**: Automatically commits and pushes `.onnx` models, training logs, and metrics to GitHub when done.

```bash
cd vmmr/code/current/mobilenet/external_dataset/train

# One-click detached tmux launcher (auto pushes to GitHub upon completion)
bash start_train_tmux.sh

# Monitor session:
tmux attach -t train_ext

# Or run directly in foreground
bash run_train.sh

# Run diagnostic suite
cd ../analysis
python output_analysis_external.py
```

---

## 4. Cross-Domain & Mixed Benchmark Evaluation (`evaluate_mixed.py`)

Compares **Model A** (PlatesMania) and **Model B** (External) on:
1. **Fine-Grained Model Level**: Top-1 Accuracy & Macro F1 across 1,235 vehicle model classes.
2. **Coarse Make Level**: Top-1 Accuracy & Macro F1 across 35 automotive makes via probability marginalization:
   $$P(\text{Make}_k) = \sum_{m \in \text{Make}_k} P(\text{Model}_m)$$

Evaluated across:
1. **In-Domain Test Sets**
2. **Out-of-Domain Cross Test Sets**
3. **Mixed Test Set** (Combined benchmark of up to 5,000 samples per dataset)

```bash
cd vmmr/code/current/mobilenet

# Run one-click evaluation
bash run_eval_mixed.sh
```

Generates:
- `mixed_benchmark_results/mixed_benchmark_summary.csv`
- `mixed_benchmark_results/cross_domain_comparison.png` (side-by-side Make Accuracy & Model Accuracy comparison)

---

## 5. Key Architectural Features

- **Framework**: PyTorch 2.14 (`pt-env`) with mixed precision (`torch.cuda.amp.autocast()`) and TF32 enabled for NVIDIA RTX 4090.
- **Backbone**: `torchvision.models.mobilenet_v2` (ImageNet pretrained, top 5 layers unfrozen).
- **Automated ONNX Export**: Automatically serializes the best checkpoint to `.onnx` whenever validation loss reaches a new minimum. The graph exposes three outputs: `predictions` (logits), `embeddings` (1280-D, for t-SNE/PCA) and `class_maps` (per-class activation maps, for Grad-CAM).
- **Diagnostic Engine**: Analytical Grad-CAM, 2D latent space projections (t-SNE & PCA), confusion matrices, and confidence calibration with rejection curves ($\tau = 0.70$).

---

## Moving to another server: `paths.sh`

All machine-specific paths live in [`paths.sh`](paths.sh) (read by every `.sh` runner and, through [`paths.py`](paths.py), by the Python scripts). The defaults are the original server's values. On a new machine create `paths.local.sh` next to it (gitignored) with only what differs:

```bash
ECON_ROOT=/data/Econ                       # everything below is derived from it
PYTHON_BIN=/opt/envs/pt-env/bin/python     # optional
# PM_IMG_DIR, PM_MANIFEST_CSV, PM_LABEL_MAP, EXT_DATASETS_DIR, AUTO_GIT_PUSH can be set individually
```

Precedence: shell environment > `paths.local.sh` > defaults. Check the result with `python paths.py --check`.

## Cross-dataset generalization options

Both training scripts share [`train_common.py`](train_common.py) so they use identical augmentation and objectives. Every option defaults to the original behaviour; extra flags passed to a runner go straight to the trainer, and `RUN_NAME=<name>` keeps each experiment in its own folders.

| Flag | Effect |
|---|---|
| `--aug-strength strong` | wide zoom/aspect (tight vs loose framing), blur, noise, random erasing, stronger colour jitter |
| `--crop-jitter 0.5` | randomizes the train-split crop (PlatesMania top crop, External bbox margin / bottom crop); val/test unchanged |
| `--label-smoothing 0.1` | smoothed cross-entropy (validation loss stays plain CE, so checkpoint selection is comparable) |
| `--make-loss-weight 0.3` | auxiliary loss on P(make) = sum of P(model) over the make's models |
| `--ema-decay 0.999` | EMA of the weights; validation, best-checkpoint selection and ONNX export use the EMA weights |

```bash
# ablation 1: augmentation only (run for both datasets, then evaluate with the same RUN_NAME)
RUN_NAME=aug bash platesmania_dataset/train/run_train.sh --aug-strength strong --crop-jitter 0.5
RUN_NAME=aug bash external_dataset/train/run_train_external.sh --aug-strength strong --crop-jitter 0.5
RUN_NAME=aug bash run_eval_mixed.sh
# ablation 2 adds:  --label-smoothing 0.1 --make-loss-weight 0.3      ablation 3 adds:  --ema-decay 0.999
```

`evaluate_mixed.py` also writes `class_coverage.csv` and `shared_class_summary.csv`: accuracy restricted to test images whose class has training images in **both** datasets (plain top-1 and top-1 with the argmax restricted to those classes). Use it to separate "the class was never seen" from "seen but misclassified under domain shift".
