# MobileNetV2 Vehicle Recognition (PyTorch & ONNX)

This directory contains the mobile-first PyTorch training pipeline, diagnostic evaluation suite, and cross-domain mixed benchmark runner for fine-grained vehicle recognition using **MobileNetV2**.

---

## Directory Overview

```text
vmmr/code/current/mobilenet/
├── merged_dataset/                       # Primary Pipeline (Model C: PlatesMania + External merged)
│   ├── train/                            # Training subsystem
│   │   ├── train_mobilenet_v2_merged.py      # Joint PyTorch training pipeline (AMP, CosineAnnealing, ONNX export)
│   │   └── train_mobilenet_v2_merged.ipynb   # Interactive training notebook
│   ├── analysis/                         # Evaluation & explainability subsystem
│   │   ├── output_analysis_merged.py         # Per-domain diagnostic evaluation adapter (--domain platesmania|external)
│   │   └── output_analysis_merged.ipynb      # Interactive dual-domain diagnostic & Grad-CAM notebook
│   ├── splits_1235models/                # External dataset train/val/test CSVs & label_map.json
│   ├── prepare_external_1235models.py    # External split builder
│   └── output_merged_run4/               # Model artifacts (.onnx), per-domain diagnostics, benchmark, plots, logs
│
├── mixed_benchmark_results/              # Latest (Run 4) mixed benchmark summary CSVs & plot
├── backup_run3/                          # Archived Run 3 single-dataset pipelines (platesmania_dataset, external_dataset)
├── backup_run2/                          # Archived Run 2 35-make diagnostics
├── backup_run1/                          # Archived Run 1 35-make outputs
├── output_analysis.py                    # Universal PyTorch & ONNX diagnostic & explainability engine
├── evaluate_mixed.py                     # Cross-domain & mixed benchmark evaluation engine
├── bash_scripts/                         # Shell runners (tmux by default); see bash_scripts/commands.md
│   ├── run_all_pipeline.sh               # train Model C -> diagnostics + mixed benchmark
│   ├── run_train_merged.sh               # train Model C (PlatesMania + External)
│   ├── run_diagnostics.sh                # per-domain diagnostics + mixed benchmark
│   └── lib/                              # helpers: paths.sh, tmux_wrap.sh, auto_git_sync.sh
├── paths.py                              # Python side of the single path config (bash_scripts/lib/paths.sh)
├── train_common.py                       # Shared datasets, model builder, ONNX export, GPU augmentation, objective & EMA
├── run_docs/                             # Benchmark reports (run_1.md, run_2.md, run_3.md)
└── README.md                             # Documentation & user guide
```

---

## 1. Quick Start

All commands, arguments and examples are in [commands.md](bash_scripts/commands.md). In short:

```bash
cd vmmr/code/current/mobilenet
bash bash_scripts/run_all_pipeline.sh     # train Model C, then diagnostics + mixed benchmark (tmux: mobilenet_pipeline)
bash bash_scripts/run_train_merged.sh     # train Model C only (tmux: train_merged)
bash bash_scripts/run_diagnostics.sh      # per-domain diagnostics + mixed benchmark (tmux: diagnostics)
```

---

## 2. Merged Dataset Pipeline (`merged_dataset/`, Model C)

Trains a single unified **MobileNetV2** classifier (`nn.Linear(1280, 1235)`) jointly across both datasets in the shared **1,235 Make/Model taxonomy** spanning 35 automotive makes:
- **PlatesMania Domain**: 1,093,501 surveillance & street-level vehicle images (`PM_MANIFEST_CSV`, `PM_IMG_DIR`) with on-the-fly top 15% banner cropping (`crop_top_pct=0.15`, `crop_bottom_pct=0.0`).
- **External Domain**: 41,880 curated images across 268 active models and 32 makes (BoxCars116k, Stanford Cars, CompCars CCTV) in `merged_dataset/splits_1235models/` (`train.csv`: 29,554, `val.csv`: 6,163, `test.csv`: 6,163) with on-the-fly bounding-box cropping (or 5% bottom crop when no bbox is present).
- **Domain-Balanced Sampling**: Uses `WeightedRandomSampler` (`--ext-fraction 0.2`) so External images constitute 20% of each epoch (~6× oversampling) while keeping epoch length equal to the PlatesMania training split.
- **Checkpoint Selection**: Best mean validation loss across PlatesMania and External validation splits (`(val_loss_pm + val_loss_ext) / 2`).

---

## 3. Per-Domain Diagnostics & Mixed Benchmark (`output_analysis.py` & `evaluate_mixed.py`)

Evaluates **Model C** on:
1. **Fine-Grained Model Level**: Top-1 / Top-5 Accuracy & Macro/Weighted F1 across 1,235 vehicle model classes.
2. **Coarse Make Level**: Top-1 Accuracy & Macro/Weighted F1 across 35 automotive makes via probability marginalization:
   $$P(\text{Make}_k) = \sum_{m \in \text{Make}_k} P(\text{Model}_m)$$

Outputs under `merged_dataset/output_merged_run4/`:
- `diagnostics_platesmania/` — Full 109,340-image PlatesMania test split reports, confusion matrices, t-SNE/PCA, calibration.
- `diagnostics_external/` — Full 6,163-image External test split reports, confusion matrices, t-SNE/PCA, calibration.
- `benchmark/` (also synced to top-level `mixed_benchmark_results/`) — Balanced 5,000-image per-domain and 10,000-image `Mixed (Combined)` summary (`mixed_benchmark_summary.csv`, `shared_class_summary.csv`, `class_coverage.csv`, `cross_domain_comparison.png`).

---

## 4. Key Architectural Features

- **Framework**: PyTorch 2.14 (`pt-env`) with mixed precision (`torch.amp.autocast`) and TF32 enabled for NVIDIA RTX 4090.
- **Backbone**: `torchvision.models.mobilenet_v2` (ImageNet pretrained, top 5 layers unfrozen, frozen BatchNorm stats in locked blocks).
- **Automated ONNX Export**: Serializes the best checkpoint to `.onnx` with three outputs: `predictions` (logits), `embeddings` (1280-D, for t-SNE/PCA) and `class_maps` (per-class activation maps, for Grad-CAM).
- **Diagnostic Engine**: Multi-worker ONNX Runtime GPU inference with selective output fetching, analytical Grad-CAM, 2D latent space projections (t-SNE & PCA), confusion matrices, and confidence calibration with rejection curves ($\tau = 0.70$).

---

## 5. Moving to Another Server: `paths.sh`

All machine-specific paths live in [`bash_scripts/lib/paths.sh`](bash_scripts/lib/paths.sh) (read by every runner and, through [`paths.py`](paths.py), by the Python scripts). On a new machine create `bash_scripts/lib/paths.local.sh` (gitignored) with only what differs:

```bash
ECON_ROOT=/data/Econ                       # everything below is derived from it
PYTHON_BIN=/opt/envs/pt-env/bin/python     # optional
# PM_IMG_DIR, PM_MANIFEST_CSV, PM_LABEL_MAP, EXT_DATASETS_DIR, AUTO_GIT_PUSH can be set individually
```

Precedence: shell environment > `paths.local.sh` > defaults. Check the result with `python paths.py --check`.

---

## 6. Generalization Options (`train_common.py`)

| Flag | Effect |
|---|---|
| `--aug-strength strong` | wide zoom/aspect (tight vs loose framing), blur, noise, random erasing, stronger colour jitter |
| `--crop-jitter 0.5` | randomizes the train-split crop (PlatesMania top crop, External bbox margin / bottom crop); val/test unchanged |
| `--label-smoothing 0.1` | smoothed cross-entropy (validation loss stays plain CE, so checkpoint selection is comparable) |
| `--make-loss-weight 0.3` | auxiliary loss on P(make) = sum of P(model) over the make's models |
| `--ema-decay 0.999` | EMA of the weights; validation, best-checkpoint selection and ONNX export use the EMA weights |
