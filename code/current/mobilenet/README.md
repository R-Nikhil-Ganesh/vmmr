# MobileNetV2 Vehicle Recognition (PyTorch & ONNX)

This directory contains the mobile-first PyTorch training pipeline, diagnostic evaluation suite, and cross-domain mixed benchmark runner for fine-grained vehicle recognition using **MobileNetV2**.

---

## Directory Overview

```text
vmmr/code/current/mobilenet/
├── run4/                                 # Primary Pipeline (Model C: PlatesMania + External merged)
│   ├── train/                            # Training subsystem
│   │   ├── train_mobilenet_v2_merged.py      # Joint PyTorch training pipeline (AMP, CosineAnnealing, ONNX export)
│   │   └── train_mobilenet_v2_merged.ipynb   # Interactive training notebook
│   ├── analysis/                         # Evaluation & explainability subsystem
│   │   ├── output_analysis_merged.py         # Per-domain diagnostic evaluation adapter (--domain platesmania|external)
│   │   └── output_analysis_merged.ipynb      # Interactive dual-domain diagnostic & Grad-CAM notebook
│   ├── splits_1235models/                # External dataset train/val/test CSVs & label_map.json
│   ├── prepare_external_1235models.py    # External split builder (51 makes + >=50 sample pruning)
│   └── output_merged_run4/               # Run 4 Model C artifacts (.onnx), per-domain diagnostics, benchmark, plots
│
├── annotation/                           # Multi-stage make/model + Qwen3-VL-8B color annotation & HTML viewer
│   ├── annotate.py                           # Stage 1-4 coordinator (ONNX, UMAP, Qwen3-VL-8B, viewer)
│   ├── color_heads.py                        # Qwen3-VL-8B 4-bit NF4 / SDPA color classifier
│   ├── build_viewer.py                       # Standalone Plotly WebGL HTML viewer builder
│   └── common.py                             # Shared crop and label-map helpers
├── annotation_runs/                      # Annotation outputs (run1 & run2 tracked via run1.dvc, run2.dvc)
├── color_distillation/                   # Distills Qwen3-VL colors into MobileNetV2 features[16] tap
│   ├── train_color_head.py                   # Feature cache extractor, MLP/Linear trainer, ONNX graph splicer
│   ├── layer_probe.py                        # 8-layer empirical linear/MLP color probe
│   ├── verify_multitask_onnx.py              # Signature, latency, and numerical equivalence verifier
│   ├── cache.dvc                             # DVC pointer for L16 feature cache (304 MB)
│   ├── output_mlp512x2/                      # [REFERENCE] Deep MLP-512x2 head & mobilenet_v2_multitask.onnx (.dvc, 75.5% test / 77.4% human GT)
│   ├── output_unweighted_mlp128/             # [ABLATION 1] Unweighted 128-unit MLP (73.2% test / 74.9% human GT)
│   └── output/                               # [BASELINE] Historical 128-unit weighted MLP head & ONNX (.dvc, 68.0% val)
│
├── backup_run3/                          # Archived Run 3 single-dataset pipelines (platesmania_dataset, external_dataset, mixed_benchmark_results_ab)
├── backup_run2/                          # Archived Run 2 35-make diagnostics
├── backup_run1/                          # Archived Run 1 35-make outputs
├── output_analysis.py                    # Universal PyTorch & ONNX diagnostic & explainability engine
├── evaluate_mixed.py                     # Cross-domain & mixed benchmark evaluation engine
├── bash_scripts/                         # Shell runners (tmux by default); see bash_scripts/commands.md
│   ├── run_all_pipeline.sh               # train Model C -> diagnostics + mixed benchmark
│   ├── run_train_merged.sh               # train Model C (PlatesMania + External)
│   ├── run_annotate.sh                   # annotate make/model/color + HTML viewer
│   ├── run_train_color_distillation.sh   # distill Qwen color labels into mobilenet_v2_multitask.onnx
│   ├── run_diagnostics.sh                # per-domain diagnostics + mixed benchmark
│   ├── dvc.sh                            # DVC CLI wrapper with Google Drive remote support
│   └── lib/                              # helpers: paths.sh, tmux_wrap.sh, auto_git_sync.sh
├── paths.py                              # Python side of the single path config (bash_scripts/lib/paths.sh)
├── train_common.py                       # Shared datasets, model builder, ONNX export, GPU augmentation, objective & EMA
├── error_docs/                           # Audit logs, error trackers, verification studies, and resolution plans
├── run_docs/                             # Benchmark & architecture reports (run_1..4, run_5_implementation_plan, color distillation, DVC)
└── README.md                             # Documentation & user guide
```

---

## 1. Quick Start

All commands, arguments and examples are in [commands.md](bash_scripts/commands.md). In short:

```bash
cd vmmr/code/current/mobilenet
bash bash_scripts/run_all_pipeline.sh              # train Model C, then diagnostics + mixed benchmark (tmux: mobilenet_pipeline)
bash bash_scripts/run_train_merged.sh              # train Model C only (tmux: train_merged)
bash bash_scripts/run_diagnostics.sh               # per-domain diagnostics + mixed benchmark (tmux: diagnostics)
bash bash_scripts/run_annotate.sh                  # annotate images (make/model + Qwen color) & viewer.html (tmux: annotate)
bash bash_scripts/run_train_color_distillation.sh  # distill color head into mobilenet_v2_multitask.onnx (tmux: color_distill)
./bash_scripts/dvc.sh status                       # check DVC-tracked datasets, caches, and ONNX models
```

---

## 2. Merged Dataset Pipeline (`run4/`, Model C)

Trains a single unified **MobileNetV2** classifier jointly across both domains:
- **PlatesMania Domain**: 1,093,501 manifest rows (`1,093,408` readable images across `1,235` classes and `35` makes; `109,331 / 109,340` readable in `test`) from `PM_MANIFEST_CSV` and `PM_IMG_DIR`, with on-the-fly top 15% banner cropping (`crop_top_pct=0.15`, `crop_bottom_pct=0.0`).
- **External Domain (`run4/splits_1235models/`)**: Pre-letterboxed 512×512 images from `external_resized_512` (BoxCars116k, Stanford Cars, CompCars CCTV) with on-the-fly bottom 5% watermark cropping (`crop_bottom_pct=0.05`; `bbox_* = -1` on all rows since vehicles were pre-cropped during 512×512 dataset construction):
  - **Run 4 Baseline (`output_merged_run4`, 35-Make / 1,235-Class Taxonomy)**: **33,329** images across **286** active classes (`train.csv`: 23,589, `val.csv`: 4,870, `test.csv`: 4,870), sharing the exact 1,235-class space (`0..1234`) with Models A and B.
  - **Expanded 51-Make Splits (Current `splits_1235models/`, 49-Make / 1,416-Class Taxonomy)**: **67,695** images across **471** active External classes (`train.csv`: 47,805, `val.csv`: 9,945, `test.csv`: 9,945) — retaining 290 overlapping PlatesMania classes (`0..1234`) plus 181 External-only classes with $\ge 50$ samples (`1235..1415`), while pruning 159 small classes ($< 50$ samples).
- **Domain-Balanced Sampling**: Uses `WeightedRandomSampler` (`--ext-fraction 0.2`) so External images constitute 20% of each epoch while keeping epoch length equal to the PlatesMania training split.
- **Checkpoint Selection**: Best mean validation loss across PlatesMania and External validation splits (`(val_loss_pm + val_loss_ext) / 2`).

---

## 3. Per-Domain Diagnostics & Mixed Benchmark (`output_analysis.py` & `evaluate_mixed.py`)

Evaluates **Model C** (`output_merged_run4`, 1,235 classes) on:
1. **Fine-Grained Model Level**: Top-1 / Top-5 Accuracy & Macro/Weighted F1 across 1,235 vehicle model classes.
2. **Coarse Make Level**: Top-1 Accuracy & Macro/Weighted F1 across 35 automotive makes via probability marginalization:
   $$P(\text{Make}_k) = \sum_{m \in \text{Make}_k} P(\text{Model}_m)$$

Outputs under `run4/output_merged_run4/`:
- `diagnostics_platesmania/` — Full 109,331-image readable PlatesMania test split reports, confusion matrices, t-SNE/PCA, calibration.
- `diagnostics_external/` — Full 4,870-image Run 4 External test split reports, confusion matrices, t-SNE/PCA, calibration.
- `benchmark/` — Balanced 5,000-image PlatesMania + 4,870-image External (`9,870` combined) summary (`mixed_benchmark_summary.csv`, `shared_class_summary.csv`, `class_coverage.csv`, `cross_domain_comparison.png`).

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
| `--crop-jitter 0.5` | randomizes the train-split crop (PlatesMania 15% top crop, External 5% bottom crop; or bbox margin when `bbox_* >= 0`); val/test unchanged |
| `--label-smoothing 0.1` | smoothed cross-entropy (validation loss stays plain CE, so checkpoint selection is comparable) |
| `--make-loss-weight 0.3` | auxiliary loss on P(make) = sum of P(model) over the make's models |
| `--ema-decay 0.999` | EMA of the weights; validation, best-checkpoint selection and ONNX export use the EMA weights |
