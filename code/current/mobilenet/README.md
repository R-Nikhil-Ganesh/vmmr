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
│   └── output_mobilenet_v2/              # Model artifacts (.pt, .onnx), metrics, plots, logs
│
├── external_dataset/                     # External Merged Dataset Pipeline (Model B)
│   ├── train/                            # Training subsystem
│   │   ├── train_mobilenet_v2_external.py    # PyTorch training pipeline (35 Makes, 52,174 images)
│   │   ├── train_mobilenet_v2_external.ipynb # Interactive training notebook
│   │   └── run_train_external.sh             # Headless training launcher
│   ├── analysis/                         # Evaluation & explainability subsystem
│   │   ├── output_analysis_external.py       # Diagnostic evaluation adapter
│   │   └── output_analysis_external.ipynb    # Full diagnostic evaluation notebook
│   └── output_mobilenet_v2_external/     # Model artifacts (.pt, .onnx), metrics, plots, logs
│
├── output_analysis.py                    # Universal PyTorch & ONNX diagnostic & explainability engine
├── evaluate_mixed.py                     # Cross-domain & mixed benchmark evaluation engine
├── run_all_pipeline.sh                   # Master chained runner for both models + eval
├── run_eval_mixed.sh                     # Mixed benchmark evaluation launcher
├── run_1.md                              # Comprehensive benchmark report for Run 1
└── README.md                             # Documentation & user guide
```

---

## 1. Quick Start: End-to-End Pipeline

Run the full pipeline (Model A training $\to$ Model B training $\to$ Mixed Benchmark evaluation):

```bash
cd vmmr/code/current/mobilenet

# Direct execution
bash run_all_pipeline.sh

# Or in a detached background tmux session
tmux new-session -d -s mobilenet_pipeline "bash run_all_pipeline.sh; exec bash"
tmux attach -t mobilenet_pipeline
```

---

## 2. PlatesMania Dataset Pipeline (`platesmania_dataset/`)

Trained on surveillance & natural street-level vehicle images (1,102,776 images across 35 makes):
- **Watermark Countermeasure**: On-the-fly cropping of the top 15% banner (`PLATESMANIA.COM`) only (`crop_top_pct=0.15`, `crop_bottom_pct=0.0`).
- **Head Strategy**: Single head with top 5 backbone layers unlocked.

```bash
# Run training directly
cd vmmr/code/current/mobilenet/platesmania_dataset/train
bash run_train.sh

# Or in background tmux
tmux new-session -d -s train_pm "bash run_train.sh; exec bash"

# Run diagnostic suite
cd ../analysis
python output_analysis_platesmania.py
```

---

## 3. External Merged Dataset Pipeline (`external_dataset/`)

Trained on 52,174 curated vehicle images spanning 35 automotive makes (BoxCars116k, Stanford Cars, CompCars CCTV/Web):
- **Watermark Countermeasure**: Zero top crop applied (`crop_top_pct=0.0`); bottom 5% border inset stripping dealership stamps (`crop_bottom_pct=0.05`).

```bash
# Run training directly
cd vmmr/code/current/mobilenet/external_dataset/train
bash run_train_external.sh

# Or in background tmux
tmux new-session -d -s train_ext "bash run_train_external.sh; exec bash"

# Run diagnostic suite
cd ../analysis
python output_analysis_external.py
```

---

## 4. Cross-Domain & Mixed Benchmark Evaluation (`evaluate_mixed.py`)

Compares **Model A** (PlatesMania) and **Model B** (External) across:
1. **In-Domain Test Sets**
2. **Out-of-Domain Cross Test Sets**
3. **Mixed Test Set** (Combined balanced benchmark across all 35 vehicle makes, 10,000 total samples)

```bash
cd vmmr/code/current/mobilenet

# Run one-click evaluation
bash run_eval_mixed.sh
```

Generates:
- `mixed_benchmark_results/mixed_benchmark_summary.csv`
- `mixed_benchmark_results/cross_domain_comparison.png`

Detailed metrics and analysis from the latest run are documented in [`run_1.md`](run_1.md).

---

## 5. Key Architectural Features

- **Framework**: PyTorch 2.14 (`pt-env`) with mixed precision (`torch.cuda.amp.autocast()`) and TF32 enabled for NVIDIA RTX 4090.
- **Backbone**: `torchvision.models.mobilenet_v2` (ImageNet pretrained, top 5 layers unfrozen).
- **Automated ONNX Export**: Automatically serializes the best checkpoint to `.onnx` whenever validation loss reaches a new minimum.
- **Diagnostic Engine**: Analytical Grad-CAM, 2D latent space projections (t-SNE & PCA), confusion matrices, and confidence calibration with rejection curves ($\tau = 0.70$).
