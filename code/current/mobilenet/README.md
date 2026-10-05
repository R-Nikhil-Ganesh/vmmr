# MobileNetV2 Vehicle Recognition (PyTorch & ONNX)

This directory contains the mobile-first PyTorch training pipelines, diagnostic evaluation suites, and cross-domain mixed benchmark runners for fine-grained vehicle recognition using **MobileNetV2**.

---

## Directory Overview

```text
vmmr/code/current/mobilenet/
├── platesmania_dataset/                  # PlatesMania Dataset Pipeline (Model A)
│   ├── train_mobilenet_v2.py             # Complete PyTorch training script (AMP, CosineAnnealing, ONNX export)
│   ├── train_mobilenet_v2.ipynb          # Interactive training notebook
│   ├── run_train.sh                      # Headless execution launcher script
│   ├── start_training_tmux.sh            # Detached tmux session launcher
│   ├── output_analysis_platesmania.py    # Diagnostic evaluation adapter
│   ├── output_analysis_platesmania.ipynb # Full diagnostic evaluation notebook
│   └── output_mobilenet_v2/              # Saved models (.pt, .onnx), reports, plots, logs
│
├── external_dataset/                     # External Merged Dataset Pipeline (Model B)
│   ├── train_mobilenet_v2_external.py    # PyTorch training pipeline (35 Makes, 52,174 images)
│   ├── train_mobilenet_v2_external.ipynb # Interactive training notebook
│   ├── run_train_external.sh             # Headless execution runner
│   ├── start_training_tmux.sh            # Detached tmux session launcher
│   ├── output_analysis_external.py       # Diagnostic evaluation adapter
│   ├── output_analysis_external.ipynb    # Full diagnostic evaluation notebook
│   └── output_mobilenet_v2_external/     # Saved models (.pt, .onnx), reports, plots, logs
│
├── output_analysis.py                    # Universal PyTorch & ONNX diagnostic & explainability engine
├── evaluate_mixed.py                     # Cross-domain & mixed benchmark evaluation engine
├── start_both_training_tmux.sh          # Master chained runner for both models + eval
├── run_eval_mixed.sh                     # Mixed benchmark one-click launcher
└── README.md                             # Documentation & user guide
```

---

## 1. PlatesMania Dataset Pipeline (`platesmania_dataset/`)

Trained on surveillance & natural street-level vehicle images:
- **Watermark Countermeasure**: On-the-fly cropping of the top 15% banner (`PLATESMANIA.COM`) only (no bottom crop).
- **Model / Make Support**: Fine-grained `Make/Model` classes with exact probability marginalization for make prediction.

```bash
cd vmmr/code/current/mobilenet/platesmania_dataset

# Option 1: Run in detached tmux session (recommended for long runs)
bash start_training_tmux.sh
# Attach anytime: tmux attach -t mobilenetv2_platesmania_train

# Option 2: Run directly
bash run_train.sh

# Option 3: Run output diagnostic suite
python output_analysis_platesmania.py
```

---

## 2. External Merged Dataset Pipeline (`external_dataset/`)

Trained on 52,174 curated vehicle images spanning 35 automotive makes (BoxCars116k, Stanford Cars, CompCars CCTV/Web):
- **Watermark Countermeasure**: Zero top crop applied (top crop is strictly unique to PlatesMania); bottom 5% border inset stripping dealership stamps.

```bash
cd vmmr/code/current/mobilenet/external_dataset

# Option 1: Run in detached tmux session
bash start_training_tmux.sh
# Attach anytime: tmux attach -t mobilenetv2_external_train

# Option 2: Run directly
bash run_train_external.sh

# Option 3: Run output diagnostic suite
python output_analysis_external.py
```

---

## 3. Cross-Domain & Mixed Benchmark Evaluation (`evaluate_mixed.py`)

Compares **Model A** (PlatesMania) and **Model B** (External) across:
1. **In-Domain Test Sets**
2. **Out-of-Domain Cross Test Sets**
3. **Mixed Test Set** (Combined balanced benchmark across all 35 vehicle makes)

```bash
cd vmmr/code/current/mobilenet

# Run one-click evaluation
bash run_eval_mixed.sh
```

Generates:
- `mixed_benchmark_results/mixed_benchmark_summary.csv`
- `mixed_benchmark_results/cross_domain_comparison.png`

---

## 4. Key Architectural Features

- **Framework**: PyTorch 2.14 (`pt-env`) with mixed precision (`torch.cuda.amp.autocast()`) and TF32 enabled for NVIDIA RTX 4090.
- **Backbone**: `torchvision.models.mobilenet_v2` (ImageNet pretrained).
- **Automated ONNX Export**: Automatically serializes the best checkpoint to `.onnx` whenever validation loss reaches a new minimum.
- **Diagnostic Engine**: Analytical Grad-CAM, 2D latent space projections (t-SNE & PCA), confusion matrices, and confidence calibration with rejection curves ($\tau = 0.70$).
