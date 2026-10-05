# Vehicle Make Classification - Production & Diagnostic Codebase

A clean, modular repository categorizing deep learning models, training pipelines, explainability frameworks, and zero-dependency inference engines for fine-grained vehicle make recognition.

> [!NOTE]
> For a comprehensive synthesis of experimental findings, multi-resolution scaling benchmarks (224×224 to 720×720), shortcut learning audits, and architectural conclusions across all models, see [efficientnet_conclusions.md](file:///home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/efficientnet_conclusions.md).

---

## Directory Overview

```text
code/current/
├── platesmania_dataset/                  # PlatesMania Dataset Pipeline & Artifacts (35 Makes)
│   ├── train_efficientnet_b0.py          # Complete training script (batch size 16 default + OnnxCheckpointCallback)
│   ├── train_efficientnet_b0.ipynb       # Interactive training notebook
│   ├── run_train.sh                      # Headless execution launcher script
│   ├── start_training_tmux.sh            # Detached tmux session launcher (tmux attach -t platesmania_train)
│   ├── output_analysis_platesmania.py    # ONNX diagnostic evaluation adapter
│   ├── output_analysis_platesmania.ipynb # Full diagnostic evaluation notebook
│   └── output_efficientnet_b0/           # Saved models (.onnx), reports, plots, and logs
│
├── external_dataset/                     # External Merged Dataset Pipeline & Artifacts (35 Makes)
│   ├── train_efficientnet_b0_merged.py   # Merged dataset training pipeline (52,174 peak-frame images)
│   ├── train_efficientnet_b0_merged.ipynb# Interactive training notebook
│   ├── run_train_merged.sh               # Headless training execution runner
│   ├── external_dataset_analysis.ipynb   # Dataset inspection & alignment notebook
│   ├── output_analysis_external.py       # ONNX diagnostic evaluation adapter
│   ├── output_analysis_external.ipynb    # Full diagnostic evaluation notebook
│   └── output_efficientnet_b0_merged/    # Saved models (.onnx), reports, plots, and logs
│
├── sample_inference/                     # Self-Sufficient Standalone Inference Testing Suite
│   ├── models/                           # Best weights (External & PlatesMania ONNX + label maps)
│   ├── images/                           # Curated batch of 15 sample test images + manifest.csv
│   ├── output/                           # Inference outputs, prediction CSVs/JSONs, and annotated HUD images
│   ├── infer.py                          # Zero-dependency standalone inference CLI & Python API
│   ├── test_all_samples.sh               # One-click test runner across both models
│   └── README.md                         # Detailed sample inference documentation
│
├── efficientnet_conclusions.md           # Comprehensive empirical conclusions, benchmarks & architectural report
├── output_analysis.py                    # Universal 100% ONNX diagnostic & explainability engine
├── infer_merged.py                       # Standalone production ONNX inference engine
├── export_onnx.py                        # Standalone ONNX conversion and validation utility
└── README.md                             # Repository documentation
```

---

## 1. Production Model Inference (`infer_merged.py`)

A production-grade, ultra-low-latency inference engine powered by **ONNX Runtime** (executing in ~15 ms per image on CPU with zero GPU memory overhead and zero TensorFlow dependency):

```bash
# Activate virtual environment
source /home/researchadmin/Econ/repo-clone/stanford-cars-model/.venv/bin/activate

# Single image inference
python infer_merged.py --image path/to/vehicle.jpg

# Batch inference on directory with CSV export
python infer_merged.py --image-dir path/to/images/ --save-csv predictions.csv

# Batch inference on CSV manifest with JSON export
python infer_merged.py --csv /path/to/manifest.csv --image-col image_path --save-json results.json

# Generate visual annotated banner (HUD prediction overlay)
python infer_merged.py --image path/to/vehicle.jpg --save-annotated annotated_vehicle.jpg
```

**Python API:**
```python
from infer_merged import MergedVehicleClassifier

clf = MergedVehicleClassifier()
res = clf.predict_image("path/to/vehicle.jpg", top_k=5)
print(f"Top-1: {res['top1_make']} ({res['top1_confidence']:.2%})")
```

---

## 2. Universal ONNX Diagnostic & Explainability Engine (`output_analysis.py`)

Evaluates any trained ONNX model across all key dimensions:
- **Test set evaluation**: Overall micro accuracy, Macro F1, Weighted F1, Cross-Entropy Loss
- **Per-class metrics**: Classification report & color-coded F1 bar ranking across all 35 makes
- **Confusion matrix**: Normalized heatmap & top confusion pairs (e.g., GMC ↔ Chevrolet, Dodge ↔ Chrysler)
- **Hardest misclassifications**: High-confidence failure analysis with visual panel grids
- **Analytical Grad-CAM**: Sub-millisecond class activation heatmaps ($L^c = \text{ReLU}\left( \sum_k w_k^c A^k \right)$)
- **Latent space visualization**: 2D t-SNE and PCA manifold projections of convolutional embeddings
- **Calibration analysis**: Confidence histograms & accuracy vs. coverage rejection curves ($\tau = 0.70$)

```bash
# Evaluate External Merged model (auto-detected default)
python output_analysis.py --dataset external --run-all

# Evaluate PlatesMania model (35 makes)
python output_analysis.py --dataset platesmania --run-all

# Custom evaluation with explicit paths
python output_analysis.py \
  --model-path platesmania_dataset/output_efficientnet_b0/models/efficientnet_b0_best.onnx \
  --output-dir platesmania_dataset/output_efficientnet_b0 \
  --splits-dir /home/researchadmin/Econ/resized_640x640/splits_filtered \
  --run-all
```

---

## 3. Training & Retraining Pipelines

Both dataset pipelines support automated training with **per-epoch ONNX export** (`OnnxCheckpointCallback`), saving `efficientnet_b0_best.onnx` whenever validation loss reaches a new minimum.

### A. PlatesMania Dataset Pipeline (`platesmania_dataset/`)
Trained on 1,102,776 real-world surveillance & street images across 35 vehicle makes (including newly added GMC, Infiniti, and Subaru):

```bash
cd platesmania_dataset

# Option 1: Launch in a persistent detached tmux session (recommended for long runs)
bash start_training_tmux.sh

# Attach to inspect live progress anytime:
tmux attach -t platesmania_train
# (To detach without stopping training: press Ctrl+B, then d)

# Option 2: Run directly in terminal (batch size 16 default)
bash run_train.sh

# Option 3: Invoke Python script directly with custom arguments
python train_efficientnet_b0.py \
  --splits-dir /home/researchadmin/Econ/resized_640x640/splits_filtered \
  --output-dir output_efficientnet_b0 \
  --img-size 640 \
  --batch-size 16 \
  --epochs 15 \
  --lr 1e-4 \
  --unfreeze-layers 5 \
  --dropout 0.4
```

### B. External Merged Dataset Pipeline (`external_dataset/`)
Trained on 52,174 curated vehicle images across 35 vehicle makes:

```bash
cd external_dataset

# Run headless training
bash run_train_merged.sh

# Or invoke directly with custom parameters
python train_efficientnet_b0_merged.py \
  --splits-dir /home/researchadmin/Econ/external_datasets/merged_data \
  --output-dir output_efficientnet_b0_merged \
  --img-size 640 \
  --batch-size 8 \
  --epochs 15 \
  --lr 1e-4 \
  --unfreeze-layers 5 \
  --dropout 0.4
```

---

## 4. Self-Sufficient Sample Inference Testing (`sample_inference/`)

A self-contained testing package containing model weights, 15 sample vehicle images across diverse brands, and an inference runner requiring zero external dependencies:

```bash
# Run one-click test on all sample images
bash sample_inference/test_all_samples.sh

# Or run interactively using the standalone Python CLI
cd sample_inference
python infer.py --model external                      # Test External model
python infer.py --model platesmania                   # Test PlatesMania model
python infer.py --image images/11_porsche.jpg         # Test single image
```

---

## 5. Architectural & Diagnostic Summary

For deep technical insights, review [efficientnet_conclusions.md](file:///home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/efficientnet_conclusions.md), which documents:
- **Multi-Resolution Empirical Frontier**: +7.99% micro-accuracy improvement from 224×224 to 640×640; capacity saturation at 720×720.
- **Watermark Shortcut Vulnerability**: Discovery that web-scraped crops exhibit border shortcut learning (e.g., Nissan 10% flip rate under border occlusion) and proven edge-masking mitigations.
- **Sister Brand Confusion**: Quantitative analysis of platform-sharing misclassifications (GMC ↔ Chevrolet, Dodge ↔ Chrysler).
- **Human-in-the-Loop Confidence Rejection**: Establishing $\tau = 0.70$ threshold filtering >76% of errors with 92%+ automated coverage.