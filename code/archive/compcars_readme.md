# Vehicle Make and Model Classification (Vehicle-Make-v1)

Fine-grained vehicle recognition on the **CompCars (Comprehensive Cars)** Showroom dataset using **EfficientNet-B0** transfer learning. This repository provides training pipelines, high-throughput bounding-box data generators, extensive evaluation toolkits, and an empirical multi-resolution benchmark scaling from 224×224 to 720×720 input resolutions.

---

## 1. Directory Structure

```text
code/
├── current_model/
│   ├── compcars_showroom_finetuned.ipynb  # End-to-end training notebook (224x224 baseline)
│   ├── functions.py                       # Modular evaluation, metric computation & visualization library
│   └── result_analysis_finetuned.ipynb    # In-depth statistical analysis & profiling notebook
├── resolution_comparison/
│   ├── train_resolutions.py               # Multi-resolution retraining & evaluation CLI pipeline
│   ├── summary_retrained_resolutions.csv  # Aggregated metrics table across all 7 resolutions
│   ├── retrained_resolution_comparison.png# Accuracy curves and training duration comparison plot
│   ├── models/                            # Saved .keras and .weights.h5 models per resolution
│   ├── histories/                         # Epoch-level training histories (.csv)
│   ├── evaluations/                       # Class-wise accuracy CSVs and summary JSONs
│   └── logs/                              # Full stdout/stderr execution logs per resolution
└── readme.md                              # Repository documentation
```

---

## 2. Dataset & Benchmark Specification

- **Dataset**: CompCars Web-Nature (Showroom) subset.
- **Classes**: 431 fine-grained vehicle model classes spanning 75 distinct vehicle makes (brands).
- **Data Splits**:
  - Official classification split: `train.txt` and `test.txt`.
  - Training split: Stratified 80/20 train/validation partition.
  - Test set: **14,939** images evaluated using ground-truth bounding box annotations (`[bbox_x1, bbox_y1, bbox_x2, bbox_y2]`).
- **Data Preprocessing**: Images are cropped to the annotated vehicle bounding box and resized using bilinear interpolation before normalization via [`preprocess_input`](file:///home/cloudyrelic/e-con/Vehicle-Make-v1/code/current_model/functions.py#L12). Data augmentation during training includes random rotation (±15°), width/height shifts (±10%), zoom (±15%), horizontal flips, and brightness variation (0.8–1.2×).

---

## 3. Architecture & Training Setup

- **Backbone**: [`EfficientNetB0`](file:///home/cloudyrelic/e-con/Vehicle-Make-v1/code/resolution_comparison/train_resolutions.py#L203-L207) pre-trained on ImageNet-1k (`include_top=False`).
- **Fine-Tuning Strategy**: Feature extractor base is frozen except for the **top 5 layers**.
- **Classification Head**:
  - `GlobalAveragePooling2D()`
  - `Dropout(rate=0.5)`
  - `Dense(431, activation="softmax", dtype="float32")`
- **Optimization**:
  - Optimizer: `Adam(learning_rate=1e-3)`
  - Loss Function: `categorical_crossentropy`
  - Callbacks:
    - `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`
    - `ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=1e-6)`
- **Data Pipeline**: Custom [`FastBBoxDataGenerator`](file:///home/cloudyrelic/e-con/Vehicle-Make-v1/code/resolution_comparison/train_resolutions.py#L157-L198) and [`BBoxDataGenerator`](file:///home/cloudyrelic/e-con/Vehicle-Make-v1/code/current_model/functions.py#L79-L109) utilizing a `ThreadPoolExecutor` (16 concurrent worker threads) for parallel disk I/O, on-the-fly bbox cropping, and resizing.

---

## 4. Multi-Resolution Empirical Benchmark

All 7 models were retrained independently from ImageNet weights using the identical training protocol on the CompCars showroom benchmark (evaluated on all 14,939 test samples across 431 classes on an NVIDIA RTX 4090):

| Resolution | Batch Size | Epochs Trained | Best Val Acc (%) | Test Micro Acc (%) | Test Macro Mean (%) | Test Macro Median (%) | Total Correct / 14,939 | Training Time (min) | Eval Time (sec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **224×224** | 32 | 20 | 81.84% | 84.23% | 82.44% | 85.00% | 12,583 | 14.54 | 47.56 |
| **256×256** | 32 | 20 | 84.36% | 86.22% | 84.78% | 86.96% | 12,881 | 16.47 | 48.82 |
| **384×384** | 32 | 20 | 87.73% | 89.30% | 88.00% | 90.62% | 13,340 | 32.91 | 47.46 |
| **512×512** | 16 | 20 | 90.26% | 90.92% | 89.89% | 92.31% | 13,582 | 53.76 | 65.90 |
| **576×576** | 16 | 20 | 87.92% | 89.76% | 88.31% | 92.31% | 13,409 | 41.43 | 46.60 |
| **640×640** | 8 | 20 | **91.01%** | **92.22%** | **90.89%** | **93.33%** | **13,777** | 58.18 | 63.85 |
| **720×720** | 8 | 20 | 90.48% | 92.05% | 90.71% | 92.86% | 13,751 | 67.66 | 66.50 |

### Key Benchmark Findings
1. **Optimal Input Resolution**: **640×640** delivers the highest overall classification accuracy:
   - **+7.99%** increase in Micro Accuracy over 224×224 baseline (92.22% vs. 84.23%).
   - **+8.45%** increase in Macro Mean Accuracy (90.89% vs. 82.44%).
   - Resolves fine-grained vehicle distinctions (grille textures, headlamp silhouettes, trim badging) that are lost at lower resolutions.
2. **Diminishing Returns**: Scaling to 720×720 shows a minor drop in accuracy (92.05% vs. 92.22%) while increasing training duration by 16.3% (67.66 min vs. 58.18 min).
3. **Efficiency Sweet Spot**: **512×512** reaches >90% Micro Accuracy (90.92%) with a batch size of 16, offering an attractive trade-off between throughput and accuracy for production deployment.

Refer to [summary_retrained_resolutions.csv](file:///home/cloudyrelic/e-con/Vehicle-Make-v1/code/resolution_comparison/summary_retrained_resolutions.csv) and [retrained_resolution_comparison.png](file:///home/cloudyrelic/e-con/Vehicle-Make-v1/code/resolution_comparison/retrained_resolution_comparison.png) for full metrics and plots.

---

## 5. Modules & Component Details

### `current_model/`
- **`compcars_showroom_finetuned.ipynb`**: Handles model training and fine-tuning on the CompCars showroom dataset with bounding-box cropping and data augmentations.
- **`result_analysis_finetuned.ipynb`**: Performs extensive performance and accuracy analysis on trained models (including class-wise accuracy, performance tiers, brand-level rankings, and CCTV cross-domain overlap) using functions imported from `functions.py`.
- **`functions.py`**: Modular library providing the helper functions for dataset loading, multi-threaded bounding-box data generation, metric calculations, and visualization routines.

### `resolution_comparison/`
- **`train_resolutions.py`**: CLI pipeline to retrain and benchmark models independently across multiple input resolutions (224×224 to 720×720), logging performance metrics and saving model weights.

---

## 6. Execution & Usage Guide

### Environment Setup
Activate the virtual environment and install dependencies:
```bash
source tf-env/bin/activate
pip install -r requirements.txt
```

### Running the Multi-Resolution Training Pipeline
Run `train_resolutions.py` from the command line:

```bash
# Sequential training across default resolutions (224, 256, 384, 512, 576, 640, 720)
python resolution_comparison/train_resolutions.py \
  --data_dir /path/to/dataset \
  --output_dir resolution_comparison \
  --epochs 20 \
  --workers 16

# Retrain specific resolutions with custom settings
python resolution_comparison/train_resolutions.py \
  --resolutions 512,640 \
  --epochs 25 \
  --reduce_lr_patience 2 \
  --workers 16
```

#### CLI Arguments:
- `--resolutions`: Comma-separated list of target resolutions (e.g., `224,384,640`).
- `--epochs`: Maximum number of epochs per resolution (default: `20`).
- `--data_dir`: Root path to the CompCars dataset directory.
- `--output_dir`: Output directory for storing models, histories, and evaluation summaries.
- `--workers`: Number of CPU data-preloading worker threads per training process (default: `16`). Handles on-the-fly bounding box cropping, resizing, and augmentation.
- `--parallel`: Number of resolution models to train concurrently (default: `1`).
- `--reduce_lr_patience`: Epoch patience before reducing learning rate on plateau (default: `1`).

### Running Notebook Analysis
To inspect and evaluate models interactively:
1. Open `result_analysis_finetuned.ipynb` in Jupyter or your IDE.
2. Select the `tf-env` Python kernel.
3. Set `model_path` to point to any desired checkpoint (e.g., `resolution_comparison/models/showroom_bbox_b0_640x640.keras`).
4. Run cells sequentially to generate class-level, brand-level, and CCTV overlap reports.

