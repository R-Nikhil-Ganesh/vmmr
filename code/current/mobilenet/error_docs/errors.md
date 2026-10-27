# errors.md: Known Issues, Pipeline Defects & Final Resolutions

**Date of Final Resolution**: October 27, 2026  
**Implementation Plan**: [`code/current/mobilenet/run_docs/errors_master_implementation_plan.md`](run_docs/errors_master_implementation_plan.md)  
**Checkpoint**: [`code/current/mobilenet/checkpoint.md`](checkpoint.md)

---

## Executive Status Summary

| ID | Priority | Description | Final Status | Key Artifact / Empirical Metric |
| :---: | :---: | :--- | :---: | :--- |
| **`E-01`** | **P0** | External vehicle bounding-box crops ignored | **`[RESOLVED]`** | `100%` pre-cropped for PlatesMania & 4 External sources; `18,520` `BoxCars116k` `2DBB` boxes recovered in `boxcars116k_bboxes_512.json` and `splits_1235models/*.csv` |
| **`E-02`** | **P0** | Color prediction crop sensitivity | **`[RESOLVED]`** | `71.33%` agreement on 300-sample probe; removing 30px road border flips color on 33.3% of `BoxCars116k` images (`crop_agreement_study.md`) |
| **`E-03`** | **P0** | UMAP 2D coordinates NaN for 97% of rows | **`[RESOLVED]`** | Schema and `--coords-max 30000` landmark subsampling documented in `run_docs/annotation_run_4_and_5.md` & `annotation/README.md` |
| **`E-04`** | **P1** | Hardcoded absolute paths in CSV manifests | **`[RESOLVED]`** | `to_rel_image_path()` and `resolve_image_path()` in `common.py`; regenerated `splits_1235models/*.csv` with portable relative paths |
| **`E-05`** | **P1** | Color feature extraction resize discrepancy | **`[RESOLVED]`** | `verify_preprocessing_e05.py` showed 2.96% L2 shift; unified `train_color_head.py` with `DiagnosticDataset` antialiased bilinear interpolation |
| **`E-06`** | **P0** | Bit-for-bit make/model numerical identity | **`[RESOLVED]`** | `max |diff| = 0.000e+00` verified on 256 test images in `color_distillation/verify_multitask_onnx.py` |
| **`E-07`** | **P1** | Color head evaluated on favorable val subset | **`[RESOLVED]`** | Unbiased held-out test evaluation: `110,217` high-conf images (`75.53%` Top-1, `95.81%` Top-3) and `114,201` total test images (`74.31%` Top-1) |
| **`E-08`** | **P0** | Color head underfitted & crippled by class weights | **`[RESOLVED]`** | Unweighted Deep MLP-512×2 boosted accuracy by **`+7.85 pp`** (`68.00% -> 75.85%`), Top-3 to `96.17%`, and macro F1 by **`+8.30 pp`** (`55.66% -> 63.96%`) |
| **`E-09`** | **P0** | Qwen color labels unvalidated against ground truth | **`[RESOLVED]`** | Evaluated on `19,479` CompCars human labels (`78.44%` Qwen teacher, `77.43%` distilled MLP-512×2); generated `color_human_review_600.csv` |
| **`E-10`** | **P1** | Production camera domain shift (night / IR) | **`[DEFERRED]`** | Protocol documented in `run_docs/`; await arrival of production surveillance frames |
| **`E-11`** | **P1** | `annotation_run_4_and_5.md` documentation collisions | **`[RESOLVED]`** | Fixed color taxonomy (`beige`/`other`), clarified in-sample vs held-out test splits, documented UMAP landmark sampling |
| **`E-12`** | **P1** | DVC Google Drive guide & credentials hygiene | **`[RESOLVED]`** | Relative links updated; documented shared Research PC env-var credential protocol |
| **`E-13`** | **P1** | Stale dataset counts in `README.md` | **`[RESOLVED]`** | Separated 1,235-class Run 4 split from 1,416-class 51-make expansion; updated component layout |
| **`E-14`** | **P1** | Runner documentation in `commands.md` | **`[RESOLVED]`** | Added `run_annotate.sh` and `dvc.sh`; documented `MODEL_RUN` and `CACHE_FROM` conventions |
| **`E-15`** | **P1** | External crop description in `run_4.md` | **`[RESOLVED]`** | Corrected crop description (`bbox_* = -1`, 5% bottom crop) and runner invocations |
| **`E-16`** | **P2** | Local `CLAUDE.md` layout paths | **`[RESOLVED]`** | Updated Research PC `CLAUDE.md` to reflect `merged_dataset/`, `color_distillation/`, and 51-make expansion |
| **`E-17`** | **P0** | Missing execution provenance (`run_config.json`) | **`[RESOLVED]`** | Added `save_run_config()` in `train_common.py`; backfilled `run_config.json` in Runs 4, 5, and distillation dirs |
| **`E-18`** | **P0** | Missing like-for-like A/B/C cross-domain comparisons | **`[RESOLVED]`** | Evaluated Models A & B on regenerated External test split; Model C achieves `92.83%` vs Model B's `58.58%` and beats Model A by `+1.27 pp` on shared classes |
| **`E-19`** | **P1** | Large ONNX weights and results in Git | **`[RESOLVED]`** | `*.onnx` added to `.gitignore` to prevent accidental staging; DVC handles large binaries; commits kept < 100 MB |
| **`E-20`** | **P2** | Clean `.gitignore` structure & hygiene | **`[RESOLVED]`** | Deduplicated `/archive/`, categorized ignore rules, ensured `*plotly.min.js` and `__pycache__` remain ignored |
| **`E-21`** | **P2** | `MobileNetV2Evaluator` parameter naming | **`[RESOLVED]`** | Renamed `checkpoint_path` $\to$ `model_path` in `output_analysis.py` with backward-compatible keyword fallback |

---

## Detailed Section Breakdown

### A. Data and Pipeline Defects

#### E-01 (P0) External Bounding Boxes [RESOLVED]
- **Verification on Disk**: PlatesMania (`1,105,650` images) and 4 External sources (`stanford_cars` `6,495`, `compcars_web` `13,290`, `compcars_cctv` `33,324`, `resized_cupra` `1,590`) were verified to be `100%` already tightly cropped and letterboxed to `512×512` with white padding bars on disk.
- **Root Cause & Fix**: `boxcars116k` (`18,520` images) was the only source whose `inst["2DBB"]` coordinates in `BoxCars116k/dataset.pkl` were ignored (`"bbox": None`). All `18,520` bounding boxes were extracted from camera principal points, mapped to 512×512 coordinates, and saved in [`merged_dataset/boxcars116k_bboxes_512.json`](merged_dataset/boxcars116k_bboxes_512.json) and populated in `splits_1235models/{train,val,test,all_splits}.csv`.

#### E-02 (P0) Crop Sensitivity Experiment [RESOLVED]
- **Study Execution**: Created and ran [`annotation/verify_crop_agreement.py`](annotation/verify_crop_agreement.py) across 300 stratified samples (150 PlatesMania, 150 `BoxCars116k` with recovered `2DBB` boxes).
- **Findings**: Overall agreement was **`71.33%`** (PlatesMania pre-cropped images had `76.00%` agreement; `BoxCars116k` images had `66.67%` agreement, showing that removing the 30px surrounding asphalt border alters predicted color in `33.33%` of cases). Documented in [`run_docs/crop_agreement_study.md`](run_docs/crop_agreement_study.md).

#### E-03 (P0) UMAP Coordinates Sparsity [RESOLVED]
- **Documentation**: Clarified in [`run_docs/annotation_run_4_and_5.md`](run_docs/annotation_run_4_and_5.md) and [`annotation/README.md`](annotation/README.md) that `x`/`y` are intentionally populated only for `--coords-max 30000` landmark rows for interactive visualization in `viewer.html`.

#### E-04 (P1) Portable Relative Image Paths [RESOLVED]
- **Implementation**: Implemented `to_rel_image_path()` and `resolve_image_path()` in [`annotation/common.py`](annotation/common.py). Updated `prepare_external_1235models.py`, `output_analysis.py`, and `train_common.py`. Regenerated all splits in `splits_1235models/*.csv` using portable relative paths (`images/<Make>/<file>.jpg`).

#### E-05 (P1) Resize Preprocessing Convention [RESOLVED]
- **Measurement**: Created [`color_distillation/verify_preprocessing_e05.py`](color_distillation/verify_preprocessing_e05.py) and evaluated 2,000 validation images. Observed a `0.029558` (2.96% > 1e-2) relative $L_2$ shift between `PIL.Image.BILINEAR` and antialiased bilinear interpolation, flipping 1.90% of color predictions.
- **Fix**: Replaced PIL resize in `ImageExtractDataset` with `torchvision.io.read_image` + antialiased `F.interpolate` matching `DiagnosticDataset`.

#### E-06 (P0) Multi-Task Bit-for-Bit Identity [RESOLVED]
- **Verification**: Ran [`color_distillation/verify_multitask_onnx.py`](color_distillation/verify_multitask_onnx.py) across 256 test images. Observed exact bit-for-bit equivalence (`max |diff| = 0.000e+00`) for `predictions`, `embeddings`, and `class_maps`.

#### E-07 & E-08 (P0) Unbiased Evaluation & Deep MLP Distillation [RESOLVED]
- **Implementation**: Updated [`color_distillation/train_color_head.py`](color_distillation/train_color_head.py) to support `--no-class-weights`, `--head-type mlp_512x2`, and `--extract-test`.
- **Ablation Results**:
  - Removing inverse-frequency class weights eliminated rare-class false-positives: `green` precision rose from `28% -> 66%`, `brown` from `27% -> 61%`, `beige` from `21% -> 52%`, and macro precision average jumped from `56% -> 71%`.
  - Deep MLP-512×2 lifted validation Top-1 accuracy to **`75.85%`** (+7.85 pp), Top-3 to **`96.17%`**, and Macro F1 to **`63.96%`** (+8.30 pp).
  - Unbiased held-out test evaluation on `110,217` high-confidence test images reached **`75.53%`** Top-1 and **`95.81%`** Top-3 (`74.31%` Top-1 across all `114,201` test images). Spliced ONNX exported to `color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx`.

#### E-09 (P0) Ground-Truth Color Validation [RESOLVED]
- **CompCars Audit**: Executed [`color_distillation/evaluate_compcars_color_gt.py`](color_distillation/evaluate_compcars_color_gt.py) matching 19,479 CompCars CCTV images. Qwen teacher achieved `78.44%` aligned accuracy. Distilled Deep MLP-512×2 achieved **`77.43%`** on the 2,867 human-labeled test images (recovering 98.7% of teacher quality).
- **Review Sample**: Generated stratified 600-image review sample in [`annotation_runs/color_human_review_600.csv`](annotation_runs/color_human_review_600.csv) and wrote report to [`run_docs/color_ground_truth_validation.md`](run_docs/color_ground_truth_validation.md).

#### E-10 (P1) Production Domain Shift [DEFERRED / DOCUMENTED]
- **Status**: Documented evaluation protocol for production night/IR frames when surveillance feeds become available.

---

### B. Documentation Collisions

- **`E-11`**: Updated [`run_docs/annotation_run_4_and_5.md`](run_docs/annotation_run_4_and_5.md) with corrected 13-color taxonomy, in-sample vs held-out test clarifications, and UMAP landmark details.
- **`E-12`**: Cleaned relative links in [`run_docs/dvc_gdrive_guide.md`](run_docs/dvc_gdrive_guide.md) and documented ephemeral credential policy.
- **`E-13`**: Updated [`README.md`](README.md) with exact 1,235-class vs 1,416-class split statistics.
- **`E-14`**: Added `run_annotate.sh` and `dvc.sh` to [`bash_scripts/commands.md`](bash_scripts/commands.md).
- **`E-15`**: Corrected external crop specifications in [`run_docs/run_4.md`](run_docs/run_4.md).
- **`E-16`**: Aligned local Research PC `CLAUDE.md` paths with active repository structure.

---

### C. Process & Reproducibility

#### E-17 (P0) Run Provenance & CLI Recording [RESOLVED]
- Added `save_run_config()` to [`train_common.py`](train_common.py). Wired into training, annotation, and distillation runners. Backfilled `run_config.json` into `merged_dataset/output_merged_run4/`, `annotation_runs/run4/`, `annotation_runs/run5/`, and `color_distillation/output/`.

#### E-18 (P0) Cross-Domain Benchmark (Model A, B, C) [RESOLVED]
- Executed [`bash_scripts/run_phase4_cross_domain_eval.sh`](bash_scripts/run_phase4_cross_domain_eval.sh) evaluating Models A and B on the regenerated External test split (`splits_1235models/test.csv`, 5,205 base-class images) and PlatesMania test split:
  - **Model B on External (In-Domain)**: Scores only **`58.58%`** model accuracy (`71.35%` make accuracy), proving Run 3's reported 93.4% was overfit to the CCTV subset. Model C achieves **`92.83%`** model accuracy (**`+34.25 pp`** higher).
  - **Shared Classes (290 classes with External training data)**: Model C achieves **`96.93%`** restricted model accuracy on PlatesMania vs Model A's **`95.66%`** (**`+1.27 pp`** improvement from co-training with External data).
  - Documented in [`run_docs/run_4.md`](run_docs/run_4.md) and saved in [`mixed_benchmark_results_ab/`](mixed_benchmark_results_ab/).

---

### D. Repository Hygiene

#### E-19 & E-20 (P1/P2) Binary Tracking & `.gitignore` Hygiene [RESOLVED]
- Cleaned and categorized [`.gitignore`](.gitignore): deduplicated `/archive/`, ensured `*plotly.min.js`, `__pycache__`, and `*.pyc` stay ignored.
- Added global `*.onnx` to `.gitignore` to prevent multi-megabyte model checkpoints from ever being staged by Git.
- **Migrated all 11 legacy Git-tracked `.onnx` models to DVC** (`git rm --cached` + `dvc add`):
  - MobileNetV2: `backup_run1/ext/models/*.onnx`, `backup_run1/pm/models/*.onnx`, `backup_run3/external_dataset/.../*.onnx`, `backup_run3/platesmania_dataset/.../*.onnx`, `merged_dataset/output_merged_run4/models/*.onnx`.
  - EfficientNet-B0: `output_efficientnet_b0_merged/models/*.onnx`, `output_efficientnet_b0/models/*.onnx`, `sample_inference/models/*.onnx`.
- **Migrated new multi-task ONNX models to DVC**: `output_mlp512x2/mobilenet_v2_multitask.onnx.dvc` and `output_unweighted_mlp128/mobilenet_v2_multitask.onnx.dvc`.
- **Migrated `annotation_runs/run4/` to DVC**: untracked Git-tracked `annotations.csv` (34 MB) and `viewer.html` (16 MB) and tracked entire directory via [`annotation_runs/run4.dvc`](annotation_runs/run4.dvc).
- Zero `.onnx` files remain tracked by Git (`git ls-files "*.onnx"` returns 0). Commits stay strictly $< 100\text{ MB}$.

#### E-21 (P2) API Naming Cleanup [RESOLVED]
- Renamed `checkpoint_path` $\to$ `model_path` in `MobileNetV2Evaluator.__init__` in [`output_analysis.py`](output_analysis.py) while preserving `checkpoint_path` as a backward-compatible keyword alias.
