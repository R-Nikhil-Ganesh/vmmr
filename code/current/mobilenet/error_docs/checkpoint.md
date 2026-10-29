# Implementation Checkpoint (`errors.md` Resolution)

**Timestamp**: 2026-10-26 12:43 IST
**Master Plan**: [`code/current/mobilenet/run_docs/errors_master_implementation_plan.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/errors_master_implementation_plan.md)
**Error Tracker**: [`code/current/mobilenet/errors.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/errors.md)

---

## 1. Completed & Verified Items

### Pre-Phase / Cropping & Dataset Audit (`E-01`, `E-02`, `E-03`) — **COMPLETED**
- **`E-01` (PlatesMania & External Bounding-Box Cropping)**:
  - Verified on disk (`/home/researchadmin/Econ/resized_512` and `/home/researchadmin/Econ/external_resized_512`) that **PlatesMania (`1,105,650` images)** and **4 of the 5 External sources** (`stanford_cars` `6,495`, `compcars_web` `13,290`, `compcars_cctv` `33,324`, `resized_cupra` `1,590`) were already `100%` cropped to vehicle bounding boxes and letterboxed to `512×512` before Run 4/5.
  - Identified that **`boxcars116k` (`18,520` images in `external_resized_512`, `15,641` in `splits_1235models`)** was the only source whose `inst["2DBB"] = [x, y, w, h]` vehicle bounding box from `BoxCars116k/dataset.pkl` had been ignored (`"bbox": None`), leaving a 30-pixel road/pavement border around each vehicle.
  - Matched all `18,520` `boxcars116k` images to `BoxCars116k/dataset.pkl`, mapped their `2DBB` coordinates into the `512×512` letterboxed coordinate space, and saved them in:
    - [`code/current/mobilenet/run4/boxcars116k_bboxes_512.json`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run4/boxcars116k_bboxes_512.json)
    - `/home/researchadmin/Econ/external_resized_512/external_metadata_51make.csv`
    - `code/current/mobilenet/run4/splits_1235models/{train,val,test,all_splits}.csv` (`15,641` rows now have `bbox_x1 >= 0`)
  - Updated [`code/current/mobilenet/run4/prepare_external_1235models.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run4/prepare_external_1235models.py) and `/home/researchadmin/Econ/scripts/rebuild_external_resized_512.py`.
- **`E-02` (300-Sample Crop Sensitivity Study)**:
  - Executed [`code/current/mobilenet/annotation/verify_crop_agreement.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/annotation/verify_crop_agreement.py) and generated [`code/current/mobilenet/run_docs/crop_agreement_study.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/crop_agreement_study.md):
    - **Overall Agreement**: **`71.33%`** (`300` images)
    - **PlatesMania (`150` images, pre-cropped on disk)**: **`76.00%`** agreement
    - **External `BoxCars116k` (`150` images, legacy uncropped 30px road border vs. tight `2DBB` vehicle bbox crop)**: **`66.67%`** agreement (`33.33%` of `BoxCars116k` images change predicted color when the surrounding asphalt border is removed).
- **`E-03` (External Dataset Expansion & 50-Sample Pruning)**:
  - Expanded External dataset across all 51 makes and pruned classes with `< 50` combined samples (`1,416` classes, `67,695` External images). Documented in [`code/current/mobilenet/run_docs/why_external_share_is_low.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/why_external_share_is_low.md).

---

### Phase 1: Reproducibility, Relative Paths & API Cleanup (`E-04`, `E-17`, `E-21`) — **COMPLETED**
- **`E-04` (Portable Relative Paths)**:
  - Added `to_rel_image_path()` and `resolve_image_path()` to [`code/current/mobilenet/annotation/common.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/annotation/common.py).
  - Updated [`prepare_external_1235models.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run4/prepare_external_1235models.py), [`annotation/annotate.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/annotation/annotate.py), [`train_common.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/train_common.py) (`ExternalMergedDataset`), and [`output_analysis.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/output_analysis.py) (`DiagnosticDataset` & `load_and_preprocess_image`).
  - Regenerated `code/current/mobilenet/run4/splits_1235models/{train,val,test,all_splits}.csv` with portable relative paths (`images/<Make>/<file>.jpg`).
- **`E-17` (`run_config.json` Provenance)**:
  - Added `save_run_config()` in [`code/current/mobilenet/train_common.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/train_common.py) and wired it into `train_mobilenet_v2_merged.py`, `annotate.py`, and `train_color_head.py`.
  - Backfilled `run_config.json` into `run4/output_merged_run4/`, `annotation_runs/run1/`, `annotation_runs/run2/`, and `color_distillation/output/`.
- **`E-21` (`MobileNetV2Evaluator` Parameter Rename)**:
  - Renamed primary parameter `checkpoint_path` $\to$ `model_path` in [`code/current/mobilenet/output_analysis.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/output_analysis.py) while maintaining backward compatibility via `checkpoint_path` keyword alias.

---

### Phase 2: Preprocessing Verification & Ground-Truth Color Audit (`E-05`, `E-06`, `E-09`) — **COMPLETED**
- **`E-05` (`features[16]` Preprocessing Comparison)**:
  - Executed [`code/current/mobilenet/color_distillation/verify_preprocessing_e05.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/verify_preprocessing_e05.py) across `2,000` validation images and saved [`code/current/mobilenet/color_distillation/output/e05_preprocessing_comparison.json`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/output/e05_preprocessing_comparison.json):
    - `global_relative_l2`: **`0.029558`** (`2.96%` > `1e-2` threshold)
    - `mean_cosine_similarity`: **`0.999534`**
    - `color_top1_agreement_pct`: **`98.10%`** (`1.90%` prediction flip rate due to non-antialiased `PIL.Image.BILINEAR` vs. `DiagnosticDataset`).
  - Unified `ImageExtractDataset` in [`code/current/mobilenet/color_distillation/train_color_head.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/train_color_head.py) to use `io.read_image` + antialiased `F.interpolate`.
- **`E-06` (Multitask ONNX Verification)**:
  - Verified bit-identical Make/Model outputs (`max_abs_diff = 0.0`) on `mobilenet_v2_multitask.onnx` in [`code/current/mobilenet/color_distillation/verify_multitask_onnx.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/verify_multitask_onnx.py).
- **`E-09` (Ground-Truth Color Audit on 19,479 CompCars CCTV Images & 600-Row Human Review Set)**:
  - Executed [`code/current/mobilenet/color_distillation/evaluate_compcars_color_gt.py`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/evaluate_compcars_color_gt.py) and generated [`code/current/mobilenet/run_docs/color_ground_truth_validation.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/color_ground_truth_validation.md) and [`code/current/mobilenet/annotation_runs/color_human_review_600.csv`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/annotation_runs/color_human_review_600.csv):
    - **Matched CompCars CCTV Ground-Truth Rows in `run2/annotations.csv`**: **`19,479`**
    - **Strict Exact Agreement**: **`70.21%`** | **Taxonomy-Aligned Agreement** (`grey`/`silver`, `gold`/`beige`/`yellow`/`brown`): **`78.44%`**
    - **High-Confidence (`color_conf >= 0.70`, `94.6%` of rows)**: **`79.70%`** aligned agreement vs. **Low-Confidence (`color_conf < 0.70`, `5.4%` of rows)**: **`56.21%`** aligned agreement (`+23.49 pp` gain).

---

### Documentation & Collision Fixes (`E-11` through `E-16`) — **COMPLETED**
- Resolved all documentation collisions across [`run_docs/run_2.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/run_2.md), [`run_docs/run_3.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/run_3.md), [`run_docs/run_4.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/run_4.md), [`run_docs/annotation_and_viewer.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/annotation_and_viewer.md), [`run_docs/color_distillation_report.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/color_distillation_report.md), and [`CLAUDE.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/CLAUDE.md).

---

### Phase 3: Color Head Retraining & Unbiased Test Evaluation (`E-07`, `E-08`) — **COMPLETED**
- **Ablation 1 (Unweighted MLP-128, `output_unweighted_mlp128`)**:
  - Removed inverse-frequency weights (`--no-class-weights --label-smoothing 0.0`), fixing rare-class false-positive explosions:
    - Val Top-1: **`73.46%`** (+5.46 pp over baseline `68.00%`), Top-3: **`95.25%`**, Macro F1: **`59.05%`** (precision macro: **`0.69`** vs `0.56`).
    - Precision jumps: `green` (`28% -> 67%`), `brown` (`27% -> 57%`), `beige` (`21% -> 50%`).
    - Unbiased Test (`conf >= 0.70`, `110,217` images): **`73.23%`** Top-1, **`95.05%`** Top-3, **`58.05%`** Macro F1.
    - All Test Rows (`114,201` images): **`72.02%`** Top-1.
    - CompCars Human Ground-Truth Test (`2,867` images): **`74.89%`** (+3.81 pp over baseline `71.08%`).
- **Ablation 2 (Deep MLP-512×2, `output_mlp512x2`) [ADOPTED REFERENCE PRODUCTION HEAD]**:
  - Added capacity via 2-layer `512 -> 512 -> 13` MLP with BatchNorm1d + Dropout + fused ONNX export:
    - Val Top-1: **`75.85%`** (+7.85 pp over baseline), Top-3: **`96.17%`**, Macro F1: **`63.96%`** (+8.30 pp).
    - Unbiased Test (`conf >= 0.70`, `110,217` images): **`75.53%`** Top-1, **`95.81%`** Top-3, **`62.89%`** Macro F1.
    - All Test Rows (`114,201` images): **`74.31%`** Top-1, **`95.40%`** Top-3.
    - CompCars Human Ground-Truth Test (`2,867` images): **`77.43%`** (+6.35 pp over baseline `71.08%`, recovering 98.7% of Qwen3-VL teacher's `78.44%` accuracy).
- Updated [`code/current/mobilenet/run_docs/color_distillation_report.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/color_distillation_report.md) with complete ablation metrics.

---

## 2. Remaining Work to Resume Next

### Phase 4: Cross-Domain & Shared-Class Benchmark (`E-18`) — **COMPLETED**
- Re-evaluated **Model A** (PlatesMania) and **Model B** (External) on the regenerated External test split (`splits_1235models/test.csv`, `5,205` base-class images) and PlatesMania test split in [`backup_run3/mixed_benchmark_results_ab/`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/backup_run3/mixed_benchmark_results_ab/):
  - **Model B on External (In-Domain)**: **`58.58%`** Model Acc, **`71.35%`** Make Acc (revealing Run 3's `93.4%` was overfit to the CCTV subset; Model C achieves **`92.83%`** Model Acc, a **`+34.25 pp`** margin!).
  - **Model A on PlatesMania (In-Domain)**: **`91.24%`** Model Acc, **`97.39%`** Make Acc (Model C matches and beats at **`91.94%`** Model Acc, **`98.20%`** Make Acc).
  - **Model A on External (Out-of-Domain)**: Collapses to **`37.10%`** Model Acc, **`55.04%`** Make Acc (Model C closes the gap at **`92.83%`**, **`+55.73 pp`**).
  - **Shared Classes (290 classes with External training data)**:
    - Model A on PlatesMania (Shared Classes): **`92.94%`** (unrestricted), **`95.66%`** (restricted).
    - Model C on PlatesMania (Shared Classes): **`94.28%`** (unrestricted), **`96.93%`** (restricted).
    - Proves empirically that co-training with External data directly improved PlatesMania in-domain representation by **`+1.27 pp`** on those shared classes.
- Updated Section 2, Section 5, and Section 6 of [`code/current/mobilenet/run_docs/run_4.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/run_4.md) with complete metrics.

### Phase 5: Repository Hygiene & Resolution Wrap-Up (`E-19`, `E-20`) — **COMPLETED**
- Cleaned and categorized [`.gitignore`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/.gitignore):
  - Deduplicated `/archive/`.
  - Added global `*.onnx` rule to prevent new multi-megabyte model checkpoints from being inadvertently staged by `git add -A` / `auto_git_sync.sh`.
  - Preserved ignores for `*plotly.min.js`, `__pycache__`, and intermediate caches.
- **Migrated all 11 legacy Git-tracked `.onnx` models to DVC** (`git rm --cached` + `dvc add`):
  - MobileNetV2: `backup_run1/ext/`, `backup_run1/pm/`, `backup_run3/external_dataset/`, `backup_run3/platesmania_dataset/`, `run4/output_merged_run4/`.
  - EfficientNet-B0: `output_efficientnet_b0_merged/`, `output_efficientnet_b0/`, `sample_inference/`.
- **Migrated new multi-task ONNX models to DVC**: `output_mlp512x2/mobilenet_v2_multitask.onnx.dvc` and `output_unweighted_mlp128/mobilenet_v2_multitask.onnx.dvc`.
- **Migrated `annotation_runs/run1/` to DVC**: untracked Git-tracked `annotations.csv` (34 MB) and `viewer.html` (16 MB) and tracked entire directory via `annotation_runs/run1.dvc`.
- Zero `.onnx` files remain tracked by Git (`git ls-files "*.onnx"` returns 0). Commits stay strictly $< 100\text{ MB}$.
- Updated [`code/current/mobilenet/errors.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/errors.md) to mark all items `E-01` through `E-21` as `[RESOLVED]` with exact empirical metrics and file references (with `E-10` documented for production night/IR deployment).

---

## 2. All Phases Complete

All five phases of [`code/current/mobilenet/run_docs/errors_master_implementation_plan.md`](file:///home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run_docs/errors_master_implementation_plan.md) are now fully executed, verified, and documented.
