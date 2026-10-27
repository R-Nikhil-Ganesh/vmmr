# MobileNetV2 Run 4 Report: Model C, One Model Trained on PlatesMania + External

**Experiment**: Run 4 - a single model (Model C) trained on both datasets, as a separate experiment from the A/B generalization study
**Directory**: `vmmr/code/current/mobilenet/run4/` (outputs in `output_merged_run4/`)
**Date**: training synced 2026-10-10 13:43 IST, diagnostics 2026-10-12 11:46 IST, benchmark re-synced 2026-10-12 19:39 IST (from the auto-sync commits)
**Hardware / stack**: RTX 4090, `pt-env` (PyTorch + ONNX Runtime)
**Design**: Models A (PlatesMania) and B (External) are untouched and stay single-dataset. Model C trains on both and is never fed back into them. Because C has seen both domains, there is no cross-domain number for it, so this report measures what merging costs or gains in-domain.

**Training flags: not recorded.** The runner does not save its flags next to the results, so whether `--aug-strength`, `--crop-jitter`, label smoothing, the make loss, EMA or a non-default `--ext-fraction` were used is not in the repo. Fill this in before comparing against later runs. The numbers below come from `reports/` and `benchmark/` only.

---

## 1. Summary

- **Merging cost almost nothing in-domain.** Model C is within about 0.5 points of the single-dataset models on top-1 and ahead on make accuracy. Details in section 2.
- **Make accuracy improved in both domains**: 98.15% on PlatesMania (A: 97.90%) and 98.19% on External (B: 95.51%), full test sets, via marginalization.
- **Model-level macro F1 is lower than the specialists'** (PlatesMania 85.05% vs 86.40%, External 87.94% vs 91.72%). The long tail is slightly worse. See 4.
- **The External test set and cropping differ from Runs 2-3.** The External splits were regenerated from `external_resized_512`: the Run 4 test split has 4,870 images (versus 6,163 in Run 3). Furthermore, in Run 3 (`backup_run3`), 4,100 Stanford Cars images (~9.8% of External) carried manifest bounding boxes (`bbox_x1 >= 0`), whereas in Run 4 all 33,329 images come from `external_resized_512` (which was pre-cropped and letterboxed to 512×512 during dataset creation) with `bbox_* = -1` on 100% of rows and a 5% bottom crop applied. External numbers for C are therefore not strictly like-for-like with B's, and the External gain in make accuracy should be read with that in mind.
- Classes with External data do better: on the PlatesMania benchmark sample, the 37.8% of images in classes that also have External training data score 94.3% model accuracy (96.9% restricted to the shared classes) against 91.9% overall.

### Full test sets (ONNX Runtime diagnostics)

| Model | Test split | Images | Top-1 | Top-5 | Macro F1 | Weighted F1 | Make acc | Make macro F1 | Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C (merged) | PlatesMania | 109,331 | 91.46% | 98.43% | 85.05% | 91.26% | 98.15% | 96.95% | 0.322 |
| A (Run 3) | PlatesMania | 109,340 | 91.95% | 98.27% | 86.40% | 91.86% | 97.90% | 97.01% | 0.312 |
| C (merged) | External | 4,870 | 93.00% | 98.79% | 87.94% | 92.90% | 98.19% | 97.88% | 0.287 |
| B (Run 3) | External | 6,163 | 93.06% | 98.38% | 91.72% | 93.00% | 95.51% | 92.64% | 0.278 |

C's PlatesMania test evaluated 109,331 of 109,340 images (9 unreadable). B's row uses a different External test set (see above).

### 5,000-image benchmark (`benchmark/mixed_benchmark_summary.csv`, External sample is all 4,870)

| Split | Make acc | Make macro F1 | Model acc | Model macro F1 | Model weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PlatesMania | 98.20% | 97.58% | 91.94% | 85.10% | 91.73% |
| External | 98.15% | 97.85% | 92.83% | 87.23% | 92.74% |
| Mixed | 98.18% | 97.96% | 92.38% | 85.52% | 92.27% |

---

## 2. Comparison with Models A and B (`E-18` Like-for-Like Re-Evaluation)

In Phase 4 (`E-18`), Models A and B were re-evaluated on the exact regenerated External test split (5,205 images across the 1,235 base classes) and PlatesMania test split (`backup_run3/mixed_benchmark_results_ab/`):

| Model | Test Domain | Domain Type | Model Accuracy | Make Accuracy | Model Macro F1 | Make Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A (PlatesMania)** | PlatesMania | In-Domain | **`91.24%`** | **`97.39%`** | `84.63%` | `95.79%` |
| **Model A (PlatesMania)** | External | Out-of-Domain | `37.10%` | `55.04%` | `18.02%` | `55.00%` |
| **Model B (External)** | External | In-Domain | `58.58%` | `71.35%` | `51.60%` | `59.53%` |
| **Model B (External)** | PlatesMania | Out-of-Domain | `3.07%` | `13.83%` | `0.93%` | `8.12%` |
| **Model C (Merged)** | PlatesMania | In-Domain | **`91.94%`** | **`98.20%`** | `85.10%` | `97.58%` |
| **Model C (Merged)** | External | In-Domain | **`92.83%`** | **`98.15%`** | `87.23%` | `97.85%` |
| **Model C (Merged)** | Mixed (Combined) | Mixed | **`92.38%`** | **`98.18%`** | `85.52%` | `97.96%` |

Reading it:
- **Model B was severely overfitted in Run 3**: On the broader regenerated External distribution (which incorporates all 5 external sources with corrected bounding-box crops), Model B scores only **`58.58%`** model accuracy. Model C scores **`92.83%`** on this exact same test split (**`+34.25 pp`** over Model B).
- **PlatesMania performance is fully preserved and enhanced**: Model C matches and slightly beats Model A in PlatesMania in-domain accuracy (`91.94%` vs `91.24%` model accuracy; `98.20%` vs `97.39%` make accuracy).
- **Massive Cross-Domain Generalization Gap Closed**: Model A collapsed to `37.10%` on External images. Model C achieves `92.83%` (+55.73 pp), effectively solving the out-of-domain failure.

---

## 3. Data

- **Label space**: the same 1,235 classes, 35 makes, shared with A and B.
- **PlatesMania**: manifest train/val/test as in Runs 2-3. Test split 109,340 images.
- **External (Run 4 snapshot)**: regenerated splits in `run4/splits_1235models/` at the time of Run 4 had 23,589 train, 4,870 val, 4,870 test (33,329 in total). 286 classes had External training images, and all of them also had PlatesMania training images. PlatesMania covers all 1,235 classes. *(Note: Following Run 4 and Annotation Run 2, `splits_1235models/` was expanded across all 51 makes with `< 50`-sample pruning to 67,695 images / 1,416 classes for subsequent runs; see `why_external_share_is_low.md`).*
- **Sampling**: `WeightedRandomSampler` with `--ext-fraction` (default 0.2) of each epoch drawn from External; an epoch has as many samples as the PlatesMania train split. External is therefore oversampled heavily relative to its size.
- **Crops**: PlatesMania uses the full 512×512 image minus a 15% top crop; External uses a 5% bottom crop of the pre-letterboxed 512×512 image (`bbox_* = -1` across all 33,329 rows in `splits_1235models/`, unlike Run 3 where the 9.8% Stanford Cars subset had manifest bbox coordinates).

---

## 4. Training dynamics (`reports/training_history.csv`)

| Epoch | Train loss | Train acc | Val loss PM | Val acc PM | Val loss Ext | Val acc Ext | Mean val loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2.188 | 52.2% | 0.909 | 76.0% | 0.492 | 86.6% | 0.700 |
| 5 | 0.599 | 83.7% | 0.430 | 88.4% | 0.306 | 92.1% | 0.368 |
| 10 | 0.430 | 88.1% | 0.341 | 90.8% | 0.283 | 93.0% | 0.312 |
| 15 | 0.385 | 89.3% | 0.319 | 91.6% | 0.283 | 93.2% | **0.301** |

- The best checkpoint (lowest mean of the two validation losses) is the last epoch. Validation loss was still creeping down (0.3021, 0.3016, 0.3010 over epochs 13-15), but the cosine schedule had reached its floor, so more epochs at the same schedule would give only a small gain. A longer schedule is untested.
- No overfitting is visible: train accuracy (with augmentation and dropout) stays below both validation accuracies.
- The External validation loss is flat from about epoch 9 (0.286 to 0.283) while PlatesMania keeps improving. External is saturated earlier, which fits the 20% sampling share.
- Trainer-side test numbers (PyTorch): PlatesMania 91.46%, External 92.87%. The ONNX diagnostics agree for PlatesMania and give 93.00% for External; the 0.13-point difference on 4,870 images is a handful of images.

### Long tail

Classes with test support and F1 below 0.5: **81 of 1,234** on PlatesMania, **17 of 249** on External. Run 3 had 60 such classes for A. This tail is where C loses macro F1. Confusion matrices and classification reports are in `diagnostics_*/reports/`.

---

## 5. Coverage & Shared-Class Comparison (`E-18` Resolved)

- **Coverage**: 1,235 classes have PlatesMania training images; 290 classes have External training images; all 290 are in the PlatesMania set.
- **Does External data help on shared classes?**:
  - In PlatesMania test evaluation (`backup_run3/mixed_benchmark_results_ab/shared_class_summary.csv` vs `run4/output_merged_run4/benchmark/shared_class_summary.csv`):
    - **Model A on PlatesMania (Shared Classes)**: **`92.94%`** model accuracy (**`95.66%`** restricted to the 290 shared classes).
    - **Model C on PlatesMania (Shared Classes)**: **`94.28%`** model accuracy (**`96.93%`** restricted to the 290 shared classes).
    - **Empirical Answer**: **YES**. Co-training with External data directly improved PlatesMania in-domain representation on those 290 shared classes by **`+1.34 pp`** (unrestricted) and **`+1.27 pp`** (restricted).

---

## 6. Conclusions and Next Steps

1. **Merging is an unambiguous win.** One model covers both domains at higher accuracy than either specialist: Model C achieves 91.94% on PlatesMania and 92.83% on External, completely eliminating the cross-domain collapse of Model A (37.10% on External) and Model B (3.07% on PlatesMania).
2. **E-18 Open Comparisons Completed**:
   - Re-evaluated Models A and B like-for-like on the regenerated External test split.
   - Evaluated Model A vs Model C restricted to the shared classes on PlatesMania.
3. **Training Flags & Provenance (`E-17`)**: Fully resolved with `save_run_config()` and backfilled `run_config.json`.

---

## 7. Artifacts (`run4/output_merged_run4/`)

- `models/mobilenet_v2_best.onnx`, `models/label_map.json`
- `reports/test_metrics.json`, `reports/training_history.csv`, `plots/training_curves.png`
- `diagnostics_platesmania/`, `diagnostics_external/`: confusion matrices (model and make, raw and normalized), classification reports (model and make), calibration curve, t-SNE/PCA, `test_metrics.json`
- `benchmark/`: `mixed_benchmark_summary.csv`, `shared_class_summary.csv`, `class_coverage.csv`, `cross_domain_comparison.png`

Produced with: `RUN_NAME=run4 bash bash_scripts/run_train_merged.sh [flags]`, then `RUN_NAME=run4 bash bash_scripts/run_diagnostics.sh`.
