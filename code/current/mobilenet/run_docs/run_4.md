# MobileNetV2 Run 4 Report: Model C, One Model Trained on PlatesMania + External

**Experiment**: Run 4 - a single model (Model C) trained on both datasets, as a separate experiment from the A/B generalization study
**Directory**: `vmmr/code/current/mobilenet/merged_dataset/` (outputs in `output_merged_run4/`)
**Date**: training synced 2026-10-10 13:43 IST, diagnostics 2026-10-12 11:46 IST, benchmark re-synced 2026-10-12 19:39 IST (from the auto-sync commits)
**Hardware / stack**: RTX 4090, `pt-env` (PyTorch + ONNX Runtime)
**Design**: Models A (PlatesMania) and B (External) are untouched and stay single-dataset. Model C trains on both and is never fed back into them. Because C has seen both domains, there is no cross-domain number for it, so this report measures what merging costs or gains in-domain.

**Training flags: not recorded.** The runner does not save its flags next to the results, so whether `--aug-strength`, `--crop-jitter`, label smoothing, the make loss, EMA or a non-default `--ext-fraction` were used is not in the repo. Fill this in before comparing against later runs. The numbers below come from `reports/` and `benchmark/` only.

---

## 1. Summary

- **Merging cost almost nothing in-domain.** Model C is within about 0.5 points of the single-dataset models on top-1 and ahead on make accuracy. Details in section 2.
- **Make accuracy improved in both domains**: 98.15% on PlatesMania (A: 97.90%) and 98.19% on External (B: 95.51%), full test sets, via marginalization.
- **Model-level macro F1 is lower than the specialists'** (PlatesMania 85.05% vs 86.40%, External 87.94% vs 91.72%). The long tail is slightly worse. See 4.
- **The External test set is not the one used in Runs 2-3.** The External splits were regenerated: the test split now has 4,870 images, versus 6,163 in Run 3. External numbers for C are therefore not like-for-like with B's, and the External gain in make accuracy should be read with that in mind.
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

## 2. Comparison with Models A and B

| Metric | A, PlatesMania | C, PlatesMania | B, External | C, External |
| :--- | :---: | :---: | :---: | :---: |
| Top-1 | 91.95% | 91.46% (-0.49) | 93.06% | 93.00% (-0.06) |
| Make accuracy | 97.90% | 98.15% (+0.25) | 95.51% | 98.19% (+2.68) |
| Macro F1 | 86.40% | 85.05% (-1.35) | 91.72% | 87.94% (-3.78) |

Reading it:
- PlatesMania is basically unchanged. The 0.5-point top-1 drop is real in the full 109k test but small.
- The External make gain is the largest change, but B's External test set was different, so treat it as likely rather than established. A re-evaluation of B on the new External test split would settle it.
- Macro F1 on External fell by about 4 points. The External test set is small (4,870 images over 249 classes with support), so a few classes with a handful of images move this metric a lot.

---

## 3. Data

- **Label space**: the same 1,235 classes, 35 makes, shared with A and B.
- **PlatesMania**: manifest train/val/test as in Runs 2-3. Test split 109,340 images.
- **External**: regenerated splits in `merged_dataset/splits_1235models/`: 23,589 train, 4,870 val, 4,870 test (33,329 in total). 286 classes have External training images, and all of them also have PlatesMania training images. PlatesMania covers all 1,235 classes.
- **Sampling**: `WeightedRandomSampler` with `--ext-fraction` (default 0.2) of each epoch drawn from External; an epoch has as many samples as the PlatesMania train split. External is therefore oversampled heavily relative to its size.
- **Crops**: PlatesMania uses the full image minus 15% top; External uses the vehicle bbox where present and a 5% bottom crop otherwise, as in A and B.

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

## 5. Coverage (`benchmark/class_coverage.csv`)

- 1,235 classes have PlatesMania training images; 286 have External training images; the 286 are all in the PlatesMania set (Run 3: 268 shared).
- On the PlatesMania benchmark sample, 1,889 of 5,000 images (37.8%) fall in those 286 classes. C scores 94.3% on them (96.9% with the argmax restricted to the 286 classes), against 91.9% overall. They are the more common classes, so this does not prove that External data helps; it needs the same split for A (A restricted to the 286 classes) to compare.
- External test images are all in classes with External training data, so the shared-class view equals the plain one there.

---

## 6. Conclusions and next steps

1. **Merging is viable.** One model covers both domains at roughly specialist accuracy and improves make accuracy. If the goal is a single deployed model, C is the candidate.
2. **It does not answer the generalization question.** C has seen both domains, so it cannot say how a model handles an unseen domain. Models A and B remain the generalization experiment, and the strong-augmentation, label-smoothing and EMA steps still need to be evaluated on them.
3. **Record the flags.** Before the next run, store the flags used in the output folder (the `run_config.json` idea in CLAUDE.md), so Run 4 can be reproduced.
4. **Open checks**:
   - Re-evaluate B and A on the new External test split so the External comparison is like-for-like.
   - Evaluate A restricted to the 286 shared classes on PlatesMania (section 5).
   - Try a different `--ext-fraction` (0.1 / 0.3) and a longer schedule.
   - The 81 weak PlatesMania classes are the first target for any further improvement.
5. **Next work**: the annotation tool (`annotation/`) uses this model for make and model.

---

## 7. Artifacts (`merged_dataset/output_merged_run4/`)

- `models/mobilenet_v2_best.onnx`, `models/label_map.json`
- `reports/test_metrics.json`, `reports/training_history.csv`, `plots/training_curves.png`
- `diagnostics_platesmania/`, `diagnostics_external/`: confusion matrices (model and make, raw and normalized), classification reports (model and make), calibration curve, t-SNE/PCA, `test_metrics.json`
- `benchmark/`: `mixed_benchmark_summary.csv`, `shared_class_summary.csv`, `class_coverage.csv`, `cross_domain_comparison.png`

Produced with: `RUN_NAME=run4 bash bash_scripts/run_train_merged.sh [flags]`, then `RUN_NAME=run4 bash bash_scripts/run_diagnostics.sh --merged`.
