# MobileNetV2 Run 3 Report: 1,235-Class Make/Model Recognition and Cross-Domain Benchmark

**Experiment**: Run 3 - fine-grained Make/Model classification (1,235 classes) instead of 35-make classification  
**Directory**: `vmmr/code/current/mobilenet/`  
**Date**: October 7, 2026 (PlatesMania training finished 18:51 IST, External 19:03, benchmark 19:10, from the auto-sync commits)  
**Hardware / stack**: RTX 4090, `pt-env` (PyTorch + ONNX Runtime), same as Run 2  
**Design (unchanged)**: two models, each trained only on its own dataset. Model A = PlatesMania, Model B = External merged. No cross-training, so cross-dataset accuracy measures generalization.

---

## 1. Summary

Run 3 moves both models from 35 makes to **1,235 Make/Model classes** (e.g. `Toyota/Corolla`). Make-level accuracy is still reported, obtained by summing model probabilities per make.

- **In-domain results are good**: model-level 92.1% (A) and 93.4% (B); make-level 97.8% and 96.0% on the 5,000-image benchmark subsets.
- **Cross-domain results are poor**: A on External 20.6% model / 39.8% make; B on PlatesMania **3.0% model / 13.2% make**.
- **Model B was effectively trained on 268 of the 1,235 classes** (see 3.2). The shared-class evaluation (5.1) shows this explains only part of the gap: restricted to classes both datasets trained on, B on PlatesMania rises from 3.0% only to 8.4%, and A on External from 20.6% to 25.0%. **Most of the cross-domain failure is genuine domain shift** (framing, image source), not missing labels.
- Run 3 diagnostics (confusion matrices, make-level reports, calibration, t-SNE) now exist for both models (section 7). The stale Run 2 files were moved to `backup_run2/`.

### Key numbers (5,000 images per dataset; Mixed = 10,000)

| Model | Test split | Make Acc | Make Macro F1 | Model Acc | Model Macro F1 | Model Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| A (PlatesMania) | PlatesMania (in-domain) | **97.84%** | 97.54% | **92.06%** | 86.18% | 91.90% |
| A (PlatesMania) | External (out-of-domain) | 39.76% | 37.53% | 20.60% | 11.43% | 23.59% |
| A (PlatesMania) | Mixed | 68.80% | 70.17% | 56.33% | 66.63% | 55.40% |
| B (External) | External (in-domain) | **95.98%** | 93.38% | **93.40%** | 92.66% | 93.34% |
| B (External) | PlatesMania (out-of-domain) | 13.18% | 7.79% | 3.00% | 0.78% | 1.40% |
| B (External) | Mixed | 54.58% | 46.47% | 48.20% | 15.35% | 45.59% |

Full-test-set metrics from the diagnostics run (`reports/test_metrics.json`, ONNX Runtime, so they differ slightly from the trainers' own PyTorch numbers: A 91.91%, B 93.15%):

| Model | Test images | Top-1 (model) | Top-5 | Macro F1 | Weighted F1 | Make acc | Make macro F1 | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A (PlatesMania) | 109,340 | 91.95% | 98.27% | 86.40% | 91.86% | 97.90% | 97.01% | 0.312 |
| B (External) | 6,163 | 93.06% | 98.38% | 91.72% | 93.00% | 95.51% | 92.64% | 0.278 |

The 5,000-image benchmark numbers above agree with these within sampling noise (A 92.06 vs 91.95, B 93.40 vs 93.06).

### Comparison with Run 2 (make-level accuracy)

| Split | Run 2 A | Run 3 A | Run 2 B | Run 3 B |
| :--- | :---: | :---: | :---: | :---: |
| PlatesMania test | 96.72% | 97.84% | 17.18% | 13.18% |
| External test | 48.50% | 39.76% | 94.34% | 95.98% |
| Mixed | 72.61% | 68.80% | 55.76% | 54.58% |

**Caution:** these are not like-for-like. Run 3's External test set only contains the 1,235-class subset (6,163 images, 264 classes) versus 7,815 images in Run 2, and each split is a 5,000-image random sample. Differences of a few points are within what a different sample and class set could produce. Read the direction, not the decimals: in-domain make accuracy went up, out-of-domain make accuracy went down.

---

## 2. What changed from Run 2

- **Label space**: 35 makes to 1,235 Make/Model classes. Both models use the same label index space (`label_map_1235models.json`, built from PlatesMania; External labels are mapped into it by `external_dataset/prepare_external_1235models.py`).
- **Evaluation**: `evaluate_mixed.py` reports model-level and make-level metrics. Make-level uses marginalization, `P(make) = sum of P(model)`.
- **Tooling**: tmux launchers and automatic commit/push after each run (`auto_git_sync.sh`).
- **Training scripts**: Model A ran with the vectorized GPU augmentation (`augment_batch`); Model B ran with the older per-image torchvision augmentation loop. Ranges are the same, but they are not identical code. (The scripts were unified afterwards in `train_common.py`, so future runs will use the same augmentation for both.)
- **Hyperparameters unchanged**: 512x512, batch 32, AdamW, LR 5e-4 head / 5e-5 backbone, cosine to 1e-6, dropout 0.25, top 5 blocks unfrozen, 15 epochs, AMP, frozen BatchNorm in frozen blocks.

---

## 3. Data

### 3.1 Model A (PlatesMania)
Manifest `dataset_1235models_splits.csv` (server only), full-image input with a 15% top crop. Test split: 109,340 images across all 1,235 classes (from the diagnostics run; train/val sizes were not recorded in a surviving Run 3 log). The benchmark used a 5,000-image random sample of the test split.

### 3.2 Model B (External merged)
Sources: BoxCars116k, Stanford Cars, CompCars (surveillance subset). Vehicle bounding boxes are used where available, otherwise a 5% bottom crop. Per-class cap of 600 images.

| Split | Images | Classes present | Makes |
| :--- | :---: | :---: | :---: |
| Train | 29,554 | **268** | 32 |
| Val | 6,163 | 264 | |
| Test | 6,163 | 264 | |

Train images by source: CompCars CCTV 13,871, BoxCars116k 12,751, Stanford Cars 2,932.

Class imbalance inside those 268 classes: median 59 train images per class, mean 110, max 420; **40 classes have fewer than 20 images and 119 fewer than 50**.

**The classifier head has 1,235 outputs, but 967 of them (78%) never saw a positive example.** Model B can essentially only predict the 268 classes it was trained on. Any PlatesMania test image of another class is unrecoverable for it.

---

## 4. Training dynamics

Both models were still improving at epoch 15 (cosine LR reached its floor), so more epochs might help in-domain, but the gap that matters is cross-domain.

| Epoch | A train acc | A val acc | A val loss | B train acc | B val acc | B val loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 62.4% | 81.5% | 0.710 | 30.5% | 58.1% | 1.889 |
| 5 | 89.4% | 89.7% | 0.386 | 87.6% | 87.7% | 0.490 |
| 10 | 92.9% | 91.5% | 0.330 | 94.8% | 92.0% | 0.312 |
| 15 | 94.1% | 92.0% | 0.317 | 96.3% | 92.9% | 0.285 |

- **A** shows a small train/val gap (2.1 points): limited more by capacity or class difficulty than by overfitting. This matches the Run 2 reading that A is capacity-limited.
- **B** shows a larger gap (3.4 points) and its best validation loss was at epoch 14; with only 29.5k training images and many tiny classes it starts to overfit.
- Epoch-1 validation accuracy is higher than train accuracy for both, expected with dropout and augmentation active during training only.

---

## 5. Cross-domain analysis

**Model A on External (20.6% model, 39.8% make).** Model A's label space contains every class in the External test set, so this is genuine domain shift: PlatesMania full-frame surveillance images versus External bounding-box crops from studio/CCTV/web sources. Make accuracy fell about 9 points versus Run 2, while in-domain make accuracy rose. This suggests fine-grained training made the features more specific to the PlatesMania look (not proven; sample and class set differ, see 1).

**Model B on PlatesMania (3.0% model, 13.2% make).** Two effects were mixed together; section 5.1 separates them. Domain shift dominates.

B's very low Model Macro F1 on Mixed (15.4%, against 48.2% accuracy) is the same effect: macro F1 averages over every class that appears in truth or prediction, and most of them are classes B never learned.

**Macro F1 versus accuracy on Mixed for A (66.6% vs 56.3%)** is higher than accuracy because the mixed sample covers few images per class, so the per-class average is dominated by easy classes; do not compare it with the in-domain macro F1.

---

### 5.1 Label coverage versus domain shift (measured)

Source: `mixed_benchmark_results/shared_class_summary.csv` and `class_coverage.csv`. **Shared classes** = classes with at least one train image in both datasets: **268** (all of B's classes are also in A's 1,235).

- Only **35.7%** of the PlatesMania test sample (1,783 of 5,000 images) belongs to a shared class. All of the External test sample is shared.
- "Restricted" = argmax over shared columns only.

| Model | Test split | Raw model acc | Shared-class acc | Shared, restricted argmax | Shared make acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| A | PlatesMania | 92.06% | 93.77% | 96.13% | 98.04% |
| A | External | 20.60% | 20.60% | 25.02% | 39.76% |
| B | External | 93.40% | 93.40% | 93.40% | 95.98% |
| B | PlatesMania | 3.00% | 8.41% | 8.41% | 20.98% |

Reading:
- **B on PlatesMania**: removing images of classes B never saw lifts accuracy only from 3.0% to 8.4% (make level 13.2% to 21.0%). B still fails on over 90% of images it has labels for. Label coverage is a minor part of the story.
- **A on External**: every External class is known to A, yet only 20.6% are right (25.0% with the argmax restricted to the 268 shared classes). This is domain shift, not coverage.
- The asymmetry (A 20-25% vs B 3-8%) is consistent with PlatesMania being the more varied training set (109k-image test split, all 1,235 classes) and External being clean vehicle-bbox crops from a few sources. Not tested directly.
- Mixed-set numbers for B (71.1% on shared classes) are inflated by the External half; do not read them as generalization.

## 6. Conclusions and next steps

1. **The cross-domain gap is mostly domain shift.** Coverage was measured (5.1) and explains only a few points. Focus on invariance, not labels.
2. **Try the generalization options** (shared by both trainers through `train_common.py`), one at a time with `RUN_NAME=<name>`: `--aug-strength strong --crop-jitter 0.5` first (it targets the framing mismatch: PlatesMania full image minus 15% top crop vs External vehicle bbox), then `--label-smoothing 0.1 --make-loss-weight 0.3`, then `--ema-decay 0.999`. See the README. Judge them on shared-class cross-domain accuracy and make accuracy, not on in-domain accuracy.
3. **External class set.** B trains on 268 classes, 40 with fewer than 20 images, so its fine-grained ceiling is set by data. More External data for the missing classes, or a min-images filter, would change what B can show. This matters for in-domain breadth, not for the cross-domain gap.
4. **Long tail in A.** 60 of A's 1,235 classes have F1 below 0.5 on the full test set (e.g. `Volkswagen/XL1`, `Nissan/Skystar`, `Honda/Supra`, `Kia/Credos`, all F1 = 0 with 1-12 test images), which is why A's macro F1 (86.4%) trails its accuracy (91.95%). In B only 2 of 264 classes fall below 0.5 (`Audi/A2` at 1 test image, `BMW/6 Series` at 0.40).
5. **Make level is the better-behaved metric in-domain.** The weakest makes in A still reach F1 0.89-0.94 (Dodge, Isuzu, Buick, Skoda, Chevrolet). In B the weakest are tiny makes (Isuzu 6 images, Lincoln 5).

---

## 7. Artifacts

Run 3 outputs (current), under `platesmania_dataset/output_mobilenet_v2/` and `external_dataset/output_mobilenet_v2_external/`:
- `models/{mobilenet_v2_best.onnx, label_map.json}`
- `reports/{test_metrics.json, training_history.csv, test_classification_report.csv, test_classification_report_make.csv, confusion_matrix_{raw,normalized}.csv, confusion_matrix_make_{raw,normalized}.csv}`
- `plots/{training_curves, confusion_matrix, confusion_matrix_make, calibration_curve, latent_space_tsne_pca}.png`
- `external_dataset/splits_1235models/` (External splits and label map)
- `mixed_benchmark_results/{mixed_benchmark_summary.csv, shared_class_summary.csv, class_coverage.csv, cross_domain_comparison.png}`

The Run 2 (35-class) diagnostics and the Run 1 `training.log` that used to sit in these folders are in `backup_run2/` (see its README). They are not Run 3 results.

**Known limitations:** train/val split sizes for PlatesMania and the training logs of Run 3 were not preserved; per-make tables like Run 2's are not reproduced here, but the per-make numbers are in `test_classification_report_make.csv`. Calibration (ECE) is plotted in `calibration_curve.png` but not summarized numerically here.
