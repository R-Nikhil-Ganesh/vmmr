# errors.md: known issues, doc mismatches and open checks

Written on the laptop from the repo contents plus a local analysis of the DVC-pulled Run 5 files. **Nothing here was run against the source images.** Items marked **VERIFY** need the research PC. Items marked **FIX** are decided, so just do them. Each item says what is wrong, the evidence, and what to do. Paths are relative to `code/current/mobilenet/`.

Priority: P0 = affects results or claims already reported; P1 = wrong or misleading docs or fragile setup; P2 = hygiene.

Rules for whoever acts on this: do not break the A/B no-cross-training rule, keep ONNX-only weights, do not print or commit DVC/Drive secrets, record the flags of anything you run (see E-17), and update `bash_scripts/commands.md` if a runner changes.

---

## A. Data and pipeline defects

### E-01 (P0) External bounding boxes are never used (VERIFY, then fix docs or data)
- Evidence: `merged_dataset/splits_1235models/all_splits.csv` has `bbox_*` = -1 for **all 33,329 rows**, every source (boxcars116k, stanford_cars, compcars_web, compcars_cctv, resized_cupra). `merged_dataset/prepare_external_1235models.py` takes the `external_resized_512` branch (line ~286) which hard-codes -1. The Stanford branch (line ~100) does read real bboxes, but the stored images are the pre-resized 512x512 ones, so those coordinates would not apply anyway.
- Consequence: Model C, `run_4.md` and both annotation runs treat External as "bottom 5% crop of the whole image", not "vehicle bbox".
- Docs that claim a bbox crop (all wrong for the current splits): `run_docs/run_4.md` section 3 ("Crops"), `merged_dataset/train/train_mobilenet_v2_merged.py` line 10, `README.md` line 108, `annotation/common.py` (`crop_like_classifier` docstring), `annotation/README.md` and `run_docs/annotation_run_4_and_5.md` ("Pre-crops vehicles to tight bounding boxes").
- Do: (1) check whether Model B / Run 3 used real bboxes (`external_dataset/train/...`, its splits) and whether C is therefore not comparable to B on crop; (2) decide: regenerate the splits with bboxes rescaled to 512, or accept full-frame and correct every doc above; (3) record the decision in `run_docs/run_4.md`.

### E-02 (P0) Qwen and the color head saw whole frames, not vehicle crops (VERIFY)
- Evidence: `annotation_runs/run4` and `run5` `annotations.csv`: `bbox_x1` = -1 on 100% of rows, both domains. PlatesMania has no bbox in its manifest at all (`annotate.py` `pm_table` hard-codes -1). So Qwen colored whole PlatesMania frames (top 15% removed) with road, shadows and other cars in view.
- Consequence: label noise at the source, and the 68% distilled-head accuracy is partly this. See E-08.
- Do: on 300 random rows compare Qwen's color on the full frame vs a detector crop (any vehicle detector on the server). Report the agreement rate. If it is below ~90%, re-annotate with crops or accept the limit and document it.

### E-03 (P0) UMAP coordinates exist for only a small fraction of rows
- Evidence: `x`/`y` are NaN for **97.3% of Run 5** (1,126,737 rows) and **87.0% of Run 4**. `annotate.py` `stage_coords` fits and projects only `--coords-max` rows (default 30,000, half per domain). `annotation_run_4_and_5.md` says Stage 2 "Projects all N points into coordinates (x, y)". That is false.
- Consequence: the viewer can only show points that have coordinates; any downstream use of `x`,`y` on all rows silently drops 97%.
- Do: either project all rows with `reducer.transform` (UMAP supports it) or fix the doc and the `annotations.csv` schema notes. Either way state in the doc how many rows have coordinates.

### E-04 (P1) Absolute `/home/researchadmin/...` paths are stored in data files
- Evidence: `merged_dataset/splits_1235models/*.csv` (`image_path`) and `annotation_runs/*/annotations.csv` (`path`) hold absolute paths of the research PC. `annotate.py` re-resolves through the `resized_512` / `external_resized_512` markers (lines ~75, ~104) but nothing else does.
- Consequence: the tables cannot be used on the laptop or the production PC without rewriting; CLAUDE.md forbids hard-coded machine paths in code, but the data does it.
- Do: store paths relative to `PM_IMG_DIR` / `EXT_DATASETS_DIR` in new outputs; add one helper in `annotation/common.py` and use it everywhere.

### E-05 (P1) Color-head feature extraction does not follow the repo's resize convention
- Evidence: `color_distillation/train_color_head.py:66` resizes with `Image.BILINEAR` (no antialias); `annotation/common.py:50` does the same for the Qwen crops. CLAUDE.md says evaluation resize must match training (antialiased bilinear). Extraction also bypasses `DiagnosticDataset`, so its preprocessing is a second implementation.
- Do: run the same 2,000 val images through (a) the extraction path and (b) `output_analysis.DiagnosticDataset` and compare features; if they differ by more than 1e-2 relative, make extraction use the shared dataset class and re-extract (cache is cheap to rebuild, ~10 min).

### E-06 (P0) "Bit-for-bit identical make/model" is claimed but not tested
- Evidence: `run_docs/color_distillation_report.md` section 1 and `color_distillation_approach.md` claim zero make/model degradation and bit-for-bit identity. `color_distillation/verify_multitask_onnx.py` contains no `assert`, `allclose` or comparison against the original ONNX; it only prints shapes and latency.
- Do: add a check that compares `predictions`, `embeddings`, `class_maps` of `mobilenet_v2_multitask.onnx` against `merged_dataset/output_merged_run4/models/mobilenet_v2_best.onnx` on 256 images (report max abs diff), and that `color_predictions` equals the PyTorch head on the cached features (argmax agreement, max logit diff). Put the numbers in the report. Fix the wording if the diff is nonzero.

### E-07 (P1) The color head is scored in a favorable way
- Evidence: `train_color_head.py` keeps only `color_conf >= 0.70` (drops 4.2% of rows, which are the hardest) for both train and val; reports against Qwen labels, not truth; picks the best epoch by val accuracy and reports that same val number; never evaluates the `test` split although the test rows exist in `annotations.csv`.
- Do: report accuracy on `split == 'test'`, on all rows (including conf < 0.7) and on the 0.70+ subset, separately; select the epoch on a split that is not the reported one.

### E-08 (P0) The shipped color head is both under-fitted and handicapped by its loss (analysis done locally, reproduce on the server)
- Evidence (laptop, cached features, 1/3 of train): same 128-unit MLP **without class weights or label smoothing: 72.4% val** (shipped: 68.0%). 512x2 MLP: 74.4%; 512x3: 74.6%; their train accuracy is 78-80%. The shipped head trained to `train_acc 66.6 < val_acc 67.6` (`color_distillation/output/training_history.csv`), so it is under-fitted. Class weights (clipped 0.25-4.0) inflate recall and wreck precision for green, brown, beige (recall 61-65%, precision 21-28% in `output/final_report.txt`).
- Likely structural causes: Model C was trained for make/model with brightness/contrast/saturation/white-balance jitter and random grayscale (`train_common.py` lines ~80-135), so even `features[16]` is partly color-invariant; global average pooling over the 16x16 grid discards where the color is; the mode color within a make/model class is only 34% on average, so the backbone has no shortcut.
- Confusions are lighting-type, not hue-type: gray->black 24%, blue->black 20%, green->black 18%, beige->silver 31%, beige->white 21%, silver->white 14%.
- Do: (1) rerun `run_train_color_distillation.sh --skip-extract` with `--label-smoothing 0` and class weights off (add a `--no-class-weights` flag, default = old behavior) and report accuracy AND macro F1; (2) try a spatial tap (4x4 average pool of `features[16]`, 160*16 = 2,560 inputs) and a 512x2 head; (3) keep the old head as the baseline in the table.

### E-09 (P0) Qwen3-VL color labels are unvalidated and overconfident
- Evidence: `color_conf` mean 0.967, median 0.99996 (it is a softmax over a 13-option list, not a calibrated accuracy). Weakest labels by Qwen confidence: other (15% below 0.7), gold (12%), beige (11%), brown (8%), gray (9%). No human-labeled color set exists anywhere in the repo.
- Do: (1) draw a stratified review sample of 500-1,000 images (stratify by Qwen color; include conf < 0.7 rows) with `annotation/build_viewer.py` or a CSV; label by hand; compute Qwen accuracy and the confusion matrix; (2) check whether the raw External data carries color labels (CompCars has a color attribute) and use it as a free validation subset; (3) then recompute the head's accuracy against the human labels.

### E-10 (P1) Production domain shift for color is not addressed
- Both training sources are daytime web/catalog photos. Production cameras (night, IR, glare, low resolution) will differ, and IR frames carry no color information at all.
- Do: nothing yet on the research PC; when production frames arrive, run the multitask ONNX on them and hand-label a small set before trusting `color_predictions`. Add the result to `run_docs/`.

---

## B. Documentation mismatches

### E-11 (P1) `run_docs/annotation_run_4_and_5.md`
- Line ~98 lists colors `pink` and `purple`; the code (`annotation/color_heads.py` `COLORS`) uses `white black silver gray red blue green yellow orange brown beige gold other`. The doc's own color table (section 5) matches the code, not the list. **FIX**: use the code's list.
- "Combined" and "Train" rows report model accuracy 93.03% / 93.39% and make accuracy 98.50% / 98.60%. 80% of those images are training data, so these are not generalization numbers; only the Test row (91.53% / 98.15%) is. The statement that val about equal to test "confirms zero data leakage" does not follow (it shows no selection bias on val, not absence of leakage). **FIX**: relabel and reword.
- Stage 2 "projects all N points" and Stage 1/3 bbox crops are false (E-01, E-02, E-03).
- Section 7 "Next Steps" proposes a `Linear(1280, 13)` probe on `embeddings.npy`; the distillation actually used `features[16]` because the embedding gave only about 53% (`color_distillation_report.md` section 2). **FIX**: mark Phase 1-3 as done and superseded.
- "Run 4 Qwen precision bfloat16, max_side=448, eager attention": the run's flags were not recorded; mark as "reported, unverified".
- Original request was Qwen3-VL-**4B**; code, docs and `paths.sh` now say 8B everywhere with no recorded reason. **VERIFY** it is intentional and record why (accuracy, speed) in the doc.

### E-12 (P1) `run_docs/dvc_gdrive_guide.md`
- Contains a `file:///home/researchadmin/Econ-n/repo-clone/...` link: machine-specific, breaks the paths rule. **FIX**: relative link.
- Says credentials are never saved to disk. On the private laptop `.dvc/config.local` does hold them (gitignored). Rule to document: **research PC (shared): env vars only; laptop: `.dvc/config.local` is acceptable.** **FIX** the text.
- `DVC_SITE_CACHE_DIR` defaults to `/tmp/dvc` in `bash_scripts/lib/paths.sh` (wiped on reboot, shared by all users of the PC). **VERIFY** whether this is intended; if not, point it to a per-user directory.
- `dvc.sh` and `paths.py`: `paths.py` `KEYS` does not include `DVC_BIN` / `DVC_SITE_CACHE_DIR`, so `python paths.py --check` does not verify DVC. **FIX** (all four places in CLAUDE.md "new path key" rule).

### E-13 (P1) `README.md` is stale
- Describes External as "41,880 images, 268 active models, 32 makes" (Run 3). The current merged splits have 33,329 images and 286 classes with External training data (`run_docs/run_4.md`). Layout section lacks `color_distillation/`, `annotation/`, `bash_scripts/dvc.sh`, `run_train_color_distillation.sh`, `errors.md`, and the DVC workflow. **FIX**.

### E-14 (P1) `bash_scripts/commands.md` and run naming
- Examples use `RUN_NAME=run5` for training (`run_train_merged.sh`) and also `RUN_NAME=run5` for annotation, and `run_train_color_distillation.sh` hard-codes `MODEL_RUN=run4`. Two different "run5"s exist (annotation run 5, a hypothetical training run 5). **FIX**: use distinct examples (`RUN_NAME=ann5`, `RUN_NAME=merged_ablation1`) and document `MODEL_RUN`, `RUN_NAME` and `--cache-from` in one table.

### E-15 (P1) `run_docs/run_4.md`
- Crop description is wrong for External (E-01).
- Training flags are not recorded (E-17).
- External numbers are not comparable to B because the test set changed (4,870 vs 6,163 images). Re-evaluate A and B on the new External test split (item E-18).

### E-16 (P2) CLAUDE.md is gitignored and per-machine
- Each environment has its own copy. After this session the laptop copy is current; copy it to the research PC and the production PC (set the "This machine" line). The server agent must not assume its copy is up to date. The earlier layout error (`external_dataset/prepare_external_1235models.py`) was corrected on the laptop: the script is at `merged_dataset/prepare_external_1235models.py`.

---

## C. Process and reproducibility

### E-17 (P0) Run flags are never recorded
- Run 4 (training), annotation run 4 and 5, and the color distillation have no record of the CLI flags, git commit, Python/package versions or seed next to their outputs.
- Do: write `run_config.json` (argv, resolved flags, `git rev-parse HEAD`, `pip freeze` subset, seed, start/end time, GPU) into every output folder from `train_common.py` (training) and from `annotate.py` / `train_color_head.py`. Backfill what is known for Runs 4/5 by hand from shell history and tmux logs (`training.log`).

### E-18 (P0) Open comparisons needed to read Run 4 correctly
- Re-evaluate Models A and B on the regenerated External test split (`merged_dataset/splits_1235models/test.csv`).
- Evaluate A restricted to the 286 classes that have External training data, on PlatesMania.
- Both are listed in `run_docs/run_4.md` section 6 and have not been run.

---

## D. Repository hygiene

### E-19 (P1) Weights and results in git
- Git still tracks about 12 ONNX files (about 20 MB each): `merged_dataset/output_merged_run4/models/`, `backup_run1/*`, `backup_run3/*`, `code/current/efficientnet/**`. The multitask ONNX is already DVC-tracked.
- `annotation_runs/run4/` (34 MB CSV + 16 MB viewer) is in git, while `run5` is in DVC and ignored. Run 4 embeddings and color cache are ignored by `.gitignore`, so a clean clone cannot reproduce `--cache-from run4`.
- Do: move `backup_run*` and the efficientnet ONNX files to DVC (`git rm --cached` + `dvc add`); decide on Run 4 model and `annotation_runs/run4` the same way; add every large artifact the runners produce to `.gitignore` or DVC before `auto_git_sync.sh` (`git add -A code/current/mobilenet/`) can add it.

### E-20 (P2) `.gitignore` hygiene
- It has both a bare `CLAUDE.md` (any level) and explicit paths for `run5/` and `color_distillation/cache/`; `dvc add` also writes its own ignore entries. Check there are no duplicates or conflicts, and that `*plotly.min.js` stays ignored (4.5 MB file in `annotation_runs/run5/`).
- `__pycache__/` directories exist in the tree; they are ignored but keep out of DVC.

### E-21 (P2) Misc
- `MobileNetV2Evaluator.checkpoint_path` actually takes an ONNX path; rename to `model_path`.
- `run_all_pipeline.sh` still excludes Model C and passes extra flags to both trainers.
- `run_docs/run_2.md` attributes both models to the same training script (Model B used an intermediate one).
- The script that built the PlatesMania manifest is not in the repo; the 50-image class floor looks like a total-per-class rule (unverified).

---

## Suggested order for the server agent
1. E-17 (record flags from now on), E-06 (verify the multitask ONNX), E-01/E-02 (decide the crop question), E-03.
2. E-08 and E-09 together: human review set, then retrain the head with the options in E-08, evaluate on `test`.
3. E-18 (A/B re-evaluation), then the doc fixes E-11 to E-15.
4. Hygiene (E-19, E-20, E-21).
