# CLAUDE.md

Vehicle Make/Model recognition (VMMR). Active work is in `code/current/mobilenet/`; `code/current/efficientnet/` and `code/archive/` are older and not maintained.

## How this project is run (important)
- **Code is edited on a dev machine. Training and evaluation run on a separate GPU server** (RTX 4090) where the data lives. The data (PlatesMania images, external datasets) is **not available locally**.
- **Do not run training or full evaluation scripts locally.** Static checks (`python -m py_compile`, `bash -n`) and synthetic-data tests of single functions are fine. Use tiny fake CSVs / images / random ONNX models for anything beyond that, and keep them out of the repo.
- The user pushes, pulls on the server and runs there. The runners auto-commit and push results (`auto_git_sync.sh`), so **pulls often conflict** on `evaluate_mixed.py`, `output_analysis.py` and READMEs. Before merging, back up local versions; prefer the server's version where it already covers the feature.
- Local Python: `/home/cloudyrelic/e-con-new/pt-env/bin/python` (torch, onnxruntime, pandas, sklearn installed).

## Experiment design (do not break)
- Two models, each trained **only on its own dataset**: Model A = PlatesMania, Model B = External merged (BoxCars116k, Stanford Cars, CompCars). **No cross-training for A and B.** The point is to measure cross-dataset generalization. Model C (`merged_dataset/`, `run_train_merged.sh`) trains on both datasets as a separate, explicitly requested experiment; never feed it into A or B.
- Classes are **1235 Make/Model labels** (e.g. `Toyota/Corolla`) over 35 makes. Both models share the same label index space. Make accuracy is obtained by summing model probabilities per make (`P(make) = sum P(model)`).
- Baseline (Run 3, see `code/current/mobilenet/run_docs/run_3.md`): in-domain model acc ~92-93% (make ~96-98%); cross-domain ~20% (A on External, make 40%) and ~3% (B on PlatesMania, make 13%). **Measured:** label coverage explains little of that gap (restricted to the 268 classes both datasets trained on, B on PlatesMania only reaches 8.4% and A on External 25%), so it is mostly domain shift (PlatesMania = full image minus 15% top crop; External = vehicle bbox). B only trains on 268 of 1235 classes. `evaluate_mixed.py` writes `class_coverage.csv` and `shared_class_summary.csv`.
- Run 4 = Model C (merged), run as `RUN_NAME=run4 bash bash_scripts/run_train_merged.sh [flags]`, then `run_diagnostics.sh --merged`. Its External share per epoch is `--ext-fraction` (default 0.2) and checkpoints are chosen on the mean of the two domain val losses.
- Weights are saved as **ONNX only** (no `.pt`). The ONNX has 3 outputs: `predictions`, `embeddings` (1280-d), `class_maps` (CAM source for Grad-CAM). Keep that contract, since `output_analysis.py` depends on it.

## Layout (`code/current/mobilenet/`)
- `platesmania_dataset/train/train_mobilenet_v2.py`, `external_dataset/train/train_mobilenet_v2_external.py`: trainers (torchvision MobileNetV2, 512px, top 5 blocks unfrozen, AdamW, cosine, AMP, 15 epochs).
- `merged_dataset/train/train_mobilenet_v2_merged.py`: Model C trainer. It imports the dataset classes, model, eval and ONNX export from the two trainers above, so changes there carry over; keep their names stable.
- `train_common.py`: shared GPU augmentation (`augment_batch`), `TrainObjective` (label smoothing, auxiliary make loss), `ModelEMA`, crop jitter. **All trainers must use this** so A, B and C stay comparable.
- `output_analysis.py`: ONNX Runtime evaluation, metrics, confusion matrices, Grad-CAM, t-SNE, calibration. Dataset adapters live in `*/analysis/output_analysis_*.py`.
- `evaluate_mixed.py`: cross-domain benchmark (in-domain / out-of-domain / mixed, model- and make-level, shared-class view).
- `external_dataset/prepare_external_1235models.py`: builds the External splits (not a runner; run it directly with Python on the server).
- `run_docs/run_1..4.md`: per-run reports (follow the `run_3.md` / `run_4.md` style for new runs). `backup_run1/`, `backup_run2/`: superseded outputs kept for reference; do not read them as current results.
- `bash_scripts/lib/paths.sh` / `paths.py`: **single source of machine-specific paths.** Put per-server values in the gitignored `bash_scripts/lib/paths.local.sh` (e.g. `ECON_ROOT=...`). Never hard-code `/home/researchadmin/...` in new code; use `paths.*`. `python paths.py --check` verifies a server.
- `bash_scripts/`: the only shell runners: `run_all_pipeline.sh`, `run_diagnostics.sh`, `run_train_pm.sh`, `run_train_external.sh`, `run_train_merged.sh`, `run_annotate.sh`. **They start in a detached tmux session by default** (`lib/tmux_wrap.sh`; `NO_TMUX=1` for foreground). New long-running scripts go in `bash_scripts/` and call `run_in_tmux` the same way; do not add separate tmux/non-tmux variants. Extra CLI flags pass through to the trainer; `RUN_NAME=<name>` writes to separate output folders so ablations do not overwrite each other. Usage and arguments are documented in `code/current/mobilenet/bash_scripts/commands.md`: keep it in sync when changing a runner.

## Conventions
- New training options must default to the **old behavior** and be flags (`--aug-strength`, `--crop-jitter`, `--label-smoothing`, `--make-loss-weight`, `--ema-decay`).
- Validation loss stays plain cross-entropy (checkpoint selection must be comparable across runs).
- Unreadable images use `IGNORE_INDEX = -100` and are reported, never silently replaced.
- Frozen backbone blocks keep BatchNorm in eval mode (`set_train_mode`).
- Evaluation resize must match training (antialiased bilinear).
- After changing a trainer, smoke-test both on a tiny synthetic dataset on CPU (`--img-size 128 --epochs 2 --num-workers 0`) before handing back.

## Data annotation (built, not yet run on the server)
`mobilenet/annotation/` (`annotate.py`, `color_heads.py`, `build_viewer.py`, `common.py`) + `bash_scripts/run_annotate.sh`. Annotates **test-split** images of both datasets: make/model from Model C (ONNX; make by marginalization, plus confidence and the 1280-d embedding), color from a pluggable head (`predict(images) -> [(color, conf)]`; default `qwen3vl` = Qwen3-VL-4B scored over a fixed color list, `classical` = no-model baseline), then a self-contained Plotly `viewer.html` (hover thumbnail panel, filters, low-confidence review tab). Output table is `annotations.csv` + `embeddings.npy` under `${ANNOT_OUT_DIR}/<run>/` (outside git; the table is the interface for a later server-backed viewer). Low-confidence rows are flagged `needs_review`; predictions are never presented as ground truth. Verified locally only with synthetic images and the `classical` head; the Qwen path and the viewer in a browser are untested. External splits have no color labels; the raw External data may (check on the server) and could validate Qwen.

## Known gaps
- `run_docs/run_2.md` attributes both models to the same training script; Model B was trained with an intermediate version.
- The pipeline (`run_all_pipeline.sh`) does not include Model C, and extra flags go to both trainers, so a flag only one trainer accepts (e.g. `--crop-top-pct`) fails the other.
- The script that built the PlatesMania manifest is not in the repo. The 50-image class floor seems to apply to the total per class (train counts reach 37-40), which is unverified.
- A cleaner single Python entrypoint with per-run `run_config.json` was proposed (to do after Run 4); runs currently do not record their flags next to the results.
- The planned distillation step (DINOv2 teacher fine-tuned per dataset) is not implemented; evidence for it in this setting is weak.
- Model-level cross-domain accuracy is capped by classes present in only one dataset; read the shared-class numbers, not only the raw ones.
