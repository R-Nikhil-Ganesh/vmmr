# Commands

Run everything from `vmmr/code/current/mobilenet/`. The runners in `bash_scripts/` **start in a detached tmux session by default** and keep the pane open afterwards. Training and diagnostics are meant for the GPU server, where the data lives.

| I want to... | Command | tmux session |
|---|---|---|
| Train Model C (merged), then per-domain diagnostics + benchmark | `bash bash_scripts/run_all_pipeline.sh [trainer flags]` | `mobilenet_pipeline` |
| Train Model C (PlatesMania + External merged) | `bash bash_scripts/run_train_merged.sh [flags]` | `train_merged` |
| Diagnostics + mixed benchmark | `bash bash_scripts/run_diagnostics.sh [options]` | `diagnostics` |
| Annotate images (make/model + Qwen3-VL color) + build `viewer.html` | `bash bash_scripts/run_annotate.sh [flags]` | `annotate` |
| Score the Qwen color teacher against CompCars human labels (one config or a sweep) | `RUN_NAME=<tag> bash bash_scripts/run_eval_color_teacher.sh [flags]` / `SWEEP=1 bash ...` | `eval_color_teacher` |
| Distill color into MobileNetV2 (`features[16]` tap) -> ONNX | `bash bash_scripts/run_train_color_distillation.sh [flags]` | `color_distill` |
| Run DVC commands (`status`, `add`, `push`, `pull`) | `./bash_scripts/dvc.sh <subcommand>` | *(foreground)* |

`bash_scripts/lib/` holds helpers that are not run directly: `paths.sh` (all machine paths), `tmux_wrap.sh`, `auto_git_sync.sh`.

## Working with tmux

```bash
tmux attach -t train_merged        # watch a run (detach: Ctrl+B, then D)
tmux ls                            # list sessions
tmux kill-session -t train_merged  # stop a run
```

A second launch with a session name that is already running is refused (it prints the attach/kill commands).

## Environment variables (set in front of any runner)

| Variable | Effect |
|---|---|
| `RUN_NAME=<name>` | Separate output folders and tmux session per experiment, so runs do not overwrite each other: `run4/output_merged_<name>` (training, diagnostics, and `benchmark/`), `annotation_runs/<name>` (annotation), session `<session>_<name>`. Use distinct names for training vs. annotation runs (e.g. `RUN_NAME=merged_51make` for training, `RUN_NAME=run2` for annotation). |
| `SWEEP=1` | `run_eval_color_teacher.sh` runs the one-factor teacher sweep (`s0_original` ... `s5_flip`) instead of a single `RUN_NAME` configuration. |
| `MODEL_RUN=<name>` | Used by `run_annotate.sh` and `run_train_color_distillation.sh` to select which `run4/output_merged_<MODEL_RUN>/models/mobilenet_v2_best.onnx` checkpoint to load (defaults to `run4`). |
| `CACHE_FROM=<dir>` | Used by `run_annotate.sh` to reuse predictions, embeddings, and Qwen color labels from an earlier annotation folder (defaults to `${ANNOT_OUT_DIR}/run1` if present). |
| `NO_TMUX=1` | Run in the foreground instead of tmux (debugging, cron, CI). |
| `AUTO_GIT_PUSH=0` | Do not commit and push results when finished (default `1`: commits `code/current/mobilenet/` and pushes). |
| `CUDA_VISIBLE_DEVICES=1` | Choose the GPU. |
| `ECON_ROOT`, `PM_IMG_DIR`, `PM_MANIFEST_CSV`, `PM_LABEL_MAP`, `EXT_DATASETS_DIR`, `PYTHON_BIN`, `DVC_BIN`, `DVC_SITE_CACHE_DIR` | Override a path for one run. Permanent per-server values go in the gitignored `bash_scripts/lib/paths.local.sh`. |
| `ANNOT_OUT_DIR`, `QWEN_MODEL_DIR` | Annotation output root (default `${MOBILENET_DIR}/annotation_runs`) and the Qwen3-VL model id or local directory (see `run_annotate.sh`). |

Example: `RUN_NAME=merged_51make AUTO_GIT_PUSH=0 bash bash_scripts/run_train_merged.sh --aug-strength strong`

---

## `run_train_merged.sh` (Model C: PlatesMania + External)

Model C trains jointly on **both** datasets across the unified Make/Model taxonomy (`1,235` base PlatesMania classes `0..1234`, automatically extended to `1,416` classes across `49` makes when `splits_1235models/label_map.json` includes the 51-make $\ge 50$-sample External classes `1235..1415`). Each image is cropped as in its own domain (PlatesMania top 15% banner crop, External bottom 5% watermark crop; `bbox_* = -1` in current 512×512 splits).

External has `47,805` train images (`23,589` in the Run 4 35-make split) against `~875k` for PlatesMania, so `WeightedRandomSampler` draws a fixed share of External images per epoch (`--ext-fraction 0.2` by default). Checkpoints are selected on the **mean of the two domain validation losses**, and final test metrics are reported per domain in `reports/test_metrics.json`.

Output: `run4/output_merged_run4/` (or `output_merged_<RUN_NAME>`), with `models/mobilenet_v2_best.onnx` (`predictions`, `embeddings`, `class_maps`), `models/label_map.json`, `reports/`, `plots/training_curves.png`, and `training.log`.

| Flag | Default | What it does |
|---|---|---|
| `--ext-fraction F` | `0.2` | Share of External images per epoch. `0` = plain concatenation. |
| `--aug-strength {base,strong}` | `base` | `strong` = wide zoom and aspect, rotation, translation, stronger colour jitter, grayscale, blur, noise, random erasing. |
| `--crop-jitter F` | `0` | 0 to 1. Randomizes the training crop (PlatesMania 15% top crop, External 5% bottom crop). |
| `--label-smoothing F` | `0` | Label smoothing on the training loss (e.g. `0.1`). |
| `--make-loss-weight F` | `0` | Weight of an auxiliary loss on P(make) = sum of P(model) over the make's models (e.g. `0.3`). |
| `--ema-decay F` | `0` | EMA of the weights (e.g. `0.999`). |
| `--epochs N`, `--batch-size N`, `--eval-batch-size N` | `15`, `32`, `64` | Schedule and batch sizes. |
| `--lr F`, `--backbone-lr F` | `5e-4`, `5e-5` | Learning rates for the head and the unfrozen backbone blocks. |
| `--dropout F`, `--unfreeze-layers N` | `0.25`, `5` | Head dropout; number of top `features` blocks trained. |
| `--img-size N`, `--num-workers N` | `512`, `10` | Input resolution and DataLoader workers. |

```bash
bash bash_scripts/run_train_merged.sh
RUN_NAME=merged_51make bash bash_scripts/run_train_merged.sh --ext-fraction 0.2 --aug-strength strong --crop-jitter 0.5
```

---

## `run_diagnostics.sh`

Evaluates Model C on:
1. Full PlatesMania test split (`run4/output_.../diagnostics_platesmania/`)
2. Full External test split (`run4/output_.../diagnostics_external/`)
3. 5,000-image per-domain and 10,000-image Mixed benchmark (`run4/output_.../benchmark/`)

| Option | What it does |
|---|---|
| (none) | PlatesMania diagnostics, External diagnostics, and the 5,000-image mixed benchmark. |
| `--quick` | Cap test images per domain diagnostic at 5,000. |
| `--max-samples N` | Cap test images per domain diagnostic at N. |
| `--pm-only` | Only PlatesMania test split diagnostics. |
| `--ext-only` | Only External test split diagnostics. |
| `--bench-only` | Only the 5,000-image mixed benchmark. |

```bash
bash bash_scripts/run_diagnostics.sh
bash bash_scripts/run_diagnostics.sh --bench-only
bash bash_scripts/run_diagnostics.sh --quick
```

---

## `run_annotate.sh` (make/model + color annotation and HTML viewer)

Annotates vehicle images across test or all splits (`train`, `val`, `test`) with make and model (merged Model C, ONNX) and color (Qwen3-VL-8B 4-bit NF4, scored over a fixed color list), then builds a self-contained `viewer.html`. tmux session `annotate_<RUN_NAME>`.

```bash
bash bash_scripts/run_annotate.sh --max-samples 200 --skip-color               # smoke test, no Qwen
bash bash_scripts/run_annotate.sh --max-samples 200                            # smoke test with Qwen
MODEL_RUN=run4 RUN_NAME=run2 bash bash_scripts/run_annotate.sh --split all     # full dataset run using Model C (run4) + run1 cache reuse
```

Needs `pip install -U "transformers>=4.57" accelerate bitsandbytes` for Qwen; `umap-learn` is used for 2D manifold projection. Output goes to `${ANNOT_OUT_DIR}/<RUN_NAME>/` (default `code/current/mobilenet/annotation_runs/<RUN_NAME>/`): `annotations.csv`, `embeddings.npy`, `color_cache.csv`, `viewer.html`.

| Option | Meaning |
|---|---|
| `--split {test,all,train,val}` | Which split(s) to annotate (default: `test`; use `all` for full dataset) |
| `--cache-from DIR` | Previous run to reuse annotations, embeddings, and colors from (default: `run1` if available) |
| `--domain both\|pm\|ext` | Which dataset(s) (default `both`) |
| `--max-samples N` | Random subsample per domain (default: all split images) |
| `--skip-color` | make/model only |
| `--color-head NAME` | `qwen3vl` (default) |
| `--qwen-model ID_OR_DIR` | Default `$QWEN_MODEL_DIR` (`Qwen/Qwen3-VL-8B-Instruct`); local directory for offline servers |
| `--color-batch-size N` | Qwen batch (default `16`) |
| `--qwen-max-side N` | Max image side for Qwen vision tokens (default: `336` for fast inference) |
| `--no-4bit` | Disable 4-bit NF4 quantization and use bfloat16 |
| `--qwen-scoring first_token\|variants\|letters` | How answers are scored. `first_token` (default, Runs 1/2) only sees lowercase, no-space color tokens, so a `White` answer is missed; `variants` sums case/space variants; `letters` scores `A)`..`M)` options |
| `--qwen-prompt short\|defined` | `defined` adds color definitions and "body paint only" instructions |
| `--qwen-center-crop F` / `--qwen-tta none\|flip` | Trim fraction F around the center; average over a horizontal flip |
| `--reuse-model-stage` | Skip the ONNX stage if `annotations.csv` and `embeddings.npy` exist |
| `--coords-max N` | Rows that get 2-D embedding coordinates (default 30000) |
| `--max-points N` | Points with thumbnails embedded in the viewer (default `15000`) |
| `--review-conf` / `--review-color-conf` | Confidence thresholds below which a row is flagged `needs_review` (`0.70` / `0.50`) |
| `--no-viewer` | Skip building the HTML |

To rebuild only the viewer: `python annotation/build_viewer.py --annotations <dir>/annotations.csv --max-points 15000 --workers 16`.

---

## `run_eval_color_teacher.sh` (Qwen color teacher vs human labels)

Scores a teacher configuration on the CompCars human-labeled rows (19,479 matched; default `--split val`, 2,859 rows) and writes `${ANNOT_OUT_DIR}/teacher_eval/<tag>/` (`predictions.csv` with the full 13-color probabilities, `metrics.json`) plus one line in `teacher_eval/summary.csv`. Tune on `val`; score `test` once at the end.

```bash
SWEEP=1 bash bash_scripts/run_eval_color_teacher.sh --max-samples 200     # dry run of every configuration
SWEEP=1 bash bash_scripts/run_eval_color_teacher.sh                       # one-factor sweep on val (s0_original ... s5_flip)
RUN_NAME=best_val   bash bash_scripts/run_eval_color_teacher.sh --no-4bit --scoring letters --max-side 672 --prompt-style defined
RUN_NAME=best_train bash bash_scripts/run_eval_color_teacher.sh --split train <same flags>          # for calibration
python annotation/calibrate_color_teacher.py --train-dir annotation_runs/teacher_eval/best_train --val-dir annotation_runs/teacher_eval/best_val
```

| Option | Meaning |
|---|---|
| `--split train\|val\|test\|all` | CompCars human-GT rows to score (default `val`) |
| `--review-csv PATH` | Also score filled-in rows of `annotation_runs/color_human_review_600.csv` (13-color, both domains) |
| `--max-samples N` | Random subsample |
| `--no-4bit`, `--max-side N`, `--scoring`, `--prompt-style`, `--center-crop F`, `--tta` | Head options, same meaning as the `--qwen-*` flags of `run_annotate.sh`; defaults = Runs 1/2 |

`calibrate_color_teacher.py` fits a logistic regression from the teacher's probabilities to the human labels on the train rows and reports val accuracy before/after. Its output is in the CompCars 10-color taxonomy (gray inside silver, beige/gold inside champagne).

---

## `run_train_color_distillation.sh` (Color Distillation -> Multi-Task ONNX)

Distills Qwen3-VL-8B color labels from `run5/annotations.csv` into a lightweight classification head branching from block `features[16]` (160 channels) of the Model C backbone (`MODEL_RUN=run4` by default). Caches extracted features to disk (`cache/features_L16_train.npy`, 262.45 MB for the 860,006 `color_conf >= 0.70` train rows) for rapid training, trains over cached vectors in <2 minutes, and automatically exports a unified `mobilenet_v2_multitask.onnx`.

> **Reference Production Head**: The production standard is the **Unweighted Deep MLP-512×2** head in `output_mlp512x2/` (**75.53% held-out test top-1**, **77.43% CompCars human ground truth**), trained via `run_phase3_color_ablations.sh`.

```bash
# 1. Train the Reference Production Head (Deep MLP-512x2, Unweighted, LS 0.0)
bash bash_scripts/run_phase3_color_ablations.sh

# 2. Or train Reference Head directly via Python
python3 color_distillation/train_color_head.py \
  --head-type mlp_512x2 \
  --no-class-weights \
  --label-smoothing 0.0 \
  --batch-size 2048 \
  --skip-extract \
  --out-dir color_distillation/output_mlp512x2

# 3. Train historical baseline 128-unit MLP head (weighted, LS 0.05)
bash bash_scripts/run_train_color_distillation.sh

# 4. Verify Reference Multi-Task ONNX model signature, latency & accuracy
python3 color_distillation/verify_multitask_onnx.py \
  --onnx-path color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx \
  --pt-head-path color_distillation/output_mlp512x2/color_head_best.pt
```

| Option | Meaning |
|---|---|
| `--head-type {mlp_512x2,mlp,linear}` | Color head architecture (`mlp_512x2`: 160 -> 512 -> 512 -> 13 [Reference]; `mlp`: 160 -> 128 -> 13; `linear`: 160 -> 13) |
| `--no-class-weights` | Disables inverse-frequency weights (crucial: fixes rare-color precision collapse [E-08]) |
| `--epochs N` | Training epochs (default `25`) |
| `--batch-size N` | Batch size for training (default `512` or `2048`) |
| `--lr F` | Learning rate with cosine annealing (default `1e-3`) |
| `--label-smoothing F` | Label smoothing for regularization (`0.0` for reference head, `0.05` for baseline) |
| `--min-conf F` | Minimum Qwen color confidence filter (default `0.70`) |
| `--extract-batch-size N`| Batch size for ONNX feature extraction (default `128`) |
| `--skip-extract` | Skip feature extraction and immediately train on cached `.npy` |
| `--out-dir DIR` | Target output directory (default: `output/`; use `output_mlp512x2/` for reference) |

Output Directories:
- **`code/current/mobilenet/color_distillation/output_mlp512x2/` [REFERENCE PRODUCTION MODEL]**:
  - `mobilenet_v2_multitask.onnx` (Unified ONNX model, 22.4 MB, DVC-tracked)
  - `color_head_best.pt` (Deep MLP-512×2 PyTorch state dict, 1.4 MB)
  - `eval_metrics.json`, `training_history.csv`, `final_report.txt`
- `code/current/mobilenet/color_distillation/output/` [HISTORICAL BASELINE]:
  - `mobilenet_v2_multitask.onnx` (128-unit weighted ONNX, 20.61 MB)
  - `color_head_best.pt` (128-unit PyTorch state dict, 93 KB)

## Building the External Splits

Only needed when the External splits change (they are committed in `run4/splits_1235models/`):

```bash
python run4/prepare_external_1235models.py
```

---

## DVC (Data Version Control) with Google Drive

Used to track heavy datasets, Qwen embeddings (`embeddings.npy`, 2.7 GB), feature caches (`features_L16_train.npy`, 263 MB), and large ONNX models without bloating Git or exceeding GitHub's 100 MB limit.

A convenience runner is located at `code/current/mobilenet/bash_scripts/dvc.sh`:

```bash
# Add a large dataset or directory to DVC
./code/current/mobilenet/bash_scripts/dvc.sh add code/current/mobilenet/annotation_runs/run2
./code/current/mobilenet/bash_scripts/dvc.sh add code/current/mobilenet/color_distillation/cache

# Push data to Google Drive (standard)
./code/current/mobilenet/bash_scripts/dvc.sh push

# Ephemeral Push for Shared Research PC (keeps Client ID, Secret, and OAuth token strictly in RAM / Temp storage)
export DVC_GDRIVE_CLIENT_ID="<your_client_id>"
export DVC_GDRIVE_CLIENT_SECRET="<your_client_secret>"
./code/current/mobilenet/bash_scripts/dvc_ephemeral.sh push

# Pull data on a new machine
./code/current/mobilenet/bash_scripts/dvc.sh pull
```
See the full guide in [`dvc_gdrive_guide.md`](../run_docs/dvc_gdrive_guide.md).

### Automated Git & DVC Integration
- **Zero Raw Binaries in Git**: `.gitignore` globally ignores `*.onnx`, `*.npy`, `*.h5`, `*.keras`, and `annotation_runs/run*/`.
- **Automatic DVC Registration**: When any runner finishes and calls `auto_git_sync.sh`, it automatically checks for new `.onnx` models and new `annotation_runs/run*` directories and runs `dvc add` on them before committing `.dvc` tracking files to Git.
- **Safety Guardrail**: `auto_git_sync.sh` contains an automated pre-commit hook that verifies staged files and blocks the commit if any raw binary files $> 50\text{ MB}$ or `*.onnx` files are staged directly into Git.


