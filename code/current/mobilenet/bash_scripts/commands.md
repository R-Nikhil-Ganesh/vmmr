# Commands

Run everything from `vmmr/code/current/mobilenet/`. There are five runners, all in `bash_scripts/`. **Each one starts in a detached tmux session by default** and keeps the pane open afterwards. Training and diagnostics are meant for the GPU server, where the data lives.

| I want to... | Command | tmux session |
|---|---|---|
| Train A, train B, then diagnostics + benchmark | `bash bash_scripts/run_all_pipeline.sh [trainer flags]` | `mobilenet_pipeline` |
| Train Model A (PlatesMania) only | `bash bash_scripts/run_train_pm.sh [trainer flags]` | `train_pm` |
| Train Model B (External) only | `bash bash_scripts/run_train_external.sh [trainer flags]` | `train_ext` |
| Train Model C (PlatesMania + External merged) | `bash bash_scripts/run_train_merged.sh [flags]` | `train_merged` |
| Diagnostics + cross-domain benchmark | `bash bash_scripts/run_diagnostics.sh [options]` | `diagnostics` |

`bash_scripts/lib/` holds helpers that are not run directly: `paths.sh` (all machine paths), `tmux_wrap.sh`, `auto_git_sync.sh`.

## Working with tmux

```bash
tmux attach -t train_pm            # watch a run (detach: Ctrl+B, then D)
tmux ls                            # list sessions
tmux kill-session -t train_pm      # stop a run
```

A second launch with a session name that is already running is refused (it prints the attach/kill commands).

## Environment variables (set in front of any runner)

| Variable | Effect |
|---|---|
| `RUN_NAME=<name>` | Separate output folders and tmux session per experiment, so runs do not overwrite each other: `platesmania_dataset/output_<name>`, `external_dataset/output_external_<name>`, `mixed_benchmark_results_<name>`, session `<session>_<name>`. Use the same name for training and diagnostics. Without it, the default `output_mobilenet_v2*` folders are used. |
| `NO_TMUX=1` | Run in the foreground instead of tmux (debugging, cron, CI). |
| `AUTO_GIT_PUSH=0` | Do not commit and push results when finished (default `1`: commits `code/current/mobilenet/` and pushes). |
| `CUDA_VISIBLE_DEVICES=1` | Choose the GPU. |
| `ECON_ROOT`, `PM_IMG_DIR`, `PM_MANIFEST_CSV`, `PM_LABEL_MAP`, `EXT_DATASETS_DIR`, `PYTHON_BIN` | Override a path for one run. Permanent per-server values go in the gitignored `bash_scripts/lib/paths.local.sh`. |

Example: `RUN_NAME=aug AUTO_GIT_PUSH=0 bash bash_scripts/run_train_pm.sh --aug-strength strong`

## Moving to another server

1. Create `bash_scripts/lib/paths.local.sh` with only what differs, for example:
   ```bash
   ECON_ROOT=/data/Econ                       # PM_IMG_DIR, PM_MANIFEST_CSV, PM_LABEL_MAP, EXT_DATASETS_DIR derive from it
   PYTHON_BIN=/opt/envs/pt-env/bin/python
   ```
2. Check: `python paths.py --check` (prints each resolved path and whether it exists).

Precedence: shell environment > `paths.local.sh` > defaults in `paths.sh`.

---

## `run_train_pm.sh` and `run_train_external.sh`

Train one model on its own dataset only (no cross-training). Output: `models/mobilenet_v2_best.onnx` (3 outputs: predictions, embeddings, class_maps), `models/label_map.json`, `reports/` (history, `test_metrics.json`), `plots/training_curves.png`, `training.log`.

Any extra arguments are passed to the trainer and **override** the runner's defaults (the last occurrence wins). Runner defaults: 512 px, batch 32, 15 epochs, lr 5e-4 (head) / 5e-5 (backbone), dropout 0.25, 5 unfrozen blocks; PlatesMania also `--crop-top-pct 0.15`, External `--crop-bottom-pct 0.05`.

Generalization options (shared by both trainers through `train_common.py`; all default to the old behaviour):

| Flag | Default | What it does |
|---|---|---|
| `--aug-strength {base,strong}` | `base` | `strong` = wide zoom and aspect, rotation, translation, stronger colour jitter, grayscale, blur, noise, random erasing. Aims to make the model robust to tight (bbox) versus loose (full frame) framing. |
| `--crop-jitter F` | `0` | 0 to 1. Randomizes the training crop (PlatesMania top crop, External bbox margin and bottom crop). Validation and test are never jittered. |
| `--label-smoothing F` | `0` | Label smoothing on the training loss (e.g. `0.1`). Validation loss stays plain cross-entropy, so checkpoints stay comparable across runs. |
| `--make-loss-weight F` | `0` | Weight of an auxiliary loss on P(make) = sum of P(model) over the make's models (e.g. `0.3`). |
| `--ema-decay F` | `0` | EMA of the weights (e.g. `0.999`). Validation, best-checkpoint selection and the ONNX export use the EMA weights. |

Common trainer flags:

| Flag | What it does |
|---|---|
| `--epochs N`, `--batch-size N`, `--eval-batch-size N` | Schedule and batch sizes. |
| `--lr F`, `--backbone-lr F` | Learning rates for the head and the unfrozen backbone blocks. |
| `--dropout F`, `--unfreeze-layers N` | Head dropout; number of top `features` blocks trained. |
| `--img-size N` | Input resolution. Evaluation must use the same value. |
| `--num-workers N` | DataLoader workers. |
| `--seed N` | Random seed (default 42). |
| `--crop-top-pct F`, `--crop-bottom-pct F` | Fixed crops (PlatesMania watermark banner / External bottom strip). |
| `--output-dir`, `--csv-path`, `--label-map-path`, `--base-img-dir`, `--splits-dir` | Locations. The runner already sets them from `paths.sh`; change only for special cases. |

Examples:

```bash
bash bash_scripts/run_train_pm.sh                                       # baseline
RUN_NAME=aug bash bash_scripts/run_train_pm.sh --aug-strength strong --crop-jitter 0.5
RUN_NAME=aug bash bash_scripts/run_train_external.sh --aug-strength strong --crop-jitter 0.5
bash bash_scripts/run_train_pm.sh --epochs 1 --num-workers 2            # quick sanity run
```

## `run_train_merged.sh` (Model C: PlatesMania + External)

A separate experiment: Model C trains on **both** datasets, while A and B stay single-dataset. Use it to see how far joint training gets compared with the cross-dataset numbers of A and B. Each image is cropped as in its own dataset (PlatesMania top 15%, External bbox / bottom 5%), and both share the 1,235-class label space (the script stops if the label maps differ).

External has about 30k train images against about 1M for PlatesMania, so plain concatenation would make it negligible. The trainer therefore samples a fixed share of External images in every epoch. An epoch has as many samples as PlatesMania's train split (same cost as Model A). Checkpoints are selected on the **mean of the two validation losses** (each domain weighted equally), and the final test is reported per domain in `reports/test_metrics.json`.

Output: `merged_dataset/output_mobilenet_v2_merged/` (or `output_merged_<RUN_NAME>`), same layout as A and B, with per-domain validation columns in `training_history.csv`.

Takes all the generalization options and common trainer flags above, plus:

| Flag | Default | What it does |
|---|---|---|
| `--ext-fraction F` | `0.2` | Share of External images per epoch. `0` = plain concatenation (about 3% External). At 0.2 each External image is drawn about 6 times per epoch, so combine with `--aug-strength strong` to limit memorizing. |
| `--pm-crop-top-pct F` | `0.15` | PlatesMania top crop. |
| `--ext-crop-bottom-pct F` | `0.05` | External bottom crop (used when no bbox). |
| `--pm-csv-path`, `--pm-img-dir`, `--ext-splits-dir`, `--label-map-path` | from `paths.sh` | Locations. |

```bash
bash bash_scripts/run_train_merged.sh
RUN_NAME=m50 bash bash_scripts/run_train_merged.sh --ext-fraction 0.5 --aug-strength strong --crop-jitter 0.5
bash bash_scripts/run_diagnostics.sh --merged            # then diagnose it (add the same RUN_NAME)
```

## `run_diagnostics.sh`

Per model: classification reports (model level and make level), confusion matrices (model and make), calibration curve, t-SNE/PCA of the embeddings, `test_metrics.json`. Then the cross-domain benchmark (in-domain, out-of-domain, mixed; 5,000 images per dataset) with the shared-class view, written to `mixed_benchmark_results[_<RUN_NAME>]/` (`mixed_benchmark_summary.csv`, `shared_class_summary.csv`, `class_coverage.csv`, `cross_domain_comparison.png`).

| Option | What it does |
|---|---|
| (none) | Model A, Model B and the benchmark, on the full test sets. |
| `--quick` | Same, but at most 5,000 test images per model diagnostic (the benchmark already samples 5,000). |
| `--max-samples N` | Cap the test images per model diagnostic at N. |
| `--pm-only` | Only Model A diagnostics. |
| `--ext-only` | Only Model B diagnostics. |
| `--bench-only` | Only the cross-domain benchmark. |
| `--merged` | Only Model C (see below). |

Use the same `RUN_NAME` as the training run to diagnose that run's models.

`--merged` diagnoses Model C: the PlatesMania and External test splits are reported separately (`merged_dataset/output_.../diagnostics_platesmania/` and `diagnostics_external/`), and the same 5,000-image benchmark as A/B is written to `.../benchmark/`. In that benchmark the "Model A" and "Model B" rows are both the merged model, so read the in-domain and out-of-domain labels as simply "PlatesMania test" and "External test".

```bash
bash bash_scripts/run_diagnostics.sh
bash bash_scripts/run_diagnostics.sh --bench-only
RUN_NAME=aug bash bash_scripts/run_diagnostics.sh --quick
```

## `run_all_pipeline.sh`

Runs, in order: train Model A, train Model B, diagnostics + benchmark for both. Extra arguments go to **both** trainers, so only use flags both accept (the five generalization flags and the common ones; `--crop-top-pct` is PlatesMania only). It pushes to git once at the end, not after every step. Stops at the first failing step.

```bash
bash bash_scripts/run_all_pipeline.sh
RUN_NAME=aug bash bash_scripts/run_all_pipeline.sh --aug-strength strong --crop-jitter 0.5
```

## Not a runner: building the External splits

Only needed when the External splits change (they are committed). Run on the server:

```bash
python external_dataset/prepare_external_1235models.py
```

Raw dataset locations come from `EXT_DATASETS_DIR` in `paths.sh`; the output is `external_dataset/splits_1235models/`.
