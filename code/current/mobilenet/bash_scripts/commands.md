# Commands

Run everything from `vmmr/code/current/mobilenet/`. The runners in `bash_scripts/` **start in a detached tmux session by default** and keep the pane open afterwards. Training and diagnostics are meant for the GPU server, where the data lives.

| I want to... | Command | tmux session |
|---|---|---|
| Train Model C (merged), then per-domain diagnostics + benchmark | `bash bash_scripts/run_all_pipeline.sh [trainer flags]` | `mobilenet_pipeline` |
| Train Model C (PlatesMania + External merged) | `bash bash_scripts/run_train_merged.sh [flags]` | `train_merged` |
| Diagnostics + mixed benchmark | `bash bash_scripts/run_diagnostics.sh [options]` | `diagnostics` |

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
| `RUN_NAME=<name>` | Separate output folders and tmux session per experiment, so runs do not overwrite each other: `merged_dataset/output_merged_<name>`, `mixed_benchmark_results_<name>`, session `<session>_<name>`. Without it, the default `merged_dataset/output_merged_run4` and `mixed_benchmark_results` folders are used. |
| `NO_TMUX=1` | Run in the foreground instead of tmux (debugging, cron, CI). |
| `AUTO_GIT_PUSH=0` | Do not commit and push results when finished (default `1`: commits `code/current/mobilenet/` and pushes). |
| `CUDA_VISIBLE_DEVICES=1` | Choose the GPU. |
| `ECON_ROOT`, `PM_IMG_DIR`, `PM_MANIFEST_CSV`, `PM_LABEL_MAP`, `EXT_DATASETS_DIR`, `PYTHON_BIN` | Override a path for one run. Permanent per-server values go in the gitignored `bash_scripts/lib/paths.local.sh`. |
| `ANNOT_OUT_DIR`, `QWEN_MODEL_DIR` | Annotation output root (default `${ECON_ROOT}/annotation_runs`) and the Qwen3-VL model id or local directory (see `run_annotate.sh`). |

Example: `RUN_NAME=run5 AUTO_GIT_PUSH=0 bash bash_scripts/run_train_merged.sh --aug-strength strong`

---

## `run_train_merged.sh` (Model C: PlatesMania + External)

Model C trains jointly on **both** datasets across the shared 1,235-class Make/Model taxonomy. Each image is cropped as in its own domain (PlatesMania top 15%, External bbox / bottom 5%).

External has ~30k train images against ~875k for PlatesMania, so `WeightedRandomSampler` draws a fixed share of External images per epoch (`--ext-fraction 0.2` by default). Checkpoints are selected on the **mean of the two domain validation losses**, and final test metrics are reported per domain in `reports/test_metrics.json`.

Output: `merged_dataset/output_merged_run4/` (or `output_merged_<RUN_NAME>`), with `models/mobilenet_v2_best.onnx` (`predictions`, `embeddings`, `class_maps`), `models/label_map.json`, `reports/`, `plots/training_curves.png`, and `training.log`.

| Flag | Default | What it does |
|---|---|---|
| `--ext-fraction F` | `0.2` | Share of External images per epoch. `0` = plain concatenation (~3% External). |
| `--aug-strength {base,strong}` | `base` | `strong` = wide zoom and aspect, rotation, translation, stronger colour jitter, grayscale, blur, noise, random erasing. |
| `--crop-jitter F` | `0` | 0 to 1. Randomizes the training crop (PlatesMania top crop, External bbox margin and bottom crop). |
| `--label-smoothing F` | `0` | Label smoothing on the training loss (e.g. `0.1`). |
| `--make-loss-weight F` | `0` | Weight of an auxiliary loss on P(make) = sum of P(model) over the make's models (e.g. `0.3`). |
| `--ema-decay F` | `0` | EMA of the weights (e.g. `0.999`). |
| `--epochs N`, `--batch-size N`, `--eval-batch-size N` | `15`, `32`, `64` | Schedule and batch sizes. |
| `--lr F`, `--backbone-lr F` | `5e-4`, `5e-5` | Learning rates for the head and the unfrozen backbone blocks. |
| `--dropout F`, `--unfreeze-layers N` | `0.25`, `5` | Head dropout; number of top `features` blocks trained. |
| `--img-size N`, `--num-workers N` | `512`, `10` | Input resolution and DataLoader workers. |

```bash
bash bash_scripts/run_train_merged.sh
RUN_NAME=run5 bash bash_scripts/run_train_merged.sh --ext-fraction 0.2 --aug-strength strong --crop-jitter 0.5
```

---

## `run_diagnostics.sh`

Evaluates Model C on:
1. Full PlatesMania test split (`merged_dataset/output_.../diagnostics_platesmania/`)
2. Full External test split (`merged_dataset/output_.../diagnostics_external/`)
3. 5,000-image per-domain and 10,000-image Mixed benchmark (`merged_dataset/output_.../benchmark/`, synced to `mixed_benchmark_results/`)

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
bash bash_scripts/run_annotate.sh --max-samples 200 --skip-color     # smoke test, no Qwen
bash bash_scripts/run_annotate.sh --max-samples 200                  # smoke test with Qwen
RUN_NAME=run5 bash bash_scripts/run_annotate.sh --split all         # full dataset run with run4 cache reuse
```

Needs `pip install -U "transformers>=4.57" accelerate bitsandbytes` for Qwen; `umap-learn` is used for 2D manifold projection. Output goes to `${ANNOT_OUT_DIR}/<RUN_NAME>/` (default `code/current/mobilenet/annotation_runs/<RUN_NAME>/`): `annotations.csv`, `embeddings.npy`, `color_cache.csv`, `viewer.html`.

| Option | Meaning |
|---|---|
| `--split {test,all,train,val}` | Which split(s) to annotate (default: `test`; use `all` for full dataset) |
| `--cache-from DIR` | Previous run to reuse annotations, embeddings, and colors from (default: `run4` if available) |
| `--domain both\|pm\|ext` | Which dataset(s) (default `both`) |
| `--max-samples N` | Random subsample per domain (default: all split images) |
| `--skip-color` | make/model only |
| `--color-head NAME` | `qwen3vl` (default) |
| `--qwen-model ID_OR_DIR` | Default `$QWEN_MODEL_DIR` (`Qwen/Qwen3-VL-8B-Instruct`); local directory for offline servers |
| `--color-batch-size N` | Qwen batch (default `16`) |
| `--qwen-max-side N` | Max image side for Qwen vision tokens (default: `336` for fast inference) |
| `--no-4bit` | Disable 4-bit NF4 quantization and use bfloat16 |
| `--reuse-model-stage` | Skip the ONNX stage if `annotations.csv` and `embeddings.npy` exist |
| `--coords-max N` | Rows that get 2-D embedding coordinates (default 30000) |
| `--max-points N` | Points with thumbnails embedded in the viewer (default `15000`) |
| `--review-conf` / `--review-color-conf` | Confidence thresholds below which a row is flagged `needs_review` (`0.70` / `0.50`) |
| `--no-viewer` | Skip building the HTML |

To rebuild only the viewer: `python annotation/build_viewer.py --annotations <dir>/annotations.csv --max-points 15000 --workers 16`.

## Building the External Splits

Only needed when the External splits change (they are committed in `merged_dataset/splits_1235models/`):

```bash
python merged_dataset/prepare_external_1235models.py
```
