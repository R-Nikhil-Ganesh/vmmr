#!/usr/bin/env bash
# Train Model A (PlatesMania). Runs in a tmux session by default. See commands.md.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux train_pm "${SCRIPT_DIR}/run_train_pm.sh" "$@"

DATASET_DIR="${MOBILENET_DIR}/platesmania_dataset"
[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

# RUN_NAME=<name> writes to output_<name> instead of output_mobilenet_v2, so ablations do not overwrite each other.
OUTPUT_DIR="${DATASET_DIR}/output_${RUN_NAME:-mobilenet_v2}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/training.log"

{
echo "=================================================================="
echo "  MobileNetV2 PlatesMania training (Model A)"
echo "  Started: $(date)"
echo "  Python:  ${PYTHON_BIN}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Extra flags: $*"
echo "=================================================================="
} | tee "${LOG_FILE}"

# Extra flags are passed through to train_mobilenet_v2.py (e.g. --aug-strength strong --ema-decay 0.999)
"${PYTHON_BIN}" -u "${DATASET_DIR}/train/train_mobilenet_v2.py" \
    --csv-path "${PM_MANIFEST_CSV}" \
    --label-map-path "${PM_LABEL_MAP}" \
    --base-img-dir "${PM_IMG_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --img-size 512 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --epochs 15 \
    --lr 5e-4 \
    --backbone-lr 5e-5 \
    --dropout 0.25 \
    --unfreeze-layers 5 \
    --crop-top-pct 0.15 \
    --crop-bottom-pct 0.0 \
    --num-workers 10 "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "  Training finished at: $(date)" | tee -a "${LOG_FILE}"

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "PlatesMania Training${RUN_NAME:+ (${RUN_NAME})}" 2>&1 | tee -a "${LOG_FILE}"
fi
