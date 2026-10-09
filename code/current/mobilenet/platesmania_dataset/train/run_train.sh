#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# MobileNetV2 PlatesMania Training Runner (PyTorch)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROOT_MOBILENET_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Machine-specific paths and Python interpreter (see paths.sh)
source "${ROOT_MOBILENET_DIR}/paths.sh"


if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

# Extra flags are passed through to the trainer (e.g. --aug-strength strong --ema-decay 0.999).
# RUN_NAME=<name> writes to output_<name> instead of output_mobilenet_v2, so ablations do not overwrite each other.
OUTPUT_DIR="${DATASET_DIR}/output_mobilenet_v2"
[ -n "${RUN_NAME:-}" ] && OUTPUT_DIR="${DATASET_DIR}/output_${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/training.log"

echo "==================================================================" | tee "${LOG_FILE}"
echo "  MobileNetV2 PlatesMania PyTorch Training Runner" | tee -a "${LOG_FILE}"
echo "  Started: $(date)" | tee -a "${LOG_FILE}"
echo "  Python:  ${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "  Output:  ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

${PYTHON_BIN} -u "${SCRIPT_DIR}/train_mobilenet_v2.py" \
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

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "  Training finished successfully at: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

if [ "${AUTO_GIT_PUSH:-1}" = "1" ] && [ -f "${ROOT_MOBILENET_DIR}/auto_git_sync.sh" ]; then
    bash "${ROOT_MOBILENET_DIR}/auto_git_sync.sh" "PlatesMania 1235 Make+Model Training" 2>&1 | tee -a "${LOG_FILE}"
fi
