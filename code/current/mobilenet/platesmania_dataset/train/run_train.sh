#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# MobileNetV2 PlatesMania Training Runner (PyTorch)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "/home/researchadmin/Econ-n/repo-clone/pt-env/bin/python" ]; then
    PYTHON_BIN="/home/researchadmin/Econ-n/repo-clone/pt-env/bin/python"
elif [ -f "${SCRIPT_DIR}/../../../../../../pt-env/bin/python" ]; then
    PYTHON_BIN="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)/pt-env/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

OUTPUT_DIR="${DATASET_DIR}/output_mobilenet_v2"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/training.log"

echo "==================================================================" | tee "${LOG_FILE}"
echo "  MobileNetV2 PlatesMania PyTorch Training Runner" | tee -a "${LOG_FILE}"
echo "  Started: $(date)" | tee -a "${LOG_FILE}"
echo "  Python:  ${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "  Output:  ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"

${PYTHON_BIN} -u "${SCRIPT_DIR}/train_mobilenet_v2.py" \
    --csv-path "/home/researchadmin/Econ/models/dataset_manifests/dataset_1235models_splits.csv" \
    --label-map-path "/home/researchadmin/Econ/models/dataset_manifests/label_map_1235models.json" \
    --base-img-dir "/home/researchadmin/Econ/resized_640x640" \
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
    --num-workers 10 2>&1 | tee -a "${LOG_FILE}"

echo "==================================================================" | tee -a "${LOG_FILE}"
echo "  Training finished successfully at: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================================" | tee -a "${LOG_FILE}"
