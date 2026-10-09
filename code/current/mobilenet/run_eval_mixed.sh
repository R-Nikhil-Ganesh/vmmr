#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Cross-Domain & Mixed Benchmark Evaluation Runner for MobileNetV2
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Machine-specific paths and Python interpreter (see paths.sh)
source "${SCRIPT_DIR}/paths.sh"


if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

# RUN_NAME=<name> evaluates the models trained with the same RUN_NAME and writes to mixed_benchmark_results_<name>
OUTPUT_DIR="${SCRIPT_DIR}/mixed_benchmark_results${RUN_NAME:+_${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

PM_RUN_DIR="${SCRIPT_DIR}/platesmania_dataset/output_${RUN_NAME:-mobilenet_v2}"
EXT_RUN_DIR="${SCRIPT_DIR}/external_dataset/output_${RUN_NAME:+external_}${RUN_NAME:-mobilenet_v2_external}"
PM_MODEL="${PM_RUN_DIR}/models/mobilenet_v2_best.onnx"
PM_MODEL_LABEL_MAP="${PM_RUN_DIR}/models/label_map.json"

EXT_MODEL="${EXT_RUN_DIR}/models/mobilenet_v2_best.onnx"
EXT_LABEL_MAP="${EXT_RUN_DIR}/models/label_map.json"

echo "=================================================================="
echo "  Running Cross-Domain & Mixed Benchmark Evaluation"
echo "  Model A (PlatesMania): ${PM_MODEL}"
echo "  Model B (External):    ${EXT_MODEL}"
echo "  Results Destination:   ${OUTPUT_DIR}"
echo "  Python:                ${PYTHON_BIN}"
echo "=================================================================="

${PYTHON_BIN} -u "${SCRIPT_DIR}/evaluate_mixed.py" \
    --pm-model-path "${PM_MODEL}" \
    --pm-label-map "${PM_MODEL_LABEL_MAP}" \
    --ext-model-path "${EXT_MODEL}" \
    --ext-label-map "${EXT_LABEL_MAP}" \
    --pm-test-csv "${PM_MANIFEST_CSV}" \
    --pm-img-dir "${PM_IMG_DIR}" \
    --ext-test-csv "${SCRIPT_DIR}/external_dataset/splits_1235models/test.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --img-size 512 \
    --batch-size 64 \
    --max-eval-per-dataset 5000

echo "=================================================================="
echo "  Evaluation completed successfully!"
echo "=================================================================="

if [ "${AUTO_GIT_PUSH:-1}" = "1" ] && [ -f "${SCRIPT_DIR}/auto_git_sync.sh" ]; then
    bash "${SCRIPT_DIR}/auto_git_sync.sh" "Mixed Benchmark Evaluation"
fi
