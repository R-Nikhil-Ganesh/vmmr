#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Cross-Domain & Mixed Benchmark Evaluation Runner for MobileNetV2
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "/home/researchadmin/Econ-n/repo-clone/pt-env/bin/python" ]; then
    PYTHON_BIN="/home/researchadmin/Econ-n/repo-clone/pt-env/bin/python"
elif [ -f "${SCRIPT_DIR}/../../../../pt-env/bin/python" ]; then
    PYTHON_BIN="$(cd "${SCRIPT_DIR}/../../../.." && pwd)/pt-env/bin/python"
else
    PYTHON_BIN="$(which python3)"
fi

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

OUTPUT_DIR="${SCRIPT_DIR}/mixed_benchmark_results"
mkdir -p "${OUTPUT_DIR}"

PM_MODEL="${SCRIPT_DIR}/platesmania_dataset/output_mobilenet_v2/models/mobilenet_v2_best.onnx"
PM_LABEL_MAP="${SCRIPT_DIR}/platesmania_dataset/output_mobilenet_v2/models/label_map.json"

EXT_MODEL="${SCRIPT_DIR}/external_dataset/output_mobilenet_v2_external/models/mobilenet_v2_best.onnx"
EXT_LABEL_MAP="${SCRIPT_DIR}/external_dataset/output_mobilenet_v2_external/models/label_map.json"

echo "=================================================================="
echo "  Running Cross-Domain & Mixed Benchmark Evaluation"
echo "  Model A (PlatesMania): ${PM_MODEL}"
echo "  Model B (External):    ${EXT_MODEL}"
echo "  Results Destination:   ${OUTPUT_DIR}"
echo "  Python:                ${PYTHON_BIN}"
echo "=================================================================="

${PYTHON_BIN} -u "${SCRIPT_DIR}/evaluate_mixed.py" \
    --pm-model-path "${PM_MODEL}" \
    --pm-label-map "${PM_LABEL_MAP}" \
    --ext-model-path "${EXT_MODEL}" \
    --ext-label-map "${EXT_LABEL_MAP}" \
    --pm-test-csv "/home/researchadmin/Econ/models/dataset_manifests/dataset_1235models_splits.csv" \
    --pm-img-dir "/home/researchadmin/Econ/resized_640x640" \
    --ext-test-csv "${SCRIPT_DIR}/external_dataset/splits_1235models/test.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --img-size 512 \
    --batch-size 64 \
    --max-eval-per-dataset 5000

echo "=================================================================="
echo "  Evaluation completed successfully!"
echo "=================================================================="
