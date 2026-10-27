#!/usr/bin/env bash
# Phase 4: Cross-Domain & Shared-Class Benchmark (E-18)
# Evaluates Model A (PlatesMania) and Model B (External) on the regenerated External test split
# and PlatesMania test split, computing shared-class metrics across all 290 shared classes.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"

export CUDA_VISIBLE_DEVICES=0

OUT_DIR="${MOBILENET_DIR}/mixed_benchmark_results_ab"
mkdir -p "${OUT_DIR}"

MODEL_A="${MOBILENET_DIR}/backup_run3/platesmania_dataset/output_mobilenet_v2/models/mobilenet_v2_best.onnx"
MAP_A="${MOBILENET_DIR}/backup_run3/platesmania_dataset/output_mobilenet_v2/models/label_map.json"

MODEL_B="${MOBILENET_DIR}/backup_run3/external_dataset/output_mobilenet_v2_external/models/mobilenet_v2_best.onnx"
MAP_B="${MOBILENET_DIR}/backup_run3/external_dataset/output_mobilenet_v2_external/models/label_map.json"

EXT_TEST="${MOBILENET_DIR}/merged_dataset/splits_1235models/test.csv"
EXT_TRAIN="${MOBILENET_DIR}/merged_dataset/splits_1235models/train.csv"

echo "=================================================================="
echo "  Phase 4: Cross-Domain Benchmark (Model A vs Model B) [E-18]"
echo "  Model A (PlatesMania): ${MODEL_A}"
echo "  Model B (External):    ${MODEL_B}"
echo "  External Test Split:   ${EXT_TEST}"
echo "  Output Directory:      ${OUT_DIR}"
echo "=================================================================="

"${PYTHON_BIN}" -u "${MOBILENET_DIR}/evaluate_mixed.py" \
  --pm-model-path "${MODEL_A}" \
  --pm-label-map "${MAP_A}" \
  --ext-model-path "${MODEL_B}" \
  --ext-label-map "${MAP_B}" \
  --pm-test-csv "${PM_MANIFEST_CSV}" \
  --pm-img-dir "${PM_IMG_DIR}" \
  --ext-test-csv "${EXT_TEST}" \
  --ext-train-csv "${EXT_TRAIN}" \
  --output-dir "${OUT_DIR}" \
  --batch-size 64 \
  --num-workers 10 \
  --max-eval-per-dataset 5205

echo; echo "=================================================================="
echo "  Phase 4 Benchmark Complete! Results saved in:"
echo "  ${OUT_DIR}/mixed_benchmark_summary.csv"
echo "  ${OUT_DIR}/shared_class_summary.csv"
echo "  ${OUT_DIR}/cross_domain_comparison.png"
echo "=================================================================="

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Cross-Domain Benchmark (Phase 4)"
fi

