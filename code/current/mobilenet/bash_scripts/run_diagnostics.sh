#!/usr/bin/env bash
# Diagnostics for trained models: per-model reports, confusion matrices (model + make), calibration, t-SNE,
# and the cross-domain / mixed benchmark. Runs in a tmux session by default. See commands.md.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux diagnostics "${SCRIPT_DIR}/run_diagnostics.sh" "$@"

[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

RUN_PM=1; RUN_EXT=1; RUN_BENCH=1; MAX_SAMPLES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)       MAX_SAMPLES=(--max-samples 5000); shift ;;
        --max-samples) MAX_SAMPLES=(--max-samples "$2"); shift 2 ;;
        --pm-only)     RUN_EXT=0; RUN_BENCH=0; shift ;;
        --ext-only)    RUN_PM=0; RUN_BENCH=0; shift ;;
        --bench-only)  RUN_PM=0; RUN_EXT=0; shift ;;
        *) echo "Unknown option: $1" >&2
           echo "Usage: bash run_diagnostics.sh [--quick] [--max-samples N] [--pm-only] [--ext-only] [--bench-only]" >&2
           exit 1 ;;
    esac
done

# RUN_NAME=<name> diagnoses the models trained with the same RUN_NAME (see run_train_*.sh)
PM_RUN_DIR="${MOBILENET_DIR}/platesmania_dataset/output_${RUN_NAME:-mobilenet_v2}"
EXT_RUN_DIR="${MOBILENET_DIR}/external_dataset/output_${RUN_NAME:+external_}${RUN_NAME:-mobilenet_v2_external}"
BENCH_DIR="${MOBILENET_DIR}/mixed_benchmark_results${RUN_NAME:+_${RUN_NAME}}"

echo "=================================================================="
echo "  MobileNetV2 diagnostics   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model A: ${PM_RUN_DIR}"
echo "  Model B: ${EXT_RUN_DIR}"
echo "  Python:  ${PYTHON_BIN}"
echo "=================================================================="

if [ "${RUN_PM}" -eq 1 ]; then
    echo; echo "=== Model A (PlatesMania) diagnostics ==="
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/platesmania_dataset/analysis/output_analysis_platesmania.py" \
        --test-csv "${PM_MANIFEST_CSV}" --base-img-dir "${PM_IMG_DIR}" \
        --model-path "${PM_RUN_DIR}/models/mobilenet_v2_best.onnx" \
        --label-map "${PM_RUN_DIR}/models/label_map.json" \
        --output-dir "${PM_RUN_DIR}" \
        --batch-size 64 --num-workers 6 "${MAX_SAMPLES[@]}"
fi

if [ "${RUN_EXT}" -eq 1 ]; then
    echo; echo "=== Model B (External) diagnostics ==="
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/external_dataset/analysis/output_analysis_external.py" \
        --model-path "${EXT_RUN_DIR}/models/mobilenet_v2_best.onnx" \
        --label-map "${EXT_RUN_DIR}/models/label_map.json" \
        --output-dir "${EXT_RUN_DIR}" \
        --batch-size 64 --num-workers 6 "${MAX_SAMPLES[@]}"
fi

if [ "${RUN_BENCH}" -eq 1 ]; then
    echo; echo "=== Cross-domain & mixed benchmark ==="
    mkdir -p "${BENCH_DIR}"
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/evaluate_mixed.py" \
        --pm-model-path "${PM_RUN_DIR}/models/mobilenet_v2_best.onnx" \
        --pm-label-map "${PM_RUN_DIR}/models/label_map.json" \
        --ext-model-path "${EXT_RUN_DIR}/models/mobilenet_v2_best.onnx" \
        --ext-label-map "${EXT_RUN_DIR}/models/label_map.json" \
        --pm-test-csv "${PM_MANIFEST_CSV}" --pm-img-dir "${PM_IMG_DIR}" \
        --ext-test-csv "${MOBILENET_DIR}/external_dataset/splits_1235models/test.csv" \
        --output-dir "${BENCH_DIR}" \
        --img-size 512 --batch-size 64 --max-eval-per-dataset 5000
fi

echo; echo "  Diagnostics finished   $(date '+%Y-%m-%d %H:%M:%S')"

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Diagnostics${RUN_NAME:+ (${RUN_NAME})}"
fi
