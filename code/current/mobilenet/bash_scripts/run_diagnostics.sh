#!/usr/bin/env bash
# Diagnostics for the merged MobileNetV2 model (Model C): per-domain reports, confusion matrices (model + make),
# calibration, t-SNE, and the 5,000-image per-domain / mixed benchmark. Runs in a tmux session by default.
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
        --merged)      shift ;;
        *) echo "Unknown option: $1" >&2
           echo "Usage: bash run_diagnostics.sh [--quick] [--max-samples N] [--pm-only] [--ext-only] [--bench-only] [--merged]" >&2
           exit 1 ;;
    esac
done

# RUN_NAME=<name> diagnoses merged_dataset/output_merged_<name> (defaults to output_merged_run4)
MERGED_RUN_DIR="${MOBILENET_DIR}/merged_dataset/output_${RUN_NAME:+merged_}${RUN_NAME:-merged_run4}"
BENCH_DIR="${MOBILENET_DIR}/mixed_benchmark_results${RUN_NAME:+_${RUN_NAME}}"
M_MODEL="${MERGED_RUN_DIR}/models/mobilenet_v2_best.onnx"
M_MAP="${MERGED_RUN_DIR}/models/label_map.json"
EXT_SPLITS_DIR="${MOBILENET_DIR}/merged_dataset/splits_1235models"

echo "=================================================================="
echo "  MobileNetV2 diagnostics (Model C Merged)   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Run Dir: ${MERGED_RUN_DIR}"
echo "  Python:  ${PYTHON_BIN}"
echo "=================================================================="

if [ "${RUN_PM}" -eq 1 ]; then
    echo; echo "=== Model C (merged) on the PlatesMania test split ==="
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/merged_dataset/analysis/output_analysis_merged.py" \
        --domain platesmania \
        --test-csv "${PM_MANIFEST_CSV}" --base-img-dir "${PM_IMG_DIR}" \
        --model-path "${M_MODEL}" --label-map "${M_MAP}" \
        --output-dir "${MERGED_RUN_DIR}/diagnostics_platesmania" \
        --batch-size 64 --num-workers 10 "${MAX_SAMPLES[@]}"
fi

if [ "${RUN_EXT}" -eq 1 ]; then
    echo; echo "=== Model C (merged) on the External test split ==="
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/merged_dataset/analysis/output_analysis_merged.py" \
        --domain external \
        --test-csv "${EXT_SPLITS_DIR}/test.csv" \
        --model-path "${M_MODEL}" --label-map "${M_MAP}" \
        --output-dir "${MERGED_RUN_DIR}/diagnostics_external" \
        --batch-size 64 --num-workers 10 "${MAX_SAMPLES[@]}"
fi

if [ "${RUN_BENCH}" -eq 1 ]; then
    echo; echo "=== Model C mixed benchmark (5,000-image samples per domain) ==="
    mkdir -p "${MERGED_RUN_DIR}/benchmark" "${BENCH_DIR}"
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/evaluate_mixed.py" \
        --pm-model-path "${M_MODEL}" --pm-label-map "${M_MAP}" \
        --ext-model-path "${M_MODEL}" --ext-label-map "${M_MAP}" \
        --pm-test-csv "${PM_MANIFEST_CSV}" --pm-img-dir "${PM_IMG_DIR}" \
        --ext-test-csv "${EXT_SPLITS_DIR}/test.csv" \
        --ext-train-csv "${EXT_SPLITS_DIR}/train.csv" \
        --output-dir "${MERGED_RUN_DIR}/benchmark" \
        --img-size 512 --batch-size 64 --num-workers 10 --max-eval-per-dataset 5000
    cp -f "${MERGED_RUN_DIR}/benchmark/"* "${BENCH_DIR}/"
fi

echo; echo "  Diagnostics finished   $(date '+%Y-%m-%d %H:%M:%S')"

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Diagnostics${RUN_NAME:+ (${RUN_NAME})}"
fi
