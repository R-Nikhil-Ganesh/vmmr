#!/usr/bin/env bash
# Annotate test images with make/model (merged Model C) and color (Qwen3-VL-8B), then build the HTML viewer.
# Runs in a tmux session by default. Output goes outside the git tree: ${ANNOT_OUT_DIR}/<RUN_NAME or run4>/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux annotate "${SCRIPT_DIR}/run_annotate.sh" "$@"

[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

# MODEL_RUN=<name> sets which model checkpoint to evaluate (defaults to run4)
# RUN_NAME=<name> sets the annotation output folder name (defaults to run5)
MODEL_RUN="${MODEL_RUN:-run4}"
RUN_NAME="${RUN_NAME:-run5}"
MERGED_RUN_DIR="${MOBILENET_DIR}/merged_dataset/output_merged_${MODEL_RUN}"
OUT_DIR="${ANNOT_OUT_DIR}/${RUN_NAME}"

# Cache directory to reuse (defaults to run4 if available)
CACHE_DIR="${CACHE_FROM:-${ANNOT_OUT_DIR}/run4}"
CACHE_FLAG=""
if [ -d "${CACHE_DIR}" ] && [ -f "${CACHE_DIR}/annotations.csv" ]; then
    CACHE_FLAG="--cache-from ${CACHE_DIR}"
fi

echo "=================================================================="
echo "  Annotation (make/model + color)   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:  ${MERGED_RUN_DIR}/models/mobilenet_v2_best.onnx"
echo "  Output: ${OUT_DIR}"
if [ -n "${CACHE_FLAG}" ]; then
    echo "  Cache:  ${CACHE_DIR}"
fi
echo "  Python: ${PYTHON_BIN}"
echo "=================================================================="

"${PYTHON_BIN}" -u "${MOBILENET_DIR}/annotation/annotate.py" \
    --model-path "${MERGED_RUN_DIR}/models/mobilenet_v2_best.onnx" \
    --label-map "${MERGED_RUN_DIR}/models/label_map.json" \
    --output-dir "${OUT_DIR}" \
    --qwen-model "${QWEN_MODEL_DIR}" \
    ${CACHE_FLAG} \
    "$@"

echo; echo "  Done. Viewer: ${OUT_DIR}/viewer.html   $(date '+%Y-%m-%d %H:%M:%S')"

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Annotation Run${RUN_NAME:+ (${RUN_NAME})}"
fi

