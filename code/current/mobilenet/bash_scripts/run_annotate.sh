#!/usr/bin/env bash
# Annotate test images with make/model (merged Model C) and color (Qwen3-VL-8B), then build the HTML viewer.
# Runs in a tmux session by default. Output goes outside the git tree: ${ANNOT_OUT_DIR}/<RUN_NAME or run4>/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux annotate "${SCRIPT_DIR}/run_annotate.sh" "$@"

[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

# RUN_NAME=<name> names the output folder. The model comes from merged_dataset/output_merged_<MODEL_RUN>; MODEL_RUN
# defaults to RUN_NAME when that model folder exists, otherwise run4 (so RUN_NAME=cam uses the run4 model).
MODEL_RUN="${MODEL_RUN:-${RUN_NAME:-run4}}"
[ -d "${MOBILENET_DIR}/merged_dataset/output_merged_${MODEL_RUN}" ] || MODEL_RUN=run4
MERGED_RUN_DIR="${MOBILENET_DIR}/merged_dataset/output_merged_${MODEL_RUN}"
OUT_DIR="${ANNOT_OUT_DIR}/${RUN_NAME:-run4}"

echo "=================================================================="
echo "  Annotation (make/model + color)   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:  ${MERGED_RUN_DIR}/models/mobilenet_v2_best.onnx"
echo "  Output: ${OUT_DIR}"
echo "  Python: ${PYTHON_BIN}"
echo "=================================================================="

"${PYTHON_BIN}" -u "${MOBILENET_DIR}/annotation/annotate.py" \
    --model-path "${MERGED_RUN_DIR}/models/mobilenet_v2_best.onnx" \
    --label-map "${MERGED_RUN_DIR}/models/label_map.json" \
    --output-dir "${OUT_DIR}" \
    --qwen-model "${QWEN_MODEL_DIR}" \
    "$@"

echo; echo "  Done. Viewer: ${OUT_DIR}/viewer.html   $(date '+%Y-%m-%d %H:%M:%S')"
