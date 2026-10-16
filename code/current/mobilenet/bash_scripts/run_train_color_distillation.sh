#!/usr/bin/env bash
# Distills Qwen3-VL-8B color labels into MobileNetV2 by branching a color head
# from block features[16] (160-d). Exports a unified mobilenet_v2_multitask.onnx.
# Runs inside a detached tmux session by default.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux color_distill "${SCRIPT_DIR}/run_train_color_distillation.sh" "$@"

[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

RUN_NAME="${RUN_NAME:-run5}"
MODEL_RUN="${MODEL_RUN:-run4}"

ANNOTATIONS="${ANNOT_OUT_DIR}/${RUN_NAME}/annotations.csv"
BASE_MODEL="${MOBILENET_DIR}/merged_dataset/output_merged_${MODEL_RUN}/models/mobilenet_v2_best.onnx"
OUT_DIR="${MOBILENET_DIR}/color_distillation/output"

echo "=================================================================="
echo "  MobileNetV2 Color Distillation (features[16] tap)"
echo "  Annotations: ${ANNOTATIONS}"
echo "  Base Model:  ${BASE_MODEL}"
echo "  Output Dir:  ${OUT_DIR}"
echo "  Python:      ${PYTHON_BIN}"
echo "=================================================================="

"${PYTHON_BIN}" -u "${MOBILENET_DIR}/color_distillation/train_color_head.py" \
    --annotations "${ANNOTATIONS}" \
    --model-path "${BASE_MODEL}" \
    --out-dir "${OUT_DIR}" \
    --head-type mlp \
    --epochs 25 \
    --batch-size 512 \
    --extract-batch-size 128 \
    --num-workers 10 \
    "$@"

echo; echo "  Distillation complete! Model: ${OUT_DIR}/mobilenet_v2_multitask.onnx"

