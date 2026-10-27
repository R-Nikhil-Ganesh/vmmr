#!/usr/bin/env bash
# Phase 3 Color Head Distillation & Evaluation (E-07 & E-08)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"

export CUDA_VISIBLE_DEVICES=0

echo "=================================================================="
echo "  Phase 3 Color Distillation Ablations"
echo "  Python: ${PYTHON_BIN}"
echo "=================================================================="

echo; echo ">>> [1/2] Training Ablation 1: Unweighted 128-unit MLP (with test extraction)..."
"${PYTHON_BIN}" -u "${MOBILENET_DIR}/color_distillation/train_color_head.py" \
  --head-type mlp \
  --no-class-weights \
  --label-smoothing 0.0 \
  --batch-size 2048 \
  --extract-test \
  --extract-batch-size 128 \
  --num-workers 10 \
  --out-dir "${MOBILENET_DIR}/color_distillation/output_unweighted_mlp128"

echo; echo ">>> [2/2] Training Ablation 2: Unweighted Deep 512x2 MLP..."
"${PYTHON_BIN}" -u "${MOBILENET_DIR}/color_distillation/train_color_head.py" \
  --head-type mlp_512x2 \
  --no-class-weights \
  --label-smoothing 0.0 \
  --batch-size 2048 \
  --skip-extract \
  --out-dir "${MOBILENET_DIR}/color_distillation/output_mlp512x2"

echo; echo "=================================================================="
echo "  Phase 3 Complete! Models and reports saved in:"
echo "  1) ${MOBILENET_DIR}/color_distillation/output_unweighted_mlp128"
echo "  2) ${MOBILENET_DIR}/color_distillation/output_mlp512x2"
echo "=================================================================="

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Color Distillation Ablations (Phase 3)"
fi
