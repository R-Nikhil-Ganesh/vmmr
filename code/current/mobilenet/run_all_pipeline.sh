#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_RUNNER="${SCRIPT_DIR}/platesmania_dataset/train/run_train.sh"
EXT_RUNNER="${SCRIPT_DIR}/external_dataset/train/run_train_external.sh"
EVAL_RUNNER="${SCRIPT_DIR}/run_eval_mixed.sh"

echo "=================================================================="
echo "  Starting Full MobileNetV2 End-to-End Pipeline"
echo "  Started: $(date)"
echo "=================================================================="

echo "=== [Step 1/3] Training Model A (PlatesMania) ==="
if ! bash "${PM_RUNNER}"; then
    echo "[ERROR] Step 1 (PlatesMania) failed! Check logs above."
    exit 1
fi

echo "=== [Step 2/3] Training Model B (External Merged) ==="
if ! bash "${EXT_RUNNER}"; then
    echo "[ERROR] Step 2 (External) failed! Check logs above."
    exit 1
fi

echo "=== [Step 3/3] Running Cross-Domain & Mixed Benchmark ==="
if ! bash "${EVAL_RUNNER}"; then
    echo "[ERROR] Step 3 (Evaluation) failed! Check logs above."
    exit 1
fi

echo "=================================================================="
echo "=== All MobileNetV2 Training & Evaluation Completed! ==="
echo "  Finished: $(date)"
echo "=================================================================="
