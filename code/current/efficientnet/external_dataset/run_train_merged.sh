#!/usr/bin/env bash
set -e

# ==============================================================================
# EfficientNet-B0 Merged 35-Make Training Launcher
# ==============================================================================
# Environment: repo-clone/stanford-cars-model/.venv (TensorFlow 2.21 + CUDA)
# Dataset:     external_datasets/merged_data (52,174 peak-frame images, 35 makes)
# ==============================================================================

REPO_DIR="/home/researchadmin/Econ/repo-clone/stanford-cars-model"
VENV_DIR="$REPO_DIR/.venv"
SCRIPT_PATH="$REPO_DIR/code/current/train_efficientnet_b0_merged.py"
OUTPUT_DIR="$REPO_DIR/code/current/output_efficientnet_b0_merged"
LOG_FILE="$OUTPUT_DIR/training.log"
SPLITS_DIR="/home/researchadmin/Econ/external_datasets/merged_data"

mkdir -p "$OUTPUT_DIR"

# 1. Activate strictly the .venv environment created with uv in stanford-cars-model
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[Error] Virtual environment not found at: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# 2. Hardware and cache configuration
export MPLCONFIGDIR="/tmp/matplotlib_cache"
export KERAS_HOME="/home/researchadmin/Econ/.keras"
export TF_CPP_MIN_LOG_LEVEL="1"
export TF_GPU_ALLOCATOR="cuda_malloc_async"

echo "========================================================================"
echo " Starting EfficientNet-B0 Training (35 Vehicle Makes)"
echo " Virtual Env:      $VIRTUAL_ENV"
echo " Python Binary:    $(which python)"
echo " TensorFlow:       $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo " Training Script:  $SCRIPT_PATH"
echo " Dataset Splits:   $SPLITS_DIR"
echo " Output Directory: $OUTPUT_DIR"
echo " Log File:         $LOG_FILE"
echo " Start Timestamp:  $(date)"
echo "========================================================================"

python -u "$SCRIPT_PATH"     --splits-dir "$SPLITS_DIR"     --output-dir "$OUTPUT_DIR"     --img-size 640     --batch-size 8     --epochs 15     --lr 1e-4     --unfreeze-layers 5     --dropout 0.4     --opset 13     2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================================================"
echo " Training Finished at: $(date)"
echo " Final Checkpoints:    $OUTPUT_DIR/models"
echo "========================================================================"
