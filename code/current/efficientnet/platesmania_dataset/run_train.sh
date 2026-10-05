#!/usr/bin/env bash
set -e

# ==============================================================================
# EfficientNet-B0 PlatesMania Dataset (35 Makes) Training Launcher
# ==============================================================================
# Environment: repo-clone/stanford-cars-model/.venv (TensorFlow 2.21 + CUDA)
# Dataset:     resized_640x640/splits_filtered (1,102,776 images, 35 makes)
# Batch Size:  16
# ==============================================================================

REPO_DIR="/home/researchadmin/Econ/repo-clone/stanford-cars-model"
VENV_DIR="$REPO_DIR/.venv"
SCRIPT_PATH="$REPO_DIR/code/current/platesmania_dataset/train_efficientnet_b0.py"
OUTPUT_DIR="$REPO_DIR/code/current/platesmania_dataset/output_efficientnet_b0"
LOG_FILE="$OUTPUT_DIR/training.log"
SPLITS_DIR="/home/researchadmin/Econ/resized_640x640/splits_filtered"

mkdir -p "$OUTPUT_DIR"

# 1. Activate repo virtual environment
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
echo " Starting EfficientNet-B0 Training (PlatesMania Dataset - 35 Makes)"
echo " Virtual Env:      $VIRTUAL_ENV"
echo " Python Binary:    $(which python)"
echo " TensorFlow:       $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo " Training Script:  $SCRIPT_PATH"
echo " Dataset Splits:   $SPLITS_DIR"
echo " Output Directory: $OUTPUT_DIR"
echo " Batch Size:       16"
echo " Log File:         $LOG_FILE"
echo " Start Timestamp:  $(date)"
echo "========================================================================"

python -u "$SCRIPT_PATH" \
    --splits-dir "$SPLITS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --img-size 640 \
    --batch-size 16 \
    --epochs 15 \
    --lr 1e-4 \
    --unfreeze-layers 5 \
    --dropout 0.4 \
    --opset 13 \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================================================"
echo " Training Finished at: $(date)"
echo " Final Checkpoints:    $OUTPUT_DIR/models"
echo "========================================================================"
