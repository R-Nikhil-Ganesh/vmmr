#!/usr/bin/env bash
set -e

# ==============================================================================
# Detached tmux Launcher for PlatesMania Dataset Training (35 Makes)
# ==============================================================================

SESSION_NAME="platesmania_train"
WORK_DIR="/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/platesmania_dataset"

# 1. Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[Info] Tmux session '${SESSION_NAME}' is already running!"
    echo "  Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

echo "Starting detached tmux session '${SESSION_NAME}'..."

# 2. Launch training in detached session and keep window open upon completion
tmux new-session -d -s "${SESSION_NAME}" -c "${WORK_DIR}" bash -c "
    bash run_train.sh
    echo ''
    echo '======================================================================'
    echo ' Training completed or stopped. Press Enter to close tmux session.'
    echo '======================================================================'
    read -r
"

echo "========================================================================"
echo "  Tmux session '${SESSION_NAME}' created successfully in background!"
echo "========================================================================"
echo "  * To view live training:  tmux attach -t ${SESSION_NAME}"
echo "  * To detach anytime:      Ctrl + b, then d"
echo "  * To view live log:       tail -f ${WORK_DIR}/output_efficientnet_b0/training.log"
echo "========================================================================"
