#!/usr/bin/env bash
# ==============================================================================
# Launch MobileNetV2 PlatesMania Training in Detached tmux Session
# ==============================================================================
# Usage:
#   bash start_train_tmux.sh
# To monitor:
#   tmux attach -t train_pm
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="train_pm"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not installed. Run directly with: bash ${SCRIPT_DIR}/run_train.sh" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[WARN] tmux session '${SESSION_NAME}' is already active!"
    echo "  To attach: tmux attach -t ${SESSION_NAME}"
    echo "  To kill:   tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

echo "=================================================================="
echo "  Launching PlatesMania Training in tmux session: ${SESSION_NAME}"
echo "  Runner: ${SCRIPT_DIR}/run_train.sh"
echo "  Auto Git Push: ENABLED (runs upon completion)"
echo "=================================================================="

tmux new-session -d -s "${SESSION_NAME}" -c "${SCRIPT_DIR}" \
    "bash run_train.sh; echo ''; echo 'Training finished. Press enter or close pane.'; exec bash"

echo "  tmux session '${SESSION_NAME}' started successfully in background!"
echo ""
echo "  To attach and monitor training in real time:"
echo "    tmux attach -t ${SESSION_NAME}"
echo ""
echo "  To detach at any time:"
echo "    Press Ctrl+B then D"
echo "=================================================================="
