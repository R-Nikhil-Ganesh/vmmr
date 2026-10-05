#!/usr/bin/env bash
set -e

# ==============================================================================
# Detached Tmux Launcher for MobileNetV2 PlatesMania Training
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="mobilenetv2_platesmania_train"
RUN_SCRIPT="${SCRIPT_DIR}/run_train.sh"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[!] Active tmux session '${SESSION_NAME}' already exists."
    echo "    Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

echo "[+] Starting detached tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "bash ${RUN_SCRIPT}; exec bash"

echo "[✓] Session launched successfully!"
echo "    Attach with: tmux attach -t ${SESSION_NAME}"
echo "    Detach with: Ctrl+b, then d"
