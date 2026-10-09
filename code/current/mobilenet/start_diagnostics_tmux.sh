#!/usr/bin/env bash
# ==============================================================================
# Launch Run 3 Diagnostics in Detached tmux Session with Auto Git Push
# ==============================================================================
# Usage:
#   bash start_diagnostics_tmux.sh
# To monitor:
#   tmux attach -t mobilenet_diagnostics
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="mobilenet_diagnostics"

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not installed. Run directly with: bash ${SCRIPT_DIR}/run_diagnostics.sh" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[WARN] tmux session '${SESSION_NAME}' is already active!"
    echo "  To attach and view logs: tmux attach -t ${SESSION_NAME}"
    echo "  To kill existing session: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

echo "=================================================================="
echo "  Launching Run 3 Diagnostics in tmux session: ${SESSION_NAME}"
echo "  Runner: ${SCRIPT_DIR}/run_diagnostics.sh"
echo "  Auto Git Push: ENABLED (runs upon completion)"
echo "=================================================================="

tmux new-session -d -s "${SESSION_NAME}" -c "${SCRIPT_DIR}" \
    "bash run_diagnostics.sh; echo ''; echo 'Diagnostics complete. Press enter or close pane.'; exec bash"

echo "  tmux session '${SESSION_NAME}' started successfully in background!"
echo ""
echo "  To attach and monitor in real time:"
echo "    tmux attach -t ${SESSION_NAME}"
echo ""
echo "  To detach at any time:"
echo "    Press Ctrl+B then D"
echo "=================================================================="

