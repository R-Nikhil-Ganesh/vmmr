#!/usr/bin/env bash
# ==============================================================================
# Launch MobileNetV2 Pipeline in Detached tmux Session with Auto Git Push
# ==============================================================================
# Usage:
#   bash start_pipeline_tmux.sh
# To monitor:
#   tmux attach -t mobilenet_pipeline
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="mobilenet_pipeline"

# Check if tmux is installed
if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux is not installed. Run directly with: bash ${SCRIPT_DIR}/run_all_pipeline.sh" >&2
    exit 1
fi

# If session already exists, notify user
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[WARN] tmux session '${SESSION_NAME}' is already active!"
    echo "  To attach and view logs: tmux attach -t ${SESSION_NAME}"
    echo "  To kill existing session: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

echo "=================================================================="
echo "  Launching MobileNetV2 Pipeline in tmux session: ${SESSION_NAME}"
echo "  Runner: ${SCRIPT_DIR}/run_all_pipeline.sh"
echo "  Auto Git Push: ENABLED (runs upon completion)"
echo "=================================================================="

tmux new-session -d -s "${SESSION_NAME}" -c "${SCRIPT_DIR}" \
    "bash run_all_pipeline.sh; echo ''; echo 'Session complete. Press enter or close pane.'; exec bash"

echo "  tmux session '${SESSION_NAME}' started successfully in background!"
echo ""
echo "  To attach and monitor training in real time:"
echo "    tmux attach -t ${SESSION_NAME}"
echo ""
echo "  To detach at any time:"
echo "    Press Ctrl+B then D"
echo "=================================================================="
