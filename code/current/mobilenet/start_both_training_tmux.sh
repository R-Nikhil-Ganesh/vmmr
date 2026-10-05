#!/usr/bin/env bash
set -e

# ==============================================================================
# Master Tmux Launcher for Both MobileNetV2 Models (PlatesMania & External)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="mobilenetv2_all_training"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_all_pipeline.sh"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[!] Active tmux session '${SESSION_NAME}' already exists."
    echo "    Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

echo "[+] Starting master training pipeline in tmux session: ${SESSION_NAME}"
echo "    1. Model A: PlatesMania Dataset (Watermark top-crop 15%, 512x512)"
echo "    2. Model B: External Merged Dataset (35 Makes, 512x512)"
echo "    3. Mixed & Cross-Domain Benchmark (evaluate_mixed.py)"

tmux new-session -d -s "${SESSION_NAME}" "bash ${PIPELINE_SCRIPT}; exec bash"

echo "[✓] Master training session launched successfully!"
echo "    Attach with : tmux attach -t ${SESSION_NAME}"
echo "    Detach with : Ctrl+b, then d"
