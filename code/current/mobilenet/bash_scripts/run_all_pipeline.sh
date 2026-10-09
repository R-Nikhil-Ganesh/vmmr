#!/usr/bin/env bash
# Full pipeline: train Model A, train Model B, then diagnostics + cross-domain benchmark for both.
# Runs in a tmux session by default. See commands.md.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux mobilenet_pipeline "${SCRIPT_DIR}/run_all_pipeline.sh" "$@"

echo "=================================================================="
echo "  MobileNetV2 end-to-end pipeline   started $(date)"
echo "  Extra trainer flags (applied to both trainers): $*"
echo "=================================================================="

# Inside tmux already, so the sub-runners do not open further sessions.
# Auto-push is done once at the end instead of after every step.
export AUTO_GIT_PUSH_FINAL="${AUTO_GIT_PUSH:-1}"
export AUTO_GIT_PUSH=0

echo "=== [1/3] Train Model A (PlatesMania) ==="
bash "${SCRIPT_DIR}/run_train_pm.sh" "$@"

echo "=== [2/3] Train Model B (External) ==="
bash "${SCRIPT_DIR}/run_train_external.sh" "$@"

echo "=== [3/3] Diagnostics + cross-domain benchmark ==="
bash "${SCRIPT_DIR}/run_diagnostics.sh"

echo "=================================================================="
echo "  Pipeline finished   $(date)"
echo "=================================================================="

if [ "${AUTO_GIT_PUSH_FINAL}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Full Pipeline${RUN_NAME:+ (${RUN_NAME})}"
fi
