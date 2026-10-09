#!/usr/bin/env bash
# ==============================================================================
# Auto Git Sync Helper for MobileNetV2 Pipelines
# ==============================================================================
# Stages relevant code, models, reports, and plots, commits and pushes to origin.
# Usage: bash auto_git_sync.sh "Task / Experiment Description"
# ==============================================================================

set -e

TASK_DESC="${1:-MobileNetV2 Training Run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

cd "${REPO_ROOT}"

echo "=================================================================="
echo "  [Git Auto-Sync] Syncing changes to GitHub"
echo "  Repository: ${REPO_ROOT}"
echo "  Task:       ${TASK_DESC}"
echo "  Timestamp:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# Stage all mobilenet files (code, configs, reports, plots, onnx models, manifests)
# Note: *__pycache__, *.btr, and *paths.local.sh are strictly handled by .gitignore
git add -A code/current/mobilenet/

# Check if there is anything to commit
if git diff --staged --quiet; then
    echo "  [Git Auto-Sync] No staged changes detected. Working tree clean."
    exit 0
fi

# Show staged summary
echo "  [Git Auto-Sync] Staged changes:"
git diff --staged --stat | sed 's/^/    /'

BRANCH="$(git branch --show-current 2>/dev/null || echo "main")"
COMMIT_MSG="feat(mobilenet): auto-sync ${TASK_DESC} [$(date '+%Y-%m-%d %H:%M:%S')]"

git commit -m "${COMMIT_MSG}"

echo "  [Git Auto-Sync] Pushing to origin ${BRANCH}..."
if git push origin "${BRANCH}"; then
    echo "  [Git Auto-Sync] Push successful! Commit: $(git rev-parse --short HEAD)"
else
    echo "  [WARN] Direct push failed. Attempting git pull --rebase origin ${BRANCH}..."
    if git pull --rebase origin "${BRANCH}" && git push origin "${BRANCH}"; then
        echo "  [Git Auto-Sync] Rebase and push succeeded! Commit: $(git rev-parse --short HEAD)"
    else
        echo "  [ERROR] Git push failed. Please inspect manually." >&2
        exit 1
    fi
fi
echo "=================================================================="
