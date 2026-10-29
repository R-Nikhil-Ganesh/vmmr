#!/usr/bin/env bash
# ==============================================================================
# Auto Git Sync Helper for MobileNetV2 Pipelines
# ==============================================================================
# Stages relevant code, models, reports, and plots, commits and pushes to origin.
# Usage: bash bash_scripts/lib/auto_git_sync.sh "Task / Experiment Description"
# ==============================================================================

set -e

TASK_DESC="${1:-MobileNetV2 Training Run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Source paths to get DVC_BIN and environment configuration
if [ -f "${SCRIPT_DIR}/paths.sh" ]; then
    source "${SCRIPT_DIR}/paths.sh"
fi

cd "${REPO_ROOT}"

echo "=================================================================="
echo "  [Git & DVC Auto-Sync] Syncing changes"
echo "  Repository: ${REPO_ROOT}"
echo "  Task:       ${TASK_DESC}"
echo "  Timestamp:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

# 1. Automatic DVC Tracking for Large Artifacts (Future Run Protection)
DVC_CMD="${DVC_BIN:-dvc}"
if command -v "${DVC_CMD}" >/dev/null 2>&1 || [ -x "${DVC_CMD}" ]; then
    # Auto-register any new .onnx models in DVC (skip scratch/git-ignored directories)
    while IFS= read -r onnx_path; do
        if [ -n "${onnx_path}" ] && [ ! -f "${onnx_path}.dvc" ]; then
            if git check-ignore -q "${onnx_path}.dvc" 2>/dev/null; then
                continue
            fi
            echo "  [DVC Auto-Track] Tracking new model in DVC: ${onnx_path}"
            "${DVC_CMD}" add "${onnx_path}"
        fi
    done < <(find code/current/mobilenet/ -type d \( -name "*scratch*" -o -name ".*" \) -prune -o -type f -name "*.onnx" -print 2>/dev/null || true)

    # Auto-register any new annotation_runs/run* directories in DVC
    while IFS= read -r run_dir; do
        if [ -n "${run_dir}" ] && [ ! -f "${run_dir}.dvc" ]; then
            if git check-ignore -q "${run_dir}.dvc" 2>/dev/null; then
                continue
            fi
            echo "  [DVC Auto-Track] Tracking new annotation run directory in DVC: ${run_dir}"
            "${DVC_CMD}" add "${run_dir}"
        fi
    done < <(find code/current/mobilenet/annotation_runs/ -mindepth 1 -maxdepth 1 -type d -name "run*" 2>/dev/null || true)

    # Ensure color distillation cache is up to date in DVC
    if [ -d "code/current/mobilenet/color_distillation/cache" ]; then
        "${DVC_CMD}" add code/current/mobilenet/color_distillation/cache >/dev/null 2>&1 || true
    fi
else
    echo "  [WARN] DVC binary not found. Skipping automated DVC check."
fi

# 2. Stage all mobilenet tracking files, code, configs, reports, plots, and .dvc pointers
git add -A code/current/mobilenet/ .gitignore

# 3. Guardrail: Verify no raw binaries > 50 MB or .onnx files slipped into git staging
STAGED_VIOLATIONS=$(git diff --cached --name-only | grep -E '\.(onnx|npy|h5|keras)$' || true)
if [ -n "${STAGED_VIOLATIONS}" ]; then
    echo "  [ERROR] Safety guardrail triggered! The following binary files cannot be committed directly to Git:" >&2
    echo "${STAGED_VIOLATIONS}" | sed 's/^/    /' >&2
    echo "  Please track them with DVC: ./code/current/mobilenet/bash_scripts/dvc.sh add <file>" >&2
    git reset HEAD ${STAGED_VIOLATIONS}
    exit 1
fi

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
