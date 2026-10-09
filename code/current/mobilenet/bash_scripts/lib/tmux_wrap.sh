# ==============================================================================
# Run every long-running script inside a detached tmux session by default.
#
# Usage (near the top of a runner, after sourcing paths.sh):
#     source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
#     run_in_tmux <session_name> "${SCRIPT_DIR}/<this_script>.sh" "$@"
#
# If not already inside tmux, this starts the script in a new detached session and exits;
# the session stays open afterwards so the output remains readable. Otherwise it returns and the script continues.
#
#   NO_TMUX=1 bash run_x.sh   run in the foreground instead (e.g. under cron, CI, or when debugging)
#   RUN_NAME=<name>           also suffixes the session name, so ablations can run side by side
# ==============================================================================

run_in_tmux() {
    local base="$1" script="$2"
    shift 2

    # Already inside tmux (including the session this function starts), or explicitly disabled
    if [ -n "${TMUX:-}" ] || [ "${NO_TMUX:-0}" = "1" ]; then
        return 0
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        echo "[WARN] tmux is not installed; running in the foreground." >&2
        return 0
    fi

    local session="${base}${RUN_NAME:+_${RUN_NAME}}"
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "[ERROR] tmux session '${session}' is already running." >&2
        echo "  Attach: tmux attach -t ${session}" >&2
        echo "  Kill:   tmux kill-session -t ${session}" >&2
        exit 1
    fi

    # The tmux server has its own environment, so forward what the script needs
    local envs="" k a args=""
    for k in PATH VIRTUAL_ENV CUDA_VISIBLE_DEVICES RUN_NAME AUTO_GIT_PUSH ECON_ROOT PM_IMG_DIR PM_MANIFEST_CSV \
             PM_LABEL_MAP EXT_DATASETS_DIR PYTHON_BIN REPO_ROOT MOBILENET_DIR; do
        if [ -n "${!k+x}" ]; then envs="${envs} ${k}=$(printf '%q' "${!k}")"; fi
    done
    for a in "$@"; do args="${args} $(printf '%q' "${a}")"; done

    tmux new-session -d -s "${session}" -c "$(dirname "${script}")" \
        "env${envs} bash $(printf '%q' "${script}")${args}; echo; echo '[finished] press enter to close this pane'; read -r _; exec bash"

    echo "=================================================================="
    echo "  Started in tmux session: ${session}"
    echo "  Attach:  tmux attach -t ${session}      (detach: Ctrl+B then D)"
    echo "  Kill:    tmux kill-session -t ${session}"
    echo "  Run in the foreground instead: NO_TMUX=1 bash ${script}"
    echo "=================================================================="
    exit 0
}
