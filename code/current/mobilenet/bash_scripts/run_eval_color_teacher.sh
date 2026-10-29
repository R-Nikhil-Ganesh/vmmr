#!/usr/bin/env bash
# Score the Qwen3-VL color teacher against CompCars human color labels (annotation/eval_color_teacher.py).
# Runs in a tmux session by default. Output: ${ANNOT_OUT_DIR}/teacher_eval/<tag>/ plus teacher_eval/summary.csv.
#
#   RUN_NAME=bf16_letters bash bash_scripts/run_eval_color_teacher.sh --no-4bit --scoring letters   # one config
#   SWEEP=1 bash bash_scripts/run_eval_color_teacher.sh                                             # one-factor sweep on val
#   SWEEP=1 bash bash_scripts/run_eval_color_teacher.sh --max-samples 200                           # dry run of the sweep
# Extra flags are appended to every configuration of the sweep.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux eval_color_teacher "${SCRIPT_DIR}/run_eval_color_teacher.sh" "$@"

[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

run_one() {
    local tag="$1"; shift
    echo; echo "=== ${tag}: $*   $(date '+%Y-%m-%d %H:%M:%S')"
    "${PYTHON_BIN}" -u "${MOBILENET_DIR}/annotation/eval_color_teacher.py" \
        --tag "${tag}" --qwen-model "${QWEN_MODEL_DIR:-Qwen/Qwen3-VL-8B-Instruct}" "$@"
}

if [ "${SWEEP:-0}" = "1" ]; then
    # Each step changes one thing relative to s2_variants (bf16 + case-insensitive scoring); combine the winners after
    BASE=(--no-4bit --scoring variants)
    run_one s0_original                                          "$@"   # Annotation Runs 1/2 settings
    run_one s1_bf16              --no-4bit                       "$@"
    run_one s2_variants          "${BASE[@]}"                    "$@"
    run_one s2_letters           --no-4bit --scoring letters     "$@"
    run_one s3_side672           "${BASE[@]}" --max-side 672     "$@"
    run_one s3_side1024          "${BASE[@]}" --max-side 1024 --batch-size 4 "$@"
    run_one s3_center15          "${BASE[@]}" --center-crop 0.15 "$@"
    run_one s4_defined           "${BASE[@]}" --prompt-style defined "$@"
    run_one s5_flip              "${BASE[@]}" --tta flip         "$@"
    echo; column -s, -t < "${ANNOT_OUT_DIR}/teacher_eval/summary.csv" || cat "${ANNOT_OUT_DIR}/teacher_eval/summary.csv"
else
    run_one "${RUN_NAME:?set RUN_NAME=<tag> for a single configuration, or SWEEP=1}" "$@"
fi

echo; echo "  Done. Results: ${ANNOT_OUT_DIR}/teacher_eval/summary.csv   $(date '+%Y-%m-%d %H:%M:%S')"

if [ "${AUTO_GIT_PUSH:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Color teacher eval${RUN_NAME:+ (${RUN_NAME})}"
fi
