#!/usr/bin/env bash
# Run 5 make/model experiment queue (see run_docs/run_5_implementation_plan.md). Runs sequentially in one tmux session.
#   bash bash_scripts/run_run5_ablations.sh [--only R5-0,R5-1] [--skip-diag]
# Each job = RUN_NAME=run5_<id> run_train_merged.sh <flags>, then evaluate_open_set.py and run_diagnostics.sh --merged.
# Every job uses the Run 4 recipe (--taxonomy base, strong aug, crop jitter 0.08, LS 0.1, make loss 0.3, EMA 0.999) + its own flags.
#   R5-0  control (Run 4 recipe on the current data)
#   R5-1  --unfreeze-layers 19 --layer-lr-decay 0.85
#   R5-2  R5-1 + --pool gem --make-gating per_class
#   R5-3  R5-2 + KD label smoothing (matrix built from KD_TEACHER_PT, default the R5-1 checkpoint)
# Later experiments (consistency loss, RandAugment, PMAL) are not implemented; add them only if R5-3 leaves a train/val gap.
# Env: KD_TEACHER_PT=<.pt> KD_ALPHA=0.1
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux run5_ablations "${SCRIPT_DIR}/run_run5_ablations.sh" "$@"

[ -x "${PYTHON_BIN}" ] || { echo "[ERROR] Python not found or not executable: ${PYTHON_BIN}" >&2; exit 1; }

ONLY=""; RUN_DIAG=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)      ONLY="$2"; shift 2 ;;
        --skip-diag) RUN_DIAG=0; shift ;;
        *) echo "Usage: bash run_run5_ablations.sh [--only R5-0,R5-1] [--skip-diag]" >&2; exit 1 ;;
    esac
done

RUN5_DIR="${MOBILENET_DIR}/run5"
BASE_FLAGS=(--taxonomy base --save-pt --aug-strength strong --crop-jitter 0.08 --label-smoothing 0.1 --make-loss-weight 0.3 --ema-decay 0.999)
FULL_FLAGS=(--unfreeze-layers 19 --layer-lr-decay 0.85)
HEAD_FLAGS=(--pool gem --make-gating per_class)

wanted() { [ -z "${ONLY}" ] || [[ ",${ONLY}," == *",$1,"* ]]; }

run_job() {                      # run_job <ID> <flags...>
    local id="$1"; shift
    local name="run5_${id,,}"; name="${name//-/}"
    local out="${RUN5_DIR}/output_merged_${name}"
    echo "################ ${id} -> ${out}"
    NO_TMUX=1 AUTO_GIT_PUSH=0 RUN_NAME="${name}" bash "${SCRIPT_DIR}/run_train_merged.sh" "$@"
    "${PYTHON_BIN}" "${MOBILENET_DIR}/evaluate_open_set.py" --onnx "${out}/models/mobilenet_v2_best.onnx" \
        --ext-splits-dir "${RUN5_DIR}/splits_1235models" 2>&1 | tee "${out}/open_set.log" || echo "[WARN] open-set eval failed for ${id}"
    if [ "${RUN_DIAG}" -eq 1 ]; then
        NO_TMUX=1 AUTO_GIT_PUSH=0 RUN_NAME="${name}" bash "${SCRIPT_DIR}/run_diagnostics.sh" --merged || echo "[WARN] diagnostics failed for ${id}"
    fi
}

wanted R5-0 && run_job R5-0 "${BASE_FLAGS[@]}"
wanted R5-1 && run_job R5-1 "${BASE_FLAGS[@]}" "${FULL_FLAGS[@]}"
wanted R5-2 && run_job R5-2 "${BASE_FLAGS[@]}" "${FULL_FLAGS[@]}" "${HEAD_FLAGS[@]}"
if wanted R5-3; then
    TEACHER="${KD_TEACHER_PT:-${RUN5_DIR}/output_merged_run5_r51/models/mobilenet_v2_best.pt}"
    KD_MATRIX="${RUN5_DIR}/kd_matrix_run5.npy"
    [ -f "${TEACHER}" ] || { echo "[ERROR] KD teacher checkpoint not found: ${TEACHER} (set KD_TEACHER_PT)" >&2; exit 1; }
    "${PYTHON_BIN}" "${MOBILENET_DIR}/build_kd_smoothing_matrix.py" --teacher-pt "${TEACHER}" --out "${KD_MATRIX}"
    run_job R5-3 "${BASE_FLAGS[@]}" "${FULL_FLAGS[@]}" "${HEAD_FLAGS[@]}" --kd-ls-matrix "${KD_MATRIX}" --kd-ls-alpha "${KD_ALPHA:-0.1}"
fi
echo "[done] Run 5 queue finished $(date)"
