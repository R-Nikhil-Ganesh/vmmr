#!/usr/bin/env bash
# Overnight color-teacher queue on one GPU (about 12 h on a 4090), in one tmux session:
#   1. ~15 min  eval_color_teacher: best config (bf16 + letters) on the test split, plus two letters combos on val
#   2. ~11 h    relabel every image (--split all) with bf16 + letters into ${ANNOT_OUT_DIR}/${RELABEL_NAME},
#               reusing make/model predictions and embeddings from CACHE_FROM (default run2) but NOT its colors.
#               color_cache.csv keeps the 13-color probabilities, so calibrate_color_teacher.py's calibrator can be applied later.
# The relabel resumes from color_cache.csv: if it is stopped, rerun the same command.
# RELABEL_NAME must start with 'run': annotation_runs/run*/ is gitignored, other names would be auto-committed (GBs).
#
#   bash bash_scripts/run_overnight_color.sh
#   SKIP_EVAL=1 bash bash_scripts/run_overnight_color.sh            # relabel only
#   RELABEL_NAME=run3_letters CACHE_FROM=/path/to/run2 bash bash_scripts/run_overnight_color.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux overnight_color "${SCRIPT_DIR}/run_overnight_color.sh" "$@"

RELABEL_NAME="${RELABEL_NAME:-run3_letters}"
[[ "${RELABEL_NAME}" == run* ]] || { echo "[ERROR] RELABEL_NAME must start with 'run' (gitignored)" >&2; exit 1; }
CACHE_FROM="${CACHE_FROM:-${ANNOT_OUT_DIR}/run2}"
[ -f "${CACHE_FROM}/annotations.csv" ] || { echo "[ERROR] ${CACHE_FROM}/annotations.csv not found (set CACHE_FROM)" >&2; exit 1; }

echo "=================================================================="
echo "  Overnight color queue   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Relabel: ${ANNOT_OUT_DIR}/${RELABEL_NAME}   (make/model from ${CACHE_FROM})"
echo "=================================================================="

# Sub-runners stay in this session and do not push; one sync at the end
export NO_TMUX=1 AUTO_GIT_PUSH_OUTER="${AUTO_GIT_PUSH:-1}" AUTO_GIT_PUSH=0
EVAL="${SCRIPT_DIR}/run_eval_color_teacher.sh"
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    # A failed eval should not cost the relabel
    RUN_NAME=s2_letters_test bash "${EVAL}" --no-4bit --scoring letters --split test             || echo "[WARN] s2_letters_test failed"
    RUN_NAME=s6_letters_flip bash "${EVAL}" --no-4bit --scoring letters --tta flip               || echo "[WARN] s6_letters_flip failed"
    RUN_NAME=s6_letters_672  bash "${EVAL}" --no-4bit --scoring letters --max-side 672 --batch-size 4 || echo "[WARN] s6_letters_672 failed"
fi

RUN_NAME="${RELABEL_NAME}" CACHE_FROM="${CACHE_FROM}" bash "${SCRIPT_DIR}/run_annotate.sh" \
    --split all --no-4bit --qwen-scoring letters --fresh-color "$@"

echo; echo "  Overnight queue done   $(date '+%Y-%m-%d %H:%M:%S')"
if [ "${AUTO_GIT_PUSH_OUTER}" = "1" ]; then
    bash "${SCRIPT_DIR}/lib/auto_git_sync.sh" "Overnight color queue (${RELABEL_NAME})"
fi
