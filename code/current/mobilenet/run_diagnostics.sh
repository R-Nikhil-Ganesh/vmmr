#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# Run Diagnostics Suite for Run 3 (1,235 Make+Model MobileNetV2)
# ==============================================================================
# Evaluates in-domain diagnostic reports, confusion matrices, calibration curves,
# latent manifold projections (t-SNE/PCA), and cross-domain mixed benchmarks.
#
# Usage:
#   bash run_diagnostics.sh              # Full test sets (Model A, B & Mixed Benchmark)
#   bash run_diagnostics.sh --quick      # Quick diagnostics (5k samples per dataset)
#   bash run_diagnostics.sh --pm-only    # Only Model A (PlatesMania)
#   bash run_diagnostics.sh --ext-only   # Only Model B (External Merged)
#   bash run_diagnostics.sh --bench-only # Only Cross-Domain Mixed Benchmark
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

RUN_PM=1
RUN_EXT=1
RUN_BENCH=1
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            MAX_SAMPLES="--max-samples 5000"
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
            shift 2
            ;;
        --pm-only)
            RUN_PM=1
            RUN_EXT=0
            RUN_BENCH=0
            shift
            ;;
        --ext-only)
            RUN_PM=0
            RUN_EXT=1
            RUN_BENCH=0
            shift
            ;;
        --bench-only)
            RUN_PM=0
            RUN_EXT=0
            RUN_BENCH=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: bash run_diagnostics.sh [--quick] [--pm-only] [--ext-only] [--bench-only] [--max-samples N]" >&2
            exit 1
            ;;
    esac
done

echo "=================================================================="
echo "  Starting MobileNetV2 Run 3 Diagnostics Suite (1,235 Classes)"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python:    ${PYTHON_BIN}"
echo "=================================================================="

# 1. Model A (PlatesMania) Diagnostics
if [ "${RUN_PM}" -eq 1 ]; then
    echo ""
    echo "=== [1/3] Running Model A (PlatesMania) In-Domain Diagnostics ==="
    ${PYTHON_BIN} -u "${SCRIPT_DIR}/platesmania_dataset/analysis/output_analysis_platesmania.py" \
        --batch-size 64 \
        --num-workers 6 \
        ${MAX_SAMPLES}
fi

# 2. Model B (External Merged) Diagnostics
if [ "${RUN_EXT}" -eq 1 ]; then
    echo ""
    echo "=== [2/3] Running Model B (External) In-Domain Diagnostics ==="
    ${PYTHON_BIN} -u "${SCRIPT_DIR}/external_dataset/analysis/output_analysis_external.py" \
        --batch-size 64 \
        --num-workers 6 \
        ${MAX_SAMPLES}
fi

# 3. Cross-Domain Mixed Benchmark
if [ "${RUN_BENCH}" -eq 1 ]; then
    echo ""
    echo "=== [3/3] Running Cross-Domain & Mixed Benchmark Evaluation ==="
    bash "${SCRIPT_DIR}/run_eval_mixed.sh"
fi

echo ""
echo "=================================================================="
echo "  All Run 3 Diagnostics Completed Successfully!"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

if [ "${AUTO_GIT_PUSH:-1}" = "1" ] && [ -f "${SCRIPT_DIR}/auto_git_sync.sh" ]; then
    bash "${SCRIPT_DIR}/auto_git_sync.sh" "Run 3 Diagnostics Suite (1235 Make+Model)"
fi
