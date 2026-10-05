#!/usr/bin/env bash
# ==============================================================================
# Self-Sufficient Sample Inference Test Runner
# ==============================================================================
# Tests both the 35-make and 32-make models on the sample image batch.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/researchadmin/Econ/repo-clone/stanford-cars-model/.venv/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

echo "========================================================================"
echo " Running Sample Inference Test on 35-Make Production Model"
echo "========================================================================"
"$PYTHON_BIN" "$SCRIPT_DIR/infer.py" \
    --model 35 \
    --save-csv "$SCRIPT_DIR/output/predictions_35makes.csv" \
    --save-json "$SCRIPT_DIR/output/predictions_35makes.json" \
    --save-annotated "$SCRIPT_DIR/output/annotated_35makes"

echo ""
echo "========================================================================"
echo " Running Sample Inference Test on 32-Make Model"
echo "========================================================================"
"$PYTHON_BIN" "$SCRIPT_DIR/infer.py" \
    --model 32 \
    --save-csv "$SCRIPT_DIR/output/predictions_32makes.csv" \
    --save-json "$SCRIPT_DIR/output/predictions_32makes.json" \
    --save-annotated "$SCRIPT_DIR/output/annotated_32makes"

echo ""
echo "========================================================================"
echo " [Complete] All tests finished! Artifacts saved in:"
echo "   $SCRIPT_DIR/output/"
echo "========================================================================"
