#!/usr/bin/env python3
"""
EfficientNet-B0 (35 Vehicle Makes): Diagnostic & Explainability Analysis Engine (PlatesMania)
=============================================================================================
Adapter module for the PlatesMania 35-Make dataset, delegating to the universal
ONNX diagnostic & explainability engine in output_analysis.py.
"""

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
ROOT_EFFICIENTNET_DIR = DATASET_DIR.parent
if str(ROOT_EFFICIENTNET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_EFFICIENTNET_DIR))

import output_analysis as _oa

# Configure analysis context specifically for PlatesMania
_oa.set_context(
    model_path=DATASET_DIR / "output_efficientnet_b0" / "models" / "efficientnet_b0_best.onnx",
    output_dir=DATASET_DIR / "output_efficientnet_b0",
    splits_dir=Path("/home/researchadmin/Econ-n/resized_640x640/splits_filtered"),
    label_map=DATASET_DIR / "output_efficientnet_b0" / "models" / "label_map.json"
)

# Re-export all functions, classes, and globals from the core output_analysis engine
from output_analysis import *

if __name__ == "__main__":
    _oa.main()
