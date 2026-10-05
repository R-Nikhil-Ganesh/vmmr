#!/usr/bin/env python3
"""
EfficientNet-B0 (35 Vehicle Makes): Diagnostic & Explainability Analysis Engine (External)
==========================================================================================
Adapter module for the External Merged 35-Make dataset, delegating to the universal
ONNX diagnostic & explainability engine in output_analysis.py.
"""

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
ROOT_EFFICIENTNET_DIR = DATASET_DIR.parent
if str(ROOT_EFFICIENTNET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_EFFICIENTNET_DIR))

import output_analysis as _oa

# Configure analysis context specifically for the External Dataset
_oa.set_context(
    model_path=DATASET_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_best.onnx",
    output_dir=DATASET_DIR / "output_efficientnet_b0_merged",
    splits_dir=Path("/home/researchadmin/Econ-n/external_datasets/merged_data"),
    label_map=DATASET_DIR / "output_efficientnet_b0_merged" / "models" / "label_map.json"
)

# Re-export all functions, classes, and globals from the core output_analysis engine
from output_analysis import *

if __name__ == "__main__":
    _oa.main()
