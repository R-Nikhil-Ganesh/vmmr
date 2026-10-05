#!/usr/bin/env python3
"""
MobileNetV2 PlatesMania Diagnostic & Explainability Analysis Adapter
=====================================================================
Adapter module for the PlatesMania dataset, delegating to the universal
PyTorch & ONNX diagnostic & explainability engine in output_analysis.py.
"""

import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
ROOT_MOBILENET_DIR = DATASET_DIR.parent
if str(ROOT_MOBILENET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_MOBILENET_DIR))

import output_analysis as _oa

# Configure analysis context specifically for PlatesMania
_oa.set_context(
    model_path=DATASET_DIR / "output_mobilenet_v2" / "models" / "mobilenet_v2_best.pt",
    output_dir=DATASET_DIR / "output_mobilenet_v2",
    test_csv=Path("/home/researchadmin/Econ/dataset_split_640x640.csv"),
    label_map=DATASET_DIR / "output_mobilenet_v2" / "models" / "label_map.json",
    base_img_dir=Path("/home/researchadmin/Econ/resized_640x640"),
    img_size=512,
    crop_top_pct=0.15,
    crop_bottom_pct=0.0
)

# Re-export all functions, classes, and globals
from output_analysis import *

if __name__ == "__main__":
    _oa.main()
