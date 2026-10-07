#!/usr/bin/env python3
"""
MobileNetV2 External Dataset Diagnostic & Explainability Analysis Adapter
=========================================================================
Adapter module for the External Merged 35-Make dataset, delegating to the universal
PyTorch & ONNX diagnostic & explainability engine in output_analysis.py.
"""

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
DATASET_DIR = ANALYSIS_DIR.parent
ROOT_MOBILENET_DIR = DATASET_DIR.parent
if str(ROOT_MOBILENET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_MOBILENET_DIR))

import output_analysis as _oa

# Configure analysis context specifically for the External Dataset
_oa.set_context(
    model_path=DATASET_DIR / "output_mobilenet_v2_external" / "models" / "mobilenet_v2_best.onnx",
    output_dir=DATASET_DIR / "output_mobilenet_v2_external",
    test_csv_path=Path("/home/researchadmin/Econ/external_datasets/merged_data/test.csv"),
    label_map_path=DATASET_DIR / "output_mobilenet_v2_external" / "models" / "label_map.json",
    base_img_dir=None,
    img_size=512,
    crop_top_pct=0.0,
    crop_bottom_pct=0.05
)

# Re-export all functions, classes, and globals
from output_analysis import *

if __name__ == "__main__":
    _oa.main()
