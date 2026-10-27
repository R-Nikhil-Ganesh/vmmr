#!/usr/bin/env python3
"""
MobileNetV2 Merged Dataset (Model C) Diagnostic & Explainability Analysis Adapter
=================================================================================
Adapter module for the jointly trained PlatesMania + External model (Model C),
supporting per-domain diagnostic evaluation (--domain platesmania | external)
via the universal PyTorch & ONNX diagnostic engine in output_analysis.py.
"""

import argparse
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
DATASET_DIR = ANALYSIS_DIR.parent
ROOT_MOBILENET_DIR = DATASET_DIR.parent
if str(ROOT_MOBILENET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_MOBILENET_DIR))

import output_analysis as _oa
import paths

DEFAULT_RUN_DIR = DATASET_DIR / "output_merged_run4"


def configure_domain(domain: str = "platesmania", run_dir: Path | str = DEFAULT_RUN_DIR):
    """Configures output_analysis.context for either 'platesmania' or 'external' evaluation of Model C."""
    run_dir = Path(run_dir)
    model_path = run_dir / "models" / "mobilenet_v2_best.onnx"
    label_map_path = run_dir / "models" / "label_map.json"

    domain = domain.lower().strip()
    if domain in ("platesmania", "pm"):
        _oa.set_context(
            model_path=model_path,
            output_dir=run_dir / "diagnostics_platesmania",
            test_csv_path=Path(paths.PM_MANIFEST_CSV),
            label_map_path=label_map_path,
            base_img_dir=Path(paths.PM_IMG_DIR),
            img_size=512,
            crop_top_pct=0.15,
            crop_bottom_pct=0.0,
        )
    elif domain in ("external", "ext"):
        _oa.set_context(
            model_path=model_path,
            output_dir=run_dir / "diagnostics_external",
            test_csv_path=DATASET_DIR / "splits_1235models" / "test.csv",
            label_map_path=label_map_path,
            base_img_dir=None,
            img_size=512,
            crop_top_pct=0.0,
            crop_bottom_pct=0.05,
        )
    else:
        raise ValueError(f"Unknown domain '{domain}'. Expected 'platesmania' or 'external'.")
    return _oa.context


# Initialize default context to PlatesMania on import
configure_domain("platesmania", DEFAULT_RUN_DIR)

# Re-export all functions, classes, and globals from output_analysis
from output_analysis import *  # noqa: E402,F401,F403


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--domain", choices=["platesmania", "pm", "external", "ext"], default="platesmania",
                            help="Target evaluation domain for Model C (default: platesmania)")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    configure_domain(pre_args.domain, DEFAULT_RUN_DIR)
    sys.argv = [sys.argv[0]] + remaining_argv
    _oa.main()

