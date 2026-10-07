#!/usr/bin/env python3
"""
Cross-Domain & Mixed Benchmark Evaluation Engine for MobileNetV2 (Make + Model)
================================================================================
Compares two models:
1. Model A (PlatesMania-trained on 1,235 Make/Model classes)
2. Model B (External-trained on 1,235 Make/Model classes)

Evaluated across three test splits:
- PlatesMania Test Set (In-domain for Model A, Out-of-domain for Model B)
- External Test Set    (In-domain for Model B, Out-of-domain for Model A)
- Mixed Test Set       (Stratified combination of both domains)

Dual-Level Metrics:
1. Fine-Grained Model Level: Top-1 Accuracy & Macro F1 (1,235 classes)
2. Coarse Make Level: Top-1 Accuracy & Macro F1 (marginalized via P(Make_k) = sum_{m in Make_k} P(Model_m))
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import output_analysis as oa

TARGET_MAKES = [
    "Acura", "Alfa Romeo", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler",
    "Cupra", "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Isuzu",
    "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln", "MINI", "Mazda", "Mitsubishi",
    "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Skoda", "Subaru", "Suzuki",
    "Toyota", "Volkswagen", "Volvo"
]
MAKE_TO_IDX = {m: i for i, m in enumerate(TARGET_MAKES)}
IDX_TO_MAKE = {i: m for i, m in enumerate(TARGET_MAKES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 Models on In-Domain, Out-of-Domain, and Mixed Benchmarks (Make + Model).")
    parser.add_argument("--pm-model-path", type=str,
        default=str(CURRENT_DIR / "platesmania_dataset" / "output_mobilenet_v2" / "models" / "mobilenet_v2_best.onnx"),
        help="Path to PlatesMania model (.onnx)")
    parser.add_argument("--pm-label-map", type=str,
        default=str(CURRENT_DIR / "platesmania_dataset" / "output_mobilenet_v2" / "models" / "label_map.json"),
        help="Path to PlatesMania label map")
    parser.add_argument("--ext-model-path", type=str,
        default=str(CURRENT_DIR / "external_dataset" / "output_mobilenet_v2_external" / "models" / "mobilenet_v2_best.onnx"),
        help="Path to External model (.onnx)")
    parser.add_argument("--ext-label-map", type=str,
        default=str(CURRENT_DIR / "external_dataset" / "splits_1235models" / "label_map.json"),
        help="Path to External label map")
    parser.add_argument("--pm-test-csv", type=str,
        default="/home/researchadmin/Econ/models/dataset_manifests/dataset_1235models_splits.csv",
        help="Path to PlatesMania test CSV")
    parser.add_argument("--pm-img-dir", type=str,
        default="/home/researchadmin/Econ/resized_640x640",
        help="Base image folder for PlatesMania")
    parser.add_argument("--ext-test-csv", type=str,
        default=str(CURRENT_DIR / "external_dataset" / "splits_1235models" / "test.csv"),
        help="Path to External test CSV")
    parser.add_argument("--output-dir", type=str,
        default=str(CURRENT_DIR / "mixed_benchmark_results"),
        help="Destination directory for benchmark tables and plots")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-eval-per-dataset", type=int, default=5000,
        help="Max samples to evaluate per dataset to keep Mixed benchmark balanced")
    return parser.parse_args()


def build_make_projection_matrix(model_label_map: Dict[str, int]) -> np.ndarray:
    """Creates a projection matrix P of shape (N_models, 35) to aggregate model probabilities into make probabilities."""
    n_models = len(model_label_map)
    proj = np.zeros((n_models, len(TARGET_MAKES)), dtype=np.float32)

    for cls_name, idx in model_label_map.items():
        make = cls_name.split("/")[0] if "/" in cls_name else cls_name
        if make in MAKE_TO_IDX:
            proj[idx, MAKE_TO_IDX[make]] = 1.0

    return proj


def evaluate_model_on_split(model: nn.Module, paths: List[str], true_make_indices: List[int],
                            true_model_indices: Optional[List[int]] = None,
                            proj_matrix: Optional[np.ndarray] = None,
                            bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
                            img_size: int = 512,
                            crop_top: float = 0.0, crop_bottom: float = 0.0,
                            batch_size: int = 64, device: str = "cuda") -> Dict[str, Any]:
    model.eval()
    model.to(device)

    all_pred_makes = []
    all_true_makes = []
    all_pred_models = []
    all_true_models = []
    n_failed = 0

    n_samples = len(paths)
    pbar = tqdm(range(0, n_samples, batch_size), desc="  Evaluating", unit="batch", leave=False)
    with torch.no_grad():
        for i in pbar:
            b_paths = paths[i:i + batch_size]
            b_make_labels = true_make_indices[i:i + batch_size]
            b_model_labels = true_model_indices[i:i + batch_size] if true_model_indices is not None else None
            b_bboxes = bboxes[i:i + batch_size] if bboxes is not None else None

            tensors = []
            valid_idx = []
            is_top_list = isinstance(crop_top, (list, tuple))
            is_bot_list = isinstance(crop_bottom, (list, tuple))
            for j, p in enumerate(b_paths):
                idx_g = i + j
                ct = crop_top[idx_g] if is_top_list else crop_top
                cb = crop_bottom[idx_g] if is_bot_list else crop_bottom
                bb = b_bboxes[j] if b_bboxes is not None else None
                try:
                    t, _ = oa.load_and_preprocess_image(p, img_size, ct, cb, bbox=bb)
                    tensors.append(t)
                    valid_idx.append(j)
                except Exception as e:
                    n_failed += 1
                    continue

            if not tensors:
                continue

            batch_tensor = torch.cat(tensors, dim=0).to(device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # 1. Model predictions (1235 classes)
            pred_models = np.argmax(probs, axis=1)
            all_pred_models.extend(pred_models)
            if b_model_labels is not None:
                all_true_models.extend([b_model_labels[k] for k in valid_idx])

            # 2. Make predictions (marginalized over models)
            if proj_matrix is not None:
                make_probs = probs @ proj_matrix
            else:
                make_probs = probs

            pred_makes = np.argmax(make_probs, axis=1)
            all_pred_makes.extend(pred_makes)
            all_true_makes.extend([b_make_labels[k] for k in valid_idx])

    if n_failed:
        print(f"[WARN] {n_failed:,} of {n_samples:,} images could not be read and were excluded.")

    y_pred_make = np.array(all_pred_makes)
    y_true_make = np.array(all_true_makes)

    make_acc = float(np.mean(y_true_make == y_pred_make))
    _, _, make_macro_f1, _ = precision_recall_fscore_support(y_true_make, y_pred_make, average="macro", zero_division=0)
    _, _, make_weighted_f1, _ = precision_recall_fscore_support(y_true_make, y_pred_make, average="weighted", zero_division=0)

    res = {
        "make_accuracy": make_acc,
        "make_macro_f1": float(make_macro_f1),
        "make_weighted_f1": float(make_weighted_f1)
    }

    if all_true_models:
        y_pred_model = np.array(all_pred_models)
        y_true_model = np.array(all_true_models)
        model_acc = float(np.mean(y_true_model == y_pred_model))
        _, _, model_macro_f1, _ = precision_recall_fscore_support(y_true_model, y_pred_model, average="macro", zero_division=0)
        _, _, model_weighted_f1, _ = precision_recall_fscore_support(y_true_model, y_pred_model, average="weighted", zero_division=0)
        res["model_accuracy"] = model_acc
        res["model_macro_f1"] = float(model_macro_f1)
        res["model_weighted_f1"] = float(model_weighted_f1)
    else:
        res["model_accuracy"] = None
        res["model_macro_f1"] = None
        res["model_weighted_f1"] = None

    return res


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Cross-Domain & Mixed Benchmark Evaluation (Make + Model)")
    print(f"  Model A (PlatesMania): {args.pm_model_path}")
    print(f"  Model B (External):    {args.ext_model_path}")
    print(f"  Output Directory:      {output_dir}")
    print("=" * 70 + "\n")

    # Load label maps
    with open(args.pm_label_map) as f:
        pm_lm_raw = json.load(f)
    pm_lm = pm_lm_raw["class_to_idx"] if "class_to_idx" in pm_lm_raw else pm_lm_raw
    pm_proj = build_make_projection_matrix(pm_lm) if len(pm_lm) > len(TARGET_MAKES) else None

    with open(args.ext_label_map) as f:
        ext_lm_raw = json.load(f)
    ext_lm = ext_lm_raw["class_to_idx"] if "class_to_idx" in ext_lm_raw else ext_lm_raw
    ext_proj = build_make_projection_matrix(ext_lm) if len(ext_lm) > len(TARGET_MAKES) else None

    # Load models
    model_a = oa.MobileNetV2Evaluator(num_classes=len(pm_lm), checkpoint_path=args.pm_model_path).to(device)
    model_b = oa.MobileNetV2Evaluator(num_classes=len(ext_lm), checkpoint_path=args.ext_model_path).to(device)

    # 1. Prepare PlatesMania test split
    df_pm = pd.read_csv(args.pm_test_csv)
    if "split" in df_pm.columns:
        df_pm = df_pm[df_pm["split"] == "test"].reset_index(drop=True)
    if len(df_pm) > args.max_eval_per_dataset:
        df_pm = df_pm.sample(n=args.max_eval_per_dataset, random_state=42).reset_index(drop=True)

    pm_col = "image_path" if "image_path" in df_pm.columns else "image_rel_path"
    pm_paths = [os.path.join(args.pm_img_dir, p) if not os.path.isabs(p) else p for p in df_pm[pm_col].values]
    pm_make_names = [str(r["make"]).strip() if "make" in r and pd.notna(r["make"]) else str(r["class_name"]).split("/")[0] for _, r in df_pm.iterrows()]
    pm_make_indices = [MAKE_TO_IDX.get(m) for m in pm_make_names]
    pm_model_indices = df_pm["label"].values.astype(int).tolist()

    pm_keep = [i for i, m in enumerate(pm_make_indices) if m is not None]
    pm_paths = [pm_paths[i] for i in pm_keep]
    pm_make_indices = [pm_make_indices[i] for i in pm_keep]
    pm_model_indices = [pm_model_indices[i] for i in pm_keep]
    pm_bboxes = [None] * len(pm_paths)

    # 2. Prepare External test split
    df_ext = pd.read_csv(args.ext_test_csv)
    if "split" in df_ext.columns:
        df_ext = df_ext[df_ext["split"] == "test"].reset_index(drop=True)
    if len(df_ext) > args.max_eval_per_dataset:
        df_ext = df_ext.sample(n=args.max_eval_per_dataset, random_state=42).reset_index(drop=True)

    ext_paths = df_ext["image_path"].tolist()
    ext_make_names = [str(r["make"]).strip() if "make" in r and pd.notna(r["make"]) else str(r["class_name"]).split("/")[0] for _, r in df_ext.iterrows()]
    ext_make_indices = [MAKE_TO_IDX.get(m) for m in ext_make_names]
    ext_model_indices = df_ext["label"].values.astype(int).tolist()

    ext_bboxes = None
    if "bbox_x1" in df_ext.columns:
        ext_bboxes = [
            (int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"]))
            if int(r["bbox_x1"]) >= 0 else None
            for _, r in df_ext.iterrows()
        ]

    ext_keep = [i for i, m in enumerate(ext_make_indices) if m is not None]
    ext_paths = [ext_paths[i] for i in ext_keep]
    ext_make_indices = [ext_make_indices[i] for i in ext_keep]
    ext_model_indices = [ext_model_indices[i] for i in ext_keep]
    if ext_bboxes is not None:
        ext_bboxes = [ext_bboxes[i] for i in ext_keep]
    else:
        ext_bboxes = [None] * len(ext_paths)

    # 3. Prepare Mixed test split
    mixed_paths = pm_paths + ext_paths
    mixed_make_indices = pm_make_indices + ext_make_indices
    mixed_model_indices = pm_model_indices + ext_model_indices
    mixed_bboxes = pm_bboxes + ext_bboxes
    mixed_crop_top = [0.15] * len(pm_paths) + [0.0] * len(ext_paths)
    mixed_crop_bot = [0.0] * len(pm_paths) + [0.05] * len(ext_paths)

    print(f"[Datasets] PlatesMania Test: {len(pm_paths):,} | External Test: {len(ext_paths):,} | Mixed Test: {len(mixed_paths):,}")

    results = []

    # Benchmark Model A (PlatesMania)
    print("\n--- Evaluating Model A (PlatesMania) ---")
    res_a_pm = evaluate_model_on_split(model_a, pm_paths, pm_make_indices, pm_model_indices, pm_proj, pm_bboxes, args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size, device=device)
    print(f"Model A on PlatesMania (In-Domain):     Make Acc={res_a_pm['make_accuracy']:.2%} | Model Acc={res_a_pm['model_accuracy']:.2%} | Make Macro F1={res_a_pm['make_macro_f1']:.2%}")

    res_a_ext = evaluate_model_on_split(model_a, ext_paths, ext_make_indices, ext_model_indices, pm_proj, ext_bboxes, args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size, device=device)
    print(f"Model A on External    (Out-of-Domain): Make Acc={res_a_ext['make_accuracy']:.2%} | Model Acc={res_a_ext['model_accuracy']:.2%} | Make Macro F1={res_a_ext['make_macro_f1']:.2%}")

    res_a_mix = evaluate_model_on_split(model_a, mixed_paths, mixed_make_indices, mixed_model_indices, pm_proj, mixed_bboxes, args.img_size, crop_top=mixed_crop_top, crop_bottom=mixed_crop_bot, batch_size=args.batch_size, device=device)
    print(f"Model A on Mixed       (Combined):      Make Acc={res_a_mix['make_accuracy']:.2%} | Model Acc={res_a_mix['model_accuracy']:.2%} | Make Macro F1={res_a_mix['make_macro_f1']:.2%}")

    results.extend([
        {"model": "Model A (PlatesMania)", "test_split": "PlatesMania (In-Domain)", "domain_type": "In-Domain", **res_a_pm},
        {"model": "Model A (PlatesMania)", "test_split": "External (Out-of-Domain)", "domain_type": "Out-of-Domain", **res_a_ext},
        {"model": "Model A (PlatesMania)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", **res_a_mix},
    ])

    # Benchmark Model B (External)
    print("\n--- Evaluating Model B (External Merged) ---")
    res_b_ext = evaluate_model_on_split(model_b, ext_paths, ext_make_indices, ext_model_indices, ext_proj, ext_bboxes, args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size, device=device)
    print(f"Model B on External    (In-Domain):     Make Acc={res_b_ext['make_accuracy']:.2%} | Model Acc={res_b_ext['model_accuracy']:.2%} | Make Macro F1={res_b_ext['make_macro_f1']:.2%}")

    res_b_pm = evaluate_model_on_split(model_b, pm_paths, pm_make_indices, pm_model_indices, ext_proj, pm_bboxes, args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size, device=device)
    print(f"Model B on PlatesMania (Out-of-Domain): Make Acc={res_b_pm['make_accuracy']:.2%} | Model Acc={res_b_pm['model_accuracy']:.2%} | Make Macro F1={res_b_pm['make_macro_f1']:.2%}")

    res_b_mix = evaluate_model_on_split(model_b, mixed_paths, mixed_make_indices, mixed_model_indices, ext_proj, mixed_bboxes, args.img_size, crop_top=mixed_crop_top, crop_bottom=mixed_crop_bot, batch_size=args.batch_size, device=device)
    print(f"Model B on Mixed       (Combined):      Make Acc={res_b_mix['make_accuracy']:.2%} | Model Acc={res_b_mix['model_accuracy']:.2%} | Make Macro F1={res_b_mix['make_macro_f1']:.2%}")

    results.extend([
        {"model": "Model B (External)", "test_split": "External (In-Domain)", "domain_type": "In-Domain", **res_b_ext},
        {"model": "Model B (External)", "test_split": "PlatesMania (Out-of-Domain)", "domain_type": "Out-of-Domain", **res_b_pm},
        {"model": "Model B (External)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", **res_b_mix},
    ])

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "mixed_benchmark_summary.csv", index=False)

    # Plot dual comparison bar chart (Make Accuracy & Model Accuracy side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    splits = ["PlatesMania Split", "External Split", "Mixed Split"]
    x = np.arange(len(splits))
    width = 0.35

    # 1. Make Accuracy Plot
    make_a_vals = [res_a_pm["make_accuracy"], res_a_ext["make_accuracy"], res_a_mix["make_accuracy"]]
    make_b_vals = [res_b_pm["make_accuracy"], res_b_ext["make_accuracy"], res_b_mix["make_accuracy"]]
    r1 = ax1.bar(x - width/2, [v * 100 for v in make_a_vals], width, label="Model A (PlatesMania)", color="#2563EB")
    r2 = ax1.bar(x + width/2, [v * 100 for v in make_b_vals], width, label="Model B (External)", color="#16A34A")
    ax1.set_ylabel("Make Accuracy (%)", fontsize=12)
    ax1.set_title("Make Accuracy (Marginalized over Models)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(fontsize=10)
    for r in r1 + r2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 2. Model Accuracy Plot
    model_a_vals = [res_a_pm["model_accuracy"], res_a_ext["model_accuracy"], res_a_mix["model_accuracy"]]
    model_b_vals = [res_b_pm["model_accuracy"], res_b_ext["model_accuracy"], res_b_mix["model_accuracy"]]
    r3 = ax2.bar(x - width/2, [v * 100 for v in model_a_vals], width, label="Model A (PlatesMania)", color="#2563EB")
    r4 = ax2.bar(x + width/2, [v * 100 for v in model_b_vals], width, label="Model B (External)", color="#16A34A")
    ax2.set_ylabel("Model Accuracy (%)", fontsize=12)
    ax2.set_title("Fine-Grained Model Accuracy (1,235 Classes)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits, fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(fontsize=10)
    for r in r3 + r4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(output_dir / "cross_domain_comparison.png"), dpi=150)
    plt.close()

    print(f"\n[Done] Benchmark results and dual comparison charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
