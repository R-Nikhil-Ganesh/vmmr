#!/usr/bin/env python3
"""
Cross-Domain & Mixed Benchmark Evaluation Engine for MobileNetV2
================================================================
Compares two models:
1. Model A (PlatesMania-trained)
2. Model B (External Merged-trained)

Evaluated across three test splits:
- PlatesMania Test Set (In-domain for Model A, Out-of-domain for Model B)
- External Test Set    (In-domain for Model B, Out-of-domain for Model A)
- Mixed Test Set       (Stratified combination of both domains)

Evaluates on the 35 standardized vehicle makes using exact probability marginalization
for fine-grained models: P(Make_k) = sum_{m in Make_k} P(Model_m).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, precision_recall_fscore_support

import torch
import torch.nn as nn
from torchvision import models

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
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 Models on In-Domain, Out-of-Domain, and Mixed Benchmarks.")
    parser.add_argument("--pm-model-path", type=str,
        default=str(CURRENT_DIR / "platesmania_dataset" / "output_mobilenet_v2" / "models" / "mobilenet_v2_best.pt"),
        help="Path to PlatesMania model checkpoint (.pt)")
    parser.add_argument("--pm-label-map", type=str,
        default=str(CURRENT_DIR / "platesmania_dataset" / "output_mobilenet_v2" / "models" / "label_map.json"),
        help="Path to PlatesMania label map")
    parser.add_argument("--ext-model-path", type=str,
        default=str(CURRENT_DIR / "external_dataset" / "output_mobilenet_v2_external" / "models" / "mobilenet_v2_best.pt"),
        help="Path to External model checkpoint (.pt)")
    parser.add_argument("--ext-label-map", type=str,
        default=str(CURRENT_DIR / "external_dataset" / "output_mobilenet_v2_external" / "models" / "label_map.json"),
        help="Path to External label map")
    parser.add_argument("--pm-test-csv", type=str,
        default="/home/researchadmin/Econ/dataset_split_640x640.csv",
        help="Path to PlatesMania test CSV")
    parser.add_argument("--pm-img-dir", type=str,
        default="/home/researchadmin/Econ/resized_640x640",
        help="Base image folder for PlatesMania")
    parser.add_argument("--ext-test-csv", type=str,
        default="/home/researchadmin/Econ/external_datasets/merged_data/test.csv",
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
        # Class name format can be 'Make/Model' or just 'Make'
        make = cls_name.split("/")[0] if "/" in cls_name else cls_name
        if make in MAKE_TO_IDX:
            proj[idx, MAKE_TO_IDX[make]] = 1.0

    return proj


def evaluate_model_on_split(model: nn.Module, paths: List[str], true_make_indices: List[int],
                            proj_matrix: Optional[np.ndarray], img_size: int = 512,
                            crop_top: float = 0.0, crop_bottom: float = 0.0,
                            batch_size: int = 64, device: str = "cuda") -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    all_pred_makes = []
    all_true_makes = []
    all_make_probs = []

    n_samples = len(paths)
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_paths = paths[i:i + batch_size]
            b_labels = true_make_indices[i:i + batch_size]

            tensors = []
            valid_idx = []
            is_top_list = isinstance(crop_top, (list, tuple))
            is_bot_list = isinstance(crop_bottom, (list, tuple))
            for j, p in enumerate(b_paths):
                idx_g = i + j
                ct = crop_top[idx_g] if is_top_list else crop_top
                cb = crop_bottom[idx_g] if is_bot_list else crop_bottom
                try:
                    t, _ = oa.load_and_preprocess_image(p, img_size, ct, cb)
                    tensors.append(t)
                    valid_idx.append(j)
                except Exception:
                    continue

            if not tensors:
                continue

            batch_tensor = torch.cat(tensors, dim=0).to(device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            if proj_matrix is not None:
                # Aggregate model probabilities into 35 make probabilities
                make_probs = probs @ proj_matrix
            else:
                make_probs = probs

            pred_makes = np.argmax(make_probs, axis=1)

            all_pred_makes.extend(pred_makes)
            all_true_makes.extend([b_labels[k] for k in valid_idx])
            all_make_probs.extend(make_probs)

    y_pred = np.array(all_pred_makes)
    y_true = np.array(all_true_makes)
    y_probs = np.array(all_make_probs)

    acc = float(np.mean(y_true == y_pred))
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    return acc, float(macro_f1), float(weighted_f1), y_true, y_pred


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Cross-Domain & Mixed Benchmark Evaluation for MobileNetV2")
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
    
    # Extract make labels for PlatesMania
    pm_makes = []
    for _, row in df_pm.iterrows():
        if "make" in row and pd.notna(row["make"]):
            m = str(row["make"])
        elif "class_name" in row and pd.notna(row["class_name"]):
            m = str(row["class_name"]).split("/")[0]
        else:
            m = "Unknown"
        pm_makes.append(MAKE_TO_IDX.get(m, 0))

    # 2. Prepare External test split
    df_ext = pd.read_csv(args.ext_test_csv)
    if len(df_ext) > args.max_eval_per_dataset:
        df_ext = df_ext.sample(n=args.max_eval_per_dataset, random_state=42).reset_index(drop=True)

    ext_paths = df_ext["image_path"].tolist()
    ext_makes = [MAKE_TO_IDX.get(str(m), 0) for m in df_ext["make"].values]

    # 3. Prepare Mixed test split (combination)
    mixed_paths = pm_paths + ext_paths
    mixed_makes = pm_makes + ext_makes
    # Top crop is ONLY applied to PlatesMania images (0.15) to strip PLATESMANIA.COM banner.
    # External images NEVER have top crop (0.0).
    mixed_crop_top = [0.15] * len(pm_paths) + [0.0] * len(ext_paths)
    mixed_crop_bot = [0.0] * len(pm_paths) + [0.05] * len(ext_paths)

    print(f"[Datasets] PlatesMania Test: {len(pm_paths):,} | External Test: {len(ext_paths):,} | Mixed Test: {len(mixed_paths):,}")

    results = []

    # Benchmark Model A (PlatesMania)
    print("\n--- Evaluating Model A (PlatesMania) ---")
    acc_a_pm, f1_a_pm, wf1_a_pm, _, _ = evaluate_model_on_split(model_a, pm_paths, pm_makes, pm_proj, args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size, device=device)
    print(f"Model A on PlatesMania (In-Domain):     Acc={acc_a_pm:.2%} | Macro F1={f1_a_pm:.2%}")

    acc_a_ext, f1_a_ext, wf1_a_ext, _, _ = evaluate_model_on_split(model_a, ext_paths, ext_makes, pm_proj, args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size, device=device)
    print(f"Model A on External    (Out-of-Domain): Acc={acc_a_ext:.2%} | Macro F1={f1_a_ext:.2%}")

    acc_a_mix, f1_a_mix, wf1_a_mix, _, _ = evaluate_model_on_split(model_a, mixed_paths, mixed_makes, pm_proj, args.img_size, crop_top=mixed_crop_top, crop_bottom=mixed_crop_bot, batch_size=args.batch_size, device=device)
    print(f"Model A on Mixed       (Combined):      Acc={acc_a_mix:.2%} | Macro F1={f1_a_mix:.2%}")

    results.extend([
        {"model": "Model A (PlatesMania)", "test_split": "PlatesMania (In-Domain)", "domain_type": "In-Domain", "accuracy": acc_a_pm, "macro_f1": f1_a_pm, "weighted_f1": wf1_a_pm},
        {"model": "Model A (PlatesMania)", "test_split": "External (Out-of-Domain)", "domain_type": "Out-of-Domain", "accuracy": acc_a_ext, "macro_f1": f1_a_ext, "weighted_f1": wf1_a_ext},
        {"model": "Model A (PlatesMania)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", "accuracy": acc_a_mix, "macro_f1": f1_a_mix, "weighted_f1": wf1_a_mix},
    ])

    # Benchmark Model B (External)
    print("\n--- Evaluating Model B (External Merged) ---")
    acc_b_ext, f1_b_ext, wf1_b_ext, _, _ = evaluate_model_on_split(model_b, ext_paths, ext_makes, ext_proj, args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size, device=device)
    print(f"Model B on External    (In-Domain):     Acc={acc_b_ext:.2%} | Macro F1={f1_b_ext:.2%}")

    acc_b_pm, f1_b_pm, wf1_b_pm, _, _ = evaluate_model_on_split(model_b, pm_paths, pm_makes, ext_proj, args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size, device=device)
    print(f"Model B on PlatesMania (Out-of-Domain): Acc={acc_b_pm:.2%} | Macro F1={f1_b_pm:.2%}")

    acc_b_mix, f1_b_mix, wf1_b_mix, _, _ = evaluate_model_on_split(model_b, mixed_paths, mixed_makes, ext_proj, args.img_size, crop_top=mixed_crop_top, crop_bottom=mixed_crop_bot, batch_size=args.batch_size, device=device)
    print(f"Model B on Mixed       (Combined):      Acc={acc_b_mix:.2%} | Macro F1={f1_b_mix:.2%}")

    results.extend([
        {"model": "Model B (External)", "test_split": "External (In-Domain)", "domain_type": "In-Domain", "accuracy": acc_b_ext, "macro_f1": f1_b_ext, "weighted_f1": wf1_b_ext},
        {"model": "Model B (External)", "test_split": "PlatesMania (Out-of-Domain)", "domain_type": "Out-of-Domain", "accuracy": acc_b_pm, "macro_f1": f1_b_pm, "weighted_f1": wf1_b_pm},
        {"model": "Model B (External)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", "accuracy": acc_b_mix, "macro_f1": f1_b_mix, "weighted_f1": wf1_b_mix},
    ])

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "mixed_benchmark_summary.csv", index=False)

    # Plot comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    splits = ["PlatesMania Split", "External Split", "Mixed Split"]
    x = np.arange(len(splits))
    width = 0.35

    acc_a_vals = [acc_a_pm, acc_a_ext, acc_a_mix]
    acc_b_vals = [acc_b_pm, acc_b_ext, acc_b_mix]

    rects1 = ax.bar(x - width/2, [v * 100 for v in acc_a_vals], width, label="Model A (PlatesMania)", color="#2563EB")
    rects2 = ax.bar(x + width/2, [v * 100 for v in acc_b_vals], width, label="Model B (External)", color="#16A34A")

    ax.set_ylabel("Make Accuracy (%)", fontsize=12)
    ax.set_title("Cross-Domain & Mixed Benchmark Make Accuracy (MobileNetV2)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(fontsize=11)

    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(output_dir / "cross_domain_comparison.png"), dpi=150)
    plt.close()

    print(f"\n[Done] Benchmark results and comparison chart saved to: {output_dir}")


if __name__ == "__main__":
    main()
