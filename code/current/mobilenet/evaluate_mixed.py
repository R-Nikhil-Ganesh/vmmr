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
import gc
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
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import output_analysis as oa
import paths

TARGET_MAKES = [
    "Acura", "Alfa Romeo", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler",
    "Cupra", "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Isuzu",
    "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln", "MINI", "Mazda", "Mitsubishi",
    "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Skoda", "Subaru", "Suzuki",
    "Toyota", "Volkswagen", "Volvo"
]
MAKE_TO_IDX = {m: i for i, m in enumerate(TARGET_MAKES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 Models on In-Domain, Out-of-Domain, and Mixed Benchmarks (Make + Model).")
    parser.add_argument("--pm-model-path", type=str,
        default=str(CURRENT_DIR / "merged_dataset" / "output_merged_run4" / "models" / "mobilenet_v2_best.onnx"),
        help="Path to PlatesMania / Merged model (.onnx)")
    parser.add_argument("--pm-label-map", type=str,
        default=str(CURRENT_DIR / "merged_dataset" / "output_merged_run4" / "models" / "label_map.json"),
        help="Path to PlatesMania / Merged label map")
    parser.add_argument("--ext-model-path", type=str,
        default=str(CURRENT_DIR / "merged_dataset" / "output_merged_run4" / "models" / "mobilenet_v2_best.onnx"),
        help="Path to External / Merged model (.onnx)")
    parser.add_argument("--ext-label-map", type=str,
        default=str(CURRENT_DIR / "merged_dataset" / "splits_1235models" / "label_map.json"),
        help="Path to External label map")
    parser.add_argument("--pm-test-csv", type=str,
        default=paths.PM_MANIFEST_CSV,
        help="Path to PlatesMania test CSV")
    parser.add_argument("--pm-img-dir", type=str,
        default=paths.PM_IMG_DIR,
        help="Base image folder for PlatesMania")
    parser.add_argument("--ext-test-csv", type=str,
        default=str(CURRENT_DIR / "merged_dataset" / "splits_1235models" / "test.csv"),
        help="Path to External test CSV")
    parser.add_argument("--ext-train-csv", type=str,
        default=str(CURRENT_DIR / "merged_dataset" / "splits_1235models" / "train.csv"),
        help="External train CSV (used only to find which classes each model could have seen)")
    parser.add_argument("--output-dir", type=str,
        default=str(CURRENT_DIR / "mixed_benchmark_results"),
        help="Destination directory for benchmark tables and plots")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=10, help="DataLoader workers (default: 10)")
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


class MixedBenchmarkDataset(Dataset):
    """Multi-worker dataset calling oa.load_and_preprocess_image for bit-exact preprocessing."""
    def __init__(self, paths: List[str], make_indices: List[int],
                 model_indices: Optional[List[int]],
                 bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]],
                 img_size: int, crop_top: Any, crop_bottom: Any):
        self.paths = paths
        self.make_indices = make_indices
        self.model_indices = model_indices
        self.bboxes = bboxes
        self.img_size = img_size
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.is_top_list = isinstance(crop_top, (list, tuple))
        self.is_bot_list = isinstance(crop_bottom, (list, tuple))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        ct = self.crop_top[idx] if self.is_top_list else self.crop_top
        cb = self.crop_bottom[idx] if self.is_bot_list else self.crop_bottom
        bb = self.bboxes[idx] if self.bboxes is not None else None
        mk_lbl = int(self.make_indices[idx])
        md_lbl = int(self.model_indices[idx]) if self.model_indices is not None else -1
        try:
            t, _ = oa.load_and_preprocess_image(p, self.img_size, ct, cb, bbox=bb)
            return t.squeeze(0), mk_lbl, md_lbl, True
        except Exception:
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32), mk_lbl, md_lbl, False


def compute_split_metrics(all_pred_makes: List[int], all_true_makes: List[int],
                          all_pred_models: List[int], all_true_models: List[int],
                          all_pred_restricted: List[int],
                          shared_classes: Optional[List[int]] = None) -> Dict[str, Any]:
    shared_arr = np.array(shared_classes, dtype=int) if shared_classes else None
    y_pred_make = np.array(all_pred_makes)
    y_true_make = np.array(all_true_makes)

    make_acc = float(np.mean(y_true_make == y_pred_make))
    _, _, make_macro_f1, _ = precision_recall_fscore_support(y_true_make, y_pred_make, average="macro", zero_division=0)
    _, _, make_weighted_f1, _ = precision_recall_fscore_support(y_true_make, y_pred_make, average="weighted", zero_division=0)

    res: Dict[str, Any] = {
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

    if shared_arr is not None and all_true_models:
        in_shared = np.isin(y_true_model, shared_arr)
        res["shared_n_images"] = int(in_shared.sum())
        res["shared_frac_of_test"] = float(in_shared.mean())
        if in_shared.any():
            yt = y_true_model[in_shared]
            yp_full = y_pred_model[in_shared]
            yp_restr = np.array(all_pred_restricted)[in_shared]
            _, _, f1_restr, _ = precision_recall_fscore_support(yt, yp_restr, average="macro", zero_division=0)
            res["shared_model_accuracy"] = float(np.mean(yt == yp_full))
            res["shared_model_accuracy_restricted"] = float(np.mean(yt == yp_restr))
            res["shared_model_macro_f1_restricted"] = float(f1_restr)
            res["shared_make_accuracy"] = float(np.mean(y_true_make[in_shared] == y_pred_make[in_shared]))
        else:
            for k in ("shared_model_accuracy", "shared_model_accuracy_restricted", "shared_model_macro_f1_restricted", "shared_make_accuracy"):
                res[k] = None

    return res


def combine_split_metrics(raw_a: Dict[str, List[int]], raw_b: Dict[str, List[int]],
                          shared_classes: Optional[List[int]] = None) -> Dict[str, Any]:
    return compute_split_metrics(
        raw_a["pred_makes"] + raw_b["pred_makes"],
        raw_a["true_makes"] + raw_b["true_makes"],
        raw_a["pred_models"] + raw_b["pred_models"],
        raw_a["true_models"] + raw_b["true_models"],
        raw_a["pred_restricted"] + raw_b["pred_restricted"],
        shared_classes=shared_classes,
    )


def evaluate_model_on_split(model: nn.Module, paths: List[str], true_make_indices: List[int],
                            true_model_indices: Optional[List[int]] = None,
                            proj_matrix: Optional[np.ndarray] = None,
                            bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
                            img_size: int = 512,
                            crop_top: Any = 0.0, crop_bottom: Any = 0.0,
                            batch_size: int = 64, device: str = "cuda",
                            shared_classes: Optional[List[int]] = None,
                            num_workers: int = 10,
                            return_raw: bool = False):
    """Make-level and Model-level evaluation using a multi-worker DataLoader."""
    model.eval()

    all_pred_makes: List[int] = []
    all_true_makes: List[int] = []
    all_pred_models: List[int] = []
    all_true_models: List[int] = []
    all_pred_restricted: List[int] = []
    shared_arr = np.array(shared_classes, dtype=int) if shared_classes else None
    n_failed = 0

    n_samples = len(paths)
    dataset = MixedBenchmarkDataset(paths, true_make_indices, true_model_indices, bboxes, img_size, crop_top, crop_bottom)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    pbar = tqdm(loader, desc="  Evaluating", unit="batch", leave=False)
    with torch.no_grad():
        for batch_tensor, b_make_labels, b_model_labels, valid_mask in pbar:
            valid_np = valid_mask.numpy()
            if not np.any(valid_np):
                n_failed += len(valid_np)
                continue
            if not np.all(valid_np):
                batch_tensor = batch_tensor[valid_np]
                b_make_labels = b_make_labels[valid_np]
                b_model_labels = b_model_labels[valid_np]
                n_failed += int(np.sum(~valid_np))

            if isinstance(model, oa.MobileNetV2Evaluator):
                logits = model(batch_tensor, return_embeddings=False, return_cam=False)
                probs = torch.softmax(logits, dim=-1).numpy()
            else:
                logits = model(batch_tensor.to(device, non_blocking=True))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # 1. Model predictions (1235 classes)
            pred_models = np.argmax(probs, axis=1)
            all_pred_models.extend(pred_models.tolist())
            if shared_arr is not None:
                all_pred_restricted.extend(shared_arr[np.argmax(probs[:, shared_arr], axis=1)].tolist())
            if true_model_indices is not None:
                all_true_models.extend(b_model_labels.numpy().tolist())

            # 2. Make predictions (marginalized over models)
            if proj_matrix is not None:
                make_probs = probs @ proj_matrix
            else:
                make_probs = probs

            pred_makes = np.argmax(make_probs, axis=1)
            all_pred_makes.extend(pred_makes.tolist())
            all_true_makes.extend(b_make_labels.numpy().tolist())

    if n_failed:
        print(f"[WARN] {n_failed:,} of {n_samples:,} images could not be read and were excluded.")

    res = compute_split_metrics(
        all_pred_makes, all_true_makes, all_pred_models, all_true_models, all_pred_restricted,
        shared_classes=shared_classes,
    )
    if return_raw:
        raw = {
            "pred_makes": all_pred_makes,
            "true_makes": all_true_makes,
            "pred_models": all_pred_models,
            "true_models": all_true_models,
            "pred_restricted": all_pred_restricted,
        }
        return res, raw
    return res


def _extract_make_names(df: pd.DataFrame) -> List[str]:
    if "make" in df.columns:
        makes = df["make"].fillna("").astype(str).str.strip().tolist()
        if all(makes):
            return makes
        cls_makes = df["class_name"].astype(str).str.split("/").str[0].tolist()
        return [m if m else cm for m, cm in zip(makes, cls_makes)]
    return df["class_name"].astype(str).str.split("/").str[0].tolist()


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

    # Class coverage: which model classes have train images in each dataset (label indices are shared by both maps)
    df_pm_all = pd.read_csv(args.pm_test_csv)
    pm_train_counts = (df_pm_all[df_pm_all["split"] == "train"] if "split" in df_pm_all.columns else df_pm_all)["label"].value_counts()
    ext_train_counts = pd.read_csv(args.ext_train_csv)["label"].value_counts()
    pm_train_cls, ext_train_cls = set(pm_train_counts.index.astype(int)), set(ext_train_counts.index.astype(int))
    shared_cls = sorted(pm_train_cls & ext_train_cls)
    print(f"[Coverage] classes with train images: PlatesMania {len(pm_train_cls):,} | External {len(ext_train_cls):,} "
          f"| shared {len(shared_cls):,} | of {len(pm_lm):,} total")
    if pm_lm != ext_lm:
        print("[WARN] PlatesMania and External label maps differ; shared-class metrics assume identical index spaces.")
    pd.DataFrame({"label": sorted(pm_train_cls | ext_train_cls)}).assign(
        platesmania_train_images=lambda d: d["label"].map(pm_train_counts).fillna(0).astype(int),
        external_train_images=lambda d: d["label"].map(ext_train_counts).fillna(0).astype(int),
    ).to_csv(output_dir / "class_coverage.csv", index=False)

    # 1. Prepare PlatesMania test split (reuse df_pm_all instead of re-reading 1.1M-row CSV)
    df_pm = df_pm_all[df_pm_all["split"] == "test"].reset_index(drop=True) if "split" in df_pm_all.columns else df_pm_all
    if len(df_pm) > args.max_eval_per_dataset:
        df_pm = df_pm.sample(n=args.max_eval_per_dataset, random_state=42).reset_index(drop=True)

    pm_col = "image_path" if "image_path" in df_pm.columns else "image_rel_path"
    strip_images = bool(args.pm_img_dir) and not os.path.isdir(os.path.join(args.pm_img_dir, "images"))
    pm_paths = [
        os.path.join(args.pm_img_dir, p[7:] if (strip_images and p.startswith("images/")) else p)
        if not os.path.isabs(p) else p
        for p in df_pm[pm_col].values
    ]
    pm_make_names = _extract_make_names(df_pm)
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
    ext_make_names = _extract_make_names(df_ext)
    ext_make_indices = [MAKE_TO_IDX.get(m) for m in ext_make_names]
    ext_model_indices = df_ext["label"].values.astype(int).tolist()

    ext_bboxes = None
    if "bbox_x1" in df_ext.columns:
        bx1 = df_ext["bbox_x1"].to_numpy(dtype=int)
        by1 = df_ext["bbox_y1"].to_numpy(dtype=int)
        bx2 = df_ext["bbox_x2"].to_numpy(dtype=int)
        by2 = df_ext["bbox_y2"].to_numpy(dtype=int)
        ext_bboxes = [
            (int(x1), int(y1), int(x2), int(y2)) if x1 >= 0 else None
            for x1, y1, x2, y2 in zip(bx1, by1, bx2, by2)
        ]

    ext_keep = [i for i, m in enumerate(ext_make_indices) if m is not None]
    ext_paths = [ext_paths[i] for i in ext_keep]
    ext_make_indices = [ext_make_indices[i] for i in ext_keep]
    ext_model_indices = [ext_model_indices[i] for i in ext_keep]
    if ext_bboxes is not None:
        ext_bboxes = [ext_bboxes[i] for i in ext_keep]
    else:
        ext_bboxes = [None] * len(ext_paths)

    n_mixed = len(pm_paths) + len(ext_paths)
    print(f"[Datasets] PlatesMania Test: {len(pm_paths):,} | External Test: {len(ext_paths):,} | Mixed Test: {n_mixed:,}")

    results = []
    same_model = (os.path.abspath(args.pm_model_path) == os.path.abspath(args.ext_model_path) and pm_lm == ext_lm)

    if same_model:
        print("\n--- Evaluating Model C (Merged) ---")
        model_c = oa.MobileNetV2Evaluator(num_classes=len(pm_lm), checkpoint_path=args.pm_model_path)
        res_a_pm, raw_a_pm = evaluate_model_on_split(
            model_c, pm_paths, pm_make_indices, pm_model_indices, pm_proj, pm_bboxes,
            args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size,
            device=device, shared_classes=shared_cls, num_workers=args.num_workers, return_raw=True
        )
        print(f"Model C on PlatesMania:      Make Acc={res_a_pm['make_accuracy']:.2%} | Model Acc={res_a_pm['model_accuracy']:.2%} | Make Macro F1={res_a_pm['make_macro_f1']:.2%}")

        res_a_ext, raw_a_ext = evaluate_model_on_split(
            model_c, ext_paths, ext_make_indices, ext_model_indices, pm_proj, ext_bboxes,
            args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size,
            device=device, shared_classes=shared_cls, num_workers=args.num_workers, return_raw=True
        )
        print(f"Model C on External:         Make Acc={res_a_ext['make_accuracy']:.2%} | Model Acc={res_a_ext['model_accuracy']:.2%} | Make Macro F1={res_a_ext['make_macro_f1']:.2%}")

        res_a_mix = combine_split_metrics(raw_a_pm, raw_a_ext, shared_classes=shared_cls)
        print(f"Model C on Mixed (Combined): Make Acc={res_a_mix['make_accuracy']:.2%} | Model Acc={res_a_mix['model_accuracy']:.2%} | Make Macro F1={res_a_mix['make_macro_f1']:.2%}")

        results.extend([
            {"model": "Model C (Merged)", "test_split": "PlatesMania", "domain_type": "In-Domain", **res_a_pm},
            {"model": "Model C (Merged)", "test_split": "External", "domain_type": "In-Domain", **res_a_ext},
            {"model": "Model C (Merged)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", **res_a_mix},
        ])
        res_b_pm = res_b_ext = res_b_mix = None
        del model_c
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # Benchmark Model A (PlatesMania)
        print("\n--- Evaluating Model A (PlatesMania) ---")
        model_a = oa.MobileNetV2Evaluator(num_classes=len(pm_lm), checkpoint_path=args.pm_model_path)
        res_a_pm, raw_a_pm = evaluate_model_on_split(
            model_a, pm_paths, pm_make_indices, pm_model_indices, pm_proj, pm_bboxes,
            args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size,
            device=device, shared_classes=shared_cls, num_workers=args.num_workers, return_raw=True
        )
        print(f"Model A on PlatesMania (In-Domain):     Make Acc={res_a_pm['make_accuracy']:.2%} | Model Acc={res_a_pm['model_accuracy']:.2%} | Make Macro F1={res_a_pm['make_macro_f1']:.2%}")

        res_a_ext, raw_a_ext = evaluate_model_on_split(
            model_a, ext_paths, ext_make_indices, ext_model_indices, pm_proj, ext_bboxes,
            args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size,
            device=device, shared_classes=shared_cls, num_workers=args.num_workers, return_raw=True
        )
        print(f"Model A on External    (Out-of-Domain): Make Acc={res_a_ext['make_accuracy']:.2%} | Model Acc={res_a_ext['model_accuracy']:.2%} | Make Macro F1={res_a_ext['make_macro_f1']:.2%}")

        res_a_mix = combine_split_metrics(raw_a_pm, raw_a_ext, shared_classes=shared_cls)
        print(f"Model A on Mixed       (Combined):      Make Acc={res_a_mix['make_accuracy']:.2%} | Model Acc={res_a_mix['model_accuracy']:.2%} | Make Macro F1={res_a_mix['make_macro_f1']:.2%}")

        results.extend([
            {"model": "Model A (PlatesMania)", "test_split": "PlatesMania (In-Domain)", "domain_type": "In-Domain", **res_a_pm},
            {"model": "Model A (PlatesMania)", "test_split": "External (Out-of-Domain)", "domain_type": "Out-of-Domain", **res_a_ext},
            {"model": "Model A (PlatesMania)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", **res_a_mix},
        ])

        del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Benchmark Model B (External)
        print("\n--- Evaluating Model B (External Merged) ---")
        model_b = oa.MobileNetV2Evaluator(num_classes=len(ext_lm), checkpoint_path=args.ext_model_path)
        res_b_ext, raw_b_ext = evaluate_model_on_split(
            model_b, ext_paths, ext_make_indices, ext_model_indices, ext_proj, ext_bboxes,
            args.img_size, crop_top=0.0, crop_bottom=0.05, batch_size=args.batch_size,
            device=device, shared_classes=shared_cls, num_workers=args.num_workers, return_raw=True
        )
        print(f"Model B on External    (In-Domain):     Make Acc={res_b_ext['make_accuracy']:.2%} | Model Acc={res_b_ext['model_accuracy']:.2%} | Make Macro F1={res_b_ext['make_macro_f1']:.2%}")

        res_b_pm, raw_b_pm = evaluate_model_on_split(
            model_b, pm_paths, pm_make_indices, pm_model_indices, ext_proj, pm_bboxes,
            args.img_size, crop_top=0.15, crop_bottom=0.0, batch_size=args.batch_size,
            device=device, shared_classes=shared_cls, num_workers=args.num_workers, return_raw=True
        )
        print(f"Model B on PlatesMania (Out-of-Domain): Make Acc={res_b_pm['make_accuracy']:.2%} | Model Acc={res_b_pm['model_accuracy']:.2%} | Make Macro F1={res_b_pm['make_macro_f1']:.2%}")

        res_b_mix = combine_split_metrics(raw_b_pm, raw_b_ext, shared_classes=shared_cls)
        print(f"Model B on Mixed       (Combined):      Make Acc={res_b_mix['make_accuracy']:.2%} | Model Acc={res_b_mix['model_accuracy']:.2%} | Make Macro F1={res_b_mix['make_macro_f1']:.2%}")

        del model_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.extend([
            {"model": "Model B (External)", "test_split": "External (In-Domain)", "domain_type": "In-Domain", **res_b_ext},
            {"model": "Model B (External)", "test_split": "PlatesMania (Out-of-Domain)", "domain_type": "Out-of-Domain", **res_b_pm},
            {"model": "Model B (External)", "test_split": "Mixed (Combined)", "domain_type": "Mixed", **res_b_mix},
        ])

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "mixed_benchmark_summary.csv", index=False)

    # Shared-class view: separates "class never seen in training" from "seen but misclassified under domain shift"
    shared_cols = ["model", "test_split", "model_accuracy", "shared_frac_of_test", "shared_n_images", "shared_model_accuracy",
                   "shared_model_accuracy_restricted", "shared_model_macro_f1_restricted", "shared_make_accuracy", "make_accuracy"]
    shared_df = results_df[[c for c in shared_cols if c in results_df.columns]]
    shared_df.to_csv(output_dir / "shared_class_summary.csv", index=False)
    print(f"\n[Shared classes: {len(shared_cls):,}] top-1 on test images whose class has train images in both datasets")
    print(shared_df.to_string(index=False, float_format=lambda v: f"{v:.2%}"))

    plot_benchmark_comparison(output_dir, same_model, res_a_pm, res_a_ext, res_a_mix, res_b_pm, res_b_ext, res_b_mix)
    print(f"\n[Done] Benchmark results and dual comparison charts saved to: {output_dir}")


def plot_benchmark_comparison(output_dir: Path, same_model: bool, res_a_pm: Dict, res_a_ext: Dict, res_a_mix: Dict,
                              res_b_pm: Optional[Dict] = None, res_b_ext: Optional[Dict] = None, res_b_mix: Optional[Dict] = None):
    """Plots Make Accuracy and Fine-Grained Model Accuracy across PlatesMania, External, and Mixed splits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    splits = ["PlatesMania Split", "External Split", "Mixed Split"]
    x = np.arange(len(splits))

    make_a_vals = [res_a_pm["make_accuracy"], res_a_ext["make_accuracy"], res_a_mix["make_accuracy"]]
    model_a_vals = [res_a_pm["model_accuracy"], res_a_ext["model_accuracy"], res_a_mix["model_accuracy"]]

    if same_model or res_b_pm is None:
        width = 0.45
        r1 = ax1.bar(x, [v * 100 for v in make_a_vals], width, label="Model C (Merged)", color="#2563EB")
        r3 = ax2.bar(x, [v * 100 for v in model_a_vals], width, label="Model C (Merged)", color="#2563EB")
        bars_ax1, bars_ax2 = list(r1), list(r3)
    else:
        width = 0.35
        make_b_vals = [res_b_pm["make_accuracy"], res_b_ext["make_accuracy"], res_b_mix["make_accuracy"]]
        model_b_vals = [res_b_pm["model_accuracy"], res_b_ext["model_accuracy"], res_b_mix["model_accuracy"]]
        r1 = ax1.bar(x - width/2, [v * 100 for v in make_a_vals], width, label="Model A (PlatesMania)", color="#2563EB")
        r2 = ax1.bar(x + width/2, [v * 100 for v in make_b_vals], width, label="Model B (External)", color="#16A34A")
        r3 = ax2.bar(x - width/2, [v * 100 for v in model_a_vals], width, label="Model A (PlatesMania)", color="#2563EB")
        r4 = ax2.bar(x + width/2, [v * 100 for v in model_b_vals], width, label="Model B (External)", color="#16A34A")
        bars_ax1, bars_ax2 = list(r1) + list(r2), list(r3) + list(r4)

    ax1.set_ylabel("Make Accuracy (%)", fontsize=12)
    ax1.set_title("Make Accuracy (Marginalized over Models)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(fontsize=10, loc="lower right")
    for r in bars_ax1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylabel("Model Accuracy (%)", fontsize=12)
    ax2.set_title("Fine-Grained Model Accuracy (1,235 Classes)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(fontsize=10, loc="lower right")
    for r in bars_ax2:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(output_dir / "cross_domain_comparison.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
