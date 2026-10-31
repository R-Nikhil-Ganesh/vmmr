#!/usr/bin/env python3
"""
Evaluate MobileNetV2 Multitask Color Head against the vCorR (colour_dataset) Benchmark
=====================================================================================
Evaluates trained MobileNetV2 color heads on the complete 10,373-image vCorR dataset
across train (7,334), val (1,519), and test (1,520) splits.

Taxonomy mapping:
- Direct 11: white, black, silver, red, blue, green, yellow, orange, brown, beige, gold
- Spelling: grey -> gray
- Extended: pink -> other, purple -> other, tan -> beige (or other)
"""

import os
import sys
import glob
import site
import ctypes
import json
import time
import argparse
from pathlib import Path

# Preload NVIDIA CUDA libs for onnxruntime CUDAExecutionProvider
for _sp in site.getsitepackages():
    for _lib in sorted(glob.glob(os.path.join(_sp, "nvidia", "*", "lib", "*.so*"))):
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

COLORS = [
    "white", "black", "silver", "gray", "red", "blue", "green",
    "yellow", "orange", "brown", "beige", "gold", "other"
]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

# Direct & Extended Mappings for vCorR classes
VCORR_DIRECT_CLASSES = {
    "black": "black",
    "white": "white",
    "silver": "silver",
    "grey": "gray",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "beige": "beige",
    "gold": "gold",
}

VCORR_FULL_MAPPING = {
    **VCORR_DIRECT_CLASSES,
    "pink": "other",
    "purple": "other",
    "tan": "beige",  # standard automotive definition
}

VCORR_ALT_MAPPING = {
    **VCORR_DIRECT_CLASSES,
    "pink": "other",
    "purple": "other",
    "tan": "other",  # alternative: tan as other
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class VCorrDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img = Image.open(r["filepath"]).convert("RGB").resize((512, 512), Image.BILINEAR)
            arr = (np.array(img, dtype=np.float32) / 255.0).transpose((2, 0, 1))
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            return torch.from_numpy(arr), idx, True
        except Exception:
            return torch.zeros((3, 512, 512), dtype=torch.float32), idx, False


def scan_dataset(dataset_dir):
    records = []
    base = Path(dataset_dir)
    for split in ["train", "val", "test"]:
        s_dir = base / split
        if not s_dir.exists():
            continue
        for c_dir in sorted([d for d in s_dir.iterdir() if d.is_dir()]):
            vcorr_class = c_dir.name
            for img_path in sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpeg"))):
                records.append({
                    "split": split,
                    "vcorr_class": vcorr_class,
                    "filepath": str(img_path),
                    "filename": img_path.name,
                    "mapped_class": VCORR_FULL_MAPPING.get(vcorr_class, "other"),
                    "alt_mapped_class": VCORR_ALT_MAPPING.get(vcorr_class, "other"),
                    "is_direct": vcorr_class in VCORR_DIRECT_CLASSES,
                })
    return records


def evaluate_model(onnx_path, records, batch_size=64, num_workers=8):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    output_names = [o.name for o in session.get_outputs()]
    color_idx = output_names.index("color_predictions")

    dataset = VCorrDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_preds = []
    all_confs = []
    all_top3 = []
    all_probs = []

    t0 = time.time()
    for batch_imgs, batch_idxs, valid_mask in loader:
        b_np = batch_imgs.numpy()
        res = session.run(None, {"input_image": b_np})
        color_logits = res[color_idx]
        
        # Softmax
        exp_logits = np.exp(color_logits - np.max(color_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        top3_indices = np.argsort(-probs, axis=1)[:, :3]
        top1_preds = [COLORS[idx] for idx in top3_indices[:, 0]]
        top1_confs = probs[np.arange(len(probs)), top3_indices[:, 0]].tolist()
        top3_classes = [[COLORS[i] for i in row] for row in top3_indices]

        all_preds.extend(top1_preds)
        all_confs.extend(top1_confs)
        all_top3.extend(top3_classes)
        all_probs.append(probs)

    elapsed = time.time() - t0
    all_probs = np.vstack(all_probs)
    fps = len(records) / max(elapsed, 0.001)

    return all_preds, all_confs, all_top3, all_probs, fps


def compute_metrics(df, pred_col, gt_col):
    total = len(df)
    if total == 0:
        return {}

    correct = (df[pred_col] == df[gt_col]).sum()
    top1_acc = (correct / total) * 100.0

    # Top-3 accuracy
    top3_correct = df.apply(lambda r: r[gt_col] in r["top3_preds"], axis=1).sum()
    top3_acc = (top3_correct / total) * 100.0

    # Per-class precision, recall, f1
    classes = sorted(df[gt_col].unique())
    per_class = {}
    f1s, precs, recs = [], [], []

    for c in classes:
        tp = ((df[pred_col] == c) & (df[gt_col] == c)).sum()
        fp = ((df[pred_col] == c) & (df[gt_col] != c)).sum()
        fn = ((df[pred_col] != c) & (df[gt_col] == c)).sum()
        n_gt = (df[gt_col] == c).sum()

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "n": int(n_gt),
            "precision": round(float(prec), 2),
            "recall": round(float(rec), 2),
            "f1": round(float(f1), 2),
        }
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    return {
        "n_samples": int(total),
        "top1_acc": round(float(top1_acc), 2),
        "top3_acc": round(float(top3_acc), 2),
        "macro_f1": round(float(np.mean(f1s)), 2),
        "macro_precision": round(float(np.mean(precs)), 2),
        "macro_recall": round(float(np.mean(recs)), 2),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Color Head on vCorR")
    parser.add_argument("--vcorr-dir", default="/home/researchadmin/Econ-n/repo-clone/colour_dataset")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    vcorr_dir = Path(args.vcorr_dir)
    print(f"[vCorR] Scanning dataset in {vcorr_dir}...")
    records = scan_dataset(vcorr_dir)
    print(f"[vCorR] Found {len(records):,} images across all splits.")

    models_to_test = [
        ("Deep MLP-512x2 (Best)", "/home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx"),
        ("Unweighted MLP-128 (Ablation 1)", "/home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/output_unweighted_mlp128/mobilenet_v2_multitask.onnx"),
        ("Baseline MLP-128 (Class Weighted)", "/home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/output/mobilenet_v2_multitask.onnx"),
    ]

    all_results = {}

    for model_name, model_path in models_to_test:
        if not Path(model_path).exists():
            print(f"[WARN] Skipping {model_name}, file not found: {model_path}")
            continue

        print(f"\n==================================================================")
        print(f"  Evaluating: {model_name}")
        print(f"  Path:       {model_path}")
        print(f"==================================================================")

        preds, confs, top3, probs, fps = evaluate_model(
            model_path, records, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(f"  Inference speed: {fps:.1f} images/sec on GPU")

        df = pd.DataFrame(records)
        df["pred_color"] = preds
        df["confidence"] = confs
        df["top3_preds"] = top3

        # 1. Overall Direct Classes (12 classes: 8,348 images)
        df_direct = df[df["is_direct"]].copy()
        direct_overall = compute_metrics(df_direct, "pred_color", "mapped_class")
        
        # 2. Split Breakdown on Direct Classes
        direct_splits = {}
        for sp in ["test", "val", "train"]:
            sub = df_direct[df_direct["split"] == sp]
            direct_splits[sp] = compute_metrics(sub, "pred_color", "mapped_class")

        # 3. Overall Full 15 Classes (tan -> beige)
        full_overall = compute_metrics(df, "pred_color", "mapped_class")

        # 4. Split Breakdown on Full 15 Classes
        full_splits = {}
        for sp in ["test", "val", "train"]:
            sub = df[df["split"] == sp]
            full_splits[sp] = compute_metrics(sub, "pred_color", "mapped_class")

        # 5. Alt Full 15 Classes (tan -> other)
        alt_overall = compute_metrics(df, "pred_color", "alt_mapped_class")

        print(f"\n  [Direct 12 Classes - n={direct_overall['n_samples']:,}]")
        print(f"    All:  Top-1 Acc = {direct_overall['top1_acc']:.2f}%, Top-3 Acc = {direct_overall['top3_acc']:.2f}%, Macro F1 = {direct_overall['macro_f1']:.2f}%")
        print(f"    Test: Top-1 Acc = {direct_splits['test']['top1_acc']:.2f}%, Top-3 Acc = {direct_splits['test']['top3_acc']:.2f}%, Macro F1 = {direct_splits['test']['macro_f1']:.2f}% (n={direct_splits['test']['n_samples']:,})")
        print(f"    Val:  Top-1 Acc = {direct_splits['val']['top1_acc']:.2f}%, Top-3 Acc = {direct_splits['val']['top3_acc']:.2f}%, Macro F1 = {direct_splits['val']['macro_f1']:.2f}% (n={direct_splits['val']['n_samples']:,})")

        print(f"\n  [Full 15 Classes (tan->beige) - n={full_overall['n_samples']:,}]")
        print(f"    All:  Top-1 Acc = {full_overall['top1_acc']:.2f}%, Top-3 Acc = {full_overall['top3_acc']:.2f}%, Macro F1 = {full_overall['macro_f1']:.2f}%")
        print(f"    Test: Top-1 Acc = {full_splits['test']['top1_acc']:.2f}%, Top-3 Acc = {full_splits['test']['top3_acc']:.2f}%, Macro F1 = {full_splits['test']['macro_f1']:.2f}% (n={full_splits['test']['n_samples']:,})")

        all_results[model_name] = {
            "fps": round(fps, 1),
            "direct_overall": direct_overall,
            "direct_splits": direct_splits,
            "full_overall": full_overall,
            "full_splits": full_splits,
            "alt_overall": alt_overall,
        }

    # Save JSON if requested
    out_json = args.output_json or "/home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/output/vcorr_eval_results.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Saved full evaluation metrics to: {out_json}")


if __name__ == "__main__":
    main()
