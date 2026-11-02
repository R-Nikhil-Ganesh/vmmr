#!/usr/bin/env python3
"""
Build the class-similarity matrix S for KD label smoothing (--kd-ls-matrix).

Runs a teacher over the train split WITHOUT augmentation and accumulates the mean softmax(z/T) per ground-truth class.
The diagonal is zeroed and rows are normalized to sum to 1, so a training target becomes
    (1 - alpha) * one_hot(y) + alpha * S[y]
Classes with no train samples (or an all-zero row) fall back to a uniform spread over the other classes of the same make.

Teacher: --teacher-pt (a --save-pt checkpoint) or --teacher-onnx (rebuilt via onnx_to_torch.py).
  python build_kd_smoothing_matrix.py --teacher-onnx backup_run4/output_merged_run4/models/mobilenet_v2_best.onnx --out S_run4.npy
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import paths
from train_common import (PlatesManiaDataset, ExternalMergedDataset, build_val_transform, load_vmmr_from_pt, IGNORE_INDEX)
from onnx_to_torch import load_torch_from_onnx


def parse_args():
    p = argparse.ArgumentParser(description="Build the KD label-smoothing matrix from a teacher.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--teacher-pt")
    g.add_argument("--teacher-onnx")
    p.add_argument("--out", required=True, help="Output .npy (num_classes x num_classes, float32)")
    p.add_argument("--num-classes", type=int, default=1235)
    p.add_argument("--label-map-path", default=paths.PM_LABEL_MAP, help="Base label map (for the same-make fallback)")
    p.add_argument("--ext-splits-dir", default=str(CURRENT_DIR / "run5" / "splits_1235models"))
    p.add_argument("--pm-csv-path", default=paths.PM_MANIFEST_CSV)
    p.add_argument("--pm-img-dir", default=paths.PM_IMG_DIR)
    p.add_argument("--no-pm", action="store_true")
    p.add_argument("--temp", type=float, default=4.0)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--pm-crop-top-pct", type=float, default=0.15)
    p.add_argument("--ext-crop-bottom-pct", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-per-class", type=int, default=0, help="Cap train images per class (0 = all); useful for quick runs")
    return p.parse_args()


def finalize(acc: np.ndarray, counts: np.ndarray, class_names=None) -> np.ndarray:
    """acc[c] = summed softmax rows of class c. Returns row-normalized S with zero diagonal."""
    n = acc.shape[0]
    S = np.zeros_like(acc, dtype=np.float64)
    for c in range(n):
        row = acc[c] / counts[c] if counts[c] > 0 else np.zeros(n)
        row[c] = 0.0
        if row.sum() <= 0:
            if class_names is not None and "/" in (class_names[c] or ""):
                mk = class_names[c].split("/", 1)[0]
                row = np.array([1.0 if (j != c and (class_names[j] or "").startswith(mk + "/")) else 0.0 for j in range(n)])
            if row.sum() <= 0:
                row = np.ones(n); row[c] = 0.0
        S[c] = row / row.sum()
    return S.astype(np.float32)


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (load_vmmr_from_pt(a.teacher_pt) if a.teacher_pt else load_torch_from_onnx(a.teacher_onnx)).to(device).eval()
    n = a.num_classes
    if model.classifier[1].out_features != n:
        sys.exit(f"[ERROR] teacher has {model.classifier[1].out_features} classes, --num-classes is {n}")
    raw = json.load(open(a.label_map_path)); c2i = {k: int(v) for k, v in raw.get("class_to_idx", raw).items()}
    names = [None] * n
    for k, i in c2i.items():
        if i < n:
            names[i] = k

    ex = pd.read_csv(Path(a.ext_splits_dir) / "train.csv")
    ex = ex[ex["label"] < n].reset_index(drop=True)
    datasets = [ExternalMergedDataset(ex, a.img_size, a.ext_crop_bottom_pct)]
    if not a.no_pm:
        pm = pd.read_csv(a.pm_csv_path)
        pm = pm[pm["split"] == "train"].reset_index(drop=True)
        datasets.append(PlatesManiaDataset(pm, Path(a.pm_img_dir), a.img_size, a.pm_crop_top_pct, 0.0))
    if a.max_per_class:
        for i, ds in enumerate(datasets):
            keep = pd.Series(ds.labels).groupby(ds.labels).head(a.max_per_class).index.values
            datasets[i] = torch.utils.data.Subset(ds, keep)

    tf = build_val_transform()
    acc = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for ds in datasets:
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)
        for imgs, labels in tqdm(loader, desc="teacher"):
            valid = labels != IGNORE_INDEX
            if not valid.any():
                continue
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                z = model(tf(imgs[valid].to(device)))
            p = torch.softmax(z.float() / a.temp, dim=-1).cpu().numpy()
            y = labels[valid].numpy()
            np.add.at(acc, y, p)
            np.add.at(counts, y, 1)
    S = finalize(acc, counts, names)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, S)
    print(f"[KD-LS] saved {a.out} {S.shape}; classes with samples {int((counts > 0).sum())}/{n}; "
          f"row sums in [{S.sum(1).min():.4f}, {S.sum(1).max():.4f}]")


if __name__ == "__main__":
    main()
