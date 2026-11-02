#!/usr/bin/env python3
"""
Open-set evaluation of a Model C ONNX (predictions + embeddings outputs).

Known set   = External test rows with label < 1235 (plus the PlatesMania test split when available).
Unknown set = External test rows with label >= 1235 (classes never seen when training with --taxonomy base).
Scores (higher = more "known"): MSP, MLS (max logit), Energy (logsumexp of logits), cosine to class-mean embeddings
(means from a per-class sample of train images).
Metrics: AUROC, FPR@95%TPR, OSCR, closed-set accuracy on the known set. Writes open_set_metrics.json and a score histogram.

Usage:
  python evaluate_open_set.py --onnx backup_run4/output_merged_run4/models/mobilenet_v2_best.onnx \
      [--ext-splits-dir run5/splits_1235models] [--pm-csv-path ...] [--no-pm] [--max-per-set 2000]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import torch
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import paths
from train_common import PlatesManiaDataset, ExternalMergedDataset, IMAGENET_MEAN, IMAGENET_STD, IGNORE_INDEX

BASE_CLASSES = 1235


def parse_args():
    p = argparse.ArgumentParser(description="Open-set evaluation (known vs unknown classes) of a Model C ONNX.")
    p.add_argument("--onnx", required=True)
    p.add_argument("--ext-splits-dir", default=str(CURRENT_DIR / "run5" / "splits_1235models"))
    p.add_argument("--pm-csv-path", default=paths.PM_MANIFEST_CSV)
    p.add_argument("--pm-img-dir", default=paths.PM_IMG_DIR)
    p.add_argument("--no-pm", action="store_true", help="Known set = External only")
    p.add_argument("--base-classes", type=int, default=BASE_CLASSES, help="Labels >= this are 'unknown'")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--pm-crop-top-pct", type=float, default=0.15)
    p.add_argument("--ext-crop-bottom-pct", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-per-set", type=int, default=0, help="Cap on images per set (0 = all), for quick runs")
    p.add_argument("--train-per-class", type=int, default=20, help="Train images per class for the cosine class means")
    p.add_argument("--output-dir", default=None, help="Default: <onnx dir>/../open_set")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_session(onnx_path):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def run_onnx(sess, ds, args):
    """Returns logits (N,C), embeddings (N,D), labels (N,) for the readable samples of `ds`."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    names = [o.name for o in sess.get_outputs()]
    li, ei = names.index("predictions"), names.index("embeddings")
    inp = sess.get_inputs()[0].name
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    L, E, Y = [], [], []
    for imgs, labels in loader:
        valid = labels != IGNORE_INDEX
        if not valid.any():
            continue
        x = ((imgs[valid].float() / 255.0) - mean) / std
        out = sess.run(None, {inp: x.numpy()})
        L.append(out[li]); E.append(out[ei]); Y.append(labels[valid].numpy())
    if not L:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros(0, dtype=np.int64)
    return np.concatenate(L), np.concatenate(E), np.concatenate(Y)


def subsample(df, n, seed):
    return df if (n <= 0 or len(df) <= n) else df.sample(n=n, random_state=seed).reset_index(drop=True)


def scores_from(logits, emb, class_means):
    z = logits.astype(np.float64)
    zmax = z.max(1)
    lse = zmax + np.log(np.exp(z - zmax[:, None]).sum(1))
    msp = np.exp(zmax - lse)
    out = {"MSP": msp, "MLS": zmax, "Energy": lse}
    if class_means is not None:
        e = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        out["Cosine"] = (e @ class_means.T).max(1)
    return out


def oscr(known_scores, known_correct, unknown_scores):
    """Area under CCR (correct and score >= t, over known) vs FPR (unknown with score >= t)."""
    s = np.concatenate([known_scores, unknown_scores])
    is_known = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    corr = np.concatenate([known_correct.astype(float), np.zeros(len(unknown_scores))])
    order = np.argsort(-s, kind="stable")
    ccr = np.cumsum(corr[order]) / len(known_scores)
    fpr = np.cumsum(1 - is_known[order]) / len(unknown_scores)
    return float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(ccr, fpr))


def fpr_at_tpr(known_scores, unknown_scores, tpr_level=0.95):
    y = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([known_scores, unknown_scores]))
    return float(fpr[np.searchsorted(tpr, tpr_level, side="left")])


def main():
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.onnx).resolve().parent.parent / "open_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = Path(args.ext_splits_dir)
    ex_test = pd.read_csv(sd / "test.csv")
    ex_train = pd.read_csv(sd / "train.csv")
    sess = make_session(args.onnx)
    print(f"[ONNX] {args.onnx} | providers {sess.get_providers()}")

    def ext_ds(df):
        return ExternalMergedDataset(df, args.img_size, args.ext_crop_bottom_pct)

    ex_known = subsample(ex_test[ex_test["label"] < args.base_classes].reset_index(drop=True), args.max_per_set, args.seed)
    ex_unknown = subsample(ex_test[ex_test["label"] >= args.base_classes].reset_index(drop=True), args.max_per_set, args.seed)
    if len(ex_unknown) == 0:
        sys.exit(f"[ERROR] No External test rows with label >= {args.base_classes} in {sd / 'test.csv'}")
    print(f"[Data] External known {len(ex_known):,} | unknown {len(ex_unknown):,}")

    parts = [run_onnx(sess, ext_ds(ex_known), args)]
    if not args.no_pm:
        pm_df = pd.read_csv(args.pm_csv_path)
        pm_test = subsample(pm_df[pm_df["split"] == "test"].reset_index(drop=True), args.max_per_set, args.seed)
        pm_ds = PlatesManiaDataset(pm_test, Path(args.pm_img_dir), args.img_size, args.pm_crop_top_pct, 0.0)
        parts.append(run_onnx(sess, pm_ds, args))
        print(f"[Data] PlatesMania known {len(pm_test):,}")
    kL = np.concatenate([p[0] for p in parts]); kE = np.concatenate([p[1] for p in parts])
    kY = np.concatenate([p[2] for p in parts])
    uL, uE, _ = run_onnx(sess, ext_ds(ex_unknown), args)

    # class-mean embeddings (normalized) from a per-class train sample
    tr = ex_train[ex_train["label"] < args.base_classes]
    tr = tr.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), args.train_per_class), random_state=args.seed))
    tr = subsample(tr.reset_index(drop=True), args.max_per_set * 2 if args.max_per_set else 0, args.seed)
    _, tE, tY = run_onnx(sess, ext_ds(tr), args)
    num_classes = kL.shape[1]
    class_means = np.zeros((num_classes, tE.shape[1]), dtype=np.float64)
    tEn = tE / np.clip(np.linalg.norm(tE, axis=1, keepdims=True), 1e-8, None)
    for c in np.unique(tY):
        class_means[c] = tEn[tY == c].mean(0)
    norms = np.linalg.norm(class_means, axis=1, keepdims=True)
    class_means = np.where(norms > 0, class_means / np.clip(norms, 1e-8, None), 0.0)
    print(f"[Cosine] class means for {len(np.unique(tY))}/{num_classes} classes from {len(tY):,} train images")

    correct = kL.argmax(1) == kY
    ks, us = scores_from(kL, kE, class_means), scores_from(uL, uE, class_means)
    metrics = {"onnx": str(args.onnx), "n_known": int(len(kY)), "n_unknown": int(len(uL)),
               "closed_set_acc": float(correct.mean()), "scores": {}}
    for name in ks:
        y = np.concatenate([np.ones(len(ks[name])), np.zeros(len(us[name]))])
        metrics["scores"][name] = {
            "AUROC": float(roc_auc_score(y, np.concatenate([ks[name], us[name]]))),
            "FPR@95TPR": fpr_at_tpr(ks[name], us[name]),
            "OSCR": oscr(ks[name], correct, us[name]),
        }
        print(f"[{name:7s}] AUROC {metrics['scores'][name]['AUROC']:.4f} | FPR@95 {metrics['scores'][name]['FPR@95TPR']:.4f} "
              f"| OSCR {metrics['scores'][name]['OSCR']:.4f}")
    print(f"[Closed-set] known-set accuracy {metrics['closed_set_acc']:.2%}")

    with open(out_dir / "open_set_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 3.5))
    for ax, name in zip(np.atleast_1d(axes), ks):
        ax.hist(ks[name], bins=50, alpha=0.6, density=True, label="known")
        ax.hist(us[name], bins=50, alpha=0.6, density=True, label="unknown")
        ax.set_title(f"{name} (AUROC {metrics['scores'][name]['AUROC']:.3f})"); ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "open_set_scores.png", dpi=130)
    plt.close()
    print(f"[Out] {out_dir}")


if __name__ == "__main__":
    main()
