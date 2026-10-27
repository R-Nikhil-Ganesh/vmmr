#!/usr/bin/env python3
"""
E-05 Verification: Compare Color-Head Extraction Preprocessing vs. DiagnosticDataset
====================================================================================
Compares 2,000 validation images processed via:
  (a) Legacy ImageExtractDataset (PIL.Image.BILINEAR without antialiasing)
  (b) output_analysis.DiagnosticDataset (torchvision.io.read_image + antialiased F.interpolate)
and reports:
  - Relative L2 difference ||f_a - f_b||_2 / ||f_b||_2 on pooled features[16] (160-D)
  - Mean cosine similarity on features[16]
  - Top-1 argmax agreement on downstream color head predictions
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR))
sys.path.insert(0, str(MOBILENET_DIR / "annotation"))

import output_analysis as oa
from common import DOMAINS, load_crop
from color_distillation.train_color_head import (
    COLOR2IDX,
    COLORS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    L16_ONNX_OUTPUT,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", default=str(MOBILENET_DIR / "annotation_runs/run2/annotations.csv"))
    p.add_argument("--multitask-onnx", default=str(SCRIPT_DIR / "output/mobilenet_v2_multitask.onnx"))
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.annotations)
    val_df = df[(df["split"] == "val") & (df["color"].isin(COLORS))].sample(
        n=min(args.num_samples, len(df[df["split"] == "val"])), random_state=args.seed
    ).reset_index(drop=True)
    print(f"[E-05] Sampled {len(val_df):,} validation rows from {args.annotations}")

    model = onnx.load(args.multitask_onnx)
    if L16_ONNX_OUTPUT not in [o.name for o in model.graph.output]:
        vi = onnx.helper.ValueInfoProto()
        vi.name = L16_ONNX_OUTPUT
        model.graph.output.append(vi)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model.SerializeToString(), providers=providers)

    feats_pil, preds_pil = [], []
    feats_diag, preds_diag = [], []

    # Process by domain so DiagnosticDataset uses each domain's crop settings
    for domain, cfg in DOMAINS.items():
        sub = val_df[val_df["domain"] == domain].reset_index(drop=True)
        if sub.empty:
            continue
        bboxes = [
            (int(r.bbox_x1), int(r.bbox_y1), int(r.bbox_x2), int(r.bbox_y2)) if int(r.bbox_x1) >= 0 else None
            for r in sub.itertuples(index=False)
        ]
        diag_ds = oa.DiagnosticDataset(
            sub["path"].tolist(),
            np.zeros(len(sub), dtype=np.int64),
            img_size=512,
            crop_top_pct=cfg["crop_top_pct"],
            crop_bottom_pct=cfg["crop_bottom_pct"],
            bboxes=bboxes,
        )
        for idx in range(0, len(sub), args.batch_size):
            end = min(idx + args.batch_size, len(sub))
            batch_pil = []
            batch_diag = []
            for j in range(idx, end):
                r = sub.iloc[j]
                bb = bboxes[j]
                im = load_crop(r["path"], domain, bb)
                im_r = im.resize((512, 512), Image.Resampling.BILINEAR)
                arr = np.transpose(np.asarray(im_r, dtype=np.float32) / 255.0, (2, 0, 1))
                arr = (arr - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                batch_pil.append(arr)

                t_diag, _, _ = diag_ds[j]
                batch_diag.append(t_diag.numpy())

            x_pil = np.stack(batch_pil, axis=0)
            x_diag = np.stack(batch_diag, axis=0)

            l16_p, col_p = sess.run([L16_ONNX_OUTPUT, "color_predictions"], {"input_image": x_pil})
            l16_d, col_d = sess.run([L16_ONNX_OUTPUT, "color_predictions"], {"input_image": x_diag})

            feats_pil.append(l16_p.mean(axis=(2, 3)))
            preds_pil.append(col_p.argmax(axis=1))
            feats_diag.append(l16_d.mean(axis=(2, 3)))
            preds_diag.append(col_d.argmax(axis=1))

    f_a = np.concatenate(feats_pil, axis=0)
    f_b = np.concatenate(feats_diag, axis=0)
    p_a = np.concatenate(preds_pil, axis=0)
    p_b = np.concatenate(preds_diag, axis=0)

    rel_l2_per_sample = np.linalg.norm(f_a - f_b, axis=1) / np.maximum(np.linalg.norm(f_b, axis=1), 1e-8)
    global_rel_l2 = float(np.linalg.norm(f_a - f_b) / np.linalg.norm(f_b))
    max_abs_diff = float(np.max(np.abs(f_a - f_b)))
    cos_sim = float(np.mean(np.sum(f_a * f_b, axis=1) / (np.linalg.norm(f_a, axis=1) * np.linalg.norm(f_b, axis=1) + 1e-8)))
    pred_agree = float(np.mean(p_a == p_b) * 100.0)

    summary = {
        "num_samples": int(len(f_a)),
        "global_relative_l2": round(global_rel_l2, 6),
        "mean_sample_relative_l2": round(float(np.mean(rel_l2_per_sample)), 6),
        "median_sample_relative_l2": round(float(np.median(rel_l2_per_sample)), 6),
        "max_abs_feature_diff": round(max_abs_diff, 6),
        "mean_cosine_similarity": round(cos_sim, 6),
        "color_top1_agreement_pct": round(pred_agree, 2),
    }
    out_json = SCRIPT_DIR / "output" / "e05_preprocessing_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 65)
    print("  E-05 PREPROCESSING COMPARISON SUMMARY (2,000 Val Images)")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    main()
