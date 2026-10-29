#!/usr/bin/env python3
"""
Score a color teacher (a color_heads.py head, e.g. Qwen3-VL with given options) against human color labels.

Ground truth: CompCars surveillance human colors (external_metadata_51make.csv, color_label >= 0), joined to an
annotation run's annotations.csv for split and crop info (same join as E-09, evaluate_compcars_color_gt.py).
Optionally also the filled-in rows of the 13-color human review sheet (color_human_review_600.csv).

CompCars has 10 colors (gray inside silver, beige/gold inside champagne), so we report:
  strict  = 13-color prediction must equal the human label (gray on a silver car is wrong)
  grouped = prediction mapped to the CompCars taxonomy first (QWEN_TO_COMPCARS_GROUPED)

Outputs in --out-dir: predictions.csv (one row per image, full probability vector), probs.npy, metrics.json,
run_config.json. A one-line summary is appended to <out-dir>/../summary.csv so configurations line up.

Example (research PC, 4090):
  python annotation/eval_color_teacher.py --tag bf16_letters --split val --no-4bit --scoring letters
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MOBILENET_DIR / "color_distillation"))

import paths  # noqa: E402
from color_heads import COLORS, PROMPTS, build_head  # noqa: E402
from common import load_crop, parse_bbox, to_rel_image_path  # noqa: E402
from evaluate_compcars_color_gt import (COMPCARS_ID2COLOR, QWEN_TO_COMPCARS_GROUPED,  # noqa: E402
                                        QWEN_TO_COMPCARS_STRICT)
from train_common import save_run_config  # noqa: E402

ANNOT_DIR = Path(paths.ANNOT_OUT_DIR or MOBILENET_DIR / "annotation_runs")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="name of this configuration (output folder)")
    p.add_argument("--annotations", default=str(ANNOT_DIR / "run2" / "annotations.csv"),
                   help="annotation run with all splits (for split, bbox and the old Qwen color)")
    p.add_argument("--ext-meta", default=str(Path(paths.EXT_DATASETS_DIR) / "external_metadata_51make.csv"))
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"],
                   help="CompCars human-GT rows to score. Tune on val; touch test once at the end")
    p.add_argument("--review-csv", default=None,
                   help="also score filled-in rows (human_color != '') of the 13-color review sheet")
    p.add_argument("--max-samples", type=int, default=None, help="random subsample (quick dry runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help=f"default {ANNOT_DIR}/teacher_eval/<tag>")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8, help="image-loading threads")
    # Head options (defaults = Annotation Runs 1/2)
    p.add_argument("--color-head", default="qwen3vl")
    p.add_argument("--qwen-model", default=paths.QWEN_MODEL_DIR or "Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--max-side", type=int, default=336)
    p.add_argument("--no-4bit", action="store_true", help="bfloat16 instead of 4-bit NF4")
    p.add_argument("--scoring", default="first_token", choices=["first_token", "variants", "letters"])
    p.add_argument("--prompt-style", default="short", choices=sorted(PROMPTS))
    p.add_argument("--center-crop", type=float, default=0.0, help="fraction trimmed off, split over both sides")
    p.add_argument("--tta", default="none", choices=["none", "flip"])
    return p.parse_args()


def load_compcars_gt(args) -> pd.DataFrame:
    ann = pd.read_csv(args.annotations)
    ann = ann[ann["domain"] == "external"].copy()
    ann["rel_path"] = [to_rel_image_path(p, "external") for p in ann["path"].astype(str)]
    meta = pd.read_csv(args.ext_meta)
    meta = meta[meta["color_label"] >= 0].copy()
    meta["rel_path"] = [to_rel_image_path(p, "external") for p in meta["image_path"].astype(str)]
    meta["gt_color"] = meta["color_label"].astype(int).map(COMPCARS_ID2COLOR)
    df = ann.merge(meta[["rel_path", "gt_color"]], on="rel_path", how="inner").drop_duplicates("rel_path")
    df = df.rename(columns={"color": "old_color", "color_conf": "old_color_conf"})
    df["gt_source"] = "compcars"
    if args.split != "all":
        df = df[df["split"] == args.split]
    return df


def load_review(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"human_color": str}).fillna({"human_color": ""})
    df = df[df["human_color"].str.strip() != ""].copy()
    df["gt_color"] = df["human_color"].str.strip().str.lower()
    bad = sorted(set(df["gt_color"]) - set(COLORS))
    if bad:
        raise ValueError(f"{path}: human_color values outside COLORS: {bad}")
    df = df.rename(columns={"color": "old_color", "color_conf": "old_color_conf"})
    df["gt_source"] = "review"
    return df


def metrics(gt: pd.Series, pred: pd.Series, source: str) -> dict:
    """Accuracy of 13-color predictions. CompCars rows: strict and grouped; review rows: 13-color exact."""
    if source == "review":
        ok = pred.values == gt.values
        return {"n": int(len(gt)), "acc_pct": round(float(ok.mean() * 100), 2),
                "recall_pct": {c: round(float(ok[gt.values == c].mean() * 100), 2)
                               for c in COLORS if (gt.values == c).any()}}
    strict = pred.map(QWEN_TO_COMPCARS_STRICT).values == gt.values
    grouped_pred = pred.map(QWEN_TO_COMPCARS_GROUPED)
    grouped = grouped_pred.values == gt.values
    labels = [c for c in COMPCARS_ID2COLOR.values() if (gt.values == c).any()]
    conf = pd.crosstab(pd.Categorical(gt, labels), pd.Categorical(grouped_pred, list(COMPCARS_ID2COLOR.values())),
                       rownames=["human"], colnames=["pred_grouped"], dropna=False)
    return {
        "n": int(len(gt)),
        "strict_acc_pct": round(float(strict.mean() * 100), 2),
        "grouped_acc_pct": round(float(grouped.mean() * 100), 2),
        "grouped_recall_pct": {c: round(float(grouped[gt.values == c].mean() * 100), 2) for c in labels},
        "grouped_precision_pct": {c: round(float((gt.values[grouped_pred.values == c] == c).mean() * 100), 2)
                                  for c in labels if (grouped_pred.values == c).any()},
        "confusion_grouped": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in conf.iterrows()},
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or ANNOT_DIR / "teacher_eval" / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(out_dir, args, "annotation/eval_color_teacher.py")

    parts = [load_compcars_gt(args)]
    if args.review_csv:
        parts.append(load_review(args.review_csv))
    df = pd.concat(parts, ignore_index=True)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"[eval] {len(df):,} rows: {df['gt_source'].value_counts().to_dict()} (split={args.split})")

    head = build_head(args.color_head, model_id=args.qwen_model, max_side=args.max_side,
                      load_in_4bit=not args.no_4bit, scoring=args.scoring, prompt_style=args.prompt_style,
                      center_crop=args.center_crop, tta=args.tta)

    def load(r):
        try:
            # full resolution; the head applies center_crop then max_side, so crops don't lose pixels
            return load_crop(r["path"], r["domain"], parse_bbox(r), max_side=None)
        except Exception as e:
            print(f"[eval] cannot read {r['path']}: {e}")
            return None

    recs = df.to_dict("records")
    probs = np.full((len(df), len(COLORS)), np.nan, dtype=np.float32)
    mass = np.full(len(df), np.nan, dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(args.num_workers) as pool:
        for i in range(0, len(recs), args.batch_size):
            imgs = list(pool.map(load, recs[i:i + args.batch_size]))
            idx = [i + j for j, im in enumerate(imgs) if im is not None]
            if idx:
                p, m = head.predict_proba([im for im in imgs if im is not None])
                probs[idx], mass[idx] = p, m
            done = min(i + args.batch_size, len(recs))
            if done % (args.batch_size * 50) < args.batch_size or done == len(recs):
                print(f"[eval] {done:,}/{len(recs):,} ({done / (time.time() - t0):.1f} img/s)")
    elapsed = time.time() - t0

    valid = ~np.isnan(probs).any(1)
    df["color"] = [COLORS[int(p.argmax())] if v else "" for p, v in zip(probs, valid)]
    df["color_conf"] = np.where(valid, np.nan_to_num(probs).max(1), np.nan)
    df["option_mass"] = mass
    for k, c in enumerate(COLORS):
        df[f"p_{c}"] = probs[:, k]
    keep = ["path", "domain", "split", "gt_source", "gt_color", "old_color", "old_color_conf",
            "color", "color_conf", "option_mass"] + [f"p_{c}" for c in COLORS]
    df[keep].to_csv(out_dir / "predictions.csv", index=False)
    np.save(out_dir / "probs.npy", probs)

    res = {"tag": args.tag, "split": args.split, "n_unreadable": int((~valid).sum()),
           "img_per_s": round(float(valid.sum() / elapsed), 2),
           "mean_option_mass": round(float(np.nanmean(mass)), 4)}
    for src in df["gt_source"].unique():
        sub = df[(df["gt_source"] == src) & valid]
        res[src] = metrics(sub["gt_color"], sub["color"], src)
        old = sub[sub["old_color"].notna()]
        res[src + "_old_teacher_same_rows"] = {k: v for k, v in metrics(old["gt_color"], old["old_color"], src).items()
                                               if k.endswith("acc_pct") or k == "n"}
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))

    line = {"tag": args.tag, "split": args.split, "max_side": args.max_side, "4bit": not args.no_4bit,
            "scoring": args.scoring, "prompt": args.prompt_style, "center_crop": args.center_crop, "tta": args.tta,
            "model": Path(args.qwen_model).name, "img_per_s": res["img_per_s"]}
    if "compcars" in res:
        line.update(n=res["compcars"]["n"], strict=res["compcars"]["strict_acc_pct"],
                    grouped=res["compcars"]["grouped_acc_pct"],
                    old_grouped=res["compcars_old_teacher_same_rows"]["grouped_acc_pct"],
                    **{f"rec_{c}": r for c, r in res["compcars"]["grouped_recall_pct"].items()})
    if "review" in res:
        line.update(review_n=res["review"]["n"], review_acc=res["review"]["acc_pct"])
    summary = out_dir.parent / "summary.csv"
    pd.DataFrame([line]).to_csv(summary, mode="a", header=not summary.exists(), index=False)

    if "compcars" in res:
        c = res["compcars"]
        print(f"[eval] CompCars {args.split}: strict {c['strict_acc_pct']}%  grouped {c['grouped_acc_pct']}%  "
              f"(old teacher, same rows: grouped {res['compcars_old_teacher_same_rows']['grouped_acc_pct']}%)")
        print("[eval] grouped recall: " + "  ".join(f"{k} {v}" for k, v in c["grouped_recall_pct"].items()))
    if "review" in res:
        print(f"[eval] review sheet: {res['review']['acc_pct']}% on {res['review']['n']} rows")
    print(f"[eval] {res['img_per_s']} img/s, mean option mass {res['mean_option_mass']}. Wrote {out_dir}")


if __name__ == "__main__":
    main()
