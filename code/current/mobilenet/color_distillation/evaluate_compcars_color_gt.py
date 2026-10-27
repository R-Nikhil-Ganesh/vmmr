#!/usr/bin/env python3
"""
E-09 Ground-Truth Color Validation: Qwen3-VL-8B & Distilled Color Heads vs. CompCars Human Labels
=================================================================================================
1. Matches annotation_runs/run2/annotations.csv against the 33,310 human-labeled CompCars CCTV
   rows in external_metadata_51make.csv (19,479 matched rows in Annotation Run 2).
2. Computes Qwen3-VL-8B accuracy, per-color precision/recall, confidence calibration
   (color_conf >= 0.70 vs < 0.70), and confusion matrix against human labels.
3. Generates a stratified 600-row human review sample (annotation_runs/color_human_review_600.csv)
   covering all 13 Qwen colors and both domains (with 30% low-confidence rows).
4. Writes a markdown report to run_docs/color_ground_truth_validation.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR))
sys.path.insert(0, str(MOBILENET_DIR / "annotation"))

import paths
from common import to_rel_image_path

COMPCARS_ID2COLOR = {
    0: "black",
    1: "white",
    2: "red",
    3: "yellow",
    4: "blue",
    5: "green",
    6: "purple",
    7: "brown",
    8: "champagne",
    9: "silver",
}

# CompCars groups gray+silver into id=9 ("silver") and champagne/beige/gold into id=8 ("champagne"),
# and has id=6 ("purple") which maps to "other" in our 13-color schema.
QWEN_TO_COMPCARS_GROUPED = {
    "black": "black",
    "white": "white",
    "red": "red",
    "yellow": "yellow",
    "orange": "yellow",
    "blue": "blue",
    "green": "green",
    "brown": "brown",
    "beige": "champagne",
    "gold": "champagne",
    "silver": "silver",
    "gray": "silver",
    "other": "purple",
}

QWEN_TO_COMPCARS_STRICT = {
    "black": "black",
    "white": "white",
    "red": "red",
    "yellow": "yellow",
    "blue": "blue",
    "green": "green",
    "brown": "brown",
    "beige": "champagne",
    "silver": "silver",
    "gray": "gray",
    "gold": "gold",
    "orange": "orange",
    "other": "other",
}


def main():
    ann_path = MOBILENET_DIR / "annotation_runs/run2/annotations.csv"
    ext_meta_path = Path(paths.EXT_DATASETS_DIR) / "external_metadata_51make.csv"
    out_report_md = MOBILENET_DIR / "run_docs/color_ground_truth_validation.md"
    out_json = SCRIPT_DIR / "output/e09_compcars_gt_validation.json"
    out_review_csv = MOBILENET_DIR / "annotation_runs/color_human_review_600.csv"

    df_ann = pd.read_csv(ann_path)
    df_meta = pd.read_csv(ext_meta_path)
    df_meta_gt = df_meta[df_meta["color_label"] >= 0].copy()
    df_meta_gt["rel_path"] = [to_rel_image_path(p, "external") for p in df_meta_gt["image_path"].astype(str)]
    df_meta_gt["gt_color_id"] = df_meta_gt["color_label"].astype(int)
    df_meta_gt["gt_color_name"] = df_meta_gt["gt_color_id"].map(COMPCARS_ID2COLOR)

    df_ext_ann = df_ann[df_ann["domain"] == "external"].copy()
    df_ext_ann["rel_path"] = [to_rel_image_path(p, "external") for p in df_ext_ann["path"].astype(str)]

    merged = df_ext_ann.merge(
        df_meta_gt[["rel_path", "gt_color_id", "gt_color_name"]],
        on="rel_path",
        how="inner",
    )
    print(f"[E-09] Matched {len(merged):,} human-labeled CompCars rows in Annotation Run 2 annotations.csv")

    merged["qwen_strict"] = merged["color"].map(QWEN_TO_COMPCARS_STRICT)
    merged["qwen_grouped"] = merged["color"].map(QWEN_TO_COMPCARS_GROUPED)
    merged["strict_ok"] = merged["qwen_strict"] == merged["gt_color_name"]
    merged["grouped_ok"] = merged["qwen_grouped"] == merged["gt_color_name"]

    overall_strict = float(merged["strict_ok"].mean() * 100.0)
    overall_grouped = float(merged["grouped_ok"].mean() * 100.0)

    hi_mask = merged["color_conf"] >= 0.70
    lo_mask = merged["color_conf"] < 0.70
    hi_strict = float(merged.loc[hi_mask, "strict_ok"].mean() * 100.0)
    hi_grouped = float(merged.loc[hi_mask, "grouped_ok"].mean() * 100.0)
    lo_strict = float(merged.loc[lo_mask, "strict_ok"].mean() * 100.0) if lo_mask.any() else 0.0
    lo_grouped = float(merged.loc[lo_mask, "grouped_ok"].mean() * 100.0) if lo_mask.any() else 0.0

    split_stats = {}
    for sp, grp in merged.groupby("split"):
        split_stats[str(sp)] = {
            "n": int(len(grp)),
            "strict_acc_pct": round(float(grp["strict_ok"].mean() * 100.0), 2),
            "grouped_acc_pct": round(float(grp["grouped_ok"].mean() * 100.0), 2),
        }

    per_gt_color = []
    for cid in sorted(COMPCARS_ID2COLOR.keys()):
        cname = COMPCARS_ID2COLOR[cid]
        sub = merged[merged["gt_color_id"] == cid]
        if len(sub) == 0:
            continue
        top_qwen = sub["color"].value_counts().head(3).to_dict()
        per_gt_color.append({
            "compcars_id": cid,
            "compcars_color": cname,
            "n": int(len(sub)),
            "strict_acc_pct": round(float(sub["strict_ok"].mean() * 100.0), 2),
            "grouped_acc_pct": round(float(sub["grouped_ok"].mean() * 100.0), 2),
            "top_qwen_preds": top_qwen,
        })

    per_qwen_color = []
    for qcol, sub in merged.groupby("color"):
        top_gt = sub["gt_color_name"].value_counts().head(3).to_dict()
        per_qwen_color.append({
            "qwen_color": str(qcol),
            "n": int(len(sub)),
            "mean_conf": round(float(sub["color_conf"].mean()), 4),
            "pct_below_070": round(float((sub["color_conf"] < 0.70).mean() * 100.0), 2),
            "strict_precision_pct": round(float(sub["strict_ok"].mean() * 100.0), 2),
            "grouped_precision_pct": round(float(sub["grouped_ok"].mean() * 100.0), 2),
            "top_human_gt": top_gt,
        })
    per_qwen_color.sort(key=lambda x: -x["n"])

    # Generate stratified 600-image human review CSV (both PlatesMania & External, 30% low-conf)
    rng = np.random.RandomState(42)
    review_parts = []
    colors_list = sorted(df_ann["color"].dropna().unique())
    per_color_target = max(1, 600 // len(colors_list))
    for col in colors_list:
        sub = df_ann[df_ann["color"] == col]
        lo = sub[sub["color_conf"] < 0.70]
        hi = sub[sub["color_conf"] >= 0.70]
        n_lo = min(len(lo), int(round(per_color_target * 0.30)))
        n_hi = min(len(hi), per_color_target - n_lo)
        if len(lo) > 0 and n_lo > 0:
            review_parts.append(lo.sample(n=n_lo, random_state=rng))
        if len(hi) > 0 and n_hi > 0:
            review_parts.append(hi.sample(n=n_hi, random_state=rng))
    df_review = pd.concat(review_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_review["path"] = [to_rel_image_path(p, d) for p, d in zip(df_review["path"].astype(str), df_review["domain"].astype(str))]
    df_review["human_color"] = ""
    df_review["notes"] = ""
    keep_cols = ["image_id", "domain", "split", "path", "gt_class", "pred_class", "color", "color_conf", "human_color", "notes"]
    df_review[keep_cols].to_csv(out_review_csv, index=False)

    summary = {
        "total_matched_rows": int(len(merged)),
        "overall_strict_acc_pct": round(overall_strict, 2),
        "overall_grouped_acc_pct": round(overall_grouped, 2),
        "high_conf_ge_070": {
            "n": int(hi_mask.sum()),
            "pct_of_total": round(float(hi_mask.mean() * 100.0), 2),
            "strict_acc_pct": round(hi_strict, 2),
            "grouped_acc_pct": round(hi_grouped, 2),
        },
        "low_conf_lt_070": {
            "n": int(lo_mask.sum()),
            "pct_of_total": round(float(lo_mask.mean() * 100.0), 2),
            "strict_acc_pct": round(lo_strict, 2),
            "grouped_acc_pct": round(lo_grouped, 2),
        },
        "by_split": split_stats,
        "per_compcars_gt_color": per_gt_color,
        "per_qwen_predicted_color": per_qwen_color,
        "stratified_review_csv": str(out_review_csv.relative_to(MOBILENET_DIR)),
        "stratified_review_rows": int(len(df_review)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    # Write Markdown report
    lines = [
        "# Ground-Truth Color Validation Report (`E-09`)",
        "",
        "## 1. Overview & Ground-Truth Source",
        f"- **Human Ground-Truth Dataset**: CompCars Surveillance (`sv_data/color_list.mat`), matched via `external_metadata_51make.csv` against [`annotation_runs/run2/annotations.csv`](../annotation_runs/run2/run_config.json).",
        f"- **Matched Human-Labeled Rows in Annotation Run 2**: **{len(merged):,}** images (`train`: `{split_stats.get('train', {}).get('n', 0):,}`, `val`: `{split_stats.get('val', {}).get('n', 0):,}`, `test`: `{split_stats.get('test', {}).get('n', 0):,}`).",
        f"- **Stratified Manual Review Sheet**: Generated [`annotation_runs/color_human_review_600.csv`](../annotation_runs/color_human_review_600.csv) (`{len(df_review)}` rows stratified across all 13 Qwen colors, with 30% drawn from `color_conf < 0.70`).",
        "",
        "## 2. Qwen3-VL-8B Agreement with CompCars Human Color Labels",
        "",
        "> [!NOTE]",
        "> CompCars uses a 10-color taxonomy (`black, white, red, yellow, blue, green, purple, brown, champagne, silver`) where **gray and silver are merged into class `9: silver`** and **champagne/beige/gold are merged into class `8: champagne`**. We report both:",
        "> 1. **Strict Agreement**: Treats Qwen `gray` on a CompCars `9: silver` car as an error.",
        "> 2. **Taxonomy-Aligned (Grouped) Agreement**: Aligns Qwen `{silver, gray}` $\\to$ `silver`, `{beige, gold}` $\\to$ `champagne`, `{yellow, orange}` $\\to$ `yellow`, and `other` $\\to$ `purple`.",
        "",
        "| Subset | Rows | Share | Strict Agreement | Taxonomy-Aligned Agreement |",
        "| :--- | ---: | ---: | ---: | ---: |",
        f"| **All Matched Rows** | `{len(merged):,}` | `100.00%` | `{overall_strict:.2f}%` | **`{overall_grouped:.2f}%`** |",
        f"| **High Confidence (`color_conf >= 0.70`)** | `{ int(hi_mask.sum()):,}` | `{hi_mask.mean()*100:.2f}%` | `{hi_strict:.2f}%` | **`{hi_grouped:.2f}%`** |",
        f"| **Low Confidence (`color_conf < 0.70`)** | `{ int(lo_mask.sum()):,}` | `{lo_mask.mean()*100:.2f}%` | `{lo_strict:.2f}%` | **`{lo_grouped:.2f}%`** |",
        f"| **Train Split** | `{split_stats.get('train', {}).get('n', 0):,}` | `{split_stats.get('train', {}).get('n', 0)/len(merged)*100:.2f}%` | `{split_stats.get('train', {}).get('strict_acc_pct', 0):.2f}%` | **`{split_stats.get('train', {}).get('grouped_acc_pct', 0):.2f}%`** |",
        f"| **Val Split** | `{split_stats.get('val', {}).get('n', 0):,}` | `{split_stats.get('val', {}).get('n', 0)/len(merged)*100:.2f}%` | `{split_stats.get('val', {}).get('strict_acc_pct', 0):.2f}%` | **`{split_stats.get('val', {}).get('grouped_acc_pct', 0):.2f}%`** |",
        f"| **Test Split** | `{split_stats.get('test', {}).get('n', 0):,}` | `{split_stats.get('test', {}).get('n', 0)/len(merged)*100:.2f}%` | `{split_stats.get('test', {}).get('strict_acc_pct', 0):.2f}%` | **`{split_stats.get('test', {}).get('grouped_acc_pct', 0):.2f}%`** |",
        "",
        "## 3. Recall by CompCars Human Ground-Truth Color",
        "",
        "| CompCars ID | Human Color | Samples | Strict Recall | Aligned Recall | Top-3 Qwen3-VL Predictions |",
        "| :---: | :--- | ---: | ---: | ---: | :--- |",
    ]
    for r in per_gt_color:
        top_str = ", ".join(f"`{k}` ({v:,})" for k, v in r["top_qwen_preds"].items())
        lines.append(
            f"| `{r['compcars_id']}` | **{r['compcars_color']}** | `{r['n']:,}` | `{r['strict_acc_pct']:.2f}%` | **`{r['grouped_acc_pct']:.2f}%`** | {top_str} |"
        )

    lines.extend([
        "",
        "## 4. Precision & Confidence Calibration by Qwen3-VL Predicted Color",
        "",
        "| Qwen Predicted Color | Samples | Mean `color_conf` | `% < 0.70` | Strict Precision | Aligned Precision | Top-3 Human Ground-Truth Labels |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ])
    for r in per_qwen_color:
        top_str = ", ".join(f"`{k}` ({v:,})" for k, v in r["top_human_gt"].items())
        lines.append(
            f"| **{r['qwen_color']}** | `{r['n']:,}` | `{r['mean_conf']:.4f}` | `{r['pct_below_070']:.2f}%` | `{r['strict_precision_pct']:.2f}%` | **`{r['grouped_precision_pct']:.2f}%`** | {top_str} |"
        )

    out_report_md.write_text("\n".join(lines) + "\n")
    print(f"[E-09] Saved validation report to {out_report_md}")
    print(f"[E-09] Overall Strict: {overall_strict:.2f}% | Aligned: {overall_grouped:.2f}% | High-Conf Aligned: {hi_grouped:.2f}% | Low-Conf Aligned: {lo_grouped:.2f}%")


if __name__ == "__main__":
    main()
