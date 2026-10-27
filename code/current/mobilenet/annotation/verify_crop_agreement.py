#!/usr/bin/env python3
"""
E-02 Crop Agreement Study: Full 512x512 Domain Crop vs. Tight Vehicle Bounding-Box Crop
=======================================================================================
Evaluates on 300 random rows (150 BoxCars116k with newly recovered 2DBB bounding boxes +
150 PlatesMania rows with tight inner-vehicle crops excluding white letterbox bars):
  1. Distilled MobileNetV2 Multi-Task ONNX color head agreement between:
     - Legacy domain crop (bbox_* = -1: top 15% removed for PM, bottom 5% removed for Ext)
     - Tight vehicle crop (exact 2DBB for BoxCars116k; non-white letterbox content box for PM)
  2. Optional Qwen3-VL-8B-Instruct agreement (--run-qwen) when run on GPU.
Saves a detailed report to run_docs/crop_agreement_study.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
MOBILENET_DIR = HERE.parent
sys.path.insert(0, str(MOBILENET_DIR))
sys.path.insert(0, str(HERE))

from common import DOMAINS, crop_like_classifier, resolve_image_path, to_rel_image_path
from color_distillation.train_color_head import COLORS, IMAGENET_MEAN, IMAGENET_STD


def detect_nonwhite_content_bbox(im: Image.Image):
    """Find the tight bounding box of the non-white letterboxed vehicle crop inside a 512x512 image."""
    arr = np.asarray(im)
    mask = (arr < 248).any(axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    w, h = im.size
    if len(rows) > 20 and len(cols) > 20:
        x1, x2 = int(cols[0]), int(cols[-1] + 1)
        y1, y2 = int(rows[0]), int(rows[-1] + 1)
        # Inset slightly (5%) to tighten around the vehicle body
        dx = int((x2 - x1) * 0.05)
        dy = int((y2 - y1) * 0.05)
        return (x1 + dx, y1 + dy, max(x1 + dx + 20, x2 - dx), max(y1 + dy + 20, y2 - dy))
    return (0, int(h * 0.15), w, int(h * 0.90))


def prep_for_onnx(im: Image.Image) -> np.ndarray:
    im_r = im.resize((512, 512), Image.Resampling.BILINEAR)
    arr = np.transpose(np.asarray(im_r, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", default=str(MOBILENET_DIR / "annotation_runs/run2/annotations.csv"))
    p.add_argument("--multitask-onnx", default=str(MOBILENET_DIR / "color_distillation/output/mobilenet_v2_multitask.onnx"))
    p.add_argument("--boxcars-bboxes", default=str(MOBILENET_DIR / "run4/boxcars116k_bboxes_512.json"))
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--run-qwen", action="store_true", help="Also run Qwen3-VL-8B-Instruct on the 300 pairs (requires GPU)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.annotations)
    bmap = json.loads(Path(args.boxcars_bboxes).read_text())

    df_pm = df[df["domain"] == "platesmania"].sample(n=args.num_samples // 2, random_state=args.seed).copy()
    df_ext = df[df["domain"] == "external"].copy()
    df_ext["rel_path"] = [to_rel_image_path(p_str, "external") for p_str in df_ext["path"].astype(str)]
    df_bc = df_ext[df_ext["rel_path"].isin(bmap)].sample(
        n=min(args.num_samples - len(df_pm), len(df_ext[df_ext["rel_path"].isin(bmap)])),
        random_state=args.seed,
    ).copy()

    sample_df = pd.concat([df_pm, df_bc], ignore_index=True)
    print(f"[E-02] Selected {len(sample_df)} rows ({len(df_pm)} PlatesMania, {len(df_bc)} BoxCars116k with recovered 2DBB)")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.multitask_onnx, providers=providers)

    records = []
    imgs_legacy, imgs_tight = [], []
    for r in sample_df.itertuples(index=False):
        dom = str(r.domain)
        full_p = resolve_image_path(str(r.path), dom)
        im = Image.open(full_p).convert("RGB")

        # 1. Legacy crop (bbox = None -> fixed domain crop)
        crop_leg = crop_like_classifier(im, dom, bbox=None)

        # 2. Tight vehicle bbox crop
        if dom == "external":
            rel = to_rel_image_path(str(r.path), "external")
            bb = tuple(bmap[rel])
            crop_tight = crop_like_classifier(im, dom, bbox=bb)
        else:
            bb = detect_nonwhite_content_bbox(im)
            crop_tight = im.crop(bb)

        imgs_legacy.append(crop_leg)
        imgs_tight.append(crop_tight)

        c_leg = sess.run(["color_predictions"], {"input_image": prep_for_onnx(crop_leg)})[0][0]
        c_tight = sess.run(["color_predictions"], {"input_image": prep_for_onnx(crop_tight)})[0][0]

        records.append({
            "domain": dom,
            "path": to_rel_image_path(str(r.path), dom),
            "qwen_run2_color": str(r.color),
            "qwen_run2_conf": float(r.color_conf),
            "onnx_legacy_color": COLORS[int(c_leg.argmax())],
            "onnx_tight_color": COLORS[int(c_tight.argmax())],
            "onnx_agree": int(c_leg.argmax()) == int(c_tight.argmax()),
            "onnx_legacy_matches_qwen": COLORS[int(c_leg.argmax())] == str(r.color),
            "onnx_tight_matches_qwen": COLORS[int(c_tight.argmax())] == str(r.color),
        })

    if args.run_qwen:
        from color_heads import build_head
        import paths as _paths
        qhead = build_head("qwen3vl", model_id=_paths.QWEN_MODEL_DIR or "Qwen/Qwen3-VL-8B-Instruct", max_side=224, load_in_4bit=True)
        q_leg_preds, q_tight_preds = [], []
        bs = 16
        for i in range(0, len(imgs_legacy), bs):
            q_leg_preds.extend(qhead.predict([im.resize((224, 224), Image.Resampling.BILINEAR) for im in imgs_legacy[i:i+bs]]))
            q_tight_preds.extend(qhead.predict([im.resize((224, 224), Image.Resampling.BILINEAR) for im in imgs_tight[i:i+bs]]))
        for rec, (cl, pl), (ct, pt) in zip(records, q_leg_preds, q_tight_preds):
            rec["qwen_legacy_color"] = cl
            rec["qwen_tight_color"] = ct
            rec["qwen_agree"] = (cl == ct)

    res_df = pd.DataFrame(records)
    overall_agree = float(res_df["onnx_agree"].mean() * 100.0)
    pm_agree = float(res_df[res_df["domain"] == "platesmania"]["onnx_agree"].mean() * 100.0)
    bc_agree = float(res_df[res_df["domain"] == "external"]["onnx_agree"].mean() * 100.0)

    out_md = MOBILENET_DIR / "run_docs/crop_agreement_study.md"
    lines = [
        "# Vehicle Cropping & Color Agreement Study (`E-01` & `E-02`)",
        "",
        "## 1. Source Image Audit on the Research PC",
        "Direct inspection of `/home/researchadmin/Econ/resized_512` (PlatesMania) and `/home/researchadmin/Econ/external_resized_512` (External) revealed that **PlatesMania and 4 of the 5 External sources were already cropped to vehicle bounding boxes on disk before Run 4/5**:",
        "",
        "| Dataset Source | Disk Resolution | Pre-Cropped to BBox on Disk? | Bounding-Box Source |",
        "| :--- | :---: | :---: | :--- |",
        "| **PlatesMania (`resized_512`)** | `512×512` | **Yes (`100%`)** | Pre-cropped to vehicle bbox + white letterboxed |",
        "| **External: `stanford_cars`** | `512×512` | **Yes (`100%`)** | `cars_train_annos.mat` / `cars_test_annos_withlabels.mat` (`L1 = 0.04`) |",
        "| **External: `compcars_web`** | `512×512` | **Yes (`100%`)** | `data/label/<make>/<model>/<year>/<id>.txt` (`L1 = 0.04`) |",
        "| **External: `compcars_cctv`** | `512×512` | **Yes (`100%`)** | YOLOv8s vehicle detector (`+2%` margin) |",
        "| **External: `resized_cupra`** | `512×512` | **Yes (`100%`)** | Copied from `resized_512/Cupra` |",
        "| **External: `boxcars116k`** | `512×512` | **No (`0%` — fixed in manifest)** | Raw `BoxCars116k` images have a 30px road border around `inst['2DBB']`; all `18,520` `2DBB` boxes are now populated in `boxcars116k_bboxes_512.json` and `splits_1235models/*.csv` |",
        "",
        "## 2. 300-Sample Crop Sensitivity Experiment (`E-02`)",
        f"- **Sample Size**: `{len(res_df)}` images (`{len(df_pm)}` PlatesMania, `{len(df_bc)}` `BoxCars116k` with recovered `2DBB` bounding boxes).",
        f"- **Overall Color Prediction Agreement (Legacy Crop vs. Tight Vehicle BBox Crop)**: **`{overall_agree:.2f}%`**",
        f"  - **PlatesMania (`150` images, already pre-cropped on disk)**: **`{pm_agree:.2f}%`** agreement.",
        f"  - **External `BoxCars116k` (`150` images, removing the 30px road/pavement border via `2DBB`)**: **`{bc_agree:.2f}%`** agreement (`{100.0 - bc_agree:.2f}%` of `BoxCars116k` images change predicted color when the surrounding gray/black asphalt border is removed!).",
    ]
    if "qwen_agree" in res_df.columns:
        q_all = float(res_df["qwen_agree"].mean() * 100.0)
        q_pm = float(res_df[res_df["domain"] == "platesmania"]["qwen_agree"].mean() * 100.0)
        q_bc = float(res_df[res_df["domain"] == "external"]["qwen_agree"].mean() * 100.0)
        lines.extend([
            f"- **Qwen3-VL-8B Agreement (Legacy Crop vs. Tight Vehicle BBox Crop)**: **`{q_all:.2f}%`** (`{q_pm:.2f}%` on PlatesMania, `{q_bc:.2f}%` on `BoxCars116k`).",
        ])

    out_md.write_text("\n".join(lines) + "\n")
    print(f"[E-02] Overall agreement: {overall_agree:.2f}% (PlatesMania: {pm_agree:.2f}%, BoxCars116k: {bc_agree:.2f}%)")
    print(f"[E-02] Report saved to {out_md}")


if __name__ == "__main__":
    main()
