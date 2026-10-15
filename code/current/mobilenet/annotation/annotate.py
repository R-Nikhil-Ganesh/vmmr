#!/usr/bin/env python3
"""
Annotate test images with make, model (merged Model C, ONNX) and color (pluggable head, default Qwen3-VL-8B), then
build the self-contained HTML viewer.

Stages (each writes into --output-dir; the color stage resumes from color_cache.csv):
  1. make/model   ONNX Runtime: top-3 classes, model confidence, make by marginalization, 1280-d embedding
  2. coordinates  2-D embedding (UMAP if installed, else PCA -> t-SNE) for a subset of rows
  3. color        --color-head (default: qwen3vl), scored over a fixed color list
  4. viewer       build_viewer.build() -> viewer.html

Outputs: annotations.csv (one row per image), embeddings.npy (float16, row-aligned with annotations.csv),
         color_cache.csv, viewer.html. The table is the interface for any later server-backed viewer.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MOBILENET_DIR = HERE.parent
sys.path.insert(0, str(MOBILENET_DIR))
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from common import DOMAINS, load_crop, load_label_map, parse_bbox, split_class  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--domain", choices=["both", "pm", "ext", "cam"], default="both",
                   help="both = PlatesMania + External test splits; cam = production camera batches (no ground truth)")
    p.add_argument("--split", default="test", help="split to annotate (default: test; unseen by the model)")
    p.add_argument("--max-samples", type=int, default=None, help="random subsample per domain")
    p.add_argument("--cam-batches", nargs="*", default=None, help="camera batch folder names under CAM_ROOT (default: all batch*)")
    p.add_argument("--cam-valid-csv", default=None,
                   help="with --domain cam: annotate only the hand-filtered boxes listed in this valid_cars.csv "
                        "(columns batch, source_image, x1..y2 in source-frame pixels) instead of every detector box")
    p.add_argument("--max-samples-batch3", type=int, default=None,
                   help="random subsample of boxes for the large camera batch (batch3); other batches are kept whole")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--reuse-model-stage", action="store_true", help="skip stage 1 if annotations.csv/embeddings.npy exist")
    p.add_argument("--coords-max", type=int, default=20000, help="rows that get 2-D coordinates (half per domain)")
    p.add_argument("--skip-color", action="store_true")
    p.add_argument("--color-head", default="qwen3vl")
    p.add_argument("--qwen-model", default=paths.QWEN_MODEL_DIR or "Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--color-batch-size", type=int, default=8)
    p.add_argument("--review-conf", type=float, default=0.70, help="model confidence below this -> needs_review")
    p.add_argument("--review-color-conf", type=float, default=0.50, help="color confidence below this -> needs_review")
    p.add_argument("--max-points", type=int, default=3000, help="points (with thumbnails) embedded in the viewer")
    p.add_argument("--no-viewer", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Input tables
# ------------------------------------------------------------------------------
def _subsample(df, n, seed):
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed).sort_index()
    return df.reset_index(drop=True)


def _resolve_pm_path(raw_path: str, base_dir: str) -> str:
    p = Path(raw_path)
    if p.is_file():
        return str(p)
    base = Path(base_dir) if base_dir else Path(".")
    raw_str = str(raw_path)
    for marker in ("resized_512", "images"):
        if marker in raw_str:
            rel = raw_str.split(marker, 1)[1].lstrip("/")
            cand = base / rel
            if cand.is_file():
                return str(cand)
            if rel.startswith("images/"):
                cand_no_img = base / rel[7:]
                if cand_no_img.is_file():
                    return str(cand_no_img)
            strip = not (base / "images").is_dir() and base.is_dir()
            return str(base / (rel[7:] if (strip and rel.startswith("images/")) else rel))
    strip = not (base / "images").is_dir() and base.is_dir()
    clean_rel = raw_str[7:] if (strip and raw_str.startswith("images/")) else raw_str
    cand = base / clean_rel
    if cand.is_file():
        return str(cand)
    cand_img = base / "images" / clean_rel
    if cand_img.is_file():
        return str(cand_img)
    return str(cand)


def _resolve_ext_path(raw_path: str, ext_base_dir: str) -> str:
    p = Path(raw_path)
    if p.is_file():
        return str(p)
    base = Path(ext_base_dir) if ext_base_dir else Path(".")
    raw_str = str(raw_path)
    marker = "external_resized_512"
    if marker in raw_str:
        rel = raw_str.split(marker, 1)[1].lstrip("/")
        cand = base / rel
        if cand.is_file():
            return str(cand)
        if rel.startswith("images/"):
            cand_no_img = base / rel[7:]
            if cand_no_img.is_file():
                return str(cand_no_img)
            if not (base / "images").is_dir() and base.is_dir():
                return str(cand_no_img)
        return str(cand)
    cand = base / raw_str.lstrip("/")
    if cand.is_file():
        return str(cand)
    return str(cand) if not p.is_absolute() else raw_str


def pm_table(split, n, seed):
    df = pd.read_csv(paths.PM_MANIFEST_CSV)
    if "split" in df.columns:
        df = df[df["split"] == split]
    col = "image_path" if "image_path" in df.columns else "image_rel_path"
    base = paths.PM_IMG_DIR
    out = [_resolve_pm_path(p, base) for p in df[col].astype(str)]
    res = pd.DataFrame({"path": out, "gt_label": df["label"].astype(int).values,
                        "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1, "source": "platesmania"})
    res.insert(0, "domain", "platesmania")
    return _subsample(res, n, seed)


def ext_table(split, n, seed):
    df = pd.read_csv(MOBILENET_DIR / "merged_dataset" / "splits_1235models" / f"{split}.csv")
    base = paths.EXT_DATASETS_DIR
    out = [_resolve_ext_path(p, base) for p in df["image_path"].astype(str)]
    res = df[["label", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "source"]].rename(
        columns={"label": "gt_label"})
    res.insert(0, "path", out)
    res.insert(0, "domain", "external")
    return _subsample(res, n, seed)


VEHICLE_IDS = {2, 3, 5, 7}          # COCO: car, motorcycle, bus, truck
CAM_NOMINAL_WH = (960, 540)
CAM_MIN_SIDE_PX = 20
CAM_DEDUP_IOU = 0.7
LARGE_BATCH_TAG = "batch3"


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cam_table(batches, n_large, seed):
    """One row per kept detector box. Labels are YOLO `cls xc yc w h` (normalized, no confidence), COCO ids."""
    from PIL import Image
    root = Path(paths.CAM_ROOT)
    dirs = [root / b for b in batches] if batches else sorted(root.glob("batch*_*"))
    rows, stats = [], {}
    for d in dirs:
        kept = skipped = 0
        for lbl in sorted(d.glob("*.txt")):
            img = d / (lbl.stem + ".jpg")
            if not img.is_file():
                continue
            boxes = []
            for line in lbl.read_text().split("\n"):
                f = line.split()
                if len(f) != 5 or int(float(f[0])) not in VEHICLE_IDS:
                    skipped += bool(f)
                    continue
                c, xc, yc, w, h = int(float(f[0])), *map(float, f[1:])
                raw = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
                box = tuple(min(1.0, max(0.0, v)) for v in raw)
                if (box[2] - box[0]) * CAM_NOMINAL_WH[0] < CAM_MIN_SIDE_PX or (box[3] - box[1]) * CAM_NOMINAL_WH[1] < CAM_MIN_SIDE_PX:
                    skipped += 1
                    continue
                edge = any(abs(a - b) > 1e-6 for a, b in zip(raw, box)) or box[0] <= 0 or box[1] <= 0 or box[2] >= 1 or box[3] >= 1
                boxes.append((c, box, edge))
            boxes.sort(key=lambda t: -(t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))   # keep the larger of overlapping boxes
            uniq = []
            for b in boxes:
                if all(_iou(b[1], u[1]) <= CAM_DEDUP_IOU for u in uniq):
                    uniq.append(b)
                else:
                    skipped += 1
            for k, (c, box, edge) in enumerate(uniq):
                rows.append({"source": d.name, "path": str(img), "frame": lbl.stem, "box_idx": k, "det_class": c,
                             "touches_edge": bool(edge), "nx1": box[0], "ny1": box[1], "nx2": box[2], "ny2": box[3]})
                kept += 1
        stats[d.name] = (kept, skipped)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"[cam] no usable boxes under {root} (batches: {[d.name for d in dirs]})")
    if n_large:
        large = df["source"].str.startswith(LARGE_BATCH_TAG)
        if large.sum() > n_large:
            df = pd.concat([df[~large], df[large].sample(n=n_large, random_state=seed)]).sort_values(["source", "frame", "box_idx"])
    df = df.reset_index(drop=True)
    sizes = {}
    for p in df["path"].unique():
        with Image.open(p) as im:
            sizes[p] = im.size
    W = df["path"].map(lambda p: sizes[p][0]); H = df["path"].map(lambda p: sizes[p][1])
    out = pd.DataFrame({
        "domain": "camera", "path": df["path"], "gt_label": -1,
        "bbox_x1": (df["nx1"] * W).round().astype(int).clip(0, W - 1), "bbox_y1": (df["ny1"] * H).round().astype(int).clip(0, H - 1),
        "bbox_x2": (df["nx2"] * W).round().astype(int), "bbox_y2": (df["ny2"] * H).round().astype(int),
        "source": df["source"], "frame": df["frame"], "box_idx": df["box_idx"], "det_class": df["det_class"],
        "touches_edge": df["touches_edge"],
    })
    for b, (k, sk) in stats.items():
        print(f"[cam] {b}: kept {k:,} boxes, dropped {sk:,} (non-vehicle ids / tiny / duplicates)")
    return out.reset_index(drop=True)


def cam_valid_table(csv_path):
    """Boxes from a curated valid_cars.csv: one row per kept crop, boxes already in source-frame pixels."""
    v = pd.read_csv(csv_path)
    root = Path(paths.CAM_ROOT)
    folders = {d.name.split("_")[0]: d for d in root.glob("batch*_*") if d.is_dir()}     # batch1 -> batch1_0007...
    path = [str(folders[b] / s) if b in folders else "" for b, s in zip(v["batch"], v["source_image"])]
    out = pd.DataFrame({
        "domain": "camera", "path": path, "gt_label": -1,
        "bbox_x1": v["x1"].astype(int), "bbox_y1": v["y1"].astype(int),
        "bbox_x2": v["x2"].astype(int), "bbox_y2": v["y2"].astype(int),
        "source": [folders[b].name if b in folders else b for b in v["batch"]],
        "frame": v["source_image"].str.replace(".jpg", "", regex=False), "box_idx": v["box_idx"],
        "det_class": -1, "touches_edge": v["near_border"].astype(bool),
    })
    missing = ~out["path"].map(lambda p: bool(p) and Path(p).is_file())
    if missing.any():
        print(f"[cam] {int(missing.sum())} of {len(out)} listed frames not found under {root}, dropped")
        out = out[~missing]
    print(f"[cam] {len(out):,} curated boxes from {csv_path} ({out['source'].value_counts().to_dict()})")
    return out.reset_index(drop=True)


# ------------------------------------------------------------------------------
# Stage 1: make / model
# ------------------------------------------------------------------------------
def stage_model(table, label_map, args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import output_analysis as oa

    idx_to_class = {v: k for k, v in label_map.items()}
    n_cls = len(label_map)
    makes = sorted({split_class(c)[0] for c in label_map})
    make_to_i = {m: i for i, m in enumerate(makes)}
    proj = np.zeros((n_cls, len(makes)), dtype=np.float32)
    for c, i in label_map.items():
        proj[i, make_to_i[split_class(c)[0]]] = 1.0

    model = oa.MobileNetV2Evaluator(n_cls, args.model_path)
    model.eval()

    rows, embs = [], []
    for domain, cfg in DOMAINS.items():
        sub = table[table["domain"] == domain]
        if sub.empty:
            continue
        bboxes = [parse_bbox(r) for r in sub.to_dict("records")]
        ds = oa.DiagnosticDataset(sub["path"].tolist(), sub["gt_label"].values.astype(int), 512,
                                  cfg["crop_top_pct"], cfg["crop_bottom_pct"], bboxes=bboxes)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0,
                            prefetch_factor=4 if args.num_workers > 0 else None)
        pos = 0
        ids = sub.index.values
        failed = 0
        with torch.no_grad():
            for x, _, valid in tqdm(loader, desc=f"[{domain}] make/model", unit="batch"):
                n = len(valid)
                vm = valid.numpy()
                batch_ids = ids[pos:pos + n]
                pos += n
                failed += int((~vm).sum())
                if not vm.any():
                    continue
                logits = model(x[valid], return_embeddings=True)
                probs = torch.softmax(logits.float(), dim=-1).numpy()
                emb = model.embeddings
                mprob = probs @ proj
                top3 = np.argsort(-probs, axis=1)[:, :3]
                for j, tid in enumerate(batch_ids[vm]):
                    t = top3[j]
                    rows.append({
                        "row": int(tid), "pred_idx": int(t[0]),
                        "pred_model_conf": float(probs[j, t[0]]),
                        "top3": "|".join(f"{idx_to_class[int(c)]}:{probs[j, c]:.3f}" for c in t),
                        "pred_make": makes[int(mprob[j].argmax())], "make_conf": float(mprob[j].max()),
                    })
                    embs.append(emb[j].astype(np.float16))
        if failed:
            print(f"[{domain}] {failed} unreadable images skipped")

    res = pd.DataFrame(rows).set_index("row")
    out = table.loc[res.index].copy()
    out = pd.concat([out, res], axis=1).reset_index(drop=True)
    out["pred_class"] = out["pred_idx"].map(idx_to_class)
    out["pred_model"] = out["pred_class"].map(lambda c: split_class(c)[1])
    out["gt_class"] = out["gt_label"].map(idx_to_class)          # NaN when there is no ground truth (camera)
    out["gt_make"] = out["gt_class"].map(lambda c: split_class(c)[0] if isinstance(c, str) else np.nan)
    out["correct_model"] = out["gt_class"].notna() & (out["pred_class"] == out["gt_class"])
    out["correct_make"] = out["gt_make"].notna() & (out["pred_make"] == out["gt_make"])
    out.insert(0, "image_id", np.arange(len(out)))
    out.insert(2, "filename", out["path"].map(lambda p: Path(p).name))
    return out.drop(columns=["pred_idx", "gt_label"]), np.stack(embs)


# ------------------------------------------------------------------------------
# Stage 2: 2-D coordinates
# ------------------------------------------------------------------------------
def stage_coords(df, emb, args):
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(args.seed)
    chosen = []
    present = [d for d in DOMAINS if (df["domain"] == d).any()]
    per = args.coords_max // max(1, len(present))
    for domain in present:
        idx = np.where(df["domain"].values == domain)[0]
        if len(idx) > per:
            idx = rng.choice(idx, per, replace=False)
        chosen.append(idx)
    chosen = np.concatenate(chosen)
    if len(chosen) < 5:
        angles = np.linspace(0, 2 * np.pi, len(chosen), endpoint=False)
        df["x"] = np.nan; df["y"] = np.nan
        df.loc[chosen, "x"] = np.cos(angles)
        df.loc[chosen, "y"] = np.sin(angles)
        return df
    x = emb[chosen].astype(np.float32)
    x = PCA(n_components=min(50, x.shape[1], len(chosen) - 1), random_state=args.seed).fit_transform(x)
    try:
        import umap
        xy = umap.UMAP(n_components=2, random_state=args.seed).fit_transform(x)
        print(f"[coords] UMAP on {len(chosen):,} rows")
    except ImportError:
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, perplexity=min(30, (len(chosen) - 1) // 3), init="pca",
                  random_state=args.seed).fit_transform(x)
        print(f"[coords] t-SNE on {len(chosen):,} rows (install umap-learn for UMAP)")
    df["x"] = np.nan; df["y"] = np.nan
    df.loc[chosen, "x"] = xy[:, 0]
    df.loc[chosen, "y"] = xy[:, 1]
    return df


# ------------------------------------------------------------------------------
# Stage 3: color
# ------------------------------------------------------------------------------
COLOR_KEY = ["path", "domain", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]   # one color per box, not per frame


def stage_color(df, out_dir, args):
    from color_heads import build_head
    cache_path = out_dir / "color_cache.csv"
    cols = COLOR_KEY + ["color", "color_conf"]
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=cols)
    if not set(cols) <= set(cache.columns):
        print("[color] ignoring an old color_cache.csv that is keyed by image only (frames with several boxes shared one color)")
        cache = pd.DataFrame(columns=cols)
    done = set(map(tuple, cache[COLOR_KEY].values.tolist()))
    keys = list(map(tuple, df[COLOR_KEY].values.tolist()))
    todo = df[[k not in done for k in keys]]
    print(f"[color] {len(df) - len(todo):,} cached, {len(todo):,} to do (head={args.color_head})")

    if len(todo):
        head = build_head(args.color_head, model_id=args.qwen_model)
        new, t0 = [], time.time()
        recs = todo.to_dict("records")
        for i in range(0, len(recs), args.color_batch_size):
            chunk = recs[i:i + args.color_batch_size]
            imgs, keep = [], []
            for r in chunk:
                try:
                    imgs.append(load_crop(r["path"], r["domain"], parse_bbox(r), max_side=448))
                    keep.append(r)
                except Exception as e:
                    print(f"[color] cannot read {r['path']}: {e}")
            if imgs:
                for r, (c, p) in zip(keep, head.predict(imgs)):
                    new.append({**{k: r[k] for k in COLOR_KEY}, "color": c, "color_conf": p})
            if (i // args.color_batch_size) % 50 == 49 or i + args.color_batch_size >= len(recs):
                cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True) if len(cache) else pd.DataFrame(new)
                cache.to_csv(cache_path, index=False)
                new = []
                rate = (i + len(chunk)) / (time.time() - t0)
                print(f"[color] {i + len(chunk):,}/{len(recs):,} ({rate:.1f} img/s)")
    return df.merge(cache.drop_duplicates(COLOR_KEY), on=COLOR_KEY, how="left")


def finalize(df, args):
    df["needs_review"] = df["pred_model_conf"] < args.review_conf
    if "color_conf" in df.columns:
        df["needs_review"] |= df["color_conf"].fillna(1.0) < args.review_color_conf
    return df


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map(args.label_map)
    ann_path, emb_path = out_dir / "annotations.csv", out_dir / "embeddings.npy"

    if args.reuse_model_stage and ann_path.exists() and emb_path.exists():
        df, emb = pd.read_csv(ann_path), np.load(emb_path)
        df = df.drop(columns=[c for c in ("color", "color_conf") if c in df.columns])
        print(f"[model] reusing {len(df):,} rows from {ann_path}")
    else:
        tables = []
        if args.domain == "cam":
            tables.append(cam_valid_table(args.cam_valid_csv) if args.cam_valid_csv
                          else cam_table(args.cam_batches, args.max_samples_batch3, args.seed))
        if args.domain in ("both", "pm"):
            tables.append(pm_table(args.split, args.max_samples, args.seed))
        if args.domain in ("both", "ext"):
            tables.append(ext_table(args.split, args.max_samples, args.seed))
        table = pd.concat(tables, ignore_index=True)
        print(f"[model] {len(table):,} {'camera' if args.domain == 'cam' else args.split} images ({table['domain'].value_counts().to_dict()})")
        df, emb = stage_model(table, label_map, args)
        df = stage_coords(df, emb, args)
        np.save(emb_path, emb)
    df.to_csv(ann_path, index=False)

    if not args.skip_color:
        df = stage_color(df, out_dir, args)
    df = finalize(df, args)
    df.to_csv(ann_path, index=False)
    print(f"[done] {len(df):,} rows -> {ann_path}")

    if not args.no_viewer:
        import build_viewer
        build_viewer.build(df, out_dir / "viewer.html", max_points=args.max_points, seed=args.seed)


if __name__ == "__main__":
    main()
