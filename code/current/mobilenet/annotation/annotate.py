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
import json
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
from common import DOMAINS, load_crop, load_label_map, parse_bbox, resolve_image_path, split_class, to_rel_image_path  # noqa: E402
from train_common import save_run_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-from", default=None, help="Directory of previous run to reuse test predictions and color cache from")
    p.add_argument("--domain", choices=["both", "pm", "ext"], default="both")
    p.add_argument("--split", default="test", help="split to annotate ('test', 'val', 'train', or 'all')")
    p.add_argument("--max-samples", type=int, default=None, help="random subsample per domain")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--reuse-model-stage", action="store_true", help="skip stage 1 if annotations.csv/embeddings.npy exist")
    p.add_argument("--coords-max", type=int, default=30000, help="rows that get 2-D coordinates (half per domain)")
    p.add_argument("--skip-color", action="store_true")
    p.add_argument("--color-head", default="qwen3vl")
    p.add_argument("--qwen-model", default=paths.QWEN_MODEL_DIR or "Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--color-batch-size", type=int, default=16)
    p.add_argument("--qwen-max-side", type=int, default=336, help="Max image dimension for Qwen (default: 336 for fast inference)")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization and use bfloat16")
    p.add_argument("--qwen-scoring", default="first_token", choices=["first_token", "variants", "letters"],
                   help="how Qwen answers are scored (see color_heads.Qwen3VLHead); default = Runs 1/2")
    p.add_argument("--qwen-prompt", default="short", help="color_heads.PROMPTS key ('short' = Runs 1/2, 'defined')")
    p.add_argument("--qwen-center-crop", type=float, default=0.0, help="fraction trimmed off around the center")
    p.add_argument("--qwen-tta", default="none", choices=["none", "flip"])
    p.add_argument("--fresh-color", action="store_true",
                   help="do not copy color_cache.csv from --cache-from (use when the color settings changed)")
    p.add_argument("--review-conf", type=float, default=0.70, help="model confidence below this -> needs_review")
    p.add_argument("--review-color-conf", type=float, default=0.50, help="color confidence below this -> needs_review")
    p.add_argument("--max-points", type=int, default=15000, help="points (with thumbnails) embedded in the viewer")
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
    return resolve_image_path(raw_path, domain="platesmania", base_dir=base_dir)


def _resolve_ext_path(raw_path: str, ext_base_dir: str) -> str:
    return resolve_image_path(raw_path, domain="external", base_dir=ext_base_dir)


def pm_table(split, n, seed):
    df = pd.read_csv(paths.PM_MANIFEST_CSV)
    if split != "all" and "split" in df.columns:
        df = df[df["split"] == split]
    col = "image_path" if "image_path" in df.columns else "image_rel_path"
    base = paths.PM_IMG_DIR
    out = [to_rel_image_path(p, "platesmania", base) for p in df[col].astype(str)]
    split_vals = df["split"].astype(str).values if "split" in df.columns else np.full(len(df), split)
    res = pd.DataFrame({
        "path": out,
        "split": split_vals,
        "gt_label": df["label"].astype(int).values,
        "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
        "source": "platesmania",
    })
    res.insert(0, "domain", "platesmania")
    return _subsample(res, n, seed)


def ext_table(split, n, seed):
    splits_dir = MOBILENET_DIR / "run4" / "splits_1235models"
    if split == "all":
        parts = []
        for s in ("train", "val", "test"):
            p = splits_dir / f"{s}.csv"
            if p.exists():
                sdf = pd.read_csv(p)
                sdf["split"] = s
                parts.append(sdf)
        df = pd.concat(parts, ignore_index=True) if parts else pd.read_csv(splits_dir / "all_splits.csv")
        if "split" not in df.columns:
            df["split"] = "all"
    else:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        df["split"] = split

    base = paths.EXT_DATASETS_DIR
    out = [to_rel_image_path(p, "external", base) for p in df["image_path"].astype(str)]
    res = df[["split", "label", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "source"]].rename(
        columns={"label": "gt_label"})
    res.insert(0, "path", out)
    res.insert(0, "domain", "external")
    return _subsample(res, n, seed)


# ------------------------------------------------------------------------------
# Stage 1: make / model
# ------------------------------------------------------------------------------
def stage_model(table, label_map, args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import output_analysis as oa

    ext_lm_path = MOBILENET_DIR / "run4" / "splits_1235models" / "label_map.json"
    full_label_map = dict(label_map)
    if ext_lm_path.exists():
        raw_ext = json.load(open(ext_lm_path))
        for k, v in raw_ext.get("class_to_idx", raw_ext).items():
            full_label_map.setdefault(k, int(v))

    idx_to_class = {v: k for k, v in full_label_map.items()}
    model = oa.MobileNetV2Evaluator(len(label_map), args.model_path)
    model.eval()

    model_label_map = {c: i for c, i in full_label_map.items() if i < model.num_classes}
    n_cls = model.num_classes
    makes = sorted({split_class(c)[0] for c in model_label_map})
    make_to_i = {m: i for i, m in enumerate(makes)}
    proj = np.zeros((n_cls, len(makes)), dtype=np.float32)
    for c, i in model_label_map.items():
        proj[i, make_to_i[split_class(c)[0]]] = 1.0

    rows, embs = [], []
    total_imgs = len(table)
    embs = np.empty((total_imgs, 1280), dtype=np.float16)

    for domain, cfg in DOMAINS.items():
        sub = table[table["domain"] == domain]
        if sub.empty:
            continue
        if domain == "platesmania":
            bboxes = None
        else:
            bboxes = [tuple(b) if b[0] >= 0 else None for b in sub[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values]

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
                valid_ids = batch_ids[vm]
                for j, tid in enumerate(valid_ids):
                    t = top3[j]
                    rows.append({
                        "row": int(tid), "pred_idx": int(t[0]),
                        "pred_model_conf": float(probs[j, t[0]]),
                        "top3": "|".join(f"{idx_to_class[int(c)]}:{probs[j, c]:.3f}" for c in t),
                        "pred_make": makes[int(mprob[j].argmax())], "make_conf": float(mprob[j].max()),
                    })
                    embs[tid] = emb[j].astype(np.float16)
        if failed:
            print(f"[{domain}] {failed} unreadable images skipped")

    res = pd.DataFrame(rows).set_index("row")
    out = table.loc[res.index].copy()
    out = pd.concat([out, res], axis=1).reset_index(drop=True)
    out["pred_class"] = out["pred_idx"].map(idx_to_class)
    out["pred_model"] = out["pred_class"].map(lambda c: split_class(c)[1])
    out["gt_class"] = out["gt_label"].map(idx_to_class)
    out["gt_make"] = out["gt_class"].map(lambda c: split_class(c)[0])
    out["correct_model"] = out["pred_class"] == out["gt_class"]
    out["correct_make"] = out["pred_make"] == out["gt_make"]
    out.insert(0, "image_id", np.arange(len(out)))
    out.insert(2, "filename", out["path"].map(lambda p: Path(p).name))
    valid_rows = res.index.values
    return out.drop(columns=["pred_idx", "gt_label"]), embs[valid_rows]


# ------------------------------------------------------------------------------
# Stage 2: 2-D coordinates
# ------------------------------------------------------------------------------
def stage_coords(df, emb, args):
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(args.seed)
    chosen = []
    per = args.coords_max // 2
    for domain in DOMAINS:
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
def stage_color(df, out_dir, args):
    from color_heads import COLORS, build_head
    cache_path = out_dir / "color_cache.csv"

    # Seed color cache from cache_from if destination does not exist
    if not cache_path.exists() and args.cache_from and not args.fresh_color:
        src_cache = Path(args.cache_from) / "color_cache.csv"
        if src_cache.exists():
            import shutil
            shutil.copy2(src_cache, cache_path)
            print(f"[color] Seeded color cache with {src_cache}")

    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["path", "domain", "color", "color_conf"])
    if len(cache):
        cache["path"] = [to_rel_image_path(p, d) for p, d in zip(cache["path"].astype(str), cache["domain"].astype(str))]
    df["path"] = [to_rel_image_path(p, d) for p, d in zip(df["path"].astype(str), df["domain"].astype(str))]
    done = set(zip(cache["path"].astype(str), cache["domain"].astype(str)))
    todo = df[[(str(p), str(d)) not in done for p, d in zip(df["path"], df["domain"])]]
    print(f"[color] {len(df) - len(todo):,} cached, {len(todo):,} to do (head={args.color_head})")

    if len(todo):
        head = build_head(args.color_head, model_id=args.qwen_model, max_side=args.qwen_max_side,
                          load_in_4bit=not args.no_4bit, scoring=args.qwen_scoring, prompt_style=args.qwen_prompt,
                          center_crop=args.qwen_center_crop, tta=args.qwen_tta)
        new, t0 = [], time.time()
        recs = todo.to_dict("records")
        for i in range(0, len(recs), args.color_batch_size):
            chunk = recs[i:i + args.color_batch_size]
            imgs, keep = [], []
            for r in chunk:
                try:
                    imgs.append(load_crop(r["path"], r["domain"], parse_bbox(r), max_side=args.qwen_max_side))
                    keep.append(r)
                except Exception as e:
                    print(f"[color] cannot read {r['path']}: {e}")
            if imgs and hasattr(head, "predict_proba"):
                # Keep the full 13-color vector and option mass so a calibrator can be applied afterwards
                probs, mass = head.predict_proba(imgs)
                for r, pr, m in zip(keep, probs, mass):
                    new.append({"path": r["path"], "domain": r["domain"], "color": COLORS[int(pr.argmax())],
                                "color_conf": float(pr.max()), "option_mass": float(m),
                                **{f"p_{c}": float(v) for c, v in zip(COLORS, pr)}})
            elif imgs:
                for r, (c, p) in zip(keep, head.predict(imgs)):
                    new.append({"path": r["path"], "domain": r["domain"], "color": c, "color_conf": p})
            # Append every 500 images or at the final batch
            if new and (len(new) >= 500 or i + args.color_batch_size >= len(recs)):
                df_new = pd.DataFrame(new)
                is_empty = not cache_path.exists() or cache_path.stat().st_size == 0
                if not is_empty:   # an older cache without probability columns keeps its header
                    df_new = df_new.reindex(columns=pd.read_csv(cache_path, nrows=0).columns)
                df_new.to_csv(cache_path, mode="a", header=is_empty, index=False)
                new = []
                rate = (i + len(chunk)) / (time.time() - t0)
                rem = (len(recs) - (i + len(chunk))) / max(rate, 0.1)
                hrs = int(rem // 3600)
                mins = int((rem % 3600) // 60)
                print(f"[color] {i + len(chunk):,}/{len(recs):,} ({rate:.1f} img/s, ~{hrs}h {mins}m remaining)")

    cache = pd.read_csv(cache_path)
    if len(cache):
        cache["path"] = [to_rel_image_path(p, d) for p, d in zip(cache["path"].astype(str), cache["domain"].astype(str))]
        cache = cache.drop_duplicates(["path", "domain"])
    df = df.drop(columns=[c for c in ("color", "color_conf") if c in df.columns])
    return df.merge(cache[["path", "domain", "color", "color_conf"]], on=["path", "domain"], how="left")


def finalize(df, args):
    df["needs_review"] = df["pred_model_conf"] < args.review_conf
    if "color_conf" in df.columns:
        df["needs_review"] |= df["color_conf"].fillna(1.0) < args.review_color_conf
    return df


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(out_dir, args, "annotation/annotate.py")
    label_map = load_label_map(args.label_map)
    ann_path, emb_path = out_dir / "annotations.csv", out_dir / "embeddings.npy"

    if args.reuse_model_stage and ann_path.exists() and emb_path.exists():
        df, emb = pd.read_csv(ann_path), np.load(emb_path)
        df = df.drop(columns=[c for c in ("color", "color_conf") if c in df.columns])
        print(f"[model] reusing {len(df):,} rows from {ann_path}")
    elif ann_path.exists() and emb_path.exists():
        df, emb = pd.read_csv(ann_path), np.load(emb_path)
        print(f"[model] Reusing existing {len(df):,} rows from {ann_path}")
    else:
        tables = []
        if args.domain in ("both", "pm"):
            tables.append(pm_table(args.split, args.max_samples, args.seed))
        if args.domain in ("both", "ext"):
            tables.append(ext_table(args.split, args.max_samples, args.seed))
        table = pd.concat(tables, ignore_index=True)
        print(f"[model] {len(table):,} {args.split} images ({table['domain'].value_counts().to_dict()})")

        # Cache reuse logic
        if args.cache_from:
            c_dir = Path(args.cache_from)
            c_ann, c_emb_file = c_dir / "annotations.csv", c_dir / "embeddings.npy"
            if c_ann.exists() and c_emb_file.exists():
                print(f"[cache] Checking cached predictions and embeddings from {c_dir}")
                c_df = pd.read_csv(c_ann)
                c_df["path"] = [to_rel_image_path(p, d) for p, d in zip(c_df["path"].astype(str), c_df["domain"].astype(str))]
                c_emb = np.load(c_emb_file)

                c_keys = set(zip(c_df["path"].astype(str), c_df["domain"].astype(str)))
                table_keys = list(zip(table["path"].astype(str), table["domain"].astype(str)))
                cached_mask = np.array([k in c_keys for k in table_keys])

                cached_table = table[cached_mask].reset_index(drop=True)
                uncached_table = table[~cached_mask].reset_index(drop=True)
                print(f"[cache] {len(cached_table):,} images found in cache, {len(uncached_table):,} uncached images to process")
                for dom in table["domain"].unique():
                    m = (table["domain"] == dom).to_numpy()
                    miss = uncached_table.loc[uncached_table["domain"] == dom, "path"]
                    print(f"[cache]   {dom}: {int(cached_mask[m].sum()):,}/{int(m.sum()):,} cached"
                          + (f"; e.g. uncached {miss.iloc[0]!r}" if len(miss) else ""))

                c_idx_map = {k: i for i, k in enumerate(zip(c_df["path"].astype(str), c_df["domain"].astype(str)))}
                cached_indices = [c_idx_map[k] for k in zip(cached_table["path"].astype(str), cached_table["domain"].astype(str))]
                cached_df_subset = c_df.iloc[cached_indices].copy().reset_index(drop=True)
                cached_emb_subset = c_emb[cached_indices]
                if "split" not in cached_df_subset.columns and "split" in cached_table.columns:
                    cached_df_subset["split"] = cached_table["split"].values

                # Seed color cache if not present
                src_color = c_dir / "color_cache.csv"
                dst_color = out_dir / "color_cache.csv"
                if src_color.exists() and not dst_color.exists() and not args.fresh_color:
                    import shutil
                    shutil.copy2(src_color, dst_color)
                    print(f"[cache] Seeded color cache with {src_color}")

                if len(uncached_table) > 0:
                    uncached_df, uncached_emb = stage_model(uncached_table, label_map, args)
                    drop_cols = [c for c in ("image_id", "x", "y", "color", "color_conf", "needs_review") if c in cached_df_subset.columns]
                    clean_cached = cached_df_subset.drop(columns=drop_cols)
                    clean_uncached = uncached_df.drop(columns=[c for c in ("image_id", "x", "y") if c in uncached_df.columns], errors="ignore")
                    df = pd.concat([clean_cached, clean_uncached], ignore_index=True)
                    df.insert(0, "image_id", np.arange(len(df)))
                    emb = np.vstack([cached_emb_subset, uncached_emb])
                else:
                    df = cached_df_subset
                    emb = cached_emb_subset
            else:
                print(f"[cache] Cache directory {c_dir} missing annotations.csv or embeddings.npy; running full stage 1")
                df, emb = stage_model(table, label_map, args)
        else:
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
