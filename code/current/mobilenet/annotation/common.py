"""Light helpers shared by annotate.py and build_viewer.py (no torch / onnxruntime imports)."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Same per-domain preprocessing as the diagnostics (merged_dataset/analysis/output_analysis_merged.py)
DOMAINS = {
    "platesmania": {"crop_top_pct": 0.15, "crop_bottom_pct": 0.0},
    "external":    {"crop_top_pct": 0.0,  "crop_bottom_pct": 0.05},
}

Bbox = Optional[Tuple[int, int, int, int]]


def load_label_map(path) -> Dict[str, int]:
    """label_map.json -> {'Make/Model': idx}. Accepts {'class_to_idx': {...}, ...} or a flat dict."""
    d = json.loads(Path(path).read_text())
    c2i = d["class_to_idx"] if "class_to_idx" in d else d
    return {k: int(v) for k, v in c2i.items() if isinstance(v, int)}


def split_class(name: str) -> Tuple[str, str]:
    make, _, model = name.partition("/")
    return make, model


def crop_like_classifier(img: Image.Image, domain: str, bbox: Bbox = None) -> Image.Image:
    """Crop exactly as the classifier sees the image (bbox if present, else the domain's top/bottom crop)."""
    w, h = img.size
    if bbox is not None and bbox[0] >= 0:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w)); y2 = max(y1 + 1, min(y2, h))
        return img.crop((x1, y1, x2, y2))
    cfg = DOMAINS[domain]
    top = int(h * cfg["crop_top_pct"])
    bottom = max(top + 10, int(h * (1.0 - cfg["crop_bottom_pct"])))
    if top > 0 or cfg["crop_bottom_pct"] > 0:
        return img.crop((0, top, w, bottom))
    return img


def load_crop(path: str, domain: str, bbox: Bbox = None, max_side: Optional[int] = None) -> Image.Image:
    img = crop_like_classifier(Image.open(path).convert("RGB"), domain, bbox)
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    return img


def parse_bbox(row) -> Bbox:
    try:
        b = (int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"]))
        return b if b[0] >= 0 else None
    except Exception:
        return None
