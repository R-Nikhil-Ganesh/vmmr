"""Light helpers shared by annotate.py and build_viewer.py (no torch / onnxruntime imports)."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

_MOBILENET_DIR = Path(__file__).resolve().parent.parent
if str(_MOBILENET_DIR) not in sys.path:
    sys.path.insert(0, str(_MOBILENET_DIR))
import paths  # noqa: E402

# Same per-domain preprocessing as the diagnostics (merged_dataset/analysis/output_analysis_merged.py)
DOMAINS = {
    "platesmania": {"crop_top_pct": 0.15, "crop_bottom_pct": 0.0},
    "external":    {"crop_top_pct": 0.0,  "crop_bottom_pct": 0.05},
}

Bbox = Optional[Tuple[int, int, int, int]]


def to_rel_image_path(raw_path: str, domain: str = "", base_dir: Optional[str] = None) -> str:
    """Convert an absolute or mixed image path into a portable relative path."""
    raw_str = str(raw_path).strip()
    if not raw_str:
        return raw_str
    if domain in ("platesmania", "pm"):
        base = base_dir or getattr(paths, "PM_IMG_DIR", "")
        if base and raw_str.startswith(str(base).rstrip("/") + "/"):
            return raw_str[len(str(base).rstrip("/")) + 1:]
        for marker in ("resized_512/", "all_car_crops_512/"):
            if marker in raw_str:
                return raw_str.split(marker, 1)[1].lstrip("/")
    elif domain in ("external", "ext"):
        base = base_dir or getattr(paths, "EXT_DATASETS_DIR", "")
        if base and raw_str.startswith(str(base).rstrip("/") + "/"):
            return raw_str[len(str(base).rstrip("/")) + 1:]
        for marker in ("external_resized_512/", "external_datasets/"):
            if marker in raw_str:
                return raw_str.split(marker, 1)[1].lstrip("/")
    else:
        for marker in ("external_resized_512/", "resized_512/", "external_datasets/", "all_car_crops_512/"):
            if marker in raw_str:
                return raw_str.split(marker, 1)[1].lstrip("/")
    return raw_str


def resolve_image_path(raw_path: str, domain: str = "", base_dir: Optional[str] = None) -> str:
    """Resolve either a portable relative path or legacy absolute path against current paths.py roots."""
    raw_str = str(raw_path).strip()
    if not raw_str:
        return raw_str
    p = Path(raw_str)
    if p.is_absolute() and p.is_file():
        return str(p)

    if domain in ("platesmania", "pm") or ("resized_512" in raw_str and "external_resized_512" not in raw_str):
        base = Path(base_dir or getattr(paths, "PM_IMG_DIR", "") or ".")
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
        cand = base / clean_rel.lstrip("/")
        if cand.is_file():
            return str(cand)
        cand_img = base / "images" / clean_rel.lstrip("/")
        if cand_img.is_file():
            return str(cand_img)
        return str(cand)

    base = Path(base_dir or getattr(paths, "EXT_DATASETS_DIR", "") or ".")
    for marker in ("external_resized_512", "external_datasets"):
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
    if raw_str.startswith("images/"):
        cand_no_img = base / raw_str[7:]
        if cand_no_img.is_file():
            return str(cand_no_img)
    return str(cand) if not p.is_absolute() else raw_str


def load_label_map(path) -> Dict[str, int]:
    """label_map.json -> {'Make/Model': idx}. Accepts {'class_to_idx': {...}, ...} or a flat dict."""
    d = json.loads(Path(path).read_text())
    c2i = d["class_to_idx"] if "class_to_idx" in d else d
    return {k: int(v) for k, v in c2i.items() if isinstance(v, int)}


def split_class(name: str) -> Tuple[str, str]:
    make, _, model = name.partition("/")
    return make, model


def crop_like_classifier(img: Image.Image, domain: str, bbox: Bbox = None) -> Image.Image:
    """Crop exactly as the classifier sees the image.
    Note: In current PlatesMania and External 512x512 splits, `bbox_* = -1` (`bbox is None`), so the domain's
    fixed crop is applied (`15%` top removed for `platesmania`, `5%` bottom removed for `external`).
    """
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
    resolved = resolve_image_path(path, domain)
    img = crop_like_classifier(Image.open(resolved).convert("RGB"), domain, bbox)
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.BILINEAR)
    return img


def parse_bbox(row) -> Bbox:
    try:
        b = (int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"]))
        return b if b[0] >= 0 else None
    except Exception:
        return None
