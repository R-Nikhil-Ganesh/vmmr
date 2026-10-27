#!/usr/bin/env python3
"""
Generate External Dataset Splits for 1,235 Fine-Grained Make/Model Classification
================================================================================
Aligns Stanford Cars, BoxCars116k, and CompCars SV into the standardized
1,235 Make/Model taxonomy matching label_map_1235models.json.

Zero extra image duplication: records direct image paths and bounding boxes.
"""

import sys
import json
import re
import pickle
import random
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # machine-specific paths: see bash_scripts/lib/paths.sh / paths.local.sh
from annotation.common import to_rel_image_path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EXT_DIR = Path(paths.EXT_DATASETS_DIR)
OUTPUT_DIR = Path(__file__).resolve().parent / "splits_1235models"
LABEL_MAP_PATH = Path(paths.PM_LABEL_MAP)

MIN_SAMPLES_PER_CLASS = 50  # Prune classes with < 50 total samples (matches PlatesMania's 50-sample floor)
MAX_PER_CLASS = 600         # Cap per class to prevent heavy Skoda/Octavia or Ford/Focus imbalance
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def clean_tok(s: str) -> str:
    s_ascii = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s_ascii.lower())


def load_target_classes():
    with open(LABEL_MAP_PATH) as f:
        lmap = json.load(f)
    target_classes = lmap["class_to_idx"] if "class_to_idx" in lmap else lmap
    
    make_to_models = defaultdict(dict)
    for c, idx in target_classes.items():
        mk, mdl = c.split("/", 1)
        make_to_models[mk.lower()][clean_tok(mdl)] = (c, idx, mk, mdl)
        if "series" in mdl.lower():
            num = mdl.lower().replace("series", "").strip()
            make_to_models[mk.lower()][clean_tok(num + "er")] = (c, idx, mk, mdl)
            
    return target_classes, make_to_models


def gather_stanford_cars(make_to_models):
    items = []
    devkit_dir = EXT_DIR / "stanford-cars" / "car_devkit" / "devkit"
    meta_path = devkit_dir / "cars_meta.mat"
    if not meta_path.exists():
        print("[Stanford Cars] devkit not found, skipping.")
        return items

    sc_meta = sio.loadmat(str(meta_path))
    sc_names = [str(c[0]) for c in sc_meta["class_names"][0]]

    splits_info = [
        ("train", "cars_train_annos.mat", "cars_train/cars_train"),
        ("test", "cars_test_annos_withlabels.mat", "cars_test/cars_test"),
    ]
    for split, anno_file, img_folder in splits_info:
        anno_path = devkit_dir / anno_file
        if not anno_path.exists():
            continue
        annos = sio.loadmat(str(anno_path))["annotations"][0]
        for it in annos:
            c_idx = int(it["class"][0][0]) - 1
            name = sc_names[c_idx]
            parts = name.split()
            m = "land rover" if (parts[0] == "Land" and len(parts) > 1 and parts[1] == "Rover") else parts[0].lower()
            rest = parts[2:] if m == "land rover" else parts[1:]
            clean_rest = clean_tok(" ".join(rest))
            
            if m in make_to_models:
                for clean_mdl, (c, idx, mk, mdl) in sorted(make_to_models[m].items(), key=lambda x: len(x[0]), reverse=True):
                    if len(clean_mdl) >= 2 and clean_mdl in clean_rest:
                        fname = str(it["fname"][0])
                        img_path = EXT_DIR / "stanford-cars" / img_folder / fname
                        if img_path.exists():
                            items.append({
                                "image_path": str(img_path),
                                "label": idx,
                                "class_name": c,
                                "make": mk,
                                "model": mdl,
                                "source": "stanford_cars",
                                "bbox_x1": int(it["bbox_x1"][0][0]),
                                "bbox_y1": int(it["bbox_y1"][0][0]),
                                "bbox_x2": int(it["bbox_x2"][0][0]),
                                "bbox_y2": int(it["bbox_y2"][0][0]),
                            })
                        break
                        
    print(f"[Stanford Cars] Collected {len(items):,} mapped samples.")
    return items


def gather_compcars_sv(make_to_models):
    items = []
    sv_root = EXT_DIR / "compcars_cctv" / "extracted" / "sv_data_extracted" / "sv_data"
    meta_path = sv_root / "sv_make_model_name.mat"
    img_root = sv_root / "image"
    if not (meta_path.exists() and img_root.exists()):
        print("[CompCars SV] Metadata not found, skipping.")
        return items

    sv_meta = sio.loadmat(str(meta_path))["sv_make_model_name"]
    cc_map = {
        "bwm": "bmw", "buck": "buick", "chevy": "chevrolet", "chrey": "chery",
        "fiat": "fiat", "kia": "kia", "land-rover": "land rover",
        "mazda": "mazda", "benz": "mercedes-benz", "greatwall": "gwm"
    }

    for i in range(len(sv_meta)):
        raw_m = str(sv_meta[i][0][0]).strip().lower()
        m = cc_map.get(raw_m, raw_m)
        raw_mdl = str(sv_meta[i][1][0]).strip().lower()
        clean_mdl = clean_tok(raw_mdl)
        
        if m in make_to_models:
            for tgt_mdl, (c, idx, mk, mdl) in sorted(make_to_models[m].items(), key=lambda x: len(x[0]), reverse=True):
                if len(tgt_mdl) >= 2 and (tgt_mdl in clean_mdl or clean_mdl in tgt_mdl):
                    p_dir = img_root / str(i + 1)
                    if p_dir.is_dir():
                        for f in p_dir.glob("*.jpg"):
                            items.append({
                                "image_path": str(f),
                                "label": idx,
                                "class_name": c,
                                "make": mk,
                                "model": mdl,
                                "source": "compcars_cctv",
                                "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1
                            })
                    break
                    
    print(f"[CompCars SV] Collected {len(items):,} mapped samples.")
    return items


def gather_boxcars116k(make_to_models):
    items = []
    bc_dir = EXT_DIR / "BoxCars116k"
    pkl_path = bc_dir / "dataset.pkl"
    img_root = bc_dir / "images"
    if not (pkl_path.exists() and img_root.exists()):
        print("[BoxCars116k] Dataset not found, skipping.")
        return items

    with open(pkl_path, "rb") as f:
        bc_data = pickle.load(f, encoding="latin1")

    bc_map = {
        "alfaromeo": "alfa romeo", "audi": "audi", "bmw": "bmw", "chevrolet": "chevrolet",
        "chrysler": "chrysler", "citroen": "citroen", "dacia": "dacia", "fiat": "fiat",
        "ford": "ford", "honda": "honda", "hyundai": "hyundai", "infinity": "infiniti",
        "jaguar": "jaguar", "jeep": "jeep", "kia": "kia", "land-rover": "land rover",
        "range-rover": "land rover", "lexus": "lexus", "mazda": "mazda",
        "mercedes-benz": "mercedes-benz", "mini": "mini", "mitsubishi": "mitsubishi",
        "nissan": "nissan", "opel": "opel", "peugeot": "peugeot", "porsche": "porsche",
        "renault": "renault", "seat": "seat", "skoda": "skoda", "subaru": "subaru",
        "suzuki": "suzuki", "toyota": "toyota", "toyoto": "toyota",
        "volkswagen": "volkswagen", "volvo": "volvo"
    }

    for s in bc_data["samples"]:
        parts = s["annotation"].split()
        m = bc_map.get(parts[0].lower())
        if m and m in make_to_models:
            clean_anno = clean_tok(" ".join(parts[1:]))
            for tgt_mdl, (c, idx, mk, mdl) in sorted(make_to_models[m].items(), key=lambda x: len(x[0]), reverse=True):
                if len(tgt_mdl) >= 2 and (tgt_mdl in clean_anno or clean_anno in tgt_mdl):
                    for inst in s["instances"]:
                        p = img_root / inst["path"]
                        if p.exists():
                            x, y, bw, bh = [int(v) for v in inst["2DBB"]]
                            items.append({
                                "image_path": str(p),
                                "label": idx,
                                "class_name": c,
                                "make": mk,
                                "model": mdl,
                                "source": "boxcars116k",
                                "bbox_x1": x, "bbox_y1": y, "bbox_x2": x + bw, "bbox_y2": y + bh
                            })
                    break

    print(f"[BoxCars116k] Collected {len(items):,} mapped samples.")
    return items


def gather_compcars_web(make_to_models):
    items = []
    web_root = EXT_DIR / "compcars_cctv" / "extracted" / "data_extracted" / "data"
    meta_path = web_root / "misc" / "make_model_name.mat"
    web_img_root = web_root / "image"
    web_lbl_root = web_root / "label"
    if not (meta_path.exists() and web_img_root.exists()):
        print("[CompCars Web] Metadata not found, skipping.")
        return items

    web_meta = sio.loadmat(str(meta_path))
    make_names = [str(r[0][0]).strip() if len(r[0]) > 0 else "" for r in web_meta["make_names"]]
    model_names = [str(r[0][0]).strip() if len(r[0]) > 0 else "" for r in web_meta["model_names"]]
    cc_map = {
        "bwm": "bmw", "buck": "buick", "chevy": "chevrolet", "chrey": "chery",
        "fiat": "fiat", "kia": "kia", "land-rover": "land rover",
        "mazda": "mazda", "benz": "mercedes-benz", "greatwall": "gwm"
    }

    for make_idx, raw_mk in enumerate(make_names, 1):
        m = cc_map.get(raw_mk.lower(), raw_mk.lower())
        if m not in make_to_models:
            continue
        make_dir = web_img_root / str(make_idx)
        if not make_dir.is_dir():
            continue
        for model_dir in make_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.isdigit():
                continue
            md_idx = int(model_dir.name)
            raw_md = model_names[md_idx - 1] if 1 <= md_idx <= len(model_names) else ""
            clean_mdl = clean_tok(raw_md)
            for tgt_mdl, (c, idx, mk, mdl) in sorted(make_to_models[m].items(), key=lambda x: len(x[0]), reverse=True):
                if len(tgt_mdl) >= 2 and (tgt_mdl in clean_mdl or clean_mdl in tgt_mdl):
                    for year_dir in model_dir.iterdir():
                        if not year_dir.is_dir():
                            continue
                        for img_file in year_dir.glob("*.jpg"):
                            lbl_file = web_lbl_root / str(make_idx) / model_dir.name / year_dir.name / f"{img_file.stem}.txt"
                            bx1, by1, bx2, by2 = -1, -1, -1, -1
                            if lbl_file.exists():
                                try:
                                    lines = [ln.strip() for ln in lbl_file.read_text().splitlines() if ln.strip()]
                                    if len(lines) >= 3:
                                        coords = [int(v) for v in lines[2].split()[:4]]
                                        bx1, by1, bx2, by2 = coords[0], coords[1], coords[2], coords[3]
                                except Exception:
                                    pass
                            items.append({
                                "image_path": str(img_file),
                                "label": idx,
                                "class_name": c,
                                "make": mk,
                                "model": mdl,
                                "source": "compcars_web",
                                "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2
                            })
                    break

    print(f"[CompCars Web] Collected {len(items):,} mapped samples.")
    return items


MODEL_ALIASES = {
    "a4l": "A4", "a6l": "A6", "a8l": "A8", "tt coupe": "TT", "tts coupe": "TT",
    "glk class": "GLK-Class", "c class": "C-Class", "e class": "E-Class", "s class": "S-Class",
    "m class": "M-Class", "g class": "G-Class", "a class": "A-Class", "b class": "B-Class",
    "cls class": "CLS-Class", "slk class": "SLK-Class", "sl class": "SL-Class", "r class": "R-Class",
    "gl class": "GL-Class", "gla class": "GLA-Class", "cla class": "CLA-Class",
    "sagitar": "Jetta", "magotan": "Passat", "lavida": "Jetta", "gran lavida": "Jetta", "lavide": "Jetta",
    "bora": "Bora", "santana": "Santana", "tiguan": "Tiguan", "touareg": "Touareg",
    "passat lingyu": "Passat", "new passat": "Passat", "polo": "Polo", "golf": "Golf",
    "scirocco": "Scirocco", "cc": "CC", "phaeton": "Phaeton", "sharan": "Sharan", "touran": "Touran",
    "sylphy": "Sentra", "new sylphy": "Sentra", "teana": "Altima", "sunny": "Sunny", "sunshine": "Sunny",
    "tiida": "Tiida", "livina": "Livina", "qashqai": "Qashqai", "x-trail": "X-Trail", "murano": "Murano",
    "march": "Micra", "ma chi": "Micra", "nv200": "NV200", "gt-r": "GT-R", "quest": "Quest",
    "haorui": "Superb", "octavia": "Octavia", "fabia": "Fabia", "superb": "Superb", "yeti": "Yeti", "rapid": "Rapid",
    "excelle": "Excelle", "exclle": "Excelle", "excelle gt": "Verano", "excelle xt": "Verano", "regal": "Regal",
    "lacrosse": "LaCrosse", "gl8": "GL8", "encore": "Encore", "enclave": "Enclave",
    "cruze": "Cruze", "malibu": "Malibu", "sail": "Sail", "aveo": "Aveo", "captiva": "Captiva",
    "camaro": "Camaro", "epica": "Epica", "lova": "Aveo", "trax": "Trax",
    "focus": "Focus", "mondeo": "Mondeo", "mendeo": "Mondeo", "fiesta": "Fiesta", "kuga": "Kuga", "ecosport": "EcoSport",
    "edge": "Edge", "explorer": "Explorer", "mustang": "Mustang", "s-max": "S-Max", "classic focus": "Focus",
    "accord": "Accord", "civic": "Civic", "cr-v": "CR-V", "fit": "Fit", "city": "City", "odyssey": "Odyssey",
    "spirior": "Accord", "crider": "Crider", "jade": "Jade", "crosstour": "Accord Crosstour", "elysion": "Elysion",
    "camry": "Camry", "corolla": "Corolla", "corolla ex": "Corolla", "huaguan": "Corolla", "crown": "Crown", "reiz": "Mark X",
    "rav4": "RAV4", "highlander": "Highlander", "prado": "Land Cruiser Prado", "land cruiser": "Land Cruiser",
    "vios": "Vios", "yaris": "Yaris", "alphard": "Alphard", "previa": "Previa", "prius": "Prius", "ez": "Verso",
    "elantra": "Elantra", "elantra yuedong": "Elantra", "langdong": "Elantra", "sonata": "Sonata",
    "sonata 8": "Sonata", "lingxiang": "Sonata", "mingtu": "Mistra", "ix35": "ix35", "tucson": "Tucson",
    "santa fe": "Santa Fe", "verna": "Accent", "rena": "Accent", "accent": "Accent", "veloster": "Veloster",
    "rohens": "Genesis", "rohens coupe": "Genesis Coupe", "equus": "Equus", "veracruz": "Veracruz",
    "cerato": "Cerato", "forte": "Forte", "furuidi": "Forte", "k2": "Rio", "k3": "Forte", "k5": "Optima",
    "optima": "Optima", "sportage": "Sportage", "sportage r": "Sportage", "sorento": "Sorento", "soul": "Soul",
    "rio": "Rio", "carens": "Carens", "kaizun": "Cadenza",
    "atenza": "6", "axela": "3", "mazda6": "6", "mazda3": "3", "mazda2": "2", "mazda5": "5", "mazda8": "8",
    "cx-5": "CX-5", "cx-7": "CX-7", "cx-9": "CX-9", "rui yi": "6", "xingcheng": "3",
    "canyenne": "Cayenne", "carrera": "911",
    "yuyan": "Swift", "tianyu": "SX4_S-Cross", "sx4": "SX4_S-Cross",
    "grand voager": "Grand Voyager", "feixiang": "Viaggio",
}

MB_ALIASES = {
    "c": "C-Class", "c class": "C-Class", "c-class": "C-Class",
    "b": "B-Class", "b class": "B-Class", "b-class": "B-Class",
    "e": "E-Class", "e class": "E-Class", "e-class": "E-Class",
    "s": "S-Class", "s class": "S-Class", "s-class": "S-Class",
    "a": "A-Class", "a class": "A-Class", "a-class": "A-Class",
    "g": "G-Class", "g class": "G-Class", "g-class": "G-Class",
    "m": "M-Class", "m class": "M-Class", "m-class": "M-Class", "ml": "M-Class",
    "r": "R-Class", "r class": "R-Class", "r-class": "R-Class",
    "gl": "GL-Class", "gl class": "GL-Class", "gl-class": "GL-Class",
    "glk": "GLK-Class", "glk class": "GLK-Class", "glk-class": "GLK-Class",
    "gla": "GLA-Class", "gla class": "GLA-Class", "gla-class": "GLA-Class",
    "cla": "CLA-Class", "cla class": "CLA-Class", "cla-class": "CLA-Class",
    "cls": "CLS-Class", "cls class": "CLS-Class", "cls-class": "CLS-Class",
    "sl": "SL-Class", "sl class": "SL-Class", "sl-class": "SL-Class",
    "slk": "SLK-Class", "slk class": "SLK-Class", "slk-class": "SLK-Class",
    "gle": "GLE-Class", "gle-": "GLE-Class",
    "wei ting": "Vito", "weiting": "Vito",
}

INFINITI_ALIASES = {
    "g-class": "Q40_Q50_Q60_G-Series",
    "g series": "Q40_Q50_Q60_G-Series",
    "q50l": "Q40_Q50_Q60_G-Series",
    "q50": "Q40_Q50_Q60_G-Series",
    "q60": "Q40_Q50_Q60_G-Series",
    "m-class": "Q70_M-Series",
    "q70l": "Q70_M-Series",
    "q70": "Q70_M-Series",
    "ex-class": "QX50_EX-Series",
    "fx-class": "QX70_FX-Series",
    "jx-class": "QX60_JX-Series",
}


def gather_external_resized_512(target_classes, make_to_models, include_external_only=True, min_samples=MIN_SAMPLES_PER_CLASS):
    meta_csv = EXT_DIR / "external_metadata_51make.csv"
    df_meta = pd.read_csv(meta_csv)
    bc_bbox_file = Path(__file__).resolve().parent / "boxcars116k_bboxes_512.json"
    bc_bboxes = json.loads(bc_bbox_file.read_text()) if bc_bbox_file.exists() else {}

    def resolve_row_bbox(row_obj):
        if hasattr(row_obj, "bbox_x1") and int(row_obj.bbox_x1) >= 0:
            return int(row_obj.bbox_x1), int(row_obj.bbox_y1), int(row_obj.bbox_x2), int(row_obj.bbox_y2)
        if str(row_obj.source) == "boxcars116k":
            p_str = str(row_obj.image_path)
            rel_key = "images/" + p_str.split("/external_resized_512/images/", 1)[1] if "/external_resized_512/images/" in p_str else p_str
            bb = bc_bboxes.get(rel_key)
            if bb is not None and len(bb) == 4:
                return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        return -1, -1, -1, -1

    make_norm = {
        "genesis": "Hyundai", "mini": "MINI", "fiat": "Fiat",
        "land-rover": "Land Rover", "range-rover": "Land Rover",
        "mercedes-benz": "Mercedes-Benz", "citroen": "Citroen",
        "chery": "Chery", "gwm": "GWM", "byd": "BYD", "jaguar": "Jaguar",
        "seat": "SEAT", "haval": "Haval", "dacia": "Dacia", "mg": "MG",
        "geely": "Geely", "daihatsu": "Daihatsu", "ram": "Ram", "tesla": "Tesla",
    }
    for c in target_classes:
        mk = c.split("/", 1)[0]
        make_norm[mk.lower()] = mk

    items = []
    unmatched_groups = defaultdict(list)

    for row in df_meta.itertuples(index=False):
        raw_mk = str(row.make).strip()
        canon_mk = make_norm.get(raw_mk.lower(), raw_mk)
        m_key = canon_mk.lower()
        if not include_external_only and m_key not in make_to_models:
            continue

        raw_mdl = str(row.model).strip()
        low_mdl = raw_mdl.lower()
        cleaned_mdl = re.sub(
            r"\b(sedan|coupe|convertible|hatchback|wagon|estate|suv|cabriolet|minivan|van|pickup|gt|derivative|new|classic|cross|sport|hybrid|import|overseas)\b",
            "",
            low_mdl,
            flags=re.I,
        )
        cleaned_mdl = re.sub(r"[-’'\x27]+$", "", cleaned_mdl)
        cleaned_mdl = re.sub(r"\s+", " ", cleaned_mdl).strip()

        if m_key == "mercedes-benz":
            alias_mdl = MB_ALIASES.get(low_mdl, MB_ALIASES.get(cleaned_mdl, raw_mdl))
            alias_cleaned = MB_ALIASES.get(cleaned_mdl, alias_mdl)
        elif m_key == "infiniti":
            alias_mdl = INFINITI_ALIASES.get(low_mdl, MODEL_ALIASES.get(low_mdl, raw_mdl))
            alias_cleaned = INFINITI_ALIASES.get(cleaned_mdl, MODEL_ALIASES.get(cleaned_mdl, cleaned_mdl))
        elif m_key == "mini" and cleaned_mdl in ("mini", "cooper"):
            alias_mdl = "Hatch"
            alias_cleaned = "Hatch"
        else:
            alias_mdl = MODEL_ALIASES.get(low_mdl, raw_mdl)
            alias_cleaned = MODEL_ALIASES.get(cleaned_mdl, cleaned_mdl)

        matched = None
        if m_key in make_to_models:
            m_dict = make_to_models[m_key]
            for cand_str in (raw_mdl, alias_mdl, cleaned_mdl, alias_cleaned):
                tok = clean_tok(cand_str)
                if tok in m_dict:
                    matched = m_dict[tok]
                    break
            if matched is None:
                clean_m = clean_tok(alias_cleaned or raw_mdl)
                for tgt_mdl, info in sorted(m_dict.items(), key=lambda x: len(x[0]), reverse=True):
                    if len(tgt_mdl) >= 2 and (tgt_mdl in clean_m or clean_m in tgt_mdl):
                        matched = info
                        break

        if matched is not None:
            c, idx, mk, mdl = matched
            bx1, by1, bx2, by2 = resolve_row_bbox(row)
            items.append({
                "image_path": to_rel_image_path(str(row.image_path), "external"),
                "label": idx,
                "class_name": c,
                "make": mk,
                "model": mdl,
                "source": str(row.source),
                "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2,
            })
        elif include_external_only:
            final_mdl = alias_cleaned if alias_cleaned else (cleaned_mdl if cleaned_mdl else raw_mdl)
            final_mdl = re.sub(r"[-’'\x27]+$", "", final_mdl).strip()
            if final_mdl.islower():
                final_mdl = final_mdl.upper() if len(final_mdl) <= 3 else final_mdl.title()
            tok = clean_tok(final_mdl)
            if tok:
                unmatched_groups[(canon_mk, tok)].append((row, final_mdl))

    n_pm_matched = len(items)
    n_ext_added_classes = 0
    n_ext_added_rows = 0
    n_ext_pruned_classes = 0
    n_ext_pruned_rows = 0

    if include_external_only:
        # Sort deterministically by (make, token) and assign new class indices starting at len(target_classes)
        next_idx = max(target_classes.values()) + 1 if target_classes else 0
        for (canon_mk, tok) in sorted(unmatched_groups.keys()):
            grp_rows = unmatched_groups[(canon_mk, tok)]
            if len(grp_rows) < min_samples:
                n_ext_pruned_classes += 1
                n_ext_pruned_rows += len(grp_rows)
                continue
            best_mdl = Counter(m for _, m in grp_rows).most_common(1)[0][0]
            cls_name = f"{canon_mk}/{best_mdl}"
            if cls_name not in target_classes:
                target_classes[cls_name] = next_idx
                next_idx += 1
            cls_idx = target_classes[cls_name]
            n_ext_added_classes += 1
            n_ext_added_rows += len(grp_rows)
            for r, _ in grp_rows:
                bx1, by1, bx2, by2 = resolve_row_bbox(r)
                items.append({
                    "image_path": to_rel_image_path(str(r.image_path), "external"),
                    "label": cls_idx,
                    "class_name": cls_name,
                    "make": canon_mk,
                    "model": best_mdl,
                    "source": str(r.source),
                    "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2,
                })

    print(f"[external_resized_512] Matched to PlatesMania taxonomy: {n_pm_matched:,} samples.")
    if include_external_only:
        print(f"[external_resized_512] Added back 51-make external-only classes (>= {min_samples} samples): "
              f"{n_ext_added_rows:,} samples across {n_ext_added_classes} classes.")
        print(f"[external_resized_512] Pruned external-only classes (< {min_samples} samples): "
              f"{n_ext_pruned_rows:,} samples across {n_ext_pruned_classes} classes.")
    print(f"[external_resized_512] Total collected from {meta_csv}: {len(items):,} / {len(df_meta):,} ({len(items)/len(df_meta):.1%}).")
    return items


def main():
    parser = argparse.ArgumentParser(description="Build External dataset splits across the 51 makes.")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CLASS,
                        help=f"Prune classes with fewer than N samples (default: {MIN_SAMPLES_PER_CLASS})")
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS,
                        help=f"Cap external samples per class (default: {MAX_PER_CLASS}, 0 = no cap)")
    parser.add_argument("--pm-only", action="store_true",
                        help="Restrict strictly to the 1,235 PlatesMania classes (disable adding 51-make external classes)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for generated train/val/test CSVs and label_map.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print("=" * 70)
    print("  EXTERNAL DATASET BUILDER (51 MAKES + >=50 SAMPLE PRUNING)")
    print(f"  External Root:    {EXT_DIR}")
    print(f"  Base Label Map:   {LABEL_MAP_PATH}")
    print(f"  Min per class:    {args.min_samples} (for external-only classes)")
    print(f"  Max per class:    {args.max_per_class}")
    print(f"  Include 51 Makes: {not args.pm_only}")
    print(f"  Output directory: {out_dir}")
    print("=" * 70)

    target_classes, make_to_models = load_target_classes()
    base_num_classes = len(target_classes)
    print(f"Loaded {base_num_classes} base PlatesMania classes from label map.")

    all_candidates = []
    if (EXT_DIR / "external_metadata_51make.csv").exists():
        all_candidates.extend(
            gather_external_resized_512(
                target_classes,
                make_to_models,
                include_external_only=not args.pm_only,
                min_samples=args.min_samples,
            )
        )
    else:
        all_candidates.extend(gather_stanford_cars(make_to_models))
        all_candidates.extend(gather_compcars_sv(make_to_models))
        all_candidates.extend(gather_compcars_web(make_to_models))
        all_candidates.extend(gather_boxcars116k(make_to_models))

    df_all = pd.DataFrame(all_candidates)
    print(f"\nTotal collected candidates: {len(df_all):,} across {df_all['label'].nunique()} active external classes.")
    print(f"With explicit manifest bbox (bbox_x1 >= 0, e.g. BoxCars116k 2DBB): {(df_all['bbox_x1'] >= 0).sum():,} "
          f"(remaining {(df_all['bbox_x1'] < 0).sum():,} are already bbox-cropped in external_resized_512).")

    # Apply per-class cap for balance
    capped_groups = []
    for c_idx, grp in df_all.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        if args.max_per_class > 0 and len(grp) > args.max_per_class:
            grp = grp.iloc[:args.max_per_class]
        capped_groups.append(grp)

    df_capped = pd.concat(capped_groups, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Total after capping (max {args.max_per_class}/class): {len(df_capped):,} samples across {df_capped['label'].nunique()} classes.")

    # Stratified Train/Val/Test Split
    train_parts, val_parts, test_parts = [], [], []
    for c_idx, grp in df_capped.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(grp)
        n_test = int(n * TEST_RATIO)
        n_val = int(n * VAL_RATIO)
        if n >= 5:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
            
        test_part = grp.iloc[:n_test]
        val_part = grp.iloc[n_test:n_test + n_val]
        train_part = grp.iloc[n_test + n_val:]

        test_parts.append(test_part)
        val_parts.append(val_part)
        train_parts.append(train_part)

    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)
    df_capped.to_csv(out_dir / "all_splits.csv", index=False)

    # Save standardized label_map.json
    label_meta = {
        "class_to_idx": target_classes,
        "idx_to_class": {str(v): k for k, v in target_classes.items()},
        "num_classes": len(target_classes),
        "base_pm_classes": base_num_classes,
        "external_only_classes": len(target_classes) - base_num_classes,
    }
    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_meta, f, indent=2)

    all_makes = sorted(set(c.split("/", 1)[0] for c in target_classes))
    print("\n" + "=" * 70)
    print(f"  EXTERNAL DATASET SPLITS GENERATED SUCCESSFULLY!")
    print(f"  Output directory: {out_dir}")
    print(f"  - Train samples:    {len(df_train):,}")
    print(f"  - Val samples:      {len(df_val):,}")
    print(f"  - Test samples:     {len(df_test):,}")
    print(f"  - Total samples:    {len(df_capped):,}")
    print(f"  - Active Ext Cls:   {df_capped['label'].nunique()} (PM overlap: {(df_capped['label'] < base_num_classes).nunique() if False else df_capped[df_capped['label'] < base_num_classes]['label'].nunique()}, Ext-only: {df_capped[df_capped['label'] >= base_num_classes]['label'].nunique()})")
    print(f"  - Unified Taxonomy: {len(target_classes):,} classes across {len(all_makes)} makes")
    print("=" * 70)


if __name__ == "__main__":
    main()
