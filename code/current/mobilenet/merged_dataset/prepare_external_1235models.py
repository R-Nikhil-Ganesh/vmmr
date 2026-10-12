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
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths  # machine-specific paths: see bash_scripts/lib/paths.sh / paths.local.sh

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EXT_DIR = Path(paths.EXT_DATASETS_DIR)
OUTPUT_DIR = Path(__file__).resolve().parent / "splits_1235models"
LABEL_MAP_PATH = Path(paths.PM_LABEL_MAP)

MAX_PER_CLASS = 600  # Cap per class to prevent heavy Skoda/Octavia or Ford/Focus imbalance
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def clean_tok(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


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
                            items.append({
                                "image_path": str(p),
                                "label": idx,
                                "class_name": c,
                                "make": mk,
                                "model": mdl,
                                "source": "boxcars116k",
                                "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1
                            })
                    break

    print(f"[BoxCars116k] Collected {len(items):,} mapped samples.")
    return items


MODEL_ALIASES = {
    "a4l": "A4", "a6l": "A6", "a8l": "A8", "tt coupe": "TT", "tts coupe": "TT",
    "glk class": "GLK-Class", "c class": "C-Class", "e class": "E-Class", "s class": "S-Class",
    "m class": "M-Class", "g class": "G-Class", "a class": "A-Class", "b class": "B-Class",
    "cls class": "CLS-Class", "slk class": "SLK-Class", "sl class": "SL-Class", "r class": "R-Class",
    "gl class": "GL-Class", "gla class": "GLA-Class", "cla class": "CLA-Class",
    "sagitar": "Jetta", "magotan": "Passat", "lavida": "Jetta", "gran lavida": "Jetta",
    "bora": "Bora", "santana": "Santana", "tiguan": "Tiguan", "touareg": "Touareg",
    "passat lingyu": "Passat", "new passat": "Passat", "polo": "Polo", "golf": "Golf",
    "scirocco": "Scirocco", "cc": "CC", "phaeton": "Phaeton", "sharan": "Sharan", "touran": "Touran",
    "sylphy": "Sentra", "new sylphy": "Sentra", "teana": "Altima", "sunny": "Sunny", "sunshine": "Sunny",
    "tiida": "Tiida", "livina": "Livina", "qashqai": "Qashqai", "x-trail": "X-Trail", "murano": "Murano",
    "march": "Micra", "ma chi": "Micra", "nv200": "NV200", "gt-r": "GT-R", "quest": "Quest",
    "haorui": "Superb", "octavia": "Octavia", "fabia": "Fabia", "superb": "Superb", "yeti": "Yeti", "rapid": "Rapid",
    "excelle": "Excelle", "excelle gt": "Verano", "excelle xt": "Verano", "regal": "Regal",
    "lacrosse": "LaCrosse", "gl8": "GL8", "encore": "Encore", "enclave": "Enclave",
    "cruze": "Cruze", "malibu": "Malibu", "sail": "Sail", "aveo": "Aveo", "captiva": "Captiva",
    "camaro": "Camaro", "epica": "Epica", "lova": "Aveo", "trax": "Trax",
    "focus": "Focus", "mondeo": "Mondeo", "fiesta": "Fiesta", "kuga": "Kuga", "ecosport": "EcoSport",
    "edge": "Edge", "explorer": "Explorer", "mustang": "Mustang", "s-max": "S-Max", "classic focus": "Focus",
    "accord": "Accord", "civic": "Civic", "cr-v": "CR-V", "fit": "Fit", "city": "City", "odyssey": "Odyssey",
    "spirior": "Accord", "crider": "Crider", "jade": "Jade", "crosstour": "Accord Crosstour", "elysion": "Elysion",
    "camry": "Camry", "corolla": "Corolla", "corolla ex": "Corolla", "crown": "Crown", "reiz": "Mark X",
    "rav4": "RAV4", "highlander": "Highlander", "prado": "Land Cruiser Prado", "land cruiser": "Land Cruiser",
    "vios": "Vios", "yaris": "Yaris", "alphard": "Alphard", "previa": "Previa", "prius": "Prius", "ez": "Verso",
    "elantra": "Elantra", "elantra yuedong": "Elantra", "langdong": "Elantra", "sonata": "Sonata",
    "sonata 8": "Sonata", "lingxiang": "Sonata", "mingtu": "Mistra", "ix35": "ix35", "tucson": "Tucson",
    "santa fe": "Santa Fe", "verna": "Accent", "accent": "Accent", "veloster": "Veloster",
    "rohens": "Genesis", "rohens coupe": "Genesis", "equus": "Equus", "veracruz": "Veracruz",
    "cerato": "Cerato", "forte": "Forte", "furuidi": "Forte", "k2": "Rio", "k3": "Forte", "k5": "Optima",
    "optima": "Optima", "sportage": "Sportage", "sportage r": "Sportage", "sorento": "Sorento", "soul": "Soul",
    "rio": "Rio", "carens": "Carens",
    "atenza": "6", "axela": "3", "mazda6": "6", "mazda3": "3", "mazda2": "2", "mazda5": "5", "mazda8": "8",
    "cx-5": "CX-5", "cx-7": "CX-7", "cx-9": "CX-9", "rui yi": "6", "xingcheng": "3",
}


def gather_external_resized_512(make_to_models):
    meta_csv = EXT_DIR / "external_metadata_51make.csv"
    df_meta = pd.read_csv(meta_csv)
    make_norm = {"genesis": "hyundai", "mini": "mini", "fiat": "fiat", "land-rover": "land rover", "range-rover": "land rover"}
    items = []
    for row in df_meta.itertuples(index=False):
        raw_mk = str(row.make).strip().lower()
        m = make_norm.get(raw_mk, raw_mk)
        if m not in make_to_models:
            continue
        raw_mdl = str(row.model).strip()
        low_mdl = raw_mdl.lower()
        alias_mdl = MODEL_ALIASES.get(low_mdl, raw_mdl)
        cleaned_mdl = re.sub(
            r"\b(sedan|coupe|convertible|hatchback|wagon|estate|suv|cabriolet|minivan|van|pickup|gt|derivative|new|classic|cross|sport|hybrid|import|overseas)\b",
            "",
            low_mdl,
            flags=re.I,
        )
        cleaned_mdl = re.sub(r"\s+", " ", cleaned_mdl).strip()
        alias_cleaned = MODEL_ALIASES.get(cleaned_mdl, cleaned_mdl)

        m_dict = make_to_models[m]
        matched = None
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
            img_p = str(row.image_path)
            items.append({
                "image_path": img_p,
                "label": idx,
                "class_name": c,
                "make": mk,
                "model": mdl,
                "source": str(row.source),
                "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
            })
    print(f"[external_resized_512] Collected {len(items):,} mapped 512x512 samples from {meta_csv}.")
    return items


def main():
    print("=" * 70)
    print("  EXTERNAL DATASET BUILDER FOR 1,235 MAKE/MODEL CLASSIFICATION")
    print(f"  External Root:    {EXT_DIR}")
    print(f"  Target Label Map: {LABEL_MAP_PATH}")
    print(f"  Max per class:    {MAX_PER_CLASS}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    target_classes, make_to_models = load_target_classes()
    print(f"Loaded {len(target_classes)} target classes from label map.")

    all_candidates = []
    if (EXT_DIR / "external_metadata_51make.csv").exists():
        all_candidates.extend(gather_external_resized_512(make_to_models))
    else:
        all_candidates.extend(gather_stanford_cars(make_to_models))
        all_candidates.extend(gather_compcars_sv(make_to_models))
        all_candidates.extend(gather_boxcars116k(make_to_models))

    df_all = pd.DataFrame(all_candidates)
    print(f"\nTotal collected candidates: {len(df_all):,} across {df_all['label'].nunique()} classes.")

    # Apply per-class cap for balance
    capped_groups = []
    for c_idx, grp in df_all.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        if MAX_PER_CLASS > 0 and len(grp) > MAX_PER_CLASS:
            grp = grp.iloc[:MAX_PER_CLASS]
        capped_groups.append(grp)

    df_capped = pd.concat(capped_groups, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Total after capping (max {MAX_PER_CLASS}/class): {len(df_capped):,} samples across {df_capped['label'].nunique()} classes.")

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    df_capped.to_csv(OUTPUT_DIR / "all_splits.csv", index=False)

    # Save standardized label_map.json
    label_meta = {
        "class_to_idx": target_classes,
        "idx_to_class": {str(v): k for k, v in target_classes.items()},
        "num_classes": len(target_classes)
    }
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  EXTERNAL DATASET SPLITS GENERATED SUCCESSFULLY!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  - Train samples:  {len(df_train):,}")
    print(f"  - Val samples:    {len(df_val):,}")
    print(f"  - Test samples:   {len(df_test):,}")
    print(f"  - Total samples:  {len(df_capped):,}")
    print(f"  - Active Classes: {df_capped['label'].nunique()} / {len(target_classes)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
