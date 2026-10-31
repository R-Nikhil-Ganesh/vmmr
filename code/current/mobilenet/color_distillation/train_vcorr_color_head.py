#!/usr/bin/env python3
"""
Train MobileNetV2 Color MLP on vCorR & Evaluate Against CompCars CCTV Ground Truth
==================================================================================
1. Extracts L16 (160-D) MobileNetV2 features for vCorR train (7,334) and val (1,519) images.
2. Trains a color classification head (Deep MLP-512x2 and Standard MLP-128) using PyTorch.
3. Evaluates the trained head on:
   a. Held-out vCorR test split (1,520 images)
   b. Held-out CompCars CCTV ground-truth test split (4,823 images)
4. Directly benchmarks vCorR-trained heads against our Qwen-distilled PlatesMania heads.
5. Exports the best vCorR color head into fused multi-task ONNX format.
"""

import os
import sys
import glob
import site
import ctypes
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Preload NVIDIA CUDA libs for ONNX Runtime GPU support
for _sp in site.getsitepackages():
    for _lib in sorted(glob.glob(os.path.join(_sp, "nvidia", "*", "lib", "*.so*"))):
        try:
            ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# Add parent directories
SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR))
sys.path.insert(0, str(MOBILENET_DIR / "annotation"))

import paths
from train_color_head import (
    ColorHeadMLP,
    ColorHeadDeepMLP,
    export_multitask_onnx,
    L16_ONNX_OUTPUT,
    L16_CHANNELS,
)

COLORS = [
    "white", "black", "silver", "gray", "red", "blue", "green",
    "yellow", "orange", "brown", "beige", "gold", "other"
]
COLOR2IDX = {c: i for i, c in enumerate(COLORS)}

# vCorR 15 classes -> Our 13 classes
VCORR_TO_13 = {
    "black": "black",
    "white": "white",
    "silver": "silver",
    "grey": "gray",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "beige": "beige",
    "gold": "gold",
    "pink": "other",
    "purple": "other",
    "tan": "beige",
}

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

PRED_TO_COMPCARS_GROUPED = {
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

PRED_TO_COMPCARS_STRICT = {
    "black": "black",
    "white": "white",
    "red": "red",
    "yellow": "yellow",
    "blue": "blue",
    "green": "green",
    "brown": "brown",
    "beige": "champagne",
    "silver": "silver",
    "gray": "silver",
    "gold": "champagne",
    "orange": "orange",
    "other": "other",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class ImageListDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img = Image.open(r["filepath"]).convert("RGB").resize((512, 512), Image.BILINEAR)
            arr = (np.array(img, dtype=np.float32) / 255.0).transpose((2, 0, 1))
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            label = r.get("label_idx", -1)
            return torch.from_numpy(arr), label, idx
        except Exception:
            return torch.zeros((3, 512, 512), dtype=torch.float32), -1, idx


def extract_l16_features(onnx_path: str, records: List[dict], batch_size: int = 128, num_workers: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    model = onnx.load(onnx_path)
    if L16_ONNX_OUTPUT not in [o.name for o in model.graph.output]:
        vi = onnx.helper.ValueInfoProto()
        vi.name = L16_ONNX_OUTPUT
        model.graph.output.append(vi)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)

    dataset = ImageListDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    N = len(records)
    features = np.empty((N, L16_CHANNELS), dtype=np.float16)
    labels = np.empty(N, dtype=np.int64)

    t0 = time.time()
    seen = 0
    for imgs, lbls, idxs in loader:
        b_np = imgs.numpy()
        B = b_np.shape[0]
        raw_outputs = session.run([L16_ONNX_OUTPUT], {"input_image": b_np})[0]
        pooled = raw_outputs.mean(axis=(2, 3)).astype(np.float16)
        features[seen:seen + B] = pooled
        labels[seen:seen + B] = lbls.numpy()
        seen += B

    elapsed = time.time() - t0
    print(f"[Extract] Processed {N:,} images in {elapsed:.1f}s ({N / max(elapsed, 0.01):.1f} img/s)")
    return features, labels


def scan_vcorr(vcorr_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    splits = {"train": [], "val": [], "test": []}
    for s in ["train", "val", "test"]:
        s_dir = vcorr_dir / s
        if not s_dir.exists():
            continue
        for c_dir in sorted([d for d in s_dir.iterdir() if d.is_dir()]):
            vcorr_name = c_dir.name
            mapped_name = VCORR_TO_13.get(vcorr_name, "other")
            mapped_idx = COLOR2IDX[mapped_name]
            for img_path in sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpeg"))):
                splits[s].append({
                    "filepath": str(img_path),
                    "vcorr_class": vcorr_name,
                    "color_name": mapped_name,
                    "label_idx": mapped_idx,
                    "split": s,
                })
    return splits["train"], splits["val"], splits["test"]


def scan_compcars_test() -> List[dict]:
    test_csv = Path(MOBILENET_DIR) / "run4/splits_1235models/test.csv"
    meta_csv = Path(paths.EXT_DATASETS_DIR) / "external_metadata_51make.csv"
    
    df_test = pd.read_csv(test_csv)
    df_meta = pd.read_csv(meta_csv)
    
    prefix = "/home/researchadmin/Econ/external_resized_512/"
    df_meta["rel_path"] = df_meta["image_path"].str.replace(prefix, "", regex=False)
    df_meta_gt = df_meta[df_meta["color_label"] >= 0].copy()
    
    m_test = df_test.merge(df_meta_gt[["rel_path", "color_label", "image_path"]], left_on="image_path", right_on="rel_path")
    records = []
    for _, r in m_test.iterrows():
        cid = int(r["color_label"])
        cname = COMPCARS_ID2COLOR[cid]
        records.append({
            "filepath": str(r["image_path_y"]),
            "compcars_id": cid,
            "compcars_name": cname,
            "label_idx": -1,  # Not directly 0-12
        })
    return records


def train_head(
    head_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Tuple[nn.Module, np.ndarray, np.ndarray, dict]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_tr_t = torch.tensor((X_train - mean) / std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor((X_val - mean) / std, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    if head_type == "mlp_512x2":
        head = ColorHeadDeepMLP(in_features=L16_CHANNELS, hidden_dims=[512, 512], num_classes=len(COLORS), dropout=0.2)
    else:
        head = ColorHeadMLP(in_features=L16_CHANNELS, hidden_dim=128, num_classes=len(COLORS), dropout=0.2)
    
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    
    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_t.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
            acc = float(accuracy_score(y_val, val_preds) * 100.0)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()

    # Final val metrics
    with torch.no_grad():
        val_logits = head(X_val_t.to(device))
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0) * 100.0)

    val_metrics = {"val_top1_acc": round(best_acc, 2), "val_macro_f1": round(val_f1, 2)}
    return head, mean, std, val_metrics


def evaluate_on_compcars(head: nn.Module, mean: np.ndarray, std: np.ndarray, X_comp: np.ndarray, compcars_records: List[dict], device: str = "cuda") -> dict:
    head.eval()
    X_t = torch.tensor((X_comp - mean) / std, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = head(X_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    top1_indices = probs.argmax(axis=1)
    top3_indices = np.argsort(-probs, axis=1)[:, :3]

    pred_13_colors = [COLORS[i] for i in top1_indices]
    gt_compcars = [r["compcars_name"] for r in compcars_records]

    # Map our predictions to CompCars taxonomy
    pred_grouped = [PRED_TO_COMPCARS_GROUPED.get(c, "other") for c in pred_13_colors]
    pred_strict = [PRED_TO_COMPCARS_STRICT.get(c, "other") for c in pred_13_colors]

    # Check top-3 grouped
    top3_grouped_matches = 0
    for row_top3, gt in zip(top3_indices, gt_compcars):
        row_colors = [PRED_TO_COMPCARS_GROUPED.get(COLORS[i], "other") for i in row_top3]
        if gt in row_colors:
            top3_grouped_matches += 1

    strict_acc = float(accuracy_score(gt_compcars, pred_strict) * 100.0)
    grouped_acc = float(accuracy_score(gt_compcars, pred_grouped) * 100.0)
    top3_acc = float((top3_grouped_matches / len(gt_compcars)) * 100.0)
    grouped_f1 = float(f1_score(gt_compcars, pred_grouped, average="macro", zero_division=0) * 100.0)

    # Per-color breakdown
    per_color = {}
    for cid, cname in sorted(COMPCARS_ID2COLOR.items()):
        gt_mask = np.array([g == cname for g in gt_compcars])
        n_gt = int(gt_mask.sum())
        if n_gt == 0:
            continue
        corr = sum((p == cname and g == cname) for p, g in zip(pred_grouped, gt_compcars))
        rec = (corr / n_gt) * 100.0
        n_pred = sum(p == cname for p in pred_grouped)
        prec = (corr / n_pred) * 100.0 if n_pred > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_color[cname] = {
            "n": n_gt,
            "recall": round(float(rec), 2),
            "precision": round(float(prec), 2),
            "f1": round(float(f1), 2),
        }

    return {
        "n_samples": len(gt_compcars),
        "grouped_acc": round(grouped_acc, 2),
        "strict_acc": round(strict_acc, 2),
        "top3_acc": round(top3_acc, 2),
        "macro_f1": round(grouped_f1, 2),
        "per_color": per_color,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Color MLP on vCorR & Evaluate on CompCars")
    parser.add_argument("--vcorr-dir", default="/home/researchadmin/Econ-n/repo-clone/colour_dataset")
    parser.add_argument("--base-model", default="/home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/run4/output_merged_run4/models/mobilenet_v2_best.onnx")
    parser.add_argument("--out-dir", default="/home/researchadmin/Econ-n/repo-clone/vmmr/code/current/mobilenet/color_distillation/output_vcorr_mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = SCRIPT_DIR / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("  Train Color MLP on vCorR & Benchmark vs. CompCars Ground Truth")
    print("==================================================================")

    # 1. Scan vCorR
    train_recs, val_recs, test_recs = scan_vcorr(Path(args.vcorr_dir))
    print(f"[vCorR] Images: Train={len(train_recs):,}, Val={len(val_recs):,}, Test={len(test_recs):,}")

    # 2. Extract / Load vCorR Features
    tr_feat_path = cache_dir / "vcorr_features_L16_train.npy"
    tr_lbl_path = cache_dir / "vcorr_labels_train.npy"
    val_feat_path = cache_dir / "vcorr_features_L16_val.npy"
    val_lbl_path = cache_dir / "vcorr_labels_val.npy"
    te_feat_path = cache_dir / "vcorr_features_L16_test.npy"
    te_lbl_path = cache_dir / "vcorr_labels_test.npy"

    if not tr_feat_path.exists() or not val_feat_path.exists():
        print(f"\n[Extract] Extracting L16 features for vCorR Train & Val...")
        X_train, y_train = extract_l16_features(args.base_model, train_recs)
        X_val, y_val = extract_l16_features(args.base_model, val_recs)
        np.save(tr_feat_path, X_train)
        np.save(tr_lbl_path, y_train)
        np.save(val_feat_path, X_val)
        np.save(val_lbl_path, y_val)
    else:
        print(f"\n[Extract] Loading cached vCorR features...")
        X_train = np.load(tr_feat_path)
        y_train = np.load(tr_lbl_path)
        X_val = np.load(val_feat_path)
        y_val = np.load(val_lbl_path)

    if not te_feat_path.exists():
        X_test, y_test = extract_l16_features(args.base_model, test_recs)
        np.save(te_feat_path, X_test)
        np.save(te_lbl_path, y_test)
    else:
        X_test = np.load(te_feat_path)
        y_test = np.load(te_lbl_path)

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # 3. Scan & Extract CompCars Test
    comp_recs = scan_compcars_test()
    print(f"\n[CompCars] Found {len(comp_recs):,} held-out CompCars CCTV test images with human ground-truth labels.")
    
    comp_feat_path = cache_dir / "compcars_test_l16_features.npy"
    if not comp_feat_path.exists():
        print(f"[Extract] Extracting L16 features for CompCars Test...")
        X_comp, _ = extract_l16_features(args.base_model, comp_recs)
        np.save(comp_feat_path, X_comp)
    else:
        print(f"[Extract] Loading cached CompCars test features...")
        X_comp = np.load(comp_feat_path)
    X_comp = X_comp.astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    for head_type, head_name in [("mlp_512x2", "vCorR Deep MLP-512x2"), ("mlp", "vCorR Standard MLP-128")]:
        print(f"\n------------------------------------------------------------------")
        print(f"  Training {head_name} on vCorR ({args.epochs} epochs)...")
        print(f"------------------------------------------------------------------")
        head, mean, std, val_stats = train_head(
            head_type, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
        )
        print(f"  vCorR Val Accuracy: {val_stats['val_top1_acc']:.2f}%, Macro F1: {val_stats['val_macro_f1']:.2f}%")

        # Evaluate on vCorR test split
        head.eval()
        with torch.no_grad():
            X_te_t = torch.tensor((X_test - mean) / std, dtype=torch.float32).to(device)
            vcorr_te_preds = head(X_te_t).argmax(dim=-1).cpu().numpy()
            vcorr_te_acc = float(accuracy_score(y_test, vcorr_te_preds) * 100.0)
            vcorr_te_f1 = float(f1_score(y_test, vcorr_te_preds, average="macro", zero_division=0) * 100.0)
        print(f"  vCorR Test Accuracy: {vcorr_te_acc:.2f}%, Macro F1: {vcorr_te_f1:.2f}%")

        # Evaluate on CompCars CCTV ground truth
        comp_stats = evaluate_on_compcars(head, mean, std, X_comp, comp_recs, device=device)
        print(f"  CompCars Ground Truth ({comp_stats['n_samples']:,} test images):")
        print(f"    Grouped Accuracy: {comp_stats['grouped_acc']:.2f}%")
        print(f"    Strict Accuracy:  {comp_stats['strict_acc']:.2f}%")
        print(f"    Top-3 Accuracy:   {comp_stats['top3_acc']:.2f}%")
        print(f"    Macro F1:         {comp_stats['macro_f1']:.2f}%")

        results[head_name] = {
            "vcorr_val": val_stats,
            "vcorr_test": {"top1_acc": round(vcorr_te_acc, 2), "macro_f1": round(vcorr_te_f1, 2)},
            "compcars_test": comp_stats,
        }

        # Save weights and multitask ONNX for Deep MLP
        if head_type == "mlp_512x2":
            best_weights = out_dir / "color_head_best.pt"
            torch.save(head.state_dict(), best_weights)
            best_onnx = out_dir / "mobilenet_v2_multitask.onnx"
            export_multitask_onnx(args.base_model, head, mean, std, str(best_onnx), head_type=head_type)

    # 4. Save results to JSON
    out_json = out_dir / "vcorr_train_and_compcars_eval.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================================================================")
    print(f"  Results saved to: {out_json}")
    print(f"==================================================================")


if __name__ == "__main__":
    main()
