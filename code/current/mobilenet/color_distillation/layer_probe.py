#!/usr/bin/env python3
"""
MobileNetV2 Backbone Layer Selection Probing for Color Distillation
===================================================================
Empirically benchmarks candidate layers (L3, L6, L10, L13, L16, L17, L18, Embedding)
in the MobileNetV2 Model C backbone to determine the optimal tap point for the color head.

Extracts intermediate activations from mobilenet_v2_best.onnx, pools spatially,
and trains fast linear and MLP probes with class-weighted cross-entropy.
Outputs a comparative accuracy, macro F1, and latency benchmark table.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
from PIL import Image

import onnx
import onnxruntime as ort

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR / "annotation"))
from common import load_crop, parse_bbox

COLORS = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige", "gold", "other"]
COLOR2IDX = {c: i for i, c in enumerate(COLORS)}

# Candidate layers in MobileNetV2 to probe
CANDIDATE_LAYERS = {
    "L3": {
        "onnx_output": "/features/features.3/Add_output_0",
        "channels": 24,
        "spatial": "128x128",
        "stage": "Early Low-Level",
        "description": "Local contrast & edge/color histograms",
    },
    "L6": {
        "onnx_output": "/features/features.6/Add_output_0",
        "channels": 32,
        "spatial": "64x64",
        "stage": "Low-Mid Level",
        "description": "Texture primitives & patch color",
    },
    "L10": {
        "onnx_output": "/features/features.10/Add_output_0",
        "channels": 64,
        "spatial": "32x32",
        "stage": "Mid-Level Stage 1",
        "description": "Initial vehicle semantic boundary",
    },
    "L13": {
        "onnx_output": "/features/features.13/Add_output_0",
        "channels": 96,
        "spatial": "32x32",
        "stage": "Mid-Level Stage 2",
        "description": "Frozen backbone boundary (rich color + shape)",
    },
    "L16": {
        "onnx_output": "/features/features.16/Add_output_0",
        "channels": 160,
        "spatial": "16x16",
        "stage": "Deep Stage 1 (Unfrozen)",
        "description": "Major vehicle sub-assemblies",
    },
    "L17": {
        "onnx_output": "/features/features.17/conv/conv.2/Conv_output_0",
        "channels": 320,
        "spatial": "16x16",
        "stage": "Deep Stage 2 (Unfrozen)",
        "description": "High-level semantic vehicle body",
    },
    "L18": {
        "onnx_output": "/features/features.18/features.18.2/Clip_output_0",
        "channels": 1280,
        "spatial": "16x16",
        "stage": "Final Conv (Unfrozen)",
        "description": "Pre-pool identity representation",
    },
    "Embedding": {
        "onnx_output": "embeddings",
        "channels": 1280,
        "spatial": "1x1 (Pooled)",
        "stage": "Penultimate 1280-D",
        "description": "Standard pooled feature vector",
    },
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class ImageCropDataset(Dataset):
    def __init__(self, records: List[dict], img_size: int = 512):
        self.records = records
        self.img_size = img_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        try:
            bbox = (r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]) if r.get("bbox_x1", -1) >= 0 else None
            img = load_crop(r["path"], r["domain"], bbox)
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
            return arr, COLOR2IDX[r["color"]], idx
        except Exception as e:
            arr = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            return arr, -1, idx


def prepare_onnx_multi_output_session(onnx_path: str) -> Tuple[ort.InferenceSession, List[str]]:
    """Loads ONNX model and modifies in-memory graph to expose intermediate layer outputs."""
    model = onnx.load(onnx_path)
    output_names = []

    for layer_id, cfg in CANDIDATE_LAYERS.items():
        out_name = cfg["onnx_output"]
        if out_name not in [o.name for o in model.graph.output]:
            vi = onnx.helper.ValueInfoProto()
            vi.name = out_name
            model.graph.output.append(vi)
        output_names.append(out_name)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    return session, output_names


def extract_intermediate_features(
    session: ort.InferenceSession,
    output_names: List[str],
    dataset: ImageCropDataset,
    batch_size: int = 64,
    num_workers: int = 8,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Runs ONNX inference and applies spatial Global Average Pooling to each candidate layer."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    N = len(dataset)

    # Preallocate pooled feature matrices
    features = {
        layer_id: np.empty((N, cfg["channels"]), dtype=np.float32)
        for layer_id, cfg in CANDIDATE_LAYERS.items()
    }
    labels = np.empty(N, dtype=np.int64)

    t0 = time.time()
    seen = 0
    print(f"[Extract] Extracting candidate features for {N:,} images (batch={batch_size})...")

    for batch_imgs, batch_labels, batch_indices in loader:
        B = batch_imgs.shape[0]
        # Normalize with ImageNet mean/std
        batch_np = batch_imgs.numpy()
        batch_np = (batch_np - IMAGENET_MEAN) / IMAGENET_STD

        raw_outputs = session.run(output_names, {"input_image": batch_np})

        # Process each layer
        for out_name, out_arr in zip(output_names, raw_outputs):
            # Find layer_id corresponding to this output name
            layer_id = next(lid for lid, cfg in CANDIDATE_LAYERS.items() if cfg["onnx_output"] == out_name)
            if out_arr.ndim == 4:
                # Spatial pooling: (B, C, H, W) -> (B, C)
                pooled = out_arr.mean(axis=(2, 3))
            elif out_arr.ndim == 2:
                pooled = out_arr
            else:
                pooled = out_arr.reshape(B, -1)

            features[layer_id][seen:seen + B] = pooled

        labels[seen:seen + B] = batch_labels.numpy()
        seen += B
        if seen % 5000 == 0 or seen == N:
            elapsed = time.time() - t0
            print(f"  Processed {seen:,}/{N:,} images ({seen/elapsed:.1f} img/s)...")

    return features, labels


class LinearProbe(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 13):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, num_classes: int = 13, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_and_eval_probe(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Trains a probe using class-weighted cross-entropy and evaluates on validation set."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    train_time = time.time() - t0

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_x = X_val.to(device)
        val_logits = model(val_x)
        val_probs = F.softmax(val_logits, dim=-1)
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        top3_preds = torch.topk(val_logits, k=3, dim=-1).indices.cpu().numpy()

    y_val_np = y_val.numpy()
    top1_acc = accuracy_score(y_val_np, val_preds) * 100.0
    top3_acc = np.mean([y in top3 for y, top3 in zip(y_val_np, top3_preds)]) * 100.0
    macro_f1 = f1_score(y_val_np, val_preds, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_val_np, val_preds, average="weighted", zero_division=0) * 100.0

    # Per-class F1
    report = classification_report(y_val_np, val_preds, target_names=COLORS, output_dict=True, zero_division=0)
    per_class_f1 = {c: report[c]["f1-score"] * 100.0 for c in COLORS if c in report}

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_time_sec": train_time,
        "per_class_f1": per_class_f1,
        "param_count": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Layer Selection Probing for Color Distillation")
    parser.add_argument("--annotations", default="code/current/mobilenet/annotation_runs/run5/annotations.csv")
    parser.add_argument("--model-path", default="code/current/mobilenet/merged_dataset/output_merged_run4/models/mobilenet_v2_best.onnx")
    parser.add_argument("--out-dir", default="code/current/mobilenet/color_distillation")
    parser.add_argument("--num-train", type=int, default=25000, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=5000, help="Number of validation samples")
    parser.add_argument("--min-conf", type=float, default=0.75, help="Minimum Qwen color confidence filter")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("  MobileNetV2 Layer Selection Probing for Color Distillation")
    print(f"  Annotations: {args.annotations}")
    print(f"  Model:       {args.model_path}")
    print(f"  Output Dir:  {out_dir}")
    print("==================================================================")

    # 1. Load annotations and sample balanced high-confidence subsets
    print(f"[Data] Loading {args.annotations}...")
    df = pd.read_csv(args.annotations)
    valid_df = df[(df["color"].isin(COLORS)) & (df["color_conf"] >= args.min_conf)].copy()
    print(f"[Data] Total valid samples with color_conf >= {args.min_conf}: {len(valid_df):,} / {len(df):,}")

    train_pool = valid_df[valid_df["split"] == "train"]
    val_pool = valid_df[valid_df["split"] == "val"]

    # Stratified balanced sampling across colors
    def sample_stratified(pool: pd.DataFrame, target_n: int) -> pd.DataFrame:
        per_class = max(10, target_n // len(COLORS))
        sampled = []
        for c in COLORS:
            sub = pool[pool["color"] == c]
            n_take = min(len(sub), per_class)
            sampled.append(sub.sample(n=n_take, random_state=42))
        res = pd.concat(sampled)
        if len(res) < target_n:
            rem = target_n - len(res)
            unpicked = pool.drop(res.index)
            res = pd.concat([res, unpicked.sample(n=min(len(unpicked), rem), random_state=42)])
        return res.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df_train = sample_stratified(train_pool, args.num_train)
    df_val = sample_stratified(val_pool, args.num_val)
    print(f"[Data] Sampled {len(df_train):,} train images and {len(df_val):,} val images.")

    # Compute inverse frequency class weights
    class_counts = df_train["color"].value_counts().reindex(COLORS).fillna(1).values
    total_samples = len(df_train)
    weights = total_samples / (len(COLORS) * class_counts)
    weights = np.clip(weights, 0.2, 5.0)  # Bound extreme weights
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    # 2. Extract intermediate features using ONNX Runtime
    session, output_names = prepare_onnx_multi_output_session(args.model_path)
    combined_records = df_train.to_dict("records") + df_val.to_dict("records")
    dataset = ImageCropDataset(combined_records, img_size=512)

    features_all, labels_all = extract_intermediate_features(
        session, output_names, dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    n_train = len(df_train)
    y_train = torch.tensor(labels_all[:n_train], dtype=torch.long)
    y_val = torch.tensor(labels_all[n_train:], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Probe] Training probes on {device}...")

    results = []

    # 3. Train and evaluate probes for each candidate layer
    for layer_id, cfg in CANDIDATE_LAYERS.items():
        C = cfg["channels"]
        print(f"\n--- Evaluating Layer: {layer_id} ({cfg['stage']}) | Channels: {C} | Resolution: {cfg['spatial']} ---")
        X_train_np = features_all[layer_id][:n_train]
        X_val_np = features_all[layer_id][n_train:]

        # Standardize features using training mean & std
        mean = X_train_np.mean(axis=0, keepdims=True)
        std = X_train_np.std(axis=0, keepdims=True) + 1e-6
        X_train_norm = torch.tensor((X_train_np - mean) / std, dtype=torch.float32)
        X_val_norm = torch.tensor((X_val_np - mean) / std, dtype=torch.float32)

        # A. Linear Probe
        lin_model = LinearProbe(C, num_classes=len(COLORS))
        lin_res = train_and_eval_probe(
            lin_model, X_train_norm, y_train, X_val_norm, y_val, class_weights_t, device=device
        )
        print(f"  [Linear Probe] Top-1: {lin_res['top1_acc']:.2f}% | Top-3: {lin_res['top3_acc']:.2f}% | Macro F1: {lin_res['macro_f1']:.2f}% | Weighted F1: {lin_res['weighted_f1']:.2f}%")

        # B. MLP Probe
        mlp_model = MLPProbe(C, hidden_dim=128, num_classes=len(COLORS))
        mlp_res = train_and_eval_probe(
            mlp_model, X_train_norm, y_train, X_val_norm, y_val, class_weights_t, device=device
        )
        print(f"  [MLP Probe]    Top-1: {mlp_res['top1_acc']:.2f}% | Top-3: {mlp_res['top3_acc']:.2f}% | Macro F1: {mlp_res['macro_f1']:.2f}% | Weighted F1: {mlp_res['weighted_f1']:.2f}%")

        results.append({
            "layer": layer_id,
            "stage": cfg["stage"],
            "channels": C,
            "spatial": cfg["spatial"],
            "description": cfg["description"],
            "lin_top1": lin_res["top1_acc"],
            "lin_top3": lin_res["top3_acc"],
            "lin_macro_f1": lin_res["macro_f1"],
            "lin_weighted_f1": lin_res["weighted_f1"],
            "mlp_top1": mlp_res["top1_acc"],
            "mlp_top3": mlp_res["top3_acc"],
            "mlp_macro_f1": mlp_res["macro_f1"],
            "mlp_weighted_f1": mlp_res["weighted_f1"],
            "lin_params": lin_res["param_count"],
            "mlp_params": mlp_res["param_count"],
            "per_class_f1_lin": lin_res["per_class_f1"],
            "per_class_f1_mlp": mlp_res["per_class_f1"],
        })

    # 4. Summary Table & Report
    res_df = pd.DataFrame(results)
    csv_path = out_dir / "probe_results.csv"
    res_df.to_csv(csv_path, index=False)

    json_path = out_dir / "probe_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print("  LAYER PROBE BENCHMARK SUMMARY (Ranked by Linear Macro F1)")
    print("=" * 95)
    summary_cols = ["layer", "stage", "channels", "spatial", "lin_top1", "lin_macro_f1", "mlp_top1", "mlp_macro_f1"]
    sorted_df = res_df.sort_values(by="lin_macro_f1", ascending=False)
    print(sorted_df[summary_cols].to_string(index=False))
    print("=" * 95)
    print(f"Results saved to: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()

