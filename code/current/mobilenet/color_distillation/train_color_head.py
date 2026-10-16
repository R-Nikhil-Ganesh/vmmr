#!/usr/bin/env python3
"""
MobileNetV2 Color Head Distillation Training (features[16] tap)
===============================================================
Distills Qwen3-VL-8B color annotations (from run5/annotations.csv) into a lightweight
color classification head branching from block features[16] (160 channels) of the Model C backbone.

Features:
- Caches extracted 160-D pooled features (float16) to disk (~287 MB for 898k train images)
  enabling instantaneous loading and rapid hyperparameter iterations.
- Class-weighted cross-entropy loss with optional label smoothing.
- High-capacity MLP head (160 -> 128 -> 13) or fast Linear head (160 -> 13).
- Automatic multi-task ONNX export (mobilenet_v2_multitask.onnx) by splicing the trained
  color head onto mobilenet_v2_best.onnx with zero degradation to existing outputs.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from PIL import Image

import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR / "annotation"))
from common import load_crop, parse_bbox

COLORS = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige", "gold", "other"]
COLOR2IDX = {c: i for i, c in enumerate(COLORS)}
L16_ONNX_OUTPUT = "/features/features.16/Add_output_0"
L16_CHANNELS = 160

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class ImageExtractDataset(Dataset):
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
        except Exception:
            arr = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            return arr, -1, idx


def extract_l16_features_to_disk(
    onnx_path: str,
    records: List[dict],
    out_feat_path: Path,
    out_label_path: Path,
    batch_size: int = 128,
    num_workers: int = 10,
    img_size: int = 512,
):
    """Runs ONNX inference to extract L16 features and saves as memory-mapped float16 numpy arrays."""
    print(f"[Extract] Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)
    if L16_ONNX_OUTPUT not in [o.name for o in model.graph.output]:
        vi = onnx.helper.ValueInfoProto()
        vi.name = L16_ONNX_OUTPUT
        model.graph.output.append(vi)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model.SerializeToString(), providers=providers)

    dataset = ImageExtractDataset(records, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    N = len(dataset)

    out_feat_path.parent.mkdir(parents=True, exist_ok=True)
    # Preallocate output files
    features = np.empty((N, L16_CHANNELS), dtype=np.float16)
    labels = np.empty(N, dtype=np.int64)

    t0 = time.time()
    seen = 0
    print(f"[Extract] Extracting L16 features for {N:,} images (batch={batch_size}, workers={num_workers})...")

    for batch_imgs, batch_labels, _ in loader:
        B = batch_imgs.shape[0]
        batch_np = batch_imgs.numpy()
        batch_np = (batch_np - IMAGENET_MEAN) / IMAGENET_STD

        raw_outputs = session.run([L16_ONNX_OUTPUT], {"input_image": batch_np})[0]
        # Global Average Pooling: (B, 160, H, W) -> (B, 160)
        pooled = raw_outputs.mean(axis=(2, 3)).astype(np.float16)

        features[seen:seen + B] = pooled
        labels[seen:seen + B] = batch_labels.numpy()
        seen += B

        if seen % 20000 == 0 or seen == N:
            elapsed = time.time() - t0
            rate = seen / elapsed
            eta_min = (N - seen) / rate / 60.0
            print(f"  [{seen:,}/{N:,}] {seen/N*100:5.1f}% completed ({rate:.1f} img/s, ETA: {eta_min:.1f} min)...")

    # Filter out corrupted images (label == -1)
    valid_mask = labels >= 0
    features = features[valid_mask]
    labels = labels[valid_mask]

    np.save(out_feat_path, features)
    np.save(out_label_path, labels)
    print(f"[Extract] Saved {len(labels):,} features to {out_feat_path} ({out_feat_path.stat().st_size / 1024**2:.1f} MB)")


class ColorHeadMLP(nn.Module):
    def __init__(self, in_features: int = 160, hidden_dim: int = 128, num_classes: int = 13, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        return self.fc2(h)


class ColorHeadLinear(nn.Module):
    def __init__(self, in_features: int = 160, num_classes: int = 13):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def export_multitask_onnx(
    base_onnx_path: str,
    color_head: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    out_onnx_path: str,
    head_type: str = "mlp"
):
    """Splices the trained color head directly into the ONNX graph."""
    print(f"[Export] Splicing color head onto {base_onnx_path}...")
    model = onnx.load(base_onnx_path)

    # GlobalAveragePool on L16 output
    pool_node = helper.make_node(
        "GlobalAveragePool",
        inputs=[L16_ONNX_OUTPUT],
        outputs=["color_pool_raw"],
        name="color_head_pool"
    )

    flat_node = helper.make_node(
        "Flatten",
        inputs=["color_pool_raw"],
        outputs=["color_flat_raw"],
        name="color_head_flatten",
        axis=1
    )

    # Feature normalization node: (x - mean) / std
    # Represented as Gemm or Mul/Add: y = (x - mean) / std = x * (1/std) - (mean/std)
    scale = (1.0 / std).astype(np.float32)
    shift = (-mean / std).astype(np.float32)

    scale_init = helper.make_tensor("color_norm_scale", TensorProto.FLOAT, [L16_CHANNELS], scale.flatten())
    shift_init = helper.make_tensor("color_norm_shift", TensorProto.FLOAT, [L16_CHANNELS], shift.flatten())

    mul_node = helper.make_node("Mul", inputs=["color_flat_raw", "color_norm_scale"], outputs=["color_norm_mul"], name="color_norm_mul")
    add_node = helper.make_node("Add", inputs=["color_norm_mul", "color_norm_shift"], outputs=["color_feat_norm"], name="color_norm_add")

    new_nodes = [pool_node, flat_node, mul_node, add_node]
    new_inits = [scale_init, shift_init]

    if head_type == "linear":
        w = color_head.fc.weight.detach().cpu().numpy().astype(np.float32)
        b = color_head.fc.bias.detach().cpu().numpy().astype(np.float32)

        w_init = helper.make_tensor("color_fc_weight", TensorProto.FLOAT, w.shape, w.flatten())
        b_init = helper.make_tensor("color_fc_bias", TensorProto.FLOAT, b.shape, b.flatten())
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["color_feat_norm", "color_fc_weight", "color_fc_bias"],
            outputs=["color_predictions"],
            name="color_head_fc",
            transB=1
        )
        new_nodes.append(gemm_node)
        new_inits.extend([w_init, b_init])
    else:
        # Fuse BatchNorm into fc1:
        # y = gamma * (fc1_w * x + fc1_b - running_mean) / sqrt(running_var + eps) + beta
        fc1_w = color_head.fc1.weight.detach().cpu().numpy()
        fc1_b = color_head.fc1.bias.detach().cpu().numpy()
        bn_gamma = color_head.bn1.weight.detach().cpu().numpy()
        bn_beta = color_head.bn1.bias.detach().cpu().numpy()
        bn_mean = color_head.bn1.running_mean.detach().cpu().numpy()
        bn_var = color_head.bn1.running_var.detach().cpu().numpy()
        eps = color_head.bn1.eps

        bn_scale = bn_gamma / np.sqrt(bn_var + eps)
        fused_w1 = fc1_w * bn_scale[:, None]
        fused_b1 = (fc1_b - bn_mean) * bn_scale + bn_beta

        fc2_w = color_head.fc2.weight.detach().cpu().numpy()
        fc2_b = color_head.fc2.bias.detach().cpu().numpy()

        w1_init = helper.make_tensor("color_fc1_weight", TensorProto.FLOAT, fused_w1.shape, fused_w1.flatten())
        b1_init = helper.make_tensor("color_fc1_bias", TensorProto.FLOAT, fused_b1.shape, fused_b1.flatten())
        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["color_feat_norm", "color_fc1_weight", "color_fc1_bias"],
            outputs=["color_fc1_out"],
            name="color_head_fc1",
            transB=1
        )
        relu_node = helper.make_node("Relu", inputs=["color_fc1_out"], outputs=["color_relu_out"], name="color_head_relu")

        w2_init = helper.make_tensor("color_fc2_weight", TensorProto.FLOAT, fc2_w.shape, fc2_w.flatten())
        b2_init = helper.make_tensor("color_fc2_bias", TensorProto.FLOAT, fc2_b.shape, fc2_b.flatten())
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["color_relu_out", "color_fc2_weight", "color_fc2_bias"],
            outputs=["color_predictions"],
            name="color_head_fc2",
            transB=1
        )
        new_nodes.extend([gemm1_node, relu_node, gemm2_node])
        new_inits.extend([w1_init, b1_init, w2_init, b2_init])

    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)

    color_out_vi = helper.make_tensor_value_info("color_predictions", TensorProto.FLOAT, ["batch_size", len(COLORS)])
    model.graph.output.append(color_out_vi)

    Path(out_onnx_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_onnx_path)
    print(f"[Export] Saved multi-task ONNX model to: {out_onnx_path}")

    # Verify session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(out_onnx_path, providers=providers)
    dummy_inp = np.random.randn(1, 3, 512, 512).astype(np.float32)
    outs = sess.run(None, {"input_image": dummy_inp})
    print("[Export] ONNX verification check passed! Output shapes:")
    for out_meta, out_val in zip(sess.get_outputs(), outs):
        print(f"  {out_meta.name:20s}: {out_val.shape}")


def main():
    parser = argparse.ArgumentParser(description="MobileNetV2 Color Distillation Training")
    parser.add_argument("--annotations", default="code/current/mobilenet/annotation_runs/run5/annotations.csv")
    parser.add_argument("--model-path", default="code/current/mobilenet/merged_dataset/output_merged_run4/models/mobilenet_v2_best.onnx")
    parser.add_argument("--out-dir", default="code/current/mobilenet/color_distillation/output")
    parser.add_argument("--head-type", choices=["mlp", "linear"], default="mlp", help="Color head architecture")
    parser.add_argument("--min-conf", type=float, default=0.70, help="Minimum Qwen color confidence filter")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--extract-batch-size", type=int, default=128, help="Batch size for ONNX feature extraction")
    parser.add_argument("--num-workers", type=int, default=10, help="DataLoader workers for extraction")
    parser.add_argument("--skip-extract", action="store_true", help="Skip feature extraction and load existing .npy")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = SCRIPT_DIR / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_feat_path = cache_dir / "features_L16_train.npy"
    train_label_path = cache_dir / "labels_train.npy"
    val_feat_path = cache_dir / "features_L16_val.npy"
    val_label_path = cache_dir / "labels_val.npy"

    print("==================================================================")
    print("  MobileNetV2 Color Head Distillation (features[16] tap)")
    print(f"  Annotations:     {args.annotations}")
    print(f"  Base ONNX Model: {args.model_path}")
    print(f"  Head Type:       {args.head_type.upper()}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Output Dir:      {out_dir}")
    print("==================================================================")

    # 1. Feature Extraction / Loading
    if not args.skip_extract and (not train_feat_path.exists() or not val_feat_path.exists()):
        print(f"[Data] Loading annotations from {args.annotations}...")
        df = pd.read_csv(args.annotations)
        valid = df[(df["color"].isin(COLORS)) & (df["color_conf"] >= args.min_conf)].copy()
        print(f"[Data] Total valid samples with color_conf >= {args.min_conf}: {len(valid):,} / {len(df):,}")

        train_recs = valid[valid["split"] == "train"].to_dict("records")
        val_recs = valid[valid["split"] == "val"].to_dict("records")
        print(f"[Data] Train: {len(train_recs):,} | Val: {len(val_recs):,}")

        extract_l16_features_to_disk(
            args.model_path, train_recs, train_feat_path, train_label_path,
            batch_size=args.extract_batch_size, num_workers=args.num_workers
        )
        extract_l16_features_to_disk(
            args.model_path, val_recs, val_feat_path, val_label_path,
            batch_size=args.extract_batch_size, num_workers=args.num_workers
        )
    else:
        print(f"[Data] Loading cached features from {cache_dir}...")

    X_train_np = np.load(train_feat_path).astype(np.float32)
    y_train_np = np.load(train_label_path)
    X_val_np = np.load(val_feat_path).astype(np.float32)
    y_val_np = np.load(val_label_path)

    print(f"[Data] Loaded Train: {X_train_np.shape} | Val: {X_val_np.shape}")

    # 2. Compute Normalization Statistics from Training Split
    mean = X_train_np.mean(axis=0, keepdims=True)
    std = X_train_np.std(axis=0, keepdims=True) + 1e-6

    X_train_t = torch.tensor((X_train_np - mean) / std, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.long)
    X_val_t = torch.tensor((X_val_np - mean) / std, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.long)

    # 3. Compute Class Weights
    class_counts = np.bincount(y_train_np, minlength=len(COLORS))
    total_samples = len(y_train_np)
    weights = total_samples / (len(COLORS) * np.maximum(class_counts, 1))
    weights = np.clip(weights, 0.25, 4.0)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Training on device: {device}")

    # 4. Instantiate Model
    if args.head_type == "mlp":
        head = ColorHeadMLP(in_features=L16_CHANNELS, hidden_dim=128, num_classes=len(COLORS), dropout=0.2)
    else:
        head = ColorHeadLinear(in_features=L16_CHANNELS, num_classes=len(COLORS))

    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device), label_smoothing=args.label_smoothing)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size * 2, shuffle=False)

    best_macro_f1 = 0.0
    best_weights_path = out_dir / "color_head_best.pt"
    history = []

    print(f"[Train] Starting {args.epochs} epochs (batch={args.batch_size})...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            train_correct += (logits.argmax(dim=-1) == by).sum().item()
            train_total += len(by)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total * 100.0

        # Evaluation
        head.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                logits = head(bx)
                val_preds.append(logits.argmax(dim=-1).cpu().numpy())
                val_targets.append(by.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0

        is_best = val_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = val_macro_f1
            torch.save(head.state_dict(), best_weights_path)

        epoch_time = time.time() - t0
        print(f"  Epoch {epoch:2d}/{args.epochs:2d} ({epoch_time:4.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | Val Acc: {val_acc:5.2f}% Macro F1: {val_macro_f1:5.2f}% {'[BEST]' if is_best else ''}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "time_sec": epoch_time
        })

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    # 5. Final Evaluation on Best Checkpoint
    head.load_state_dict(torch.load(best_weights_path))
    head.eval()

    val_preds, val_targets, top3_preds = [], [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = head(bx)
            val_preds.append(logits.argmax(dim=-1).cpu().numpy())
            top3_preds.append(torch.topk(logits, k=3, dim=-1).indices.cpu().numpy())
            val_targets.append(by.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    top3_preds = np.concatenate(top3_preds)

    final_top1 = accuracy_score(val_targets, val_preds) * 100.0
    final_top3 = np.mean([y in top3 for y, top3 in zip(val_targets, top3_preds)]) * 100.0
    final_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0) * 100.0

    print("\n" + "=" * 65)
    print("  FINAL DISTILLATION EVALUATION (Validation Set)")
    print("=" * 65)
    print(f"  Top-1 Accuracy: {final_top1:.2f}%")
    print(f"  Top-3 Accuracy: {final_top3:.2f}%")
    print(f"  Macro F1:       {final_macro_f1:.2f}%")
    print("\nDetailed Per-Class Classification Report:")
    report_str = classification_report(val_targets, val_preds, target_names=COLORS, zero_division=0)
    print(report_str)

    with open(out_dir / "final_report.txt", "w") as f:
        f.write(report_str)

    # 6. Spliced ONNX Multi-Task Export
    out_onnx_path = out_dir / "mobilenet_v2_multitask.onnx"
    export_multitask_onnx(
        args.model_path, head, mean, std, str(out_onnx_path), head_type=args.head_type
    )

    print("\n==================================================================")
    print("  Distillation Complete!")
    print(f"  Weights: {best_weights_path}")
    print(f"  ONNX:    {out_onnx_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
