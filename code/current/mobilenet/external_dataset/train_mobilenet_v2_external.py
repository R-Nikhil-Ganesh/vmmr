#!/usr/bin/env python3
"""
MobileNetV2 Training Pipeline on External Merged Dataset (PyTorch)
==================================================================
Dataset: /home/researchadmin/Econ/external_datasets/merged_data
Classes: 35 Vehicle Makes (BoxCars116k, Stanford Cars, CompCars CCTV/Web)
Environment: pt-env (PyTorch 2.14 + CUDA, RTX 4090)
"""

import os
import sys
import json
import time
import copy
import argparse
from pathlib import Path

# Matplotlib & Torch cache redirection
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
CACHE_DIR = os.environ.get("TORCH_CACHE_DIR", "/tmp/torch_cache")
os.environ["TRITON_CACHE_DIR"] = os.path.join(CACHE_DIR, "triton")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CACHE_DIR, "inductor")
os.makedirs(os.path.join(CACHE_DIR, "triton"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "inductor"), exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
import torchvision.io as io
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_MOBILENET_DIR = CURRENT_DIR.parent
if str(ROOT_MOBILENET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_MOBILENET_DIR))

import output_analysis as oa


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on External Merged Dataset (35 Makes).")
    parser.add_argument("--splits-dir", type=str,
        default="/home/researchadmin/Econ/external_datasets/merged_data",
        help="Path containing train.csv, val.csv, test.csv, and label_map.json")
    parser.add_argument("--output-dir", type=str,
        default=str(CURRENT_DIR / "output_mobilenet_v2_external"),
        help="Destination directory for checkpoints, metrics, and plots")
    parser.add_argument("--img-size", type=int, default=512, help="Input resolution (default: 512)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Classifier head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-5, help="Backbone learning rate")
    parser.add_argument("--dropout", type=float, default=0.25, help="Classifier head dropout rate")
    parser.add_argument("--unfreeze-layers", type=int, default=5, help="Number of top backbone layers to unlock (default: 5)")
    parser.add_argument("--crop-bottom-pct", type=float, default=0.0, help="Bottom watermark strip crop (default: 0.0)")
    parser.add_argument("--num-workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


# ==============================================================================
# External Merged Dataset Loader
# ==============================================================================
class ExternalMergedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 512, crop_bottom_pct: float = 0.0):
        self.paths = df["image_path"].values
        self.labels = df["label"].values.astype(np.int64)
        self.img_size = img_size
        self.crop_bottom_pct = crop_bottom_pct

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = io.read_image(path, mode=io.ImageReadMode.RGB)
            _, h, w = img.shape
            # Bottom border strip crop to strip web dealership stamps
            bottom_h = max(20, int(h * (1.0 - self.crop_bottom_pct)))
            cropped = img[:, :bottom_h, :]
            resized = nn.functional.interpolate(
                cropped.unsqueeze(0).float(),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).to(torch.uint8)
            return resized, self.labels[idx]
        except Exception:
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.uint8), self.labels[idx]


def build_transforms():
    train_gpu_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=(-10, 10), translate=(0.06, 0.06), scale=(0.94, 1.06)),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        v2.Normalize(mean=oa.IMAGENET_MEAN, std=oa.IMAGENET_STD)
    ])
    val_test_gpu_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=oa.IMAGENET_MEAN, std=oa.IMAGENET_STD)
    ])
    return train_gpu_transforms, val_test_gpu_transforms


def create_model(num_classes: int, dropout: float = 0.25, unfreeze_layers: int = 5):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    # Freeze entire backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Unlock only the top N backbone layers
    for layer in model.features[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Classifier head is fully trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] MobileNetV2: {n_frozen:,} frozen parameters, {n_trainable:,} trainable parameters (top {unfreeze_layers} backbone layers unlocked)")
    return model


def export_to_onnx(model: nn.Module, num_classes: int, img_size: int, out_onnx_path: Path):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")
    cpu_model = copy.deepcopy(model).cpu()
    
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        cpu_model,
        dummy_input,
        str(out_onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["predictions"],
        dynamic_axes={"input_image": {0: "batch_size"}, "predictions": {0: "batch_size"}}
    )
    print(f"[ONNX] Exported model successfully to: {out_onnx_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for d in [models_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "label_map.json") as f:
        label_map_raw = json.load(f)
    if "class_to_idx" in label_map_raw:
        class_to_idx = {k: int(v) for k, v in label_map_raw["class_to_idx"].items()}
    else:
        class_to_idx = {k: int(v) for k, v in label_map_raw.items()}
    num_classes = len(class_to_idx)
    with open(models_dir / "label_map.json", "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": {v: k for k, v in class_to_idx.items()}}, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  MobileNetV2 External (PyTorch) | {num_classes} classes | {args.img_size}x{args.img_size} | batch={args.batch_size}")
    print(f"  LR={args.lr} | Backbone LR={args.backbone_lr} | Epochs={args.epochs}")
    print(f"  Splits: {splits_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 65 + "\n")

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df   = pd.read_csv(splits_dir / "val.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")
    print(f"[Data] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_ds = ExternalMergedDataset(train_df, args.img_size, args.crop_bottom_pct)
    val_ds   = ExternalMergedDataset(val_df, args.img_size, args.crop_bottom_pct)
    test_ds  = ExternalMergedDataset(test_df, args.img_size, args.crop_bottom_pct)

    # Optimized high-throughput DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    test_loader  = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    train_tf, val_tf = build_transforms()

    model = create_model(num_classes=num_classes, dropout=args.dropout).to(device)

    # Differential learning rate
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = []

    best_pt_path = models_dir / "mobilenet_v2_best.pt"
    best_onnx_path = models_dir / "mobilenet_v2_best.onnx"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            imgs = train_tf(imgs)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                imgs = val_tf(imgs)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * len(labels)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"\n[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_pt_path)
            print(f"[*] New best validation loss: {best_val_loss:.4f}. Saved checkpoint to {best_pt_path}")
            try:
                export_to_onnx(model, num_classes, args.img_size, best_onnx_path)
            except Exception as e:
                print(f"[Warning] ONNX export failed: {e}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(reports_dir / "training_history.csv", index=False)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["epoch"], history_df["train_acc"], "o-", label="Train Accuracy", color="#2563EB")
    axes[0].plot(history_df["epoch"], history_df["val_acc"], "s--", label="Val Accuracy", color="#16A34A")
    axes[0].set_title("Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_loss"], "o-", label="Train Loss", color="#DC2626")
    axes[1].plot(history_df["epoch"], history_df["val_loss"], "s--", label="Val Loss", color="#EA580C")
    axes[1].set_title("Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(str(plots_dir / "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Plot] Training curves saved to {plots_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
