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
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
import torchvision.io as io
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

IGNORE_INDEX = -100  # label for unreadable images; ignored by CrossEntropyLoss

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR.parent
ROOT_MOBILENET_DIR = DATASET_DIR.parent
if str(ROOT_MOBILENET_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_MOBILENET_DIR))

import output_analysis as oa


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on External Merged Dataset (1,235 Make/Model Classes).")
    parser.add_argument("--splits-dir", type=str,
        default=str(DATASET_DIR / "splits_1235models"),
        help="Path containing train.csv, val.csv, test.csv, and label_map.json")
    parser.add_argument("--output-dir", type=str,
        default=str(DATASET_DIR / "output_mobilenet_v2_external"),
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
        self.has_bbox = "bbox_x1" in df.columns
        if self.has_bbox:
            self.bboxes = df[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values.astype(int)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = io.read_image(path, mode=io.ImageReadMode.RGB)
            _, h, w = img.shape

            # If sample has bounding box annotations (e.g. Stanford Cars)
            if self.has_bbox and self.bboxes[idx][0] >= 0:
                x1, y1, x2, y2 = self.bboxes[idx]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                cropped = img[:, y1:y2, x1:x2]
            else:
                # Bottom border strip crop to strip web dealership stamps if configured
                bottom_h = max(20, int(h * (1.0 - self.crop_bottom_pct)))
                cropped = img[:, :bottom_h, :]

            resized = nn.functional.interpolate(
                cropped.unsqueeze(0).float(),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
                antialias=True
            ).squeeze(0).round().clamp(0, 255).to(torch.uint8)
            return resized, self.labels[idx]
        except Exception as e:
            # Unreadable image: report it and mark it with IGNORE_INDEX so it is excluded from loss/accuracy
            print(f"[WARN] Failed to load image {path}: {e}", file=sys.stderr, flush=True)
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.uint8), IGNORE_INDEX


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


def set_train_mode(model: nn.Module, unfreeze_layers: int):
    """model.train(), but keep BatchNorm running stats of the frozen backbone blocks fixed."""
    model.train()
    n_frozen_blocks = max(0, len(model.features) - unfreeze_layers)
    for block in model.features[:n_frozen_blocks]:
        for m in block.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def apply_per_sample(tf, imgs: torch.Tensor) -> torch.Tensor:
    """Apply a random transform independently to every image (v2 transforms draw one set of params per call)."""
    return torch.stack([tf(img) for img in imgs])


def evaluate(model: nn.Module, loader: DataLoader, tf, criterion: nn.Module, device: torch.device):
    """Returns (loss, accuracy) over valid (readable) samples."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            valid = labels != IGNORE_INDEX
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            imgs = tf(imgs)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            loss_sum += loss.item() * n_valid
            correct += ((outputs.argmax(1) == labels) & valid).sum().item()
            total += n_valid
    return loss_sum / total, correct / total


class OnnxExportWrapper(nn.Module):
    """Exposes logits plus the pooled embeddings and per-class activation maps, so t-SNE and CAM work from ONNX."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.features(x)                                              # (B, 1280, H, W)
        pooled = torch.flatten(nn.functional.adaptive_avg_pool2d(features, (1, 1)), 1)  # (B, 1280)
        logits = self.model.classifier(pooled)                                         # (B, num_classes)
        head_w = self.model.classifier[1].weight                                       # (num_classes, 1280)
        class_maps = nn.functional.conv2d(features, head_w[:, :, None, None])          # (B, num_classes, H, W)
        return logits, pooled, class_maps


def export_to_onnx(model: nn.Module, num_classes: int, img_size: int, out_onnx_path: Path):
    cpu_model = OnnxExportWrapper(copy.deepcopy(model).cpu()).eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")

    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        cpu_model,
        dummy_input,
        str(out_onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["predictions", "embeddings", "class_maps"],
        dynamic_axes={"input_image": {0: "batch_size"}, "predictions": {0: "batch_size"},
                      "embeddings": {0: "batch_size"}, "class_maps": {0: "batch_size"}},
        dynamo=False
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
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": {v: k for k, v in class_to_idx.items()},
            "num_classes": num_classes
        }, f, indent=2)

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

    model = create_model(num_classes=num_classes, dropout=args.dropout, unfreeze_layers=args.unfreeze_layers).to(device)

    # Differential learning rate
    trainable_backbone = [p for p in model.features.parameters() if p.requires_grad]
    param_groups = [
        {"params": trainable_backbone, "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = []
    best_state = None

    best_onnx_path = models_dir / "mobilenet_v2_best.onnx"

    for epoch in range(args.epochs):
        set_train_mode(model, args.unfreeze_layers)
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            valid = labels != IGNORE_INDEX
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            imgs = apply_per_sample(train_tf, imgs)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * n_valid
            correct += ((outputs.argmax(1) == labels) & valid).sum().item()
            total += n_valid

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, val_tf, criterion, device)

        print(f"\n[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            print(f"[*] New best validation loss: {best_val_loss:.4f} (Val Acc: {best_val_acc:.2%}). Exporting ONNX weights to {best_onnx_path}")
            export_to_onnx(model, num_classes, args.img_size, best_onnx_path)
            best_state = copy.deepcopy(model.state_dict())

    # Final test-set evaluation with the best (lowest val loss) weights, the same ones exported to ONNX
    if best_state is not None:
        model.load_state_dict(best_state)
        test_loss, test_acc = evaluate(model, test_loader, val_tf, criterion, device)
        print(f"[Test] Loss: {test_loss:.4f} | Acc: {test_acc:.2%}")
        with open(reports_dir / "test_metrics.json", "w") as f:
            json.dump({"test_loss": test_loss, "test_acc": test_acc, "best_val_loss": best_val_loss, "best_val_acc": best_val_acc}, f, indent=2)

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
