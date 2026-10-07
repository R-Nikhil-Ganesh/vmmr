#!/usr/bin/env python3
"""
MobileNetV2 Training Pipeline on PlatesMania Dataset (PyTorch)
==============================================================
Dataset: /home/researchadmin/Econ/resized_640x640/splits_filtered/
Environment: pt-env (PyTorch 2.14 + CUDA, RTX 4090)
"""

import os
import sys
import json
import math
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
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on PlatesMania Dataset using PyTorch.")
    parser.add_argument("--splits-dir", type=str,
        default=None,
        help="Path containing train.csv, val.csv, test.csv, and optionally label_map.json")
    parser.add_argument("--csv-path", type=str,
        default="/home/researchadmin/Econ/models/dataset_manifests/dataset_1235models_splits.csv",
        help="Path to single unified dataset split CSV (if applicable)")
    parser.add_argument("--base-img-dir", type=str,
        default="/home/researchadmin/Econ/resized_640x640",
        help="Base image folder")
    parser.add_argument("--label-map-path", type=str,
        default="/home/researchadmin/Econ/models/dataset_manifests/label_map_1235models.json",
        help="Path to label_map.json")
    parser.add_argument("--output-dir", type=str,
        default=str(DATASET_DIR / "output_mobilenet_v2"),
        help="Destination directory for checkpoints, metrics, and plots")
    parser.add_argument("--img-size", type=int, default=512, help="Input resolution (default: 512)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-5, help="Backbone learning rate")
    parser.add_argument("--dropout", type=float, default=0.25, help="Classifier dropout rate")
    parser.add_argument("--unfreeze-layers", type=int, default=5, help="Number of top backbone layers to unlock (default: 5)")
    parser.add_argument("--crop-top-pct", type=float, default=0.15, help="Top watermark crop (default: 0.15)")
    parser.add_argument("--crop-bottom-pct", type=float, default=0.0, help="Bottom watermark crop (default: 0.0)")
    parser.add_argument("--num-workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ==============================================================================
# Dataset Loader with Top Watermark Cropping
# ==============================================================================
def resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased bilinear resize of a (3,H,W) uint8 tensor to (3,size,size).
    Runs directly on uint8 (much cheaper than a float round-trip); falls back to float if unsupported by this torch build."""
    try:
        return nn.functional.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear",
                                         align_corners=False, antialias=True).squeeze(0)
    except (RuntimeError, NotImplementedError):
        out = nn.functional.interpolate(img.unsqueeze(0).float(), size=(size, size), mode="bilinear",
                                        align_corners=False, antialias=True)
        return out.squeeze(0).round().clamp(0, 255).to(torch.uint8)


class PlatesManiaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_img_dir: Path, img_size: int = 512,
                 crop_top_pct: float = 0.15, crop_bottom_pct: float = 0.0):
        col = "image_path" if "image_path" in df.columns else "image_rel_path"
        self.paths = [os.path.join(str(base_img_dir), p) if not os.path.isabs(p) else p for p in df[col].values]
        self.labels = df["label"].values.astype(np.int64)
        self.img_size = img_size
        self.crop_top_pct = crop_top_pct
        self.crop_bottom_pct = crop_bottom_pct

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = io.read_image(path, mode=io.ImageReadMode.RGB)
            _, h, w = img.shape
            top = int(h * self.crop_top_pct)
            bottom = max(top + 10, int(h * (1.0 - self.crop_bottom_pct)))
            cropped = img[:, top:bottom, :]
            return resize_uint8(cropped, self.img_size), self.labels[idx]
        except Exception as e:
            # Unreadable image: report it and mark it with IGNORE_INDEX so it is excluded from loss/accuracy
            print(f"[WARN] Failed to load image {path}: {e}", file=sys.stderr, flush=True)
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.uint8), IGNORE_INDEX


def build_val_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=oa.IMAGENET_MEAN, std=oa.IMAGENET_STD)
    ])


def _gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def augment_batch(imgs: torch.Tensor) -> torch.Tensor:
    """Vectorized GPU augmentation: uint8 (B,3,H,W) -> normalized float32 (B,3,H,W).

    Every image gets its own random draw, but with only a handful of batched kernels (a per-image loop over
    torchvision v2 transforms made the Python main thread the bottleneck). Same ranges as before:
    hflip p=0.5, rotate +-10 deg, translate +-6% of the size, scale 0.94-1.06 (zero fill), and
    brightness/contrast/saturation factors in 1 +- 0.15 (random op order per batch).
    """
    B, dev = imgs.shape[0], imgs.device
    x = imgs.float().div_(255.0)

    # Horizontal flip
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)

    # Affine (rotation + isotropic scale + translation) in one grid_sample. theta maps output -> input coordinates.
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(10.0)
    scale = 0.94 + torch.rand(B, device=dev) * 0.12
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * (0.06 * 2)  # +-6% of the image size, in [-1, 1] coordinates
    c, s = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = c, s
    theta[:, 1, 0], theta[:, 1, 1] = -s, c
    theta[:, 0, 2] = -(c * t[:, 0] + s * t[:, 1])
    theta[:, 1, 2] = -(-s * t[:, 0] + c * t[:, 1])
    grid = nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Colour jitter (brightness, contrast, saturation), per-image factors
    for op in torch.randperm(3).tolist():
        f = 0.85 + torch.rand(B, 1, 1, 1, device=dev) * 0.30
        if op == 0:
            x = (x * f).clamp_(0.0, 1.0)
        elif op == 1:
            m = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - m) * f + m).clamp_(0.0, 1.0)
        else:
            g = _gray(x)
            x = ((x - g) * f + g).clamp_(0.0, 1.0)

    mean = torch.tensor(oa.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(oa.IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


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

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for d in [models_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Resolve label map with robust multi-location fallback
    label_map_file = None
    if args.label_map_path and os.path.exists(args.label_map_path):
        label_map_file = Path(args.label_map_path)
    elif args.splits_dir and (Path(args.splits_dir) / "label_map.json").exists():
        label_map_file = Path(args.splits_dir) / "label_map.json"
    elif (models_dir / "label_map.json").exists():
        label_map_file = models_dir / "label_map.json"
    else:
        raise FileNotFoundError("Could not find label_map.json. Provide --label-map-path or place it in the splits/output directory.")

    with open(label_map_file) as f:
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
    print(f"  MobileNetV2 (PyTorch) | {num_classes} classes | {args.img_size}x{args.img_size} | batch={args.batch_size}")
    print(f"  LR={args.lr} | Backbone LR={args.backbone_lr} | Epochs={args.epochs}")
    print(f"  Output: {output_dir}")
    print("=" * 65 + "\n")

    # Load splits (supports either --csv-path or --splits-dir)
    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)
    else:
        splits_dir = Path(args.splits_dir)
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df   = pd.read_csv(splits_dir / "val.csv")
        test_df  = pd.read_csv(splits_dir / "test.csv")
    print(f"[Data] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    train_ds = PlatesManiaDataset(train_df, Path(args.base_img_dir), args.img_size, args.crop_top_pct, args.crop_bottom_pct)
    val_ds   = PlatesManiaDataset(val_df, Path(args.base_img_dir), args.img_size, args.crop_top_pct, args.crop_bottom_pct)
    test_ds  = PlatesManiaDataset(test_df, Path(args.base_img_dir), args.img_size, args.crop_top_pct, args.crop_bottom_pct)

    # Optimized high-throughput DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None
    )
    test_loader  = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    val_tf = build_val_transform()

    torch.backends.cudnn.benchmark = True
    model = create_model(num_classes=num_classes, dropout=args.dropout, unfreeze_layers=args.unfreeze_layers).to(device)

    # Differential learning rate for unlocked layers vs new head
    trainable_backbone = [p for p in model.features.parameters() if p.requires_grad]
    param_groups = [
        {"params": trainable_backbone, "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.lr}
    ]
    use_fused = (device.type == "cuda")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4, fused=use_fused)
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
            imgs = augment_batch(imgs)

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
