#!/usr/bin/env python3
"""
MobileNetV2 trained on PlatesMania + External merged (Model C)
==============================================================
A separate experiment from Model A (PlatesMania only) and Model B (External only), which stay single-dataset.
Model C sees both training sets and is evaluated on both test sets, so it shows what joint training buys
compared with the cross-dataset numbers of A and B.

Reuses the datasets, model, evaluation and ONNX export of the two single-dataset trainers, so every image is
cropped exactly as in A (PlatesMania: top 15%) or B (External: vehicle bbox / bottom 5%).

Both datasets share one label index space (1,235 Make/Model classes).

Sampling: External has ~30k train images against ~1M for PlatesMania, so plain concatenation would make it
negligible. --ext-fraction sets the share of External images drawn per epoch (default 0.2, 0 = plain
concatenation). An epoch has as many samples as the PlatesMania train split, like Model A.

Checkpoint selection: lowest MEAN of the two domain validation losses (each domain weighted equally).
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR.parent
ROOT_MOBILENET_DIR = DATASET_DIR.parent
for sub in ["", "platesmania_dataset/train", "external_dataset/train"]:
    p = str(ROOT_MOBILENET_DIR / sub) if sub else str(ROOT_MOBILENET_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)

import paths
import train_mobilenet_v2 as pm_t            # PlatesMania trainer: dataset, model, eval, export
import train_mobilenet_v2_external as ext_t  # External trainer: dataset
from train_common import augment_batch, add_robustness_args, TrainObjective, ModelEMA

IGNORE_INDEX = pm_t.IGNORE_INDEX


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on PlatesMania + External merged (Model C).")
    parser.add_argument("--pm-csv-path", default=paths.PM_MANIFEST_CSV, help="PlatesMania manifest (split column)")
    parser.add_argument("--pm-img-dir", default=paths.PM_IMG_DIR)
    parser.add_argument("--label-map-path", default=paths.PM_LABEL_MAP, help="Shared 1,235-class label map")
    parser.add_argument("--ext-splits-dir", default=str(ROOT_MOBILENET_DIR / "external_dataset" / "splits_1235models"))
    parser.add_argument("--output-dir", default=str(DATASET_DIR / "output_mobilenet_v2_merged"))
    parser.add_argument("--ext-fraction", type=float, default=0.2,
                        help="Share of External images per epoch (0 = plain concatenation, i.e. ~3%% External)")
    parser.add_argument("--pm-crop-top-pct", type=float, default=0.15)
    parser.add_argument("--ext-crop-bottom-pct", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--unfreeze-layers", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    add_robustness_args(parser)
    return parser.parse_args()


def make_loader(ds, batch_size, num_workers, **kw):
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None, **kw)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    models_dir, plots_dir, reports_dir = output_dir / "models", output_dir / "plots", output_dir / "reports"
    for d in [models_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Shared label space: PlatesMania's map; the External splits must use the same one
    def load_map(path):
        raw = json.load(open(path))
        return {k: int(v) for k, v in raw.get("class_to_idx", raw).items()}
    class_to_idx = load_map(args.label_map_path)
    ext_map_file = Path(args.ext_splits_dir) / "label_map.json"
    if ext_map_file.exists() and load_map(ext_map_file) != class_to_idx:
        raise ValueError(f"{ext_map_file} differs from {args.label_map_path}: the two datasets must share one label space.")
    num_classes = len(class_to_idx)
    with open(models_dir / "label_map.json", "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": {v: k for k, v in class_to_idx.items()},
                   "num_classes": num_classes}, f, indent=2)

    # Data
    df = pd.read_csv(args.pm_csv_path)
    pm = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    sd = Path(args.ext_splits_dir)
    ex = {s: pd.read_csv(sd / f"{s}.csv") for s in ["train", "val", "test"]}
    print(f"[Data] PlatesMania train/val/test: {len(pm['train']):,}/{len(pm['val']):,}/{len(pm['test']):,}")
    print(f"[Data] External    train/val/test: {len(ex['train']):,}/{len(ex['val']):,}/{len(ex['test']):,}")

    def pm_ds(split, jitter=0.0):
        return pm_t.PlatesManiaDataset(pm[split], Path(args.pm_img_dir), args.img_size, args.pm_crop_top_pct, 0.0, crop_jitter=jitter)

    def ex_ds(split, jitter=0.0):
        return ext_t.ExternalMergedDataset(ex[split], args.img_size, args.ext_crop_bottom_pct, crop_jitter=jitter)

    n_pm, n_ex = len(pm["train"]), len(ex["train"])
    train_ds = ConcatDataset([pm_ds("train", args.crop_jitter), ex_ds("train", args.crop_jitter)])
    if args.ext_fraction > 0:
        weights = np.concatenate([np.full(n_pm, (1 - args.ext_fraction) / n_pm), np.full(n_ex, args.ext_fraction / n_ex)])
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=n_pm, replacement=True)
        train_loader = make_loader(train_ds, args.batch_size, args.num_workers, sampler=sampler, drop_last=True)
        print(f"[Sampling] {n_pm:,} samples/epoch, {args.ext_fraction:.0%} External "
              f"(~{args.ext_fraction * n_pm / n_ex:.1f} draws per External image per epoch)")
    else:
        train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
        print(f"[Sampling] plain concatenation: {n_ex / (n_pm + n_ex):.1%} External")

    val_loaders = {"pm": make_loader(pm_ds("val"), args.eval_batch_size, args.num_workers, shuffle=False),
                   "ext": make_loader(ex_ds("val"), args.eval_batch_size, args.num_workers, shuffle=False)}
    test_loaders = {"pm": make_loader(pm_ds("test"), args.eval_batch_size, args.num_workers, shuffle=False),
                    "ext": make_loader(ex_ds("test"), args.eval_batch_size, args.num_workers, shuffle=False)}

    val_tf = pm_t.build_val_transform()
    torch.backends.cudnn.benchmark = True
    model = pm_t.create_model(num_classes, args.dropout, args.unfreeze_layers).to(device)
    param_groups = [{"params": [p for p in model.features.parameters() if p.requires_grad], "lr": args.backbone_lr},
                    {"params": model.classifier.parameters(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4, fused=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # plain CE for validation/test
    train_objective = TrainObjective(class_to_idx, args.label_smoothing, args.make_loss_weight, device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[Config] {num_classes} classes | aug={args.aug_strength} | crop_jitter={args.crop_jitter} | "
          f"label_smoothing={args.label_smoothing} | make_loss_weight={args.make_loss_weight} | ema_decay={args.ema_decay}")

    best_val, best_state, history = float("inf"), None, []
    best_onnx_path = models_dir / "mobilenet_v2_best.onnx"
    eval_model = model

    for epoch in range(args.epochs):
        pm_t.set_train_mode(model, args.unfreeze_layers)
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            valid = labels != IGNORE_INDEX
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            imgs = augment_batch(imgs, args.aug_strength)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs)
                loss = train_objective(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            run_loss += loss.item() * n_valid
            correct += ((outputs.argmax(1) == labels) & valid).sum().item()
            total += n_valid
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})
        scheduler.step()

        eval_model = ema.module if ema is not None else model
        vl_pm, va_pm = pm_t.evaluate(eval_model, val_loaders["pm"], val_tf, criterion, device)
        vl_ex, va_ex = pm_t.evaluate(eval_model, val_loaders["ext"], val_tf, criterion, device)
        val_loss = (vl_pm + vl_ex) / 2
        print(f"\n[Epoch {epoch+1:02d}] Train Loss {run_loss/total:.4f} Acc {correct/total:.2%} || "
              f"Val PM {vl_pm:.4f}/{va_pm:.2%} | Val EXT {vl_ex:.4f}/{va_ex:.2%} | mean loss {val_loss:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": run_loss / total, "train_acc": correct / total,
                        "val_loss_pm": vl_pm, "val_acc_pm": va_pm, "val_loss_ext": vl_ex, "val_acc_ext": va_ex,
                        "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            print(f"[*] New best mean validation loss {best_val:.4f}. Exporting ONNX to {best_onnx_path}")
            pm_t.export_to_onnx(eval_model, num_classes, args.img_size, best_onnx_path)
            best_state = copy.deepcopy(eval_model.state_dict())

    if best_state is not None:
        eval_model.load_state_dict(best_state)
        metrics = {"best_val_loss_mean": best_val}
        for key, name in [("pm", "platesmania"), ("ext", "external")]:
            tl, ta = pm_t.evaluate(eval_model, test_loaders[key], val_tf, criterion, device)
            metrics[f"test_loss_{name}"], metrics[f"test_acc_{name}"] = tl, ta
            print(f"[Test {name}] Loss {tl:.4f} | Acc {ta:.2%}")
        with open(reports_dir / "test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    hist = pd.DataFrame(history)
    hist.to_csv(reports_dir / "training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(hist["epoch"], hist["train_acc"], "o-", label="Train")
    ax[0].plot(hist["epoch"], hist["val_acc_pm"], "s--", label="Val PlatesMania")
    ax[0].plot(hist["epoch"], hist["val_acc_ext"], "^--", label="Val External")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].grid(True, ls="--", alpha=0.5); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_loss"], "o-", label="Train")
    ax[1].plot(hist["epoch"], hist["val_loss_pm"], "s--", label="Val PlatesMania")
    ax[1].plot(hist["epoch"], hist["val_loss_ext"], "^--", label="Val External")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].grid(True, ls="--", alpha=0.5); ax[1].legend()
    plt.tight_layout()
    plt.savefig(str(plots_dir / "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Plot] Training curves saved to {plots_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
