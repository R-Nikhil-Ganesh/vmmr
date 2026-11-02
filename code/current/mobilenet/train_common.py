#!/usr/bin/env python3
"""
Code shared by the PlatesMania and External training scripts, so both models are trained with identical
augmentation and objectives (only the dataset differs).

Contents
  resize_uint8, augment_batch      GPU augmentation ("base" = original behaviour, "strong" = domain-robust)
  add_robustness_args              CLI flags: --aug-strength --crop-jitter --label-smoothing --make-loss-weight --ema-decay
  TrainObjective                   cross-entropy with optional label smoothing + auxiliary make-level loss
  ModelEMA                         exponential moving average of the weights (cheap stand-in for SWAD-style averaging)
  jitter_*                         helpers used by the datasets for crop-geometry jitter
"""

import os
import sys
import copy
import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.io as io
import torchvision.transforms.v2 as v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = -100


# ==============================================================================
# Resize / augmentation
# ==============================================================================
def resize_uint8(img: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased bilinear resize of a (3,H,W) uint8 tensor to (3,size,size).
    Runs directly on uint8 (much cheaper than a float round-trip); falls back to float if unsupported by this torch build."""
    try:
        return F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear",
                             align_corners=False, antialias=True).squeeze(0)
    except (RuntimeError, NotImplementedError):
        out = F.interpolate(img.unsqueeze(0).float(), size=(size, size), mode="bilinear",
                            align_corners=False, antialias=True)
        return out.squeeze(0).round().clamp(0, 255).to(torch.uint8)


def _gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _uniform(B: int, lo: float, hi: float, dev) -> torch.Tensor:
    return lo + torch.rand(B, device=dev) * (hi - lo)


def _gaussian_blur_per_sample(x: torch.Tensor, sigma: torch.Tensor, k: int = 9) -> torch.Tensor:
    """Separable Gaussian blur with a different sigma per image (sigma<=0.05 -> identity). x: (B,C,H,W)."""
    B, C, H, W = x.shape
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    s = sigma.clamp_min(1e-3).view(B, 1)
    ker = torch.exp(-(ax.view(1, k) ** 2) / (2 * s ** 2))
    ker = ker / ker.sum(dim=1, keepdim=True)                       # (B, k)
    ker = ker.repeat_interleave(C, dim=0)                           # (B*C, k)
    xx = x.reshape(1, B * C, H, W)
    xx = F.conv2d(F.pad(xx, (k // 2, k // 2, 0, 0), mode="reflect"), ker.view(B * C, 1, 1, k), groups=B * C)
    xx = F.conv2d(F.pad(xx, (0, 0, k // 2, k // 2), mode="reflect"), ker.view(B * C, 1, k, 1), groups=B * C)
    return xx.view(B, C, H, W)


def augment_batch(imgs: torch.Tensor, strength: str = "base") -> torch.Tensor:
    """Vectorized GPU augmentation: uint8 (B,3,H,W) -> normalized float32 (B,3,H,W).

    Every image gets its own random draw, with only a handful of batched kernels.

    strength="base"   hflip p=0.5, rotate +-10 deg, translate +-6%, scale 0.94-1.06 (zero fill),
                      brightness/contrast/saturation 1 +- 0.15 in random order.
    strength="strong" meant for cross-domain robustness: wider zoom (0.7-1.4, log-uniform) and aspect jitter (0.85-1.18)
                      to cover tight (bbox) vs loose (full photo) framing, rotate +-12 deg, translate +-12%, reflection
                      padding, brightness/contrast/saturation 1 +- 0.40, per-channel colour gain (white balance) +-8%,
                      random grayscale (p=0.05), Gaussian blur (p=0.3, sigma 0.4-1.6), Gaussian noise (p=0.4, std<=0.04),
                      one random erasing rectangle (p=0.25, 4-16% of the area).
    """
    if strength not in ("base", "strong"):
        raise ValueError(f"unknown aug strength: {strength}")
    strong = strength == "strong"
    B, dev = imgs.shape[0], imgs.device
    x = imgs.float().div_(255.0)

    # Horizontal flip
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)

    # Affine (rotation + scale (+ aspect) + translation) in one grid_sample. theta maps output -> input coordinates.
    if strong:
        ang = _uniform(B, -12.0, 12.0, dev) * math.pi / 180.0
        zoom = torch.exp(_uniform(B, math.log(0.7), math.log(1.4), dev))     # >1 zooms in
        asp = torch.exp(_uniform(B, math.log(0.85), math.log(1.18), dev))    # horizontal / vertical zoom ratio
        zx, zy = zoom * asp.sqrt(), zoom / asp.sqrt()
        t = (torch.rand(B, 2, device=dev) * 2 - 1) * (0.12 * 2)
        pad_mode = "reflection"
    else:
        ang = _uniform(B, -10.0, 10.0, dev) * math.pi / 180.0
        zx = zy = 0.94 + torch.rand(B, device=dev) * 0.12
        t = (torch.rand(B, 2, device=dev) * 2 - 1) * (0.06 * 2)               # +-6% of the size, in [-1, 1] coordinates
        pad_mode = "zeros"
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = c / zx, s / zx
    theta[:, 1, 0], theta[:, 1, 1] = -s / zy, c / zy
    theta[:, 0, 2] = -(theta[:, 0, 0] * t[:, 0] + theta[:, 0, 1] * t[:, 1])
    theta[:, 1, 2] = -(theta[:, 1, 0] * t[:, 0] + theta[:, 1, 1] * t[:, 1])
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode=pad_mode, align_corners=False)

    # Colour jitter (brightness, contrast, saturation), per-image factors, random op order per batch
    j = 0.40 if strong else 0.15
    for op in torch.randperm(3).tolist():
        f = (1.0 - j + torch.rand(B, 1, 1, 1, device=dev) * 2 * j)
        if op == 0:
            x = (x * f).clamp_(0.0, 1.0)
        elif op == 1:
            m = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - m) * f + m).clamp_(0.0, 1.0)
        else:
            g = _gray(x)
            x = ((x - g) * f + g).clamp_(0.0, 1.0)

    if strong:
        # per-channel gain (white balance / lighting colour)
        x = (x * (1.0 + (torch.rand(B, 3, 1, 1, device=dev) * 2 - 1) * 0.08)).clamp_(0.0, 1.0)
        # random grayscale
        gs = torch.rand(B, device=dev) < 0.05
        x = torch.where(gs[:, None, None, None], _gray(x).expand(-1, 3, -1, -1), x)
        # blur (sigma ~0 -> identity for the images that are not selected)
        sigma = torch.where(torch.rand(B, device=dev) < 0.3, _uniform(B, 0.4, 1.6, dev), torch.zeros(B, device=dev))
        if bool((sigma > 0).any()):
            x = _gaussian_blur_per_sample(x, sigma)
        # sensor-like noise
        std = torch.where(torch.rand(B, device=dev) < 0.4, _uniform(B, 0.005, 0.04, dev), torch.zeros(B, device=dev))
        x = (x + torch.randn_like(x) * std.view(B, 1, 1, 1)).clamp_(0.0, 1.0)
        # random erasing: one rectangle covering 4-16% of the image, filled with the per-image mean colour
        H, W = x.shape[-2:]
        area = _uniform(B, 0.04, 0.16, dev)
        ratio = torch.exp(_uniform(B, math.log(0.5), math.log(2.0), dev))
        eh = (area * ratio).sqrt().clamp(max=1.0)                 # fractions of H and W
        ew = (area / ratio).sqrt().clamp(max=1.0)
        cy = torch.rand(B, device=dev) * (1 - eh) + eh / 2
        cx = torch.rand(B, device=dev) * (1 - ew) + ew / 2
        yy = (torch.arange(H, device=dev).view(1, H, 1) + 0.5) / H
        xx = (torch.arange(W, device=dev).view(1, 1, W) + 0.5) / W
        inside = ((yy - cy.view(B, 1, 1)).abs() < eh.view(B, 1, 1) / 2) & ((xx - cx.view(B, 1, 1)).abs() < ew.view(B, 1, 1) / 2)
        inside = inside & (torch.rand(B, device=dev) < 0.25).view(B, 1, 1)
        x = torch.where(inside[:, None], x.mean(dim=(2, 3), keepdim=True).expand_as(x), x)

    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std_ = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std_


# ==============================================================================
# Crop-geometry jitter (used by the datasets, training split only)
# ==============================================================================
def jitter_fraction(base: float, jitter: float) -> float:
    """Crop fraction sampled uniformly in base*[1-jitter, 1+jitter] (jitter in [0,1]); returns base when jitter==0."""
    if jitter <= 0 or base <= 0:
        return base
    return base * random.uniform(1.0 - jitter, 1.0 + jitter)


def jitter_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, jitter: float):
    """Expand a bounding box by a random margin of up to 30%*jitter of its size per side (looser, full-photo-like framing)."""
    if jitter <= 0:
        return x1, y1, x2, y2
    bw, bh = x2 - x1, y2 - y1
    m = 0.30 * jitter
    x1 = max(0, int(x1 - bw * random.uniform(0, m)))
    x2 = min(w, int(x2 + bw * random.uniform(0, m)))
    y1 = max(0, int(y1 - bh * random.uniform(0, m)))
    y2 = min(h, int(y2 + bh * random.uniform(0, m)))
    return x1, y1, x2, y2


# ==============================================================================
# CLI flags
# ==============================================================================
def add_robustness_args(parser) -> None:
    g = parser.add_argument_group("generalization options (all default to the original behaviour)")
    g.add_argument("--aug-strength", choices=["base", "strong"], default="base",
                   help="GPU augmentation preset. 'strong' = wide zoom/aspect, blur, noise, erasing, stronger colour jitter")
    g.add_argument("--crop-jitter", type=float, default=0.0,
                   help="0-1. Randomizes the dataset-specific crop (top/bottom crop fraction, bbox margin) on the TRAIN split, "
                        "so the model sees both tight and loose framing. Val/test always use the fixed crop")
    g.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for the training loss (e.g. 0.1)")
    g.add_argument("--make-loss-weight", type=float, default=0.0,
                   help="Weight of the auxiliary make-level loss (needs 'Make/Model' class names), e.g. 0.3")
    g.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA of the weights (e.g. 0.999). Validation, best-checkpoint selection and ONNX export use the EMA weights")


# ==============================================================================
# Objective
# ==============================================================================
class TrainObjective:
    """Cross-entropy (optional label smoothing) + optional auxiliary make-level loss.

    The make-level loss is the NLL of P(make) = sum of P(model) over the make's models, so a prediction of the right make
    but the wrong model is penalized less than a wrong make. It needs class names of the form 'Make/Model'.
    """

    def __init__(self, class_to_idx: Dict[str, int], label_smoothing: float = 0.0, make_loss_weight: float = 0.0,
                 device: Optional[torch.device] = None, kd_ls_matrix: Optional[np.ndarray] = None, kd_ls_alpha: float = 0.1):
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)
        self.kd_S, self.kd_alpha = None, kd_ls_alpha
        if kd_ls_matrix is not None:
            S = torch.as_tensor(kd_ls_matrix, dtype=torch.float32, device=device)
            if S.shape != (len(class_to_idx), len(class_to_idx)):
                raise ValueError(f"--kd-ls-matrix is {tuple(S.shape)} but the taxonomy has {len(class_to_idx)} classes")
            self.kd_S = S
            print(f"[Loss] KD label smoothing enabled (alpha {kd_ls_alpha}); replaces uniform label smoothing")
        self.make_loss_weight = make_loss_weight
        self.class_to_make = None
        if make_loss_weight > 0:
            names = [None] * len(class_to_idx)
            for k, i in class_to_idx.items():
                names[i] = k
            if not all(n and "/" in n for n in names):
                print("[WARN] --make-loss-weight ignored: class names are not all 'Make/Model'")
                self.make_loss_weight = 0.0
            else:
                makes = sorted({n.split("/", 1)[0] for n in names})
                m_idx = {m: i for i, m in enumerate(makes)}
                self.class_to_make = torch.tensor([m_idx[n.split("/", 1)[0]] for n in names], device=device)
                self.make_proj = torch.zeros(len(names), len(makes), device=device)
                self.make_proj[torch.arange(len(names)), self.class_to_make] = 1.0
                print(f"[Loss] auxiliary make loss enabled: {len(makes)} makes, weight {make_loss_weight}")

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.kd_S is not None:
            valid = labels != IGNORE_INDEX
            y = labels.clamp_min(0)
            target = (1 - self.kd_alpha) * F.one_hot(y, logits.shape[1]).float() + self.kd_alpha * self.kd_S[y]
            loss = -(target * F.log_softmax(logits.float(), dim=-1)).sum(1)
            loss = (loss * valid).sum() / valid.sum().clamp_min(1)
        else:
            loss = self.ce(logits, labels)
        if self.make_loss_weight > 0:
            p_make = torch.softmax(logits.float(), dim=-1) @ self.make_proj
            make_labels = torch.where(labels == IGNORE_INDEX, torch.full_like(labels, IGNORE_INDEX),
                                      self.class_to_make[labels.clamp_min(0)])
            loss = loss + self.make_loss_weight * F.nll_loss(torch.log(p_make.clamp_min(1e-8)), make_labels,
                                                             ignore_index=IGNORE_INDEX)
        return loss


# ==============================================================================
# EMA
# ==============================================================================
class ModelEMA:
    """Exponential moving average of the model weights. Buffers (BatchNorm statistics) are copied from the live model."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))   # warm-up so early steps are not dominated by init
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


# ==============================================================================
# Shared Datasets, Model Builder, Evaluation & ONNX Export
# ==============================================================================
class PlatesManiaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_img_dir: Path, img_size: int = 512,
                 crop_top_pct: float = 0.15, crop_bottom_pct: float = 0.0, crop_jitter: float = 0.0):
        col = "image_path" if "image_path" in df.columns else "image_rel_path"
        base_str = str(base_img_dir)
        strip_images = bool(base_str) and not os.path.isdir(os.path.join(base_str, "images"))
        paths = []
        for p in df[col].values:
            full_p = os.path.join(base_str, p[7:] if (strip_images and p.startswith("images/")) else p) if not os.path.isabs(p) else p
            if not os.path.exists(full_p):
                # Handle trailing dot folder discrepancy on disk (e.g. 'Alfa Romeo/S.Z' -> 'Alfa Romeo/S.Z.')
                parent, fname = os.path.split(full_p)
                dot_p = os.path.join(parent + ".", fname)
                if os.path.exists(dot_p):
                    full_p = dot_p
                elif parent.endswith(".") and os.path.exists(os.path.join(parent[:-1], fname)):
                    full_p = os.path.join(parent[:-1], fname)
            paths.append(full_p)
        self.paths = paths
        self.labels = df["label"].values.astype(np.int64)
        self.img_size = img_size
        self.crop_top_pct = crop_top_pct
        self.crop_bottom_pct = crop_bottom_pct
        self.crop_jitter = crop_jitter

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = io.read_image(path, mode=io.ImageReadMode.RGB)
            _, h, _ = img.shape
            top = int(h * jitter_fraction(self.crop_top_pct, self.crop_jitter))
            bottom = max(top + 10, int(h * (1.0 - jitter_fraction(self.crop_bottom_pct, self.crop_jitter))))
            cropped = img[:, top:bottom, :]
            return resize_uint8(cropped, self.img_size), self.labels[idx]
        except Exception as e:
            print(f"[WARN] Failed to load image {path}: {e}", file=sys.stderr, flush=True)
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.uint8), IGNORE_INDEX


class ExternalMergedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 512, crop_bottom_pct: float = 0.0, crop_jitter: float = 0.0,
                 base_img_dir: Optional[Path] = None):
        if base_img_dir is None:
            try:
                import paths as _paths
                base_str = str(getattr(_paths, "EXT_DATASETS_DIR", "") or "")
            except Exception:
                base_str = ""
        else:
            base_str = str(base_img_dir)
        resolved_paths = []
        for raw_p in df["image_path"].astype(str).values:
            if os.path.isabs(raw_p) and os.path.exists(raw_p):
                resolved_paths.append(raw_p)
            elif base_str and "/external_resized_512/" in raw_p:
                resolved_paths.append(os.path.join(base_str, raw_p.split("/external_resized_512/", 1)[1].lstrip("/")))
            elif base_str and not os.path.isabs(raw_p):
                resolved_paths.append(os.path.join(base_str, raw_p.lstrip("/")))
            else:
                resolved_paths.append(raw_p)
        self.paths = np.array(resolved_paths)
        self.labels = df["label"].values.astype(np.int64)
        self.img_size = img_size
        self.crop_bottom_pct = crop_bottom_pct
        self.crop_jitter = crop_jitter
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
            if self.has_bbox and self.bboxes[idx][0] >= 0:
                x1, y1, x2, y2 = self.bboxes[idx]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))
                x1, y1, x2, y2 = jitter_bbox(x1, y1, x2, y2, w, h, self.crop_jitter)
                cropped = img[:, y1:y2, x1:x2]
            else:
                bottom_h = max(20, int(h * (1.0 - jitter_fraction(self.crop_bottom_pct, self.crop_jitter))))
                cropped = img[:, :bottom_h, :]
            return resize_uint8(cropped, self.img_size), self.labels[idx]
        except Exception as e:
            print(f"[WARN] Failed to load image {path}: {e}", file=sys.stderr, flush=True)
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.uint8), IGNORE_INDEX


# ==============================================================================
# Model with overridable pooling and an optional make branch (Run 5)
# ==============================================================================
class VMMRNet(nn.Module):
    """torchvision MobileNetV2 body with overridable pooling and an optional make-gated classifier.

    With pool='gap' and make_gating='none' the state_dict keys (features.*, classifier.1.*) and the exported ONNX graph are
    identical to the plain torchvision model. Extra parameters exist only when a flag asks for them:
      pool='gem'        -> gem_p (learnable exponent, init 3)
      make_gating!=none -> make_fc (Linear 1280->num_makes), beta (gate strength, init 0), make_proj buffer
    Pooling keeps the embedding 1280-D: 'gap_gmp' = avg + max, 'gem' = generalized mean.
    """

    def __init__(self, backbone: nn.Module, pool: str = "gap", make_gating: str = "none",
                 make_proj: Optional[torch.Tensor] = None):
        super().__init__()
        self.features, self.classifier = backbone.features, backbone.classifier
        self.pool, self.make_gating = pool, make_gating
        n_cls, dim = self.classifier[1].weight.shape
        if pool == "gem":
            self.gem_p = nn.Parameter(torch.tensor(3.0))
        if make_gating != "none":
            if make_proj is None:
                raise ValueError("--make-gating needs class names of the form 'Make/Model'")
            self.make_fc = nn.Linear(dim, make_proj.shape[1])
            self.beta = nn.Parameter(torch.zeros(1 if make_gating == "scalar" else n_cls))
            self.register_buffer("make_proj", make_proj.float().clone())

    def _pool(self, f: torch.Tensor) -> torch.Tensor:
        if self.pool == "gem":
            p = self.gem_p.clamp(1.0, 8.0)
            g = F.adaptive_avg_pool2d(f.float().clamp_min(1e-6).pow(p), (1, 1)).pow(1.0 / p)
            return torch.flatten(g, 1).to(f.dtype)
        avg = torch.flatten(F.adaptive_avg_pool2d(f, (1, 1)), 1)
        if self.pool == "gap_gmp":
            return avg + torch.flatten(F.adaptive_max_pool2d(f, (1, 1)), 1)
        return avg

    def _heads(self, e: torch.Tensor):
        logits = self.classifier(e)
        make_logits = None
        if self.make_gating != "none":
            make_logits = self.make_fc(e)
            log_pm = F.log_softmax(make_logits.float(), dim=-1) @ self.make_proj.t()
            logits = logits + (self.beta * log_pm).to(logits.dtype)
        return logits, make_logits

    def forward_train(self, x: torch.Tensor):
        """(logits, make_logits-or-None); make_logits feeds the dedicated make recognizer loss."""
        return self._heads(self._pool(self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_train(x)[0]

    def forward_export(self, x: torch.Tensor):
        features = self.features(x)
        e = self._pool(features)
        logits, _ = self._heads(e)
        class_maps = F.conv2d(features, self.classifier[1].weight[:, :, None, None])
        return logits, e, class_maps


def create_vmmr_model(num_classes: int, dropout: float = 0.25, unfreeze_layers: int = 5, pool: str = "gap",
                      make_gating: str = "none", class_to_idx: Optional[Dict[str, int]] = None) -> VMMRNet:
    backbone = create_model(num_classes, dropout, unfreeze_layers)
    make_proj = build_make_proj(class_to_idx) if (make_gating != "none" and class_to_idx) else None
    model = VMMRNet(backbone, pool, make_gating, make_proj)
    for name in ("gem_p", "make_fc", "beta"):                      # new heads always train
        mod = getattr(model, name, None)
        if mod is not None:
            for prm in (mod.parameters() if isinstance(mod, nn.Module) else [mod]):
                prm.requires_grad = True
    return model


def model_cfg_of(model: VMMRNet, img_size: int, dropout: float, unfreeze_layers: int) -> dict:
    return {"num_classes": model.classifier[1].out_features, "dropout": dropout, "unfreeze_layers": unfreeze_layers,
            "img_size": img_size, "pool": model.pool, "make_gating": model.make_gating,
            "num_makes": (model.make_proj.shape[1] if model.make_gating != "none" else 0)}


def load_vmmr_from_pt(path, device="cpu") -> VMMRNet:
    """Rebuilds a VMMRNet from a `--save-pt` checkpoint (state_dict + model_cfg). Does not need class names."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ck["model_cfg"]
    backbone = create_model(cfg["num_classes"], cfg.get("dropout", 0.25), cfg.get("unfreeze_layers", 5))
    proj = torch.zeros(cfg["num_classes"], cfg["num_makes"]) if cfg.get("make_gating", "none") != "none" else None
    model = VMMRNet(backbone, cfg.get("pool", "gap"), cfg.get("make_gating", "none"), proj)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def build_param_groups(model: VMMRNet, backbone_lr: float, head_lr: float, layer_lr_decay: float = 1.0):
    """AdamW param groups. layer_lr_decay=1 -> one backbone group (the original behaviour); otherwise block i of
    features gets backbone_lr * decay**(n_blocks-1-i), so shallow blocks move less."""
    head = [p for n, p in model.named_parameters() if not n.startswith("features.") and p.requires_grad]
    if layer_lr_decay == 1.0:
        groups = [{"params": [p for p in model.features.parameters() if p.requires_grad], "lr": backbone_lr}]
    else:
        n = len(model.features)
        groups = [{"params": [p for p in blk.parameters() if p.requires_grad],
                   "lr": backbone_lr * layer_lr_decay ** (n - 1 - i)} for i, blk in enumerate(model.features)]
        groups = [g for g in groups if g["params"]]
    return groups + [{"params": head, "lr": head_lr}]


def build_val_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_model(num_classes: int, dropout: float = 0.25, unfreeze_layers: int = 5):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    for param in model.features.parameters():
        param.requires_grad = False
    for layer in model.features[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] MobileNetV2: {n_frozen:,} frozen parameters, {n_trainable:,} trainable parameters (top {unfreeze_layers} backbone layers unlocked)")
    return model


def set_train_mode(model: nn.Module, unfreeze_layers: int, freeze_bn_blocks: int = 0):
    """model.train(), but keep BatchNorm running stats of the frozen backbone blocks fixed.
    `freeze_bn_blocks` additionally freezes the BN statistics of features[:N] even when their weights train."""
    model.train()
    n_frozen_blocks = max(max(0, len(model.features) - unfreeze_layers), freeze_bn_blocks)
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


def build_make_proj(class_to_idx: Dict[str, int], device: Optional[torch.device] = None):
    """(num_classes, num_makes) one-hot class->make matrix, or None if class names are not all 'Make/Model'."""
    names = [None] * len(class_to_idx)
    for k, i in class_to_idx.items():
        names[i] = k
    if not all(n and "/" in n for n in names):
        return None
    makes = sorted({n.split("/", 1)[0] for n in names})
    m_idx = {m: i for i, m in enumerate(makes)}
    proj = torch.zeros(len(names), len(makes), device=device)
    proj[torch.arange(len(names)), torch.tensor([m_idx[n.split("/", 1)[0]] for n in names])] = 1.0
    return proj


def evaluate_with_make(model: nn.Module, loader: DataLoader, tf, criterion: nn.Module, device: torch.device,
                       make_proj: Optional[torch.Tensor]):
    """Like `evaluate`, plus make accuracy by marginalization (P(make) = softmax(logits) @ make_proj).
    Returns (loss, model_acc, make_acc); make_acc is None without a make projection."""
    model.eval()
    loss_sum, correct, make_correct, total = 0.0, 0, 0, 0
    proj = make_proj.to(device) if make_proj is not None else None
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            valid = labels != IGNORE_INDEX
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(tf(imgs))
                loss = criterion(outputs, labels)
            loss_sum += loss.item() * n_valid
            correct += ((outputs.argmax(1) == labels) & valid).sum().item()
            if proj is not None:
                p_make = torch.softmax(outputs.float(), dim=-1) @ proj
                true_make = proj[labels.clamp_min(0)].argmax(1)
                make_correct += ((p_make.argmax(1) == true_make) & valid).sum().item()
            total += n_valid
    return loss_sum / total, correct / total, (make_correct / total if proj is not None else None)


def save_pt_checkpoint(model: nn.Module, path: Path, model_cfg: dict) -> None:
    """state_dict + a small config dict next to the best ONNX (only with --save-pt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg}, str(path))
    print(f"[PT] Saved checkpoint to {path}")


class OnnxExportWrapper(nn.Module):
    """Exposes logits plus the pooled embeddings and per-class activation maps, so t-SNE and CAM work from ONNX."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        if hasattr(self.model, "forward_export"):
            return self.model.forward_export(x)
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


def save_run_config(output_dir: Path, args, script_name: str, extra: Optional[dict] = None) -> Path:
    """Save CLI arguments, git commit, and environment metadata to <output_dir>/run_config.json (E-17)."""
    import datetime
    import getpass
    import json
    import platform
    import socket
    import subprocess

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "run_config.json"

    repo_dir = Path(__file__).resolve().parent
    git_commit, git_dirty = "unknown", False
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True
        ).strip()
        git_dirty = bool(status_out)
    except Exception:
        pass

    ort_ver = None
    try:
        import onnxruntime as _ort
        ort_ver = _ort.__version__
    except Exception:
        pass

    cuda_dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    args_dict = vars(args).copy() if hasattr(args, "__dict__") else (dict(args) if isinstance(args, dict) else {})
    for k, v in list(args_dict.items()):
        if isinstance(v, Path):
            args_dict[k] = str(v)

    payload = {
        "script": script_name,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": args_dict,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "environment": {
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "onnxruntime_version": ort_ver,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": cuda_dev,
        },
    }
    if extra:
        payload.update(extra)

    with open(cfg_path, "w") as f:
        json.dump(payload, f, indent=2)
    return cfg_path

