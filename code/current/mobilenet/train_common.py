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

import copy
import math
import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 device: Optional[torch.device] = None):
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)
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
