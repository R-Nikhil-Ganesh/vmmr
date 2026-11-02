#!/usr/bin/env python3
"""
Rebuild a torchvision MobileNetV2 (as `train_common.VMMRNet`, pool=gap, no gating) from a BN-folded Model C ONNX.

The exported ONNX has BatchNorm folded into the Conv weights (53 Conv nodes: the 52 backbone convs plus the 1x1 class_maps
conv, no BN nodes). The i-th backbone Conv node maps to the i-th torchvision Conv2d; each BatchNorm2d becomes the identity.
`classifier.1` comes from the Gemm node. A parity check against ONNX Runtime must pass (max |diff| < 1e-3) or the script
exits with an error.

Library use:  from onnx_to_torch import load_torch_from_onnx;  model = load_torch_from_onnx(path)
CLI:          python onnx_to_torch.py --onnx <model.onnx> [--check] [--images-dir DIR --n 64] [--save-pt out.pt]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from train_common import VMMRNet, IMAGENET_MEAN, IMAGENET_STD


def load_torch_from_onnx(onnx_path) -> VMMRNet:
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(onnx_path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    gemm = [n for n in m.graph.node if n.op_type == "Gemm"]
    if len(gemm) != 1:
        raise ValueError(f"expected 1 Gemm node, found {len(gemm)}")
    if any(n.op_type == "BatchNormalization" for n in m.graph.node):
        raise ValueError("ONNX still has BatchNormalization nodes; only BN-folded exports are supported")

    w_gemm = init[gemm[0].input[1]]                       # (num_classes, 1280) (transB=1) from torch export
    num_classes = w_gemm.shape[0]
    backbone = models.mobilenet_v2(weights=None)
    backbone.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(1280, num_classes))
    tconvs = [mod for mod in backbone.features.modules() if isinstance(mod, nn.Conv2d)]
    if len(convs) != len(tconvs) + 1:
        raise ValueError(f"{len(convs)} ONNX convs vs {len(tconvs)} backbone convs (+1 class_maps conv expected)")

    for node, mod in zip(convs, tconvs):
        w = init[node.input[1]]
        if tuple(w.shape) != tuple(mod.weight.shape):
            raise ValueError(f"shape mismatch at {node.name}: {w.shape} vs {tuple(mod.weight.shape)}")
        mod.weight.data = torch.from_numpy(w.copy())
        if mod.bias is None:
            mod.bias = nn.Parameter(torch.zeros(mod.out_channels))
        mod.bias.data = torch.from_numpy(init[node.input[2]].copy()) if len(node.input) > 2 else torch.zeros(mod.out_channels)
    for mod in backbone.features.modules():
        if isinstance(mod, nn.BatchNorm2d):               # identity: (x - 0) / sqrt(1 - eps + eps) * 1 + 0
            mod.weight.data.fill_(1.0); mod.bias.data.zero_()
            mod.running_mean.zero_(); mod.running_var.fill_(1.0 - mod.eps)
    backbone.classifier[1].weight.data = torch.from_numpy(w_gemm.copy())
    backbone.classifier[1].bias.data = torch.from_numpy(init[gemm[0].input[2]].copy())
    return VMMRNet(backbone, "gap", "none").eval()


def parity_check(model: VMMRNet, onnx_path, x: np.ndarray, tol: float = 1e-3):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    ref = dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x})))
    with torch.no_grad():
        logits, emb, _ = model.forward_export(torch.from_numpy(x))
    d_logits = float(np.abs(logits.numpy() - ref["predictions"]).max())
    d_emb = float(np.abs(emb.numpy() - ref["embeddings"]).max())
    print(f"[Parity] max|dlogits| {d_logits:.2e} | max|dembeddings| {d_emb:.2e} (tol {tol:g})")
    return d_logits < tol and d_emb < tol


def real_batch(images_dir: Path, n: int, img_size: int) -> np.ndarray:
    import torchvision.io as io
    from train_common import resize_uint8
    files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    if not files:
        raise SystemExit(f"[ERROR] no images under {images_dir}")
    xs = [resize_uint8(io.read_image(str(f), mode=io.ImageReadMode.RGB), img_size) for f in files]
    x = torch.stack(xs).float() / 255.0
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--check", action="store_true", help="Run the parity check (exit 1 on failure)")
    ap.add_argument("--images-dir", default=None, help="Real images for the parity check (default: random tensors)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--save-pt", default=None, help="Save a --save-pt compatible checkpoint (state_dict + model_cfg)")
    a = ap.parse_args()
    model = load_torch_from_onnx(a.onnx)
    print(f"[Rebuild] {a.onnx}: {model.classifier[1].out_features} classes")
    if a.check:
        if a.images_dir:
            x = real_batch(Path(a.images_dir), a.n, a.img_size)
        else:
            x = np.random.default_rng(0).standard_normal((min(a.n, 8), 3, a.img_size, a.img_size)).astype(np.float32)
            print("[Parity] no --images-dir: using random tensors (run again with real images for the official check)")
        if not parity_check(model, a.onnx, x):
            sys.exit("[ERROR] parity check failed")
    if a.save_pt:
        from train_common import save_pt_checkpoint, model_cfg_of
        save_pt_checkpoint(model, Path(a.save_pt), model_cfg_of(model, a.img_size, 0.0, 19))


if __name__ == "__main__":
    main()
