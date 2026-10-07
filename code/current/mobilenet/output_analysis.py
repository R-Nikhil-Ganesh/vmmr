#!/usr/bin/env python3
"""
ONNX Diagnostic Analysis Engine for MobileNetV2
===================================================================
Features:
- ONNX Runtime evaluation (weights are stored as .onnx only)
- Native evaluation of both Make and Model classifiers
- Hierarchical Make extraction from Make/Model labels (P(Make) = sum_{m in Make} P(Model_m))
- Per-class classification reports, normalized confusion matrices, and top confusion pairs
- Analytical Class Activation Mapping (Grad-CAM / CAM) on MobileNetV2 feature maps
- Latent feature space extraction (1,280-D) and 2D t-SNE / PCA manifold visualizations
- Confidence calibration analysis and automated human-in-the-loop rejection curves (tau = 0.70)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

torch.set_float32_matmul_precision("high")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ==============================================================================
# 1. Global Context Management
# ==============================================================================
class AnalysisContext:
    def __init__(self):
        self.model_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.models_dir: Optional[Path] = None
        self.reports_dir: Optional[Path] = None
        self.plots_dir: Optional[Path] = None
        self.pred_dir: Optional[Path] = None
        self.test_csv_path: Optional[Path] = None
        self.label_map_path: Optional[Path] = None
        self.base_img_dir: Optional[Path] = None

        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.num_classes: int = 0
        self.img_size: int = 512
        self.crop_top_pct: float = 0.0
        self.crop_bottom_pct: float = 0.0
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def configure(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, Path(v) if isinstance(v, (str, Path)) and "dir" in k or "path" in k else v)
        
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.models_dir = self.output_dir / "models"
            self.reports_dir = self.output_dir / "reports"
            self.plots_dir = self.output_dir / "plots"
            self.pred_dir = self.output_dir / "predictions"
            for d in [self.models_dir, self.reports_dir, self.plots_dir, self.pred_dir]:
                d.mkdir(parents=True, exist_ok=True)

        if self.label_map_path and Path(self.label_map_path).exists():
            self._load_label_map(self.label_map_path)

    def _load_label_map(self, path: Union[str, Path]):
        with open(path) as f:
            data = json.load(f)
        if "class_to_idx" in data and "idx_to_class" in data:
            self.class_to_idx = {k: int(v) for k, v in data["class_to_idx"].items()}
            self.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
        elif "class_to_idx" in data:
            self.class_to_idx = {k: int(v) for k, v in data["class_to_idx"].items()}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            self.class_to_idx = {k: int(v) for k, v in data.items()}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        self.num_classes = len(self.class_names)


context = AnalysisContext()

def set_context(**kwargs):
    context.configure(**kwargs)
    return context


# ==============================================================================
# 2. ONNX Model Loader
# ==============================================================================
class MobileNetV2Evaluator(nn.Module):
    """ONNX Runtime wrapper around an exported MobileNetV2 classifier (weights are only stored as .onnx).

    The exported graph has up to three outputs (see export_to_onnx in the training scripts):
      predictions  (B, num_classes)        logits
      embeddings   (B, 1280)               pooled penultimate features (t-SNE / PCA)
      class_maps   (B, num_classes, H, W)  per-class activation maps (CAM / Grad-CAM)
    After forward(), `embeddings` and `class_maps` hold the latest batch (None for older single-output models).
    """
    def __init__(self, num_classes: int, checkpoint_path: str):
        super().__init__()
        self.num_classes = num_classes
        self.checkpoint_path = str(checkpoint_path)

        if not self.checkpoint_path.endswith(".onnx"):
            raise ValueError(f"Only .onnx models are supported, got: {self.checkpoint_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"ONNX model not found: {self.checkpoint_path}")
        if not HAS_ORT:
            raise ImportError("onnxruntime is required to evaluate .onnx models.")

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.ort_session = ort.InferenceSession(self.checkpoint_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.ort_session.get_outputs()]
        print(f"[Model] Initialized ONNX Runtime session ({self.ort_session.get_providers()[0]}) from {self.checkpoint_path}")
        if "embeddings" not in self.output_names:
            print("[WARN] ONNX model has no 'embeddings'/'class_maps' outputs (exported by an older script); "
                  "latent-space and Grad-CAM analysis are unavailable. Re-export with the current training script.")

        self.embeddings: Optional[np.ndarray] = None
        self.class_maps: Optional[np.ndarray] = None

    def forward(self, x):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        out = dict(zip(self.output_names, self.ort_session.run(None, {self.input_name: x_np})))
        self.embeddings = out.get("embeddings")
        self.class_maps = out.get("class_maps")
        return torch.from_numpy(out["predictions"]).to(x.device)


# ==============================================================================
# 3. Image Preprocessing & Batch Evaluation
# ==============================================================================
def load_and_preprocess_image(path: str, img_size: int = 512, crop_top_pct: float = 0.0, crop_bottom_pct: float = 0.0):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    top = int(h * crop_top_pct)
    bottom = max(top + 10, int(h * (1.0 - crop_bottom_pct)))
    if top > 0 or crop_bottom_pct > 0:
        img = img.crop((0, top, w, bottom))
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, img


def evaluate_test_dataset(model: MobileNetV2Evaluator, test_df: pd.DataFrame, batch_size: int = 64):
    device = context.device
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_embeddings = []
    n_failed = 0

    paths = test_df["image_path"].values if "image_path" in test_df.columns else test_df["image_rel_path"].values
    if context.base_img_dir:
        paths = [str(Path(context.base_img_dir) / p) if not os.path.isabs(p) else p for p in paths]
    labels = test_df["label"].values.astype(int)

    n_samples = len(paths)
    print(f"[Eval] Running evaluation on {n_samples:,} samples (batch_size={batch_size})...")

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            tensors = []
            valid_idx = []
            for j, p in enumerate(batch_paths):
                try:
                    t, _ = load_and_preprocess_image(p, context.img_size, context.crop_top_pct, context.crop_bottom_pct)
                    tensors.append(t)
                    valid_idx.append(j)
                except Exception as e:
                    n_failed += 1
                    print(f"[WARN] Skipping unreadable image {p}: {e}", file=sys.stderr, flush=True)
                    continue

            if not tensors:
                continue

            batch_tensor = torch.cat(tensors, dim=0).to(device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch_labels[valid_idx])
            if model.embeddings is not None:
                all_embeddings.append(model.embeddings.astype(np.float16))  # float16 keeps full-test-set memory modest

    if n_failed:
        print(f"[WARN] {n_failed:,} of {n_samples:,} images could not be read and were excluded from the metrics.")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_embeddings = np.concatenate(all_embeddings).astype(np.float32) if all_embeddings else np.empty((0, 1280), dtype=np.float32)

    return all_labels, all_preds, all_probs, all_embeddings


# ==============================================================================
# 4. Metrics, Classification Report & Confusion Matrix
# ==============================================================================
def compute_and_save_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, output_dir: Path):
    acc = float(np.mean(y_true == y_pred))
    
    # Top-5 Accuracy
    top5_acc = None
    if y_probs.shape[1] >= 5:
        top5_preds = np.argsort(y_probs, axis=1)[:, -5:]
        top5_acc = float(np.mean([y in top5_preds[i] for i, y in enumerate(y_true)]))

    prec, rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Cross-entropy loss
    eps = 1e-12
    probs_clamped = np.clip(y_probs, eps, 1.0 - eps)
    ce_loss = float(-np.mean(np.log(probs_clamped[np.arange(len(y_true)), y_true])))

    metrics = {
        "test_accuracy": acc,
        "test_top5_accuracy": top5_acc,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "cross_entropy_loss": ce_loss,
        "num_evaluated_samples": int(len(y_true)),
        "num_classes": int(context.num_classes)
    }

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Classification report CSV
    target_names = [context.idx_to_class.get(i, f"class_{i}") for i in range(context.num_classes)]
    present_classes = sorted(list(set(y_true) | set(y_pred)))
    present_names = [target_names[i] for i in present_classes]

    clf_report = classification_report(y_true, y_pred, labels=present_classes, target_names=present_names, output_dict=True, zero_division=0)
    pd.DataFrame(clf_report).transpose().to_csv(reports_dir / "test_classification_report.csv")

    # Confusion matrix CSVs
    cm_raw = confusion_matrix(y_true, y_pred, labels=present_classes)
    cm_norm = cm_raw.astype(float) / (cm_raw.sum(axis=1, keepdims=True) + 1e-12)
    pd.DataFrame(cm_raw, index=present_names, columns=present_names).to_csv(reports_dir / "confusion_matrix_raw.csv")
    pd.DataFrame(cm_norm, index=present_names, columns=present_names).to_csv(reports_dir / "confusion_matrix_normalized.csv")

    print("\n" + "=" * 60)
    print(f"  Accuracy:      {acc:.2%}" + (f" | Top-5: {top5_acc:.2%}" if top5_acc else ""))
    print(f"  Macro F1:      {macro_f1:.2%} | Weighted F1: {weighted_f1:.2%}")
    print(f"  Loss:          {ce_loss:.4f} | Samples: {len(y_true):,}")
    print("=" * 60)

    return metrics


def plot_confusion_matrix(cm_norm_path: Path, out_path: Path, top_n: int = 35):
    if not cm_norm_path.exists():
        return
    df = pd.read_csv(cm_norm_path, index_col=0)
    if len(df) > top_n:
        # Show top confusion pairs or subset
        df = df.iloc[:top_n, :top_n]

    fig, ax = plt.subplots(figsize=(14, 12))
    cax = ax.matshow(df.values, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_title("Normalized Confusion Matrix (Top Classes)", fontsize=14, pad=20, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved confusion matrix heatmap to: {out_path}")


# ==============================================================================
# 5. Analytical Grad-CAM on MobileNetV2
# ==============================================================================
def compute_gradcam(model: MobileNetV2Evaluator, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Class activation map L_c = ReLU(sum_k w_k^c * A^k), computed inside the ONNX graph as the `class_maps` output."""
    model.eval()
    with torch.no_grad():
        model(image_tensor.to(context.device))
    if model.class_maps is None:
        raise RuntimeError("ONNX model has no 'class_maps' output; re-export it with the current training script.")

    cam = np.maximum(model.class_maps[0, class_idx], 0)  # (H_feat, W_feat)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    w, h = img_pil.size
    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heatmap = plt.cm.jet(np.array(cam_pil) / 255.0)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)
    blended = Image.blend(img_pil.convert("RGBA"), heatmap_pil.convert("RGBA"), alpha=alpha)
    return blended.convert("RGB")


# ==============================================================================
# 6. Latent Space Visualization (t-SNE & PCA)
# ==============================================================================
def plot_latent_space(embeddings: np.ndarray, labels: np.ndarray, out_path: Path, max_samples: int = 1500):
    if len(embeddings) == 0:
        return
    if len(embeddings) > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    pca_2d = PCA(n_components=2).fit_transform(embeddings)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("MobileNetV2 Latent Space Representation (1,280-D Penultimate Features)", fontsize=14, fontweight="bold")

    axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap="tab20", s=15, alpha=0.7)
    axes[0].set_title("PCA Projection (2D)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    scatter = axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=labels, cmap="tab20", s=15, alpha=0.7)
    axes[1].set_title("t-SNE Manifold Projection (2D)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved latent space visualizations to: {out_path}")


# ==============================================================================
# 7. Calibration & Rejection Analysis
# ==============================================================================
def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, out_path: Path, tau_default: float = 0.70):
    max_conf = np.max(y_probs, axis=1)
    is_correct = (y_true == y_pred)

    thresholds = np.linspace(0.1, 0.99, 50)
    accuracies = []
    coverages = []

    for tau in thresholds:
        accepted = max_conf >= tau
        cov = float(np.mean(accepted))
        coverages.append(cov)
        if np.sum(accepted) > 0:
            accuracies.append(float(np.mean(is_correct[accepted])))
        else:
            accuracies.append(1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coverages, accuracies, "o-", color="#2563EB", linewidth=2, label="Accuracy vs Coverage")
    
    # Highlight tau default point
    idx_tau = np.argmin(np.abs(thresholds - tau_default))
    ax.scatter([coverages[idx_tau]], [accuracies[idx_tau]], color="#DC2626", s=100, zorder=5, 
               label=f"τ = {tau_default} (Acc: {accuracies[idx_tau]:.1%}, Cov: {coverages[idx_tau]:.1%})")

    ax.set_title("Confidence Calibration & Rejection Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coverage (Proportion of Accepted Inferences)", fontsize=11)
    ax.set_ylabel("Accuracy on Accepted Inferences", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved calibration curve to: {out_path}")


# ==============================================================================
# 8. Main CLI Runner
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Universal Diagnostic Analysis for MobileNetV2")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint (.onnx)")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test.csv split")
    parser.add_argument("--label-map", type=str, required=True, help="Path to label_map.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for reports & plots")
    parser.add_argument("--base-img-dir", type=str, default=None, help="Root folder for relative image paths")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--crop-top-pct", type=float, default=0.0)
    parser.add_argument("--crop-bottom-pct", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    set_context(
        model_path=args.model_path,
        output_dir=args.output_dir,
        test_csv=args.test_csv,
        label_map_path=args.label_map,
        base_img_dir=args.base_img_dir,
        img_size=args.img_size,
        crop_top_pct=args.crop_top_pct,
        crop_bottom_pct=args.crop_bottom_pct
    )

    test_df = pd.read_csv(args.test_csv)
    model = MobileNetV2Evaluator(num_classes=context.num_classes, checkpoint_path=args.model_path)
    
    y_true, y_pred, y_probs, embeddings = evaluate_test_dataset(model, test_df, batch_size=args.batch_size)
    metrics = compute_and_save_metrics(y_true, y_pred, y_probs, context.output_dir)

    plot_confusion_matrix(context.reports_dir / "confusion_matrix_normalized.csv", context.plots_dir / "confusion_matrix.png")
    plot_calibration_curve(y_true, y_pred, y_probs, context.plots_dir / "calibration_curve.png")
    plot_latent_space(embeddings, y_true, context.plots_dir / "latent_space_tsne_pca.png")

    print(f"\n[Done] All diagnostic artifacts successfully saved to: {context.output_dir}")


if __name__ == "__main__":
    main()
