#!/usr/bin/env python3
"""
Standalone Self-Sufficient ONNX Vehicle Inference Engine
=========================================================
Self-contained script to run inference using the fine-tuned EfficientNet-B0
models for vehicle make classification (32-make and 35-make versions).

Zero external dependencies beyond standard scientific packages:
- onnxruntime
- numpy
- pillow (PIL)

Usage Examples:
---------------
1. Run on default sample images with the 35-make model:
   python infer.py

2. Run on default sample images with the 32-make model:
   python infer.py --model 32

3. Run on a single image:
   python infer.py --image images/11_porsche.jpg

4. Save predictions to CSV and JSON:
   python infer.py --save-csv output/predictions.csv --save-json output/predictions.json

5. Generate annotated images with HUD confidence banner:
   python infer.py --save-annotated output/annotated/

6. Python API usage:
   from infer import SampleClassifier
   clf = SampleClassifier(model_type="35")  # or "32"
   result = clf.predict_image("images/02_bmw.jpg", top_k=3)
   print(result["top1_make"], result["top1_confidence"])
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
IMAGES_DIR = SCRIPT_DIR / "images"
OUTPUT_DIR = SCRIPT_DIR / "output"


class SampleClassifier:
    """
    Self-contained ONNX vehicle make classifier.
    """
    def __init__(
        self,
        model_type: str = "35",
        model_path: Optional[Union[str, Path]] = None,
        label_map_path: Optional[Union[str, Path]] = None,
        img_size: int = 640,
        device: str = "auto"
    ):
        self.img_size = img_size
        self.model_type = str(model_type).strip()

        # 1. Resolve model file
        if model_path:
            self.model_path = Path(model_path).resolve()
        elif self.model_type == "32":
            self.model_path = MODELS_DIR / "efficientnet_b0_32makes_best.onnx"
        else:
            self.model_path = MODELS_DIR / "efficientnet_b0_35makes_best.onnx"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 2. Resolve label map
        if label_map_path:
            self.label_map_path = Path(label_map_path).resolve()
        elif self.model_type == "32":
            self.label_map_path = MODELS_DIR / "label_map_32makes.json"
        else:
            self.label_map_path = MODELS_DIR / "label_map_35makes.json"

        self.class_names: List[str] = []
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self._load_label_map()

        # 3. Initialize ONNX Runtime Session
        self.session, self.active_provider = self._init_session(device)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.num_classes = len(self.class_names)

    def _load_label_map(self):
        if self.label_map_path and self.label_map_path.exists():
            with open(self.label_map_path, "r") as f:
                data = json.load(f)
            if "idx_to_class" in data:
                self.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
                self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
                return

        # Fallback default names
        n = 32 if self.model_type == "32" else 35
        self.class_names = [f"Class_{i}" for i in range(n)]
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

    def _init_session(self, device: str) -> Tuple[ort.InferenceSession, str]:
        available = ort.get_available_providers()
        providers = []
        if device.lower() in ("auto", "cuda", "gpu") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
        return session, session.get_providers()[0]

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Preprocesses PIL image to normalized float32 tensor."""
        rgb = img.convert("RGB")
        if rgb.size != (self.img_size, self.img_size):
            rgb = rgb.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        arr = np.array(rgb, dtype=np.float32)
        # EfficientNet-B0 native normalization is within model graph; raw [0, 255] float32
        return arr[np.newaxis, ...]

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Runs inference on a single image and returns ranked predictions."""
        t0 = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            pil_img = Image.open(path_str)
        else:
            path_str = "in-memory"
            pil_img = image_input

        inp = self.preprocess(pil_img)
        outputs = self.session.run(None, {self.input_name: inp})
        probs = outputs[0][0]

        top_indices = np.argsort(-probs)[:top_k]
        top_predictions = [
            {
                "rank": rank + 1,
                "make": self.idx_to_class.get(int(idx), f"Class_{idx}"),
                "class_idx": int(idx),
                "confidence": float(probs[idx])
            }
            for rank, idx in enumerate(top_indices)
        ]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "image_path": path_str,
            "top1_make": top_predictions[0]["make"],
            "top1_confidence": top_predictions[0]["confidence"],
            "predictions": top_predictions,
            "latency_ms": latency_ms,
            "model_type": f"{self.num_classes}-makes",
            "provider": self.active_provider
        }

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        top_k: int = 5,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """Runs batched inference on multiple image paths."""
        results = []
        n_total = len(image_paths)

        for b_start in range(0, n_total, batch_size):
            b_paths = image_paths[b_start : b_start + batch_size]
            b_imgs = []
            valid_paths = []

            for p in b_paths:
                try:
                    img = Image.open(str(p)).convert("RGB")
                    if img.size != (self.img_size, self.img_size):
                        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                    b_imgs.append(np.array(img, dtype=np.float32))
                    valid_paths.append(str(p))
                except Exception as e:
                    print(f"[Warning] Failed to read {p}: {e}")

            if not b_imgs:
                continue

            t0 = time.perf_counter()
            b_tensor = np.stack(b_imgs, axis=0)
            outputs = self.session.run(None, {self.input_name: b_tensor})
            b_probs = outputs[0]
            lat_per_img = ((time.perf_counter() - t0) * 1000.0) / len(b_imgs)

            for p_str, probs in zip(valid_paths, b_probs):
                top_indices = np.argsort(-probs)[:top_k]
                preds = [
                    {
                        "rank": r + 1,
                        "make": self.idx_to_class.get(int(idx), f"Class_{idx}"),
                        "class_idx": int(idx),
                        "confidence": float(probs[idx])
                    }
                    for r, idx in enumerate(top_indices)
                ]
                results.append({
                    "image_path": p_str,
                    "top1_make": preds[0]["make"],
                    "top1_confidence": preds[0]["confidence"],
                    "predictions": preds,
                    "latency_ms": lat_per_img,
                    "model_type": f"{self.num_classes}-makes",
                    "provider": self.active_provider
                })

        return results

    def annotate_image(
        self,
        image_input: Union[str, Path, Image.Image],
        result: Dict[str, Any]
    ) -> Image.Image:
        """Overlays a clean HUD prediction banner on the image."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")

        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Banner layout
        banner_h = max(50, int(h * 0.12))
        overlay = Image.new("RGBA", (w, banner_h), (15, 23, 42, 220))
        img.paste(overlay, (0, 0), overlay)

        make_text = f"{result['top1_make'].upper()}"
        conf_text = f"Confidence: {result['top1_confidence']*100:.1f}%  |  Model: {self.num_classes} Makes ({result['latency_ms']:.1f}ms)"

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(banner_h * 0.42))
            font_sub   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(banner_h * 0.28))
        except Exception:
            font_title = ImageFont.load_default()
            font_sub   = font_title

        # Accent bar
        draw.rectangle([(0, 0), (6, banner_h)], fill=(16, 185, 129))
        draw.text((16, int(banner_h * 0.12)), make_text, fill=(255, 255, 255), font=font_title)
        draw.text((16, int(banner_h * 0.58)), conf_text, fill=(148, 163, 184), font=font_sub)

        return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Self-Sufficient ONNX Vehicle Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, default=None, help="Path to a single image file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to directory of images.")
    parser.add_argument("--model", type=str, choices=["35", "32"], default="35", help="Model version to test (35 or 32 makes).")
    parser.add_argument("--model-path", type=str, default=None, help="Custom ONNX model checkpoint path.")
    parser.add_argument("--label-map", type=str, default=None, help="Custom label map JSON path.")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="Execution hardware.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked predictions to return.")
    parser.add_argument("--save-csv", type=str, default=None, help="File path to save results as CSV.")
    parser.add_argument("--save-json", type=str, default=None, help="File path to save results as JSON.")
    parser.add_argument("--save-annotated", type=str, default=None, help="Directory to save annotated visualization images.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"  Self-Sufficient ONNX Vehicle Classifier (Model: {args.model}-Makes)")
    print("=" * 70)

    clf = SampleClassifier(
        model_type=args.model,
        model_path=args.model_path,
        label_map_path=args.label_map,
        device=args.device
    )

    print(f"  Checkpoint: {clf.model_path.name} ({clf.model_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"  Classes:    {clf.num_classes} vehicle makes")
    print(f"  Provider:   {clf.active_provider}")
    print("=" * 70)

    # Resolve target images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        target_dir = Path(args.image_dir) if args.image_dir else IMAGES_DIR
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in valid_exts])

    if not image_paths:
        print("[Error] No image files found to process.")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} sample image(s)...\n")

    # Run inference
    results = clf.predict_batch(image_paths, top_k=args.top_k)

    # Print summary table
    print(f"{'#':2s} | {'Sample Image':20s} | {'Top-1 Make':15s} | {'Confidence':10s} | {'Latency':8s} | {'Runner-ups (Top-3)':25s}")
    print("-" * 95)
    for i, r in enumerate(results):
        fn = Path(r["image_path"]).name
        alt = ", ".join([f"{p['make']} ({p['confidence']*100:.0f}%)" for p in r["predictions"][1:3]])
        print(f"{i+1:02d} | {fn:20s} | {r['top1_make']:15s} | {r['top1_confidence']*100:6.1f}%    | {r['latency_ms']:5.1f} ms | {alt}")
    print("-" * 95)

    avg_lat = np.mean([r["latency_ms"] for r in results])
    print(f"\nAverage Inference Latency: {avg_lat:.2f} ms / image on {clf.active_provider}")

    # Export outputs if requested
    if args.save_csv:
        csv_p = Path(args.save_csv)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in results:
            row = {
                "image_path": r["image_path"],
                "filename": Path(r["image_path"]).name,
                "top1_make": r["top1_make"],
                "top1_confidence": r["top1_confidence"],
                "latency_ms": r["latency_ms"]
            }
            for p in r["predictions"]:
                row[f"top_{p['rank']}_make"] = p["make"]
                row[f"top_{p['rank']}_confidence"] = p["confidence"]
            rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_p, index=False)
        print(f"[Export Saved] CSV predictions written to: {csv_p}")

    if args.save_json:
        json_p = Path(args.save_json)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Export Saved] JSON predictions written to: {json_p}")

    if args.save_annotated:
        ann_dir = Path(args.save_annotated)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            fn = Path(r["image_path"]).name
            annotated_img = clf.annotate_image(r["image_path"], r)
            out_img_path = ann_dir / f"annotated_{fn}"
            annotated_img.save(out_img_path, quality=92)
        print(f"[Export Saved] {len(results)} annotated visualization image(s) written to: {ann_dir}/")


if __name__ == "__main__":
    main()
