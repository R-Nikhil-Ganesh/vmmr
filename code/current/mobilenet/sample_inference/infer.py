#!/usr/bin/env python3
"""
Standalone Self-Sufficient MobileNetV2 Vehicle Recognition Inference Engine
=============================================================================
Self-contained script to run production inference using the fine-tuned
MobileNetV2 multitask vehicle classifier (1,235 model classes across 35 makes).

Zero external dependencies beyond standard scientific packages:
- onnxruntime (CPU or CUDA)
- numpy
- pillow (PIL)

Capabilities:
-------------
1. Fine-Grained Model Recognition: Predicts across 1,235 fine-grained vehicle models.
2. Coarse Make Marginalization: Accurately computes vehicle make probabilities by
   marginalizing over model posteriors: P(Make_k) = sum_{m in Make_k} P(Model_m).
3. 1280-D Embeddings: Optionally retrieves normalized feature embeddings.
4. HUD Visualization: Generates publication-ready annotated images with prediction HUD.
5. Batch & Single-Image Processing: Fast execution via ONNX Runtime GPU/CPU.

Usage Examples:
---------------
1. Run on default sample images:
   python infer.py

2. Run on a single image:
   python infer.py --image images/01_audi.jpg

3. Save predictions to CSV and JSON:
   python infer.py --save-csv output/predictions.csv --save-json output/predictions.json

4. Generate annotated images with HUD confidence banner:
   python infer.py --save-annotated output/annotated/

5. Python API usage:
   from infer import MobileNetClassifier
   clf = MobileNetClassifier()
   res = clf.predict_image("images/01_audi.jpg", top_k=3)
   print(res["top1_model"], res["top1_model_confidence"])
   print(res["top1_make"], res["top1_make_confidence"])
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


class MobileNetClassifier:
    """
    Self-contained MobileNetV2 vehicle classifier (1,235 models, 35 makes).
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        label_map_path: Optional[Union[str, Path]] = None,
        img_size: int = 512,
        device: str = "auto"
    ):
        self.img_size = img_size

        # 1. Resolve model file
        if model_path:
            self.model_path = Path(model_path).resolve()
        else:
            candidates = [
                MODELS_DIR / "mobilenet_v2_best.onnx",
                SCRIPT_DIR / "mobilenet_v2_best.onnx",
                MODELS_DIR / "model.onnx"
            ]
            self.model_path = next((c for c in candidates if c.exists()), candidates[0])

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        # 2. Resolve label map
        if label_map_path:
            self.label_map_path = Path(label_map_path).resolve()
        else:
            candidates = [
                MODELS_DIR / "label_map.json",
                SCRIPT_DIR / "label_map.json"
            ]
            self.label_map_path = next((c for c in candidates if c.exists()), candidates[0])

        self.class_names: List[str] = []
        self.idx_to_class: Dict[int, str] = {}
        self.class_to_idx: Dict[str, int] = {}
        self._load_label_map()

        # 3. Initialize ONNX Runtime Session
        self.session, self.active_provider = self._init_session(device)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.num_classes = len(self.class_names)

        # Precompute make mappings for fast probability marginalization
        self.idx_to_make: List[str] = []
        self.make_to_indices: Dict[str, List[int]] = {}
        for idx, name in enumerate(self.class_names):
            make = name.split("/")[0] if "/" in name else name
            self.idx_to_make.append(make)
            if make not in self.make_to_indices:
                self.make_to_indices[make] = []
            self.make_to_indices[make].append(idx)
        self.makes = sorted(self.make_to_indices.keys())

    def _load_label_map(self):
        if self.label_map_path and self.label_map_path.exists():
            with open(self.label_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "idx_to_class" in data:
                self.idx_to_class = {int(k): str(v) for k, v in data["idx_to_class"].items()}
                self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
                return

        # Fallback default names
        self.class_names = [f"Class_{i}" for i in range(1235)]
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
        """
        Preprocesses PIL image to normalized float32 tensor [1, 3, H, W]
        with ImageNet mean and standard deviation.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Bilinear resize to model resolution (512x512)
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Scale to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # Transpose to [C, H, W] and add batch dimension -> [1, C, H, W]
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def marginalize_makes(self, model_probs: np.ndarray) -> Dict[str, float]:
        """
        Computes make probabilities by summing over member fine-grained models:
        P(Make) = sum_{m in Make} P(Model_m)
        """
        make_probs = {}
        for make, indices in self.make_to_indices.items():
            make_probs[make] = float(np.sum(model_probs[indices]))
        return make_probs

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 5,
        return_embedding: bool = False
    ) -> Dict[str, Any]:
        """
        Runs inference on a single image.
        """
        image_path = None
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            img = Image.open(image_path)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError("image_input must be a file path or PIL Image.")

        tensor = self.preprocess(img)

        # Run ONNX Runtime session
        t0 = time.perf_counter()
        feed_dict = {self.input_name: tensor}
        
        # Determine output fetches
        fetches = ["predictions"]
        if return_embedding and "embeddings" in self.output_names:
            fetches.append("embeddings")
        
        outputs = self.session.run(fetches, feed_dict)
        logits = outputs[0][0]
        embedding = outputs[1][0] if len(outputs) > 1 else None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Probabilities
        probs = self._softmax(logits)

        # Top-K Fine-grained Models
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        predictions = []
        for rank, idx in enumerate(top_k_indices, 1):
            full_name = self.idx_to_class[idx]
            make_name = full_name.split("/")[0] if "/" in full_name else full_name
            model_name = full_name.split("/")[1] if "/" in full_name else full_name
            predictions.append({
                "rank": rank,
                "class_idx": int(idx),
                "full_class": full_name,
                "make": make_name,
                "model": model_name,
                "confidence": float(probs[idx])
            })

        # Marginalized Makes
        make_probs = self.marginalize_makes(probs)
        top_makes = sorted(make_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        make_predictions = [
            {"rank": r, "make": mk, "confidence": conf}
            for r, (mk, conf) in enumerate(top_makes, 1)
        ]

        result = {
            "image_path": image_path,
            "latency_ms": round(latency_ms, 2),
            "top1_model": predictions[0]["full_class"],
            "top1_model_name": predictions[0]["model"],
            "top1_model_confidence": predictions[0]["confidence"],
            "top1_make": make_predictions[0]["make"],
            "top1_make_confidence": make_predictions[0]["confidence"],
            "predictions": predictions,
            "make_predictions": make_predictions,
            "provider": self.active_provider
        }
        if embedding is not None:
            result["embedding"] = embedding.tolist()

        return result

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on a collection of images.
        """
        results = []
        for inp in image_inputs:
            try:
                results.append(self.predict_image(inp, top_k=top_k))
            except Exception as e:
                path_str = str(inp) if isinstance(inp, (str, Path)) else "<PIL.Image>"
                print(f"[Warning] Failed to infer {path_str}: {e}", file=sys.stderr)
        return results

    def annotate_image(
        self,
        image_input: Union[str, Path, Image.Image],
        pred_result: Dict[str, Any]
    ) -> Image.Image:
        """
        Renders HUD prediction overlay banner on the image.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")

        w, h = img.size
        banner_h = max(70, int(h * 0.16))

        overlay = Image.new("RGBA", (w, banner_h), (15, 23, 42, 230))
        img.paste(overlay, (0, h - banner_h), overlay)

        draw = ImageDraw.Draw(img)
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(banner_h * 0.32))
            font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(banner_h * 0.22))
        except Exception:
            font_title = ImageFont.load_default()
            font_sub = font_title

        top_model = pred_result["top1_model"]
        top_make = pred_result["top1_make"]
        model_conf = pred_result["top1_model_confidence"]
        make_conf = pred_result["top1_make_confidence"]
        lat = pred_result["latency_ms"]

        title_text = f"{top_model}  ({model_conf * 100:.1f}%)"
        sub_text = f"Make: {top_make} ({make_conf * 100:.1f}%)  |  Latency: {lat:.1f}ms  |  {pred_result['provider']}"

        draw.text((16, h - banner_h + int(banner_h * 0.12)), title_text, fill=(255, 255, 255), font=font_title)
        draw.text((16, h - banner_h + int(banner_h * 0.55)), sub_text, fill=(148, 163, 184), font=font_sub)

        return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileNetV2 1,235-Class Vehicle Classifier Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, default=None, help="Path to a single image file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to directory of images.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to ONNX model.")
    parser.add_argument("--label-map", type=str, default=None, help="Path to label map JSON.")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="Hardware device.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of ranked predictions to return.")
    parser.add_argument("--save-csv", type=str, default=None, help="Path to save results as CSV.")
    parser.add_argument("--save-json", type=str, default=None, help="Path to save results as JSON.")
    parser.add_argument("--save-annotated", type=str, default=None, help="Directory to save annotated images.")
    parser.add_argument("--extract-embeddings", action="store_true", help="Extract and include 1280-D feature embeddings.")
    parser.add_argument("--self-test", action="store_true", help="Run automated self-test on a synthetic in-memory image without any external image files.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 78)
    print("  MobileNetV2 Production Vehicle Recognition Inference Engine (1,235 Models)")
    print("=" * 78)

    clf = MobileNetClassifier(
        model_path=args.model_path,
        label_map_path=args.label_map,
        device=args.device
    )

    print(f"  Checkpoint: {clf.model_path.name} ({clf.model_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"  Taxonomy:   {clf.num_classes} Fine-Grained Models across {len(clf.makes)} Vehicle Makes")
    print(f"  Provider:   {clf.active_provider}")
    print("=" * 78)

    # 1. Self-Test Mode or Automatic Health Check (Zero dataset / Zero external files required)
    if args.self_test or (args.image is None and args.image_dir is None and (not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()))):
        print("\n[Self-Test] Running self-test with synthetic in-memory image (zero dataset required)...")
        dummy_img = Image.new("RGB", (512, 512), color=(128, 140, 160))
        res = clf.predict_image(dummy_img, top_k=args.top_k, return_embedding=args.extract_embeddings)

        print(f"  Forward Pass:    SUCCESS")
        print(f"  Top-1 Model:     {res['top1_model']}")
        print(f"  Top-1 Make:      {res['top1_make']} ({res['top1_make_confidence']*100:.1f}%)")
        print(f"  Latency:         {res['latency_ms']:.2f} ms on {clf.active_provider}")
        print("\n[Ready] The model and inference pipeline are operational without any dataset.")
        print("To classify your own vehicle images, run:")
        print("  python infer.py --image /path/to/vehicle.jpg")
        print("  python infer.py --image-dir /path/to/folder/")
        print("  python infer.py --image /path/to/vehicle.jpg --save-annotated output/")
        return

    # 2. Resolve target images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        target_dir = Path(args.image_dir) if args.image_dir else IMAGES_DIR
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in valid_exts]) if target_dir.exists() else []

    if not image_paths:
        print(f"[Error] No image files found in {args.image or target_dir}.")
        print("Please provide --image <path> or --image-dir <path>, or run with --self-test.")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} image(s)...\n")

    results = []
    for p in image_paths:
        res = clf.predict_image(p, top_k=args.top_k, return_embedding=args.extract_embeddings)
        results.append(res)

    # Print summary table
    print(f"{'#':2s} | {'Sample Image':18s} | {'Top-1 Model':22s} | {'Conf':7s} | {'Make (Marg.)':14s} | {'Lat':7s}")
    print("-" * 88)
    for i, r in enumerate(results):
        fn = Path(r["image_path"]).name if r["image_path"] else f"img_{i+1}"
        mk_str = f"{r['top1_make']} ({r['top1_make_confidence']*100:.0f}%)"
        print(f"{i+1:02d} | {fn[:18]:18s} | {r['top1_model'][:22]:22s} | {r['top1_model_confidence']*100:5.1f}%  | {mk_str:14s} | {r['latency_ms']:5.1f}ms")
    print("-" * 88)

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
                "filename": Path(r["image_path"]).name if r["image_path"] else "",
                "top1_model": r["top1_model"],
                "top1_model_confidence": r["top1_model_confidence"],
                "top1_make": r["top1_make"],
                "top1_make_confidence": r["top1_make_confidence"],
                "latency_ms": r["latency_ms"]
            }
            for p in r["predictions"]:
                row[f"model_rank_{p['rank']}"] = p["full_class"]
                row[f"model_rank_{p['rank']}_conf"] = p["confidence"]
            for m in r["make_predictions"]:
                row[f"make_rank_{m['rank']}"] = m["make"]
                row[f"make_rank_{m['rank']}_conf"] = m["confidence"]
            rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_p, index=False)
        print(f"[Export Saved] CSV predictions written to: {csv_p}")

    if args.save_json:
        json_p = Path(args.save_json)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Export Saved] JSON predictions written to: {json_p}")

    if args.save_annotated:
        ann_dir = Path(args.save_annotated)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            fn = Path(r["image_path"]).name if r["image_path"] else "annotated.jpg"
            annotated_img = clf.annotate_image(r["image_path"], r)
            out_img_path = ann_dir / f"annotated_{fn}"
            annotated_img.save(out_img_path, quality=92)
        print(f"[Export Saved] {len(results)} annotated visualization image(s) written to: {ann_dir}/")


if __name__ == "__main__":
    main()
