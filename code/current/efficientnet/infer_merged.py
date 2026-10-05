#!/usr/bin/env python3
"""
EfficientNet-B0 Merged (35 Vehicle Makes) - Inference Engine
============================================================
Fast, production-grade inference script for the fine-tuned EfficientNet-B0
vehicle classifier trained on the merged 35-make dataset.

Supports both ONNX Runtime (recommended for speed and low memory footprint)
and native TensorFlow/Keras checkpoints.

Usage Examples:
---------------
1. Single image classification:
   python infer_merged.py --image path/to/car.jpg

2. Batch classification on a directory:
   python infer_merged.py --image-dir path/to/folder/ --save-csv predictions.csv

3. Batch classification from a CSV manifest:
   python infer_merged.py --csv /path/to/test.csv --image-col image_path --save-json results.json

4. Generate annotated visualization images:
   python infer_merged.py --image path/to/car.jpg --save-annotated output_annotated/

5. Python API usage:
   from infer_merged import MergedVehicleClassifier
   clf = MergedVehicleClassifier()
   res = clf.predict_image("path/to/car.jpg", top_k=5)
   print(f"Prediction: {res['top1_make']} ({res['top1_confidence']:.2%})")
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

# Set clean logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Default Class Labels (35 Makes)
DEFAULT_CLASSES = [
    "Acura", "Alfa Romeo", "Audi", "BMW", "Buick", "Cadillac", "Chevrolet", "Chrysler",
    "Cupra", "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Isuzu",
    "Jeep", "Kia", "Land Rover", "Lexus", "Lincoln", "MINI", "Mazda", "Mitsubishi",
    "Nissan", "Opel", "Peugeot", "Porsche", "Renault", "Skoda", "Subaru", "Suzuki",
    "Toyota", "Volkswagen", "Volvo"
]

# Standard Search Paths for Best Checkpoint & Label Map
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

DEFAULT_MODEL_CANDIDATES = [
    SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_best.onnx",
    SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_best.keras",
    SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_final.onnx",
    SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_final.keras",
    Path("/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged/models/efficientnet_b0_best.onnx"),
    Path("/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged/models/efficientnet_b0_best.keras"),
]

DEFAULT_LABEL_MAP_CANDIDATES = [
    SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "label_map.json",
    SCRIPT_DIR / "output_efficientnet_b0_merged" / "label_map.json",
    Path("/home/researchadmin/Econ/external_datasets/merged_data/label_map.json"),
    Path("/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged/models/label_map.json"),
]


def resolve_default_model() -> Path:
    """Finds the best available model checkpoint."""
    for cand in DEFAULT_MODEL_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No model checkpoint found in default search paths. "
        "Please provide --model-path explicitly."
    )


def resolve_default_label_map() -> Optional[Path]:
    """Finds the class label map JSON."""
    for cand in DEFAULT_LABEL_MAP_CANDIDATES:
        if cand.exists():
            return cand
    return None


class MergedVehicleClassifier:
    """
    High-performance vehicle make classifier for the 35-make dataset.
    Supports both ONNX Runtime and TensorFlow/Keras backends.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        label_map_path: Optional[Union[str, Path]] = None,
        img_size: int = 640,
        device: str = "auto"
    ):
        self.img_size = img_size
        self.model_path = Path(model_path) if model_path else resolve_default_model()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        # Determine backend from extension
        self.backend = "onnx" if self.model_path.suffix.lower() == ".onnx" else "keras"

        # Load label map
        self.classes, self.class_to_idx, self.idx_to_class = self._load_label_map(label_map_path)
        self.num_classes = len(self.classes)

        # Initialize backend model
        self.device = device
        self._init_backend()

    def _load_label_map(self, label_map_path: Optional[Union[str, Path]]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        resolved_map = Path(label_map_path) if label_map_path else resolve_default_label_map()
        if resolved_map and resolved_map.exists():
            with open(resolved_map, "r") as f:
                meta = json.load(f)
            if "idx_to_class" in meta:
                idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
                classes = [idx_to_class[i] for i in range(len(idx_to_class))]
                class_to_idx = {name: i for i, name in enumerate(classes)}
                return classes, class_to_idx, idx_to_class

        # Fallback to standard 35 classes
        classes = DEFAULT_CLASSES
        class_to_idx = {name: i for i, name in enumerate(classes)}
        idx_to_class = {i: name for i, name in enumerate(classes)}
        return classes, class_to_idx, idx_to_class

    def _init_backend(self):
        if self.backend == "onnx":
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = []
            if self.device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.active_provider = self.session.get_providers()[0]
        else:
            import tensorflow as tf
            if self.device == "cpu":
                tf.config.set_visible_devices([], "GPU")
            else:
                gpus = tf.config.list_physical_devices("GPU")
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
            self.keras_model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.active_provider = "TensorFlow/Keras"

    def preprocess(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocesses an image to (1, img_size, img_size, 3) in float32 [0.0, 255.0].
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        arr = np.array(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Runs inference on a single image and returns top predictions."""
        start_time = time.perf_counter()
        tensor = self.preprocess(image_input)

        if self.backend == "onnx":
            probs = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        else:
            probs = self.keras_model(tensor, training=False).numpy()[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        sorted_indices = np.argsort(probs)[::-1]

        k = min(top_k, self.num_classes)
        top_k_list = [
            {
                "rank": rank + 1,
                "make": self.idx_to_class[idx],
                "confidence": float(probs[idx])
            }
            for rank, idx in enumerate(sorted_indices[:k])
        ]

        img_path_str = str(image_input) if isinstance(image_input, (str, Path)) else None

        return {
            "image_path": img_path_str,
            "top1_make": top_k_list[0]["make"],
            "top1_confidence": top_k_list[0]["confidence"],
            "top_k": top_k_list,
            "latency_ms": elapsed_ms,
            "backend": self.backend,
            "all_probabilities": {self.idx_to_class[i]: float(probs[i]) for i in range(self.num_classes)}
        }

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Runs batched inference across a list of image paths."""
        results = []
        n_total = len(image_paths)

        for b_start in range(0, n_total, batch_size):
            b_paths = image_paths[b_start : b_start + batch_size]
            tensors = []
            valid_paths = []
            for p in b_paths:
                try:
                    t = self.preprocess(p)
                    tensors.append(t[0])
                    valid_paths.append(p)
                except Exception as e:
                    print(f"[Warning] Failed to read {p}: {e}")

            if not tensors:
                continue

            batch_arr = np.stack(tensors, axis=0)
            t0 = time.perf_counter()
            if self.backend == "onnx":
                probs_batch = self.session.run([self.output_name], {self.input_name: batch_arr})[0]
            else:
                probs_batch = self.keras_model(batch_arr, training=False).numpy()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(valid_paths)

            k = min(top_k, self.num_classes)
            for path, probs in zip(valid_paths, probs_batch):
                sorted_idx = np.argsort(probs)[::-1]
                top_k_list = [
                    {
                        "rank": r + 1,
                        "make": self.idx_to_class[idx],
                        "confidence": float(probs[idx])
                    }
                    for r, idx in enumerate(sorted_idx[:k])
                ]
                results.append({
                    "image_path": str(path),
                    "top1_make": top_k_list[0]["make"],
                    "top1_confidence": top_k_list[0]["confidence"],
                    "top_k": top_k_list,
                    "latency_ms": elapsed_ms,
                    "backend": self.backend
                })
        return results

    def annotate_image(
        self,
        image_input: Union[str, Path, Image.Image],
        pred: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> Image.Image:
        """
        Overlays a modern, clean HUD banner on the image showing predicted make and top probabilities.
        """
        if isinstance(image_input, (str, Path)):
            base_img = Image.open(str(image_input)).convert("RGB")
        else:
            base_img = image_input.convert("RGB")

        w, h = base_img.size
        # Make a copy to draw on
        annotated = base_img.copy()
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # Header banner height
        banner_h = int(min(max(h * 0.22, 110), 220))
        # Draw translucent dark gradient/banner at top
        draw_overlay.rectangle([(0, 0), (w, banner_h)], fill=(15, 23, 42, 215))
        # Draw subtle bottom border line on banner
        draw_overlay.line([(0, banner_h), (w, banner_h)], fill=(59, 130, 246, 255), width=2)

        # Merge overlay
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(annotated)

        # Choose font sizes proportionally
        title_font_size = max(18, int(banner_h * 0.28))
        detail_font_size = max(13, int(banner_h * 0.16))

        try:
            from PIL import ImageFont
            # Attempt to use standard clean font
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_font_size)
            detail_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", detail_font_size)
        except Exception:
            title_font = ImageFont.load_default()
            detail_font = ImageFont.load_default()

        # Top 1 Prediction
        top1 = pred["top1_make"]
        conf1 = pred["top1_confidence"]

        # Color coding confidence
        if conf1 >= 0.85:
            badge_color = (34, 197, 94)    # Green
        elif conf1 >= 0.60:
            badge_color = (234, 179, 8)   # Yellow
        else:
            badge_color = (249, 115, 22)   # Orange

        title_text = f"Make: {top1}"
        conf_text = f"{conf1 * 100:.1f}% Confidence"
        draw.text((16, 12), title_text, fill=(255, 255, 255), font=title_font)
        draw.text((16 + int(len(title_text) * title_font_size * 0.62) + 20, 14), f"[{conf_text}]", fill=badge_color, font=detail_font)

        # Top-K sub-ranking line
        top_k_str = " | ".join([f"#{r['rank']} {r['make']}: {r['confidence']*100:.1f}%" for r in pred["top_k"][1:4]])
        if top_k_str:
            draw.text((16, 16 + title_font_size + 8), f"Runner-ups: {top_k_str}", fill=(203, 213, 225), font=detail_font)

        # Latency & Backend watermark
        latency_str = f"Backend: {pred['backend'].upper()} ({self.active_provider}) | Latency: {pred['latency_ms']:.1f}ms"
        draw.text((16, banner_h - detail_font_size - 8), latency_str, fill=(148, 163, 184), font=detail_font)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(str(out_p))

        return annotated


def render_ascii_bar(val: float, length: int = 15) -> str:
    """Renders a text progress bar for confidence visualization."""
    filled = int(round(val * length))
    filled = min(max(filled, 0), length)
    return "█" * filled + "░" * (length - filled)


def print_single_prediction_table(res: Dict[str, Any]):
    """Pretty prints a single image prediction result."""
    print("\n" + "=" * 65)
    print(f"  VEHICLE MAKE PREDICTION RESULT")
    print("=" * 65)
    if res["image_path"]:
        print(f"  Target Image:  {res['image_path']}")
    print(f"  Backend:       {res['backend'].upper()}")
    print(f"  Inference:     {res['latency_ms']:.2f} ms")
    print(f"  Winner (Top 1): {res['top1_make']} ({res['top1_confidence']*100:.2f}%)")
    print("-" * 65)
    print(f"  {'Rank':<5} {'Vehicle Make':<18} {'Confidence':<12} {'Probability Bar':<15}")
    print("-" * 65)
    for r in res["top_k"]:
        bar = render_ascii_bar(r["confidence"], length=15)
        print(f"  #{r['rank']:<4} {r['make']:<18} {r['confidence']*100:6.2f}%     [{bar}]")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 (35 Vehicle Makes) - Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input modes
    input_group = parser.add_argument_group("Input Sources (select one)")
    input_group.add_argument("--image", type=str, help="Path to a single vehicle image.")
    input_group.add_argument("--image-dir", type=str, help="Directory containing images to process.")
    input_group.add_argument("--csv", type=str, help="CSV manifest containing image paths.")
    input_group.add_argument("--image-col", type=str, default="image_path", help="Column in CSV containing image paths.")
    input_group.add_argument("--glob", type=str, help="Glob pattern to match images (e.g. 'data/*.jpg').")

    # Model settings
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model-path", type=str, default=None, help="Custom path to .onnx or .keras checkpoint.")
    model_group.add_argument("--label-map", type=str, default=None, help="Path to label_map.json.")
    model_group.add_argument("--img-size", type=int, default=640, help="Input spatial resolution (default: 640).")
    model_group.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference hardware device.")

    # Execution & Export settings
    exec_group = parser.add_argument_group("Inference & Export Options")
    exec_group.add_argument("--top-k", type=int, default=5, help="Number of ranked predictions to output.")
    exec_group.add_argument("--batch-size", type=int, default=32, help="Batch size for directory or CSV inference.")
    exec_group.add_argument("--save-json", type=str, default=None, help="Save predictions to JSON file.")
    exec_group.add_argument("--save-csv", type=str, default=None, help="Save predictions to CSV file.")
    exec_group.add_argument("--save-annotated", type=str, default=None, help="Directory or file path to save annotated visualization(s).")
    exec_group.add_argument("--quiet", action="store_true", help="Suppress verbose stdout table.")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve inputs
    image_paths: List[str] = []
    is_single = False

    if args.image:
        image_paths = [args.image]
        is_single = True
    elif args.image_dir:
        dir_p = Path(args.image_dir)
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.JPG", "*.PNG"]
        for ext in exts:
            image_paths.extend(glob.glob(str(dir_p / "**" / ext), recursive=True))
        image_paths = sorted(list(set(image_paths)))
    elif args.glob:
        image_paths = sorted(glob.glob(args.glob))
    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if args.image_col not in df.columns:
            raise KeyError(f"Column '{args.image_col}' not found in CSV. Available columns: {df.columns.tolist()}")
        image_paths = df[args.image_col].dropna().astype(str).tolist()
    else:
        # If no input provided, show usage help
        print("[Notice] No input image specified. Running demonstration on a sample vehicle from dataset:")
        sample_cand = list(Path("/home/researchadmin/Econ/external_datasets/merged_data/images").glob("*/*.jpg"))
        if sample_cand:
            image_paths = [str(sample_cand[0])]
            is_single = True
        else:
            print("Please specify an input using --image, --image-dir, --csv, or --glob. See --help for details.")
            sys.exit(1)

    if not image_paths:
        print("[Error] No valid images found matching the provided input criteria.")
        sys.exit(1)

    # 2. Instantiate Classifier
    if not args.quiet:
        print("\n" + "=" * 65)
        print("  Initializing EfficientNet-B0 (35 Vehicle Makes) Classifier...")
        print("=" * 65)

    classifier = MergedVehicleClassifier(
        model_path=args.model_path,
        label_map_path=args.label_map,
        img_size=args.img_size,
        device=args.device
    )

    if not args.quiet:
        print(f"  Model Path:    {classifier.model_path}")
        print(f"  Backend:       {classifier.backend.upper()} (Provider: {classifier.active_provider})")
        print(f"  Classes:       {classifier.num_classes} vehicle makes")
        print(f"  Input Target:  {len(image_paths):,} image(s)")
        print("=" * 65)

    # 3. Perform Inference
    if is_single:
        pred = classifier.predict_image(image_paths[0], top_k=args.top_k)
        if not args.quiet:
            print_single_prediction_table(pred)
        results = [pred]

        # Annotated image save
        if args.save_annotated:
            out_target = Path(args.save_annotated)
            if out_target.is_dir() or str(args.save_annotated).endswith(("/", "\\")):
                out_target.mkdir(parents=True, exist_ok=True)
                out_path = out_target / f"annotated_{Path(image_paths[0]).name}"
            else:
                out_path = out_target
            classifier.annotate_image(image_paths[0], pred, output_path=out_path)
            print(f"[Annotated] Saved visualization to: {out_path}")

    else:
        if not args.quiet:
            print(f"\nProcessing {len(image_paths):,} images in batches of {args.batch_size}...")
        t_start = time.perf_counter()
        results = classifier.predict_batch(image_paths, batch_size=args.batch_size, top_k=args.top_k)
        total_time = time.perf_counter() - t_start
        fps = len(results) / total_time if total_time > 0 else 0

        if not args.quiet:
            print(f"[Done] Finished {len(results):,} inferences in {total_time:.2f}s ({fps:.1f} FPS, {1000/fps:.1f} ms/image)")
            # Print sample top 5
            print("\nPreview of First 5 Predictions:")
            print("-" * 65)
            for r in results[:5]:
                fname = Path(r["image_path"]).name
                print(f"  - {fname:<30} -> {r['top1_make']:<15} ({r['top1_confidence']*100:.2f}%)")
            print("-" * 65)

        # Annotated batch save
        if args.save_annotated:
            out_dir = Path(args.save_annotated)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving annotated visualizations to: {out_dir}")
            for r in results:
                out_path = out_dir / f"annotated_{Path(r['image_path']).name}"
                classifier.annotate_image(r["image_path"], r, output_path=out_path)
            print(f"[Annotated] All {len(results):,} images saved to: {out_dir}")

    # 4. Save JSON
    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        # Clean probabilities for export
        export_list = []
        for r in results:
            export_list.append({
                "image_path": r["image_path"],
                "top1_make": r["top1_make"],
                "top1_confidence": r["top1_confidence"],
                "top_k": r["top_k"],
                "latency_ms": r.get("latency_ms", 0.0)
            })
        with open(out_json, "w") as f:
            json.dump(export_list, f, indent=2)
        print(f"[Export] Saved JSON predictions to: {out_json}")

    # 5. Save CSV
    if args.save_csv:
        import pandas as pd
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in results:
            row = {
                "image_path": r["image_path"],
                "top1_make": r["top1_make"],
                "top1_confidence": r["top1_confidence"],
            }
            for rank_item in r["top_k"]:
                rk = rank_item["rank"]
                row[f"rank{rk}_make"] = rank_item["make"]
                row[f"rank{rk}_confidence"] = rank_item["confidence"]
            rows.append(row)
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_csv, index=False)
        print(f"[Export] Saved CSV predictions to: {out_csv}")


if __name__ == "__main__":
    main()
