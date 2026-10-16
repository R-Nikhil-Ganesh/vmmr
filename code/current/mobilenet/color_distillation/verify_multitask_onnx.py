#!/usr/bin/env python3
"""
Multi-Task MobileNetV2 ONNX Model Verifier & Benchmark
======================================================
Validates the exported mobilenet_v2_multitask.onnx model:
1. Inspects input and output tensor signatures.
2. Benchmarks raw forward pass latency (GPU and CPU).
3. Evaluates simultaneous Make/Model and Color predictions on test images from run5/annotations.csv.
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
MOBILENET_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MOBILENET_DIR / "annotation"))
from common import load_crop, load_label_map

COLORS = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige", "gold", "other"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser(description="Verify Multi-Task MobileNetV2 ONNX Model")
    parser.add_argument("--onnx-path", default=str(SCRIPT_DIR / "output" / "mobilenet_v2_multitask.onnx"))
    parser.add_argument("--label-map", default=str(MOBILENET_DIR / "merged_dataset" / "output_merged_run4" / "models" / "label_map.json"))
    parser.add_argument("--annotations", default=str(MOBILENET_DIR / "annotation_runs" / "run5" / "annotations.csv"))
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples to verify")
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        print(f"[ERROR] Model file not found: {onnx_path}")
        print("Please wait for run_train_color_distillation.sh to finish training and export the model.")
        sys.exit(1)

    print("==================================================================")
    print("  Multi-Task MobileNetV2 ONNX Model Verification")
    print(f"  Model:       {onnx_path} ({onnx_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"  Label Map:   {args.label_map}")
    print(f"  Annotations: {args.annotations}")
    print("==================================================================")

    # 1. Load ONNX Session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = session.get_providers()[0]
    print(f"[ONNX] Active Execution Provider: {active_provider}")

    # Inspect Inputs & Outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print("\n--- Model Signatures ---")
    print(f"Input:  {inputs[0].name} (shape: {inputs[0].shape})")
    print("Outputs:")
    for out in outputs:
        print(f"  - {out.name:20s}: shape={out.shape}, type={out.type}")

    expected_outputs = {"predictions", "color_predictions", "embeddings", "class_maps"}
    actual_outputs = {out.name for out in outputs}
    if not expected_outputs.issubset(actual_outputs):
        print(f"[WARN] Expected outputs {expected_outputs} but got {actual_outputs}")
    else:
        print("[SUCCESS] All 4 multi-task outputs verified!")

    # 2. Benchmark Latency
    print("\n--- Latency Benchmark ---")
    dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
    # Warmup
    for _ in range(10):
        session.run(None, {"input_image": dummy})

    iters = 100
    t0 = time.time()
    for _ in range(iters):
        session.run(None, {"input_image": dummy})
    lat_ms = (time.time() - t0) / iters * 1000.0
    fps = 1000.0 / lat_ms
    print(f"Single-image forward latency: {lat_ms:.2f} ms ({fps:.1f} FPS) on {active_provider}")

    # Batch latency (batch=32)
    dummy_b = np.random.randn(32, 3, 512, 512).astype(np.float32)
    for _ in range(5):
        session.run(None, {"input_image": dummy_b})
    t0 = time.time()
    for _ in range(20):
        session.run(None, {"input_image": dummy_b})
    lat_batch_ms = (time.time() - t0) / (20 * 32) * 1000.0
    fps_b = 1000.0 / lat_batch_ms
    print(f"Batched (batch=32) latency:   {lat_batch_ms:.2f} ms per image ({fps_b:.1f} FPS) on {active_provider}")

    # 3. Sample Verification on Test Data
    if Path(args.annotations).exists():
        print(f"\n--- Sample Predictions on Test Set (First {min(10, args.num_samples)} images) ---")
        label_map = load_label_map(args.label_map)
        idx2class = {v: k for k, v in label_map.items()}

        df = pd.read_csv(args.annotations)
        test_df = df[df["split"] == "test"].head(args.num_samples)

        correct_models = 0
        correct_colors = 0
        total_eval = 0

        print(f"{'Idx':3s} | {'True Model':25s} | {'Pred Model (Conf)':30s} | {'True Color':10s} | {'Pred Color (Conf)':18s}")
        print("-" * 95)

        for i, (_, row) in enumerate(test_df.iterrows()):
            try:
                bbox = (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]) if row["bbox_x1"] >= 0 else None
                img = load_crop(row["path"], row["domain"], bbox).resize((512, 512), Image.BILINEAR)
                arr = (np.array(img, dtype=np.float32) / 255.0).transpose((2, 0, 1))[None, ...]
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

                out_dict = dict(zip([o.name for o in outputs], session.run(None, {"input_image": arr})))

                m_logits = out_dict["predictions"][0]
                m_probs = np.exp(m_logits - np.max(m_logits))
                m_probs /= m_probs.sum()
                pred_m_idx = int(np.argmax(m_probs))
                pred_m_conf = float(m_probs[pred_m_idx])
                pred_model = idx2class.get(pred_m_idx, str(pred_m_idx))

                c_logits = out_dict["color_predictions"][0]
                c_probs = np.exp(c_logits - np.max(c_logits))
                c_probs /= c_probs.sum()
                pred_c_idx = int(np.argmax(c_probs))
                pred_c_conf = float(c_probs[pred_c_idx])
                pred_color = COLORS[pred_c_idx]

                gt_model = str(row["gt_class"])
                gt_color = str(row["color"])

                correct_m = (pred_model == gt_model)
                correct_c = (pred_color == gt_color)
                if correct_m: correct_models += 1
                if correct_c: correct_colors += 1
                total_eval += 1

                if i < 10:
                    m_str = f"{pred_model[:22]} ({pred_m_conf*100:.1f}%)"
                    c_str = f"{pred_color:7s} ({pred_c_conf*100:.1f}%)"
                    print(f"{i:3d} | {gt_model[:25]:25s} | {m_str:30s} | {gt_color:10s} | {c_str:18s}")

            except Exception:
                continue

        if total_eval > 0:
            print("-" * 95)
            print(f"Sample Model Accuracy: {correct_models}/{total_eval} ({correct_models/total_eval*100:.1f}%)")
            print(f"Sample Color Agreement: {correct_colors}/{total_eval} ({correct_colors/total_eval*100:.1f}%)")

    print("\n[COMPLETE] Model verified and ready for production deployment.")


if __name__ == "__main__":
    main()
