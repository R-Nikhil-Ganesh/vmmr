# MobileNetV2 Color Distillation

For complete architectural details, empirical layer probing benchmarks, mathematical formulations, and multi-task ONNX export specifications, see:

👉 [**Color Distillation Approach Documentation (`color_distillation_approach.md`)**](color_distillation_approach.md)  
👉 [**Multi-Task Color Distillation Report (`../run_docs/color_distillation_report.md`)**](../run_docs/color_distillation_report.md)

---

## Models & Performance Summary

| Model Tier | Architecture | Weights & Loss | Output Directory | Test Top-1 (`conf>=0.70`) | CompCars Human GT | GPU Latency |
|---|---|---|---|:---:|:---:|:---:|
| ⭐ **Reference Production** | **Deep MLP-512×2** | Unweighted, LS 0.0 | `output_mlp512x2/` | **`75.53%`** (Macro F1 62.9%) | **`77.43%`** *(98.7% of teacher)* | **1.11 ms** (~900 FPS) |
| Ablation 1 | MLP-128 | Unweighted, LS 0.0 | `output_unweighted_mlp128/` | `73.23%` (Macro F1 58.1%) | `74.89%` | 1.14 ms (~880 FPS) |
| Historical Baseline | MLP-128 | Class-weighted, LS 0.05 | `output/` | `68.00%` (val) | `71.08%` | 1.13 ms (~880 FPS) |
| *Teacher Reference* | *Qwen3-VL-8B* | *Zero-shot VLM* | *`annotation_runs/run5/`* | *—* | *`78.44%`* | *~27 ms (37 FPS)* |

> **Production Model Note**: The reference multi-task ONNX graph is **`output_mlp512x2/mobilenet_v2_multitask.onnx`** (tracked via DVC), which achieves 75.53% test accuracy and 77.43% on independent human ground truth while adding only 39 microseconds (<0.04 ms) over the 128-unit head.

---

## Quick Reference Commands

```bash
cd vmmr/code/current/mobilenet

# 1. Train Reference Deep MLP-512x2 head (Unweighted, LS 0.0)
"${PYTHON_BIN:-python3}" -u color_distillation/train_color_head.py \
  --head-type mlp_512x2 \
  --no-class-weights \
  --label-smoothing 0.0 \
  --batch-size 2048 \
  --skip-extract \
  --out-dir color_distillation/output_mlp512x2

# 2. Or run full Phase 3 color ablations script
bash bash_scripts/run_phase3_color_ablations.sh

# 3. Verify multi-task ONNX signature, latency, and numerical equivalence
python3 color_distillation/verify_multitask_onnx.py \
  --onnx-path color_distillation/output_mlp512x2/mobilenet_v2_multitask.onnx \
  --pt-head-path color_distillation/output_mlp512x2/color_head_best.pt
```
