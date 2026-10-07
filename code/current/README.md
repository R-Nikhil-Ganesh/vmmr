# Vehicle Make Classification - Experiments & Models

This directory contains the production and experimental model pipelines for fine-grained vehicle make recognition.

---

## Directory Layout

```text
code/current/
├── efficientnet/      # EfficientNet-B0 transfer learning pipelines, ONNX weights, and diagnostic suite
└── mobilenet/       # MobileNetV2 architecture experiments (in progress)
```

---

## Active Pipelines

### 1. EfficientNet-B0 (`efficientnet/`)
- **Benchmark Performance**: 94.18% Micro Accuracy, 93.96% Macro F1 across 35 vehicle makes.
- **Production Artifacts**: 8.45 MB ONNX Runtime binaries, sub-15 ms CPU inference latency.
- **Documentation**: See [`efficientnet/README.md`](efficientnet/README.md) and [`efficientnet/efficientnet_conclusions.md`](efficientnet/efficientnet_conclusions.md) for architectural trade-offs, multi-resolution scaling frontiers (224×224 to 720×720), and diagnostic Grad-CAM heatmaps.

### 2. MobileNetV2 (`mobilenet/`)
- Lightweight mobile-first architecture experiment exploring inverted residuals and linear bottlenecks for low-latency vehicle classification.
