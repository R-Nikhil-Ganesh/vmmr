# MobileNetV2 Production Vehicle Recognition Package

This package provides the standalone production **MobileNetV2** vehicle recognition model, label maps, inference engine, and comprehensive accuracy report.

---

## 1. Directory Structure

```text
code/current/mobilenet/sample_inference/
├── models/
│   ├── mobilenet_v2_best.onnx   # Production ONNX model checkpoint (20.87 MB)
│   ├── label_map.json           # 1,235-class mapping dictionary (class_to_idx, idx_to_class)
│   └── classes.txt              # Plaintext list of 1,235 vehicle classes (one per line)
├── reports/
│   ├── ACCURACY_REPORT.md       # Detailed accuracy report (F1, FN, FP, precision, recall)
│   ├── accuracy_report.json     # Machine-readable summary metrics
│   ├── per_class_metrics.csv    # 1,235 rows: class, support, TP, FP, FN, precision, recall, F1
│   └── make_metrics.csv         # 35 makes: make, support, TP, FP, FN, precision, recall, F1
├── infer.py                     # Self-sufficient standalone inference script (CLI + Python API)
├── ACCURACY_REPORT.md           # Root symlink / duplicate of evaluation report
└── README.md                    # Package documentation
```

---

## 2. Accuracy & Evaluation Summary

Evaluated on **114,545 unseen test images** (109,340 PlatesMania + 5,205 External), plus 4,740 Out-Of-Distribution (OOD) samples:

| Metric | Combined Evaluation (114,545 images) | External Split (5,205 images) | PlatesMania Split (109,340 images) |
| :--- | :---: | :---: | :---: |
| **Top-1 Fine-Grained Accuracy** | **92.28%** | **94.12%** | **92.19%** |
| **Top-5 Fine-Grained Accuracy** | **98.67%** | **99.14%** | **98.65%** |
| **Make Accuracy (Marginalized)** | **98.62%** | **98.66%** | **98.62%** |
| **Model Macro F1** | **85.87%** | **90.58%** | **85.65%** |
| **Model Weighted F1** | **92.04%** | **94.10%** | **91.94%** |
| **Make Macro F1** | **97.76%** | **98.67%** | **97.72%** |
| **True Positives (TP)** | **105,704** | **4,899** | **100,805** |
| **False Positives (FP)** | **8,841** | **306** | **8,535** |
| **False Negatives (FN)** | **8,841** | **306** | **8,535** |
| **Overall Micro Precision** | **92.28%** | **94.12%** | **92.19%** |
| **Overall Micro Recall** | **92.28%** | **94.12%** | **92.19%** |
| **Cross-Entropy Loss** | **0.353** | **0.259** | **0.358** |
| **Open-Set OOD AUROC (MSP)** | **93.13%** | — | **FPR@95TPR: 29.37%** |

---

## 3. Quick Start: Running Inference

### CLI Options

1. **Single Image**:
   ```bash
   python infer.py --image /path/to/vehicle.jpg
   ```

2. **CPU Execution**:
   ```bash
   python infer.py --image /path/to/vehicle.jpg --device cpu
   ```

3. **Batch Inference on Directory**:
   ```bash
   python infer.py --image-dir /path/to/image_folder/
   ```

4. **Export CSV, JSON, and Annotated Images**:
   ```bash
   python infer.py \
     --image-dir /path/to/image_folder/ \
     --save-csv output/predictions.csv \
     --save-json output/predictions.json \
     --save-annotated output/annotated/
   ```

---

## 4. Python API Usage

```python
from infer import MobileNetClassifier

# 1. Initialize classifier (auto-detects CUDA / CPU)
clf = MobileNetClassifier()

# 2. Predict on an image path or PIL.Image
result = clf.predict_image("/path/to/vehicle.jpg", top_k=5)

print(f"Top Model:        {result['top1_model']}")
print(f"Model Confidence: {result['top1_model_confidence']:.2%}")
print(f"Top Make:         {result['top1_make']}")
print(f"Make Confidence:  {result['top1_make_confidence']:.2%}")
print(f"Inference Time:   {result['latency_ms']:.1f} ms")

# 3. Batch prediction
batch = clf.predict_batch(["/path/to/car1.jpg", "/path/to/car2.jpg"])
```
