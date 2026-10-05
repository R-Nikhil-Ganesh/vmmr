# Self-Sufficient Sample Inference Package

This directory is an entirely self-contained, standalone inference testing suite for both the **35-make** and **32-make** EfficientNet-B0 vehicle classifiers.

It contains:
- Best ONNX model weights and label maps for both models (`models/`)
- A curated test batch of 15 sample vehicle images across diverse makes (`images/`)
- A self-contained inference script (`infer.py`) with zero dependencies on other codebase files
- A one-click automated test runner script (`test_all_samples.sh`)

---

## Directory Layout

```text
sample_inference/
├── models/
│   ├── efficientnet_b0_35makes_best.onnx   # Best 35-make ONNX model (8.05 MB)
│   ├── label_map_35makes.json              # 35-make class mapping dictionary
│   ├── efficientnet_b0_32makes_best.onnx   # Best 32-make ONNX model (8.04 MB)
│   └── label_map_32makes.json              # 32-make class mapping dictionary
│
├── images/                                 # Curated test vehicle images
│   ├── 01_audi.jpg
│   ├── 02_bmw.jpg
│   ├── 03_chevrolet.jpg
│   ├── 04_ford.jpg
│   ├── 05_gmc.jpg
│   ├── 06_honda.jpg
│   ├── 07_hyundai.jpg
│   ├── 08_infiniti.jpg
│   ├── 09_jeep.jpg
│   ├── 10_nissan.jpg
│   ├── 11_porsche.jpg
│   ├── 12_subaru.jpg
│   ├── 13_toyota.jpg
│   ├── 14_volkswagen.jpg
│   ├── 15_volvo.jpg
│   └── manifest.csv                        # Metadata and ground-truth makes
│
├── output/                                 # Prediction logs & annotated images
│   ├── predictions_35makes.csv
│   ├── predictions_35makes.json
│   ├── annotated_35makes/                  # Images with visual HUD prediction banner
│   ├── predictions_32makes.csv
│   └── annotated_32makes/
│
├── infer.py                                # Self-sufficient inference script
├── test_all_samples.sh                     # One-click test runner
└── README.md                               # This documentation
```

---

## Quick Start

### 1. One-Click Verification
Run the automated test runner to test both models on all sample images:
```bash
bash test_all_samples.sh
```

### 2. Run Inference on Default Image Batch
```bash
# Test with 35-make production model (default)
python infer.py

# Test with 32-make model
python infer.py --model 32
```

### 3. Test a Single Image
```bash
# Test an individual image
python infer.py --image images/11_porsche.jpg

# Save annotated image with confidence HUD banner
python infer.py --image images/11_porsche.jpg --save-annotated output/
```

### 4. Custom Exports
```bash
python infer.py \
  --model 35 \
  --save-csv output/my_predictions.csv \
  --save-json output/my_predictions.json \
  --save-annotated output/annotated_images/
```

---

## Python API Usage

The script can be imported directly into Python scripts or Jupyter notebooks:

```python
from infer import SampleClassifier

# 1. Initialize classifier (choose "35" or "32")
clf = SampleClassifier(model_type="35")

# 2. Predict on a single image
result = clf.predict_image("images/02_bmw.jpg", top_k=3)
print(f"Top Make:   {result['top1_make']}")
print(f"Confidence: {result['top1_confidence']:.2%}")
print(f"Latency:    {result['latency_ms']:.1f} ms")

# 3. Predict on a batch of paths
batch_results = clf.predict_batch(["images/01_audi.jpg", "images/09_jeep.jpg"])
```
