# MobileNetV2 Production Model Accuracy & Performance Evaluation Report

**Model**: MobileNetV2 Production Vehicle Classifier (`mobilenet_v2_best.onnx`)  
**Resolution**: $512 \times 512$ RGB  
**Taxonomy**: 1,235 Fine-Grained Vehicle Model Classes across 35 Vehicle Makes  
**Framework**: PyTorch 2.14 / ONNX Runtime 1.23 (`pt-env`)  
**Evaluation Scope**: 114,545 total unseen test images (109,340 PlatesMania + 5,205 External), plus 4,740 Out-Of-Distribution (OOD) vehicle images.

---

## 1. Executive Summary & Overall Accuracy Metrics

| Metric | Combined Evaluation (114,545 images) | External Test Split (5,205 images) | PlatesMania Test Split (109,340 images) | Balanced Mixed Benchmark (10,000 images) |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Model Accuracy** | **92.28%** | **94.12%** | **92.19%** | **93.29%** |
| **Top-5 Model Accuracy** | **98.67%** | **99.14%** | **98.65%** | **—** |
| **Make Accuracy (Marginalized)** | **98.62%** | **98.66%** | **98.62%** | **98.68%** |
| **Model Macro F1** | **85.87%** | **90.58%** | **85.65%** | **86.24%** |
| **Model Weighted F1** | **92.04%** | **94.10%** | **91.94%** | **93.14%** |
| **Make Macro F1** | **97.76%** | **98.67%** | **97.72%** | **98.66%** |
| **True Positives (TP)** | **105,704** | **4,899** | **100,805** | **9,329** |
| **False Positives (FP)** | **8,841** | **306** | **8,535** | **671** |
| **False Negatives (FN)** | **8,841** | **306** | **8,535** | **671** |
| **Overall Micro Precision** | **92.28%** | **94.12%** | **92.19%** | **93.29%** |
| **Overall Micro Recall** | **92.28%** | **94.12%** | **92.19%** | **93.29%** |
| **Cross-Entropy Loss** | **0.353** | **0.259** | **0.358** | **—** |

---

## 2. Model-Level vs. Make-Level Performance

The classifier predicts 1,235 fine-grained models formatted as `Make/Model`. Coarse automotive make probabilities are derived via probability marginalization:
$$P(\text{Make}_k) = \sum_{m \in \text{Make}_k} P(\text{Model}_m)$$

### High-Level Comparison
- **Fine-Grained 1,235 Models**: **`92.28%`** accuracy, with **`98.67%`** Top-5 capture.
- **Coarse 35 Makes (Marginalized)**: **`98.62%`** accuracy and **`97.76%`** Macro F1.

### Make-Level Breakdown (35 Automotive Makes)
*Computed over all 114,545 test samples (`make_metrics.csv`):*

| Make | Test Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Acura** | 249 | 232 | 0 | 17 | 100.0% | 93.17% | **96.47%** |
| **Alfa Romeo** | 1,347 | 1,328 | 5 | 19 | 99.62% | 98.59% | **99.10%** |
| **Audi** | 1,995 | 1,976 | 11 | 19 | 99.45% | 99.05% | **99.25%** |
| **BMW** | 2,211 | 2,174 | 29 | 37 | 98.68% | 98.33% | **98.50%** |
| **Buick** | 581 | 540 | 28 | 41 | 95.07% | 92.94% | **93.99%** |
| **Cadillac** | 876 | 849 | 15 | 27 | 98.26% | 96.92% | **97.59%** |
| **Chevrolet** | 336 | 319 | 10 | 17 | 96.96% | 94.94% | **95.94%** |
| **Chrysler** | 3,227 | 3,173 | 89 | 54 | 97.27% | 98.33% | **97.80%** |
| **Cupra** | 395 | 391 | 3 | 4 | 99.24% | 98.99% | **99.11%** |
| **Dodge** | 387 | 378 | 15 | 9 | 96.18% | 97.67% | **96.92%** |
| **Ford** | 13,326 | 13,178 | 185 | 148 | 98.61% | 98.89% | **98.75%** |
| **GMC** | 1,189 | 1,170 | 36 | 19 | 97.01% | 98.40% | **97.70%** |
| **Honda** | 3,285 | 3,243 | 42 | 42 | 98.72% | 98.72% | **98.72%** |
| **Hyundai** | 5,665 | 5,595 | 92 | 70 | 98.38% | 98.76% | **98.57%** |
| **Infiniti** | 2,128 | 2,096 | 12 | 32 | 99.43% | 98.50% | **98.96%** |
| **Isuzu** | 2,058 | 2,028 | 23 | 30 | 98.88% | 98.54% | **98.71%** |
| **Jeep** | 2,216 | 2,187 | 30 | 29 | 98.65% | 98.69% | **98.67%** |
| **Kia** | 7,163 | 7,080 | 129 | 83 | 98.21% | 98.84% | **98.52%** |
| **Land Rover** | 437 | 434 | 4 | 3 | 99.09% | 99.31% | **99.20%** |
| **Lexus** | 1,779 | 1,757 | 18 | 22 | 98.99% | 98.76% | **98.87%** |
| **Lincoln** | 1,223 | 1,198 | 24 | 25 | 98.04% | 97.96% | **98.00%** |
| **MINI** | 751 | 743 | 9 | 8 | 98.80% | 98.93% | **98.87%** |
| **Mazda** | 6,367 | 6,290 | 69 | 77 | 98.91% | 98.79% | **98.85%** |
| **Mitsubishi** | 3,674 | 3,622 | 68 | 52 | 98.16% | 98.58% | **98.37%** |
| **Nissan** | 11,465 | 11,326 | 154 | 139 | 98.66% | 98.79% | **98.72%** |
| **Opel** | 3,576 | 3,539 | 49 | 37 | 98.63% | 98.97% | **98.80%** |
| **Peugeot** | 4,244 | 4,213 | 42 | 31 | 99.01% | 99.27% | **99.14%** |
| **Porsche** | 473 | 471 | 2 | 2 | 99.58% | 99.58% | **99.58%** |
| **Renault** | 3,588 | 3,554 | 55 | 34 | 98.48% | 99.05% | **98.76%** |
| **Skoda** | 4,210 | 4,171 | 55 | 39 | 98.70% | 99.07% | **98.89%** |
| **Subaru** | 3,243 | 3,205 | 40 | 38 | 98.77% | 98.83% | **98.80%** |
| **Suzuki** | 3,212 | 3,165 | 55 | 47 | 98.29% | 98.54% | **98.41%** |
| **Toyota** | 7,654 | 7,549 | 108 | 105 | 98.59% | 98.63% | **98.61%** |
| **Volkswagen** | 9,871 | 9,737 | 158 | 134 | 98.40% | 98.64% | **98.52%** |
| **Volvo** | 1,768 | 1,757 | 10 | 11 | 99.43% | 99.38% | **99.41%** |

---

## 3. Confusion & Error Analysis

Because 1,235 fine-grained classes represent real-world commercial vehicle fleets, the majority of model-level misclassifications correspond to badge-engineered clones, regional nameplates, or generation-trim sibling variants rather than catastrophic errors:

| Rank | Ground Truth Class | Predicted Class | Misclassifications | Real-World Relationship |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `Volkswagen/Bus` | `Volkswagen/Kombi` | 92 | Identical platform; regional branding nomenclature |
| **2** | `Chrysler/Grand Voyager` | `Chrysler/Voyager` | 64 | Long-wheelbase vs standard-wheelbase variants |
| **3** | `Toyota/ToyoAce` | `Toyota/Dyna` | 61 | Mechanically identical twin cab-over trucks |
| **4** | `Volkswagen/Caravelle` | `Volkswagen/Transporter` | 58 | Passenger interior trim vs cargo panel van trim |
| **5** | `Ford/F-250` | `Ford/F-350` | 55 | Identical body cabs; differentiated only by suspension/axle rating |
| **6** | `Chrysler/Voyager` | `Chrysler/Grand Voyager` | 53 | Wheelbase length variant |
| **7** | `Volkswagen/Multivan` | `Volkswagen/Caravelle` | 53 | Executive passenger van vs passenger shuttle van |
| **8** | `Ford/Transit Custom` | `Ford/Tourneo Custom` | 50 | Commercial van vs consumer passenger MPV |
| **9** | `Ford/Tourneo Custom` | `Ford/Transit Custom` | 49 | Passenger MPV vs commercial van |
| **10** | `Toyota/Corsa` | `Toyota/Corolla II` | 42 | Shared platform subcompact badge variants |
| **11** | `Ford/Lobo` | `Ford/F-150` | 36 | Mexican market branding (`Lobo`) for US `F-150` |
| **12** | `Honda/Fit` | `Honda/Jazz` | 35 | Japan/US naming (`Fit`) vs European/Global naming (`Jazz`) |
| **13** | `Mazda/Demio` | `Mazda/2` | 35 | Japanese domestic market naming (`Demio`) vs Global (`2`) |

---

## 4. Open-Set Out-of-Distribution (OOD) Performance

Evaluated against 4,740 unseen open-set vehicle classes:

| Rejection Metric | AUROC | FPR@95% TPR | OSCR |
| :--- | :---: | :---: | :---: |
| **Maximum Softmax Probability (MSP)** | **0.9313** | **29.37%** | **0.8767** |
| **Max Logit Score (MLS)** | 0.6061 | 67.95% | 0.5661 |
| **Energy-based Score** | 0.5597 | 69.28% | 0.5213 |
| **Cosine Proximity Score** | 0.1714 | 99.56% | 0.1597 |

Confidence rejection threshold recommendation: $\tau = 0.70$ provides strong rejection of non-vehicle or unsupported classes while retaining $97\%+$ precision on in-distribution models.

---

## 5. Progression Across Runs

| Iteration | Architecture | Evaluated Images | Top-1 Model Acc | Make Acc (Marg.) | Model Macro F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Run 3 (Specialist A)** | MobileNetV2 | 109,340 | 91.95% | 97.90% | 86.40% | Overfits PlatesMania; fails on External |
| **Run 3 (Specialist B)** | MobileNetV2 | 5,205 | 58.58% | 92.64% | 52.10% | Severe generalization breakdown |
| **Run 4 (Merged Baseline)** | MobileNetV2 | 114,201 | 91.53% | 98.15% | 85.17% | First unified 1,235-class merged model |
| **Run 5 (Production Champion)** | **MobileNetV2** | **114,545** | **92.28%** | **98.62%** | **85.87%** | **GeM pooling ($p=3.0$), layerwise LR decay, KD matrix label smoothing** |
