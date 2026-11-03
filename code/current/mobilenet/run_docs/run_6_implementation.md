# Run 6 — Production Full-Taxonomy Upgrade: 1,416-Class Vehicle Recognition & Multi-Task Car Body Type Classification

## 1. Goal & Architectural Overview

Upgrade the production **Model C** vehicle recognition system from the 1,235-class base taxonomy to the **complete 1,416-class taxonomy** across **49 vehicle makes**, while adding direct, native **vehicle body type (car type)** recognition (SUV, Sedan, Hatchback, Coupe, Convertible, Wagon, Pickup, Van).

The resulting unified multi-task ONNX model will simultaneously output:

| Output Name | Tensor Shape | Description | Origin / Tap Point |
| :--- | :---: | :--- | :--- |
| `predictions` | `B × 1416` | Fine-grained Make/Model logits (49 makes, 1,416 models) | Make-gated classifier head |
| `body_type_predictions` | `B × 8` | Car body type logits (`sedan`, `suv`, `hatchback`, etc.) | Semantic `embeddings` (1,280-D) |
| `color_predictions` | `B × 13` | Vehicle paint color logits | Photometric `features[16]` (160-D) |
| `embeddings` | `B × 1280` | Latent vehicle feature representation | Penultimate layer after GeM pooling ($p=3.0$) |
| `class_maps` | `B × 1416 × 16 × 16` | Per-class spatial evidence heatmaps | $1\times 1$ convolution on `features[18]` |

---

## 2. Grounding Facts from the Codebase & Datasets

| Fact | Source / Verification | Architectural Effect |
| :--- | :--- | :--- |
| **Complete 1,416-Class Split** | `splits_1235models/label_map.json` (1,416 classes, 49 makes) | Run 5 restricted training to 1,235 classes (`--taxonomy base`), dropping 22,651 External train rows. Run 6 trains on all 1,416 classes (`--taxonomy auto`), utilizing the entire dataset. |
| **External Training Data Scale** | `splits_1235models/train.csv` (47,805 total rows) | 22,651 rows are in classes 1235–1415. Incorporating them fully expands vehicle coverage across European, Asian, and American markets. |
| **Run 5 Champion Weights Available** | `run5/output_run5/models/mobilenet_v2_best.pt` | Run 5 fine-tuned all 19 backbone blocks with GeM pooling ($p=3.0$) and layerwise LR decay ($\gamma=0.85$). We warm-start from this checkpoint to accelerate convergence and retain feature fidelity. |
| **Ground-Truth Body Type Datasets** | Stanford Cars (16,185 images), BoxCars116k (26,853 images), CompCars (136k images) | High-quality image-level body type annotations exist for training and validating the body type head. |
| **Unambiguous Model-to-Body Mapping** | Catalog specifications across the 1,416 classes | Over 85% of fine-grained models possess an unambiguous primary body style (e.g. RAV4 $\to$ SUV, F-150 $\to$ Pickup, Camry $\to$ Sedan, Odyssey $\to$ Van). This provides strong supervision across the ~800k PlatesMania images. |
| **Reference Production Color Head** | `color_distillation/output_mlp512x2/` (Deep MLP-512×2) | 77.43% accuracy on independent CompCars human GT. Spliced onto `features[16]` of the final ONNX model. |

---

## 3. Taxonomy Definitions

### 3.1. 8-Class Canonical Body Type Taxonomy

| Index | Name | Definition | Key Reference Models |
| :---: | :--- | :--- | :--- |
| `0` | `sedan` | 4-door / 2-door notchback vehicle with a distinct, enclosed three-box trunk | Toyota Camry, Honda Accord, BMW 3 Series Sedan, Audi A4, Mercedes C-Class |
| `1` | `suv` | Sport Utility Vehicle, Crossover, Compact SUV, 4x4, Off-Roader | Toyota RAV4, Honda CR-V, BMW X5, Audi Q5, Ford Explorer, Jeep Wrangler |
| `2` | `hatchback` | 2-box vehicle with upward-lifting rear hatch and short rear cargo overhang | VW Golf, Ford Focus Hatch, Toyota Yaris, Renault Clio, Peugeot 208 |
| `3` | `coupe` | 2-door fixed-roof sporty car with sloping fastback / notchback roofline | Ford Mustang, Porsche 911, Chevrolet Camaro, Nissan 370Z, BMW 4 Series Coupe |
| `4` | `convertible` | Open-top cabriolet, roadster, spider with retractable fabric or hard top | Mazda MX-5 Miata, Porsche Boxster, BMW Z4, Audi TT Roadster |
| `5` | `wagon` | Station wagon, estate, avant, touring, combi with extended roofline and large rear cargo | Volvo V60, BMW 3 Series Touring, Audi A4 Avant, Subaru Outback |
| `6` | `pickup` | Utility truck featuring an enclosed cab and an open rear cargo cargo bed | Ford F-150, Chevrolet Silverado, Ram 1500, Toyota Hilux, Toyota Tacoma |
| `7` | `van` | Minivan, passenger van, MPV, commercial cargo van, tall monovolume profile | Honda Odyssey, Toyota Sienna, Ford Transit, Mercedes Sprinter, VW Transporter |

### 3.2. Mapping & Normalization Utilities (`body_type/taxonomy.py`)

- **CompCars**: Mapped from `attributes.txt` integer codes (1: MPV $\to$ `van`, 2: SUV $\to$ `suv`, 3: sedan $\to$ `sedan`, 4: hatchback $\to$ `hatchback`, 5: minibus $\to$ `van`, 6: fastback $\to$ `coupe`, 7: estate $\to$ `wagon`, 8: pickup $\to$ `pickup`, 9: hardtop convertible $\to$ `convertible`, 10: sports $\to$ `coupe`, 11: crossover $\to$ `suv`, 12: convertible $\to$ `convertible`).
- **BoxCars116k**: Mapped from `types_mapping` (`combi` $\to$ `wagon`, `mpv`/`van` $\to$ `van`, `sedan`, `suv`, `hatchback`).
- **Stanford Cars**: Extracted directly from class strings (`Sedan`, `SUV`, `Hatchback`, `Coupe`, `Convertible`, `Wagon`, `Cab`/`Pickup`, `Van`).
- **1,416 Make/Model Table**: Audited model lookup mapping every class $c \in \{0 \dots 1415\}$ to its primary body type.

---

## 4. Architecture & Mathematical Formulation

```mermaid
flowchart TD
    subgraph Backbone["MobileNetV2 Fine-Tuned Backbone (Run 6)"]
        IN["Input Image: B x 3 x 512 x 512"] --> F0["features[0:15]"]
        F0 --> F16["features[16]: B x 160 x 16 x 16"]
        F16 --> F18["features[18]: B x 1280 x 16 x 16"]
        F18 --> GeM["GeM Pooling (p=3.0)"]
        GeM --> EMB["embeddings: B x 1280"]
    end

    subgraph MultiTaskHeads["Multi-Task Heads"]
        F16 --> CH["Color Head (Deep MLP-512x2)"]
        CH --> C_OUT["color_predictions: B x 13"]

        EMB --> MG["Make Head (Linear 1280 -> 49)"]
        EMB --> CLS["Classifier (Linear 1280 -> 1416)"]
        MG -.->|Beta * Log P(make)| CLS
        CLS --> M_OUT["predictions: B x 1416"]

        EMB --> BH["Body Type Head (Linear 1280 -> 8)"]
        BH --> B_OUT["body_type_predictions: B x 8"]

        F18 --> CMAP["1x1 Conv (1280 -> 1416)"]
        CMAP --> MAP_OUT["class_maps: B x 1416 x 16 x 16"]
    end
```

### 4.1. Joint Multi-Task Objective

During training, each batch provides:
- Vehicle model label $y_{\text{model}} \in \{0 \dots 1415\}$
- Vehicle make label $y_{\text{make}} \in \{0 \dots 48\}$ (from marginalization projection)
- Vehicle body type label $y_{\text{body}} \in \{0 \dots 7\}$ (from class mapping + dataset ground truth)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{model}}(z_{\text{model}}, y_{\text{model}}) + w_{\text{make}} \cdot \mathcal{L}_{\text{make}}(z_{\text{make}}, y_{\text{make}}) + w_{\text{body}} \cdot \mathcal{L}_{\text{body}}(z_{\text{body}}, y_{\text{body}})$$

where $w_{\text{make}} = 0.3$, $w_{\text{body}} = 0.3$, and $\mathcal{L}_{\text{model}}$ incorporates label smoothing ($\alpha = 0.1$).

### 4.2. Warm-Start Parameter Transfer from Run 5

To eliminate cold-start instability and maintain the high accuracy of the Run 5 model, weights are transferred via `load_warm_start_weights`:
1. `features.*` (Backbone 19 blocks): 100% copied from Run 5.
2. `gem_p` (GeM pooling parameter): Copied from Run 5.
3. `classifier.1.weight[0:1235, :]` and `classifier.1.bias[0:1235]`: Copied from Run 5. New classes ($1235 \dots 1415$) initialized with Xavier uniform.
4. `make_fc.weight` and `bias`: The 35 existing makes copied from Run 5; the 14 new makes initialized with Xavier uniform.
5. `body_fc` (Linear $1280 \to 8$): Initialized with Xavier uniform.

---

## 5. Detailed Implementation Plan by Component

### Component 0 — Body Type Taxonomy & Mapping Generator

#### [NEW] `code/current/mobilenet/body_type/taxonomy.py`
- Canonical definitions: `BODY_TYPES`, `BODY_TYPE_TO_IDX`, `IDX_TO_BODY_TYPE`.
- String parsing and alias normalization functions for external datasets.

#### [NEW] `code/current/mobilenet/body_type/build_body_type_map.py`
- Loads `splits_1235models/label_map.json` (1,416 classes).
- Maps all 1,416 classes to their body type using catalog definitions and verified keywords.
- Generates:
  - `code/current/mobilenet/body_type/data/body_type_map_1416.json`
  - `code/current/mobilenet/body_type/data/body_type_proj_1416.npy` (shape: `[1416, 8]`)

---

### Component 1 — Training Engine Upgrades

#### [MODIFY] `code/current/mobilenet/train_common.py`
- **`VMMRNet`**:
  - Add optional `body_proj: Optional[torch.Tensor] = None` and `body_head: bool = True`.
  - When enabled, instantiates `self.body_fc = nn.Linear(dim, 8)`.
  - Update `forward_train(x)` to return `(model_logits, make_logits, body_logits)`.
  - Update `forward_export(x)` to return `(model_logits, body_logits, embeddings, class_maps)`.
- **`TrainObjective`**:
  - Add `body_loss_weight: float = 0.3` and `class_to_body` mapping buffer.
  - Computes cross-entropy on `body_logits` when valid body labels are present.
- **`load_warm_start_weights(model, pt_path)`**:
  - Reconstructs state from Run 5 checkpoint, matching shapes and cleanly copying overlapping slices.

#### [NEW] `code/current/mobilenet/run6/train/train_mobilenet_v2_run6.py`
*(Or updated `train_mobilenet_v2_merged.py`)*:
- Sets default `--taxonomy auto` (1,416 classes).
- Accepts `--warm-start-pt` (pointing to Run 5 checkpoint).
- Accepts `--body-loss-weight` (default `0.3`).
- Accepts `--body-type-map` (pointing to `body_type_map_1416.json`).
- Outputs to `code/current/mobilenet/run6/output_run6/`.

#### [NEW] `code/current/mobilenet/bash_scripts/run_train_run6.sh`
- Tmux-wrapped runner executing:
  ```bash
  python train_mobilenet_v2_run6.py \
      --taxonomy auto \
      --unfreeze-layers 19 \
      --layer-lr-decay 0.85 \
      --pool gem \
      --make-gating per_class \
      --make-loss-weight 0.3 \
      --body-loss-weight 0.3 \
      --warm-start-pt code/current/mobilenet/run5/output_run5/models/mobilenet_v2_best.pt \
      --aug-strength strong \
      --crop-jitter 0.08 \
      --label-smoothing 0.1 \
      --epochs 15 \
      --save-pt \
      --output-dir code/current/mobilenet/run6/output_run6
  ```

---

### Component 2 — Multi-Task ONNX Splicing (5 Outputs)

#### [NEW] `code/current/mobilenet/body_type/export_multitask_5outputs.py`
- Loads `run6/output_run6/models/mobilenet_v2_best.onnx` (which contains `predictions`, `body_type_predictions`, `embeddings`, `class_maps`).
- Splices the reference Deep MLP-512×2 color head onto `features[16]`.
- Exports `run6/output_run6/models/mobilenet_v2_multitask_1416_5outputs.onnx`.

---

### Component 3 — Downstream Tooling & Verification

#### [MODIFY] `code/current/mobilenet/color_distillation/verify_multitask_onnx.py`
- Expects all 5 output tensors:
  `{"predictions", "color_predictions", "body_type_predictions", "embeddings", "class_maps"}`.
- Verifies ONNX Runtime graph execution and benchmarks single-image and batch latency.

#### [MODIFY] `infer_mobilenetv2_mmc.py`
- Parses `body_type_predictions` along with make, model, and color.
- Displays vehicle body type in console outputs, `--save-csv`, and visual overlay badges.

#### [NEW] `code/current/mobilenet/run_docs/run_6.md`
- Complete experiment report detailing:
  - 1,416-class test set accuracy on PlatesMania (109,340 images) and External (9,945 images).
  - 8-class body type test set accuracy, Macro F1, and confusion matrix.
  - Multi-task latency benchmarks on GPU and CPU.

---

## 6. Verification & Quality Gates

| Checkpoint | Target Metric / Acceptance Criteria | Tool / Command |
| :--- | :--- | :--- |
| **Warm-Start Verification** | Epoch 1 train loss starts at $\le 0.80$ (demonstrating smooth parameter transfer vs $>3.0$ from scratch) | `training.log` inspect |
| **PlatesMania Top-1 Accuracy** | $\ge 92.20\%$ across 109,340 test images | `run_diagnostics.sh` |
| **External Top-1 Accuracy** | $\ge 93.50\%$ across all 9,945 test images (covering all 1,416 classes) | `run_diagnostics.sh` |
| **Body Type Top-1 Accuracy** | $\ge 92.00\%$ Top-1, $\ge 88.00\%$ Macro F1 | `eval_body_type.py` |
| **ONNX Signature & Parity** | 5 outputs matching shapes; parity difference $< 1\times 10^{-4}$ | `verify_multitask_onnx.py` |
| **Inference Latency** | GPU latency $< 2.0$ ms / frame at batch=1; isolated body type overhead $< 0.03$ ms | `verify_multitask_onnx.py` |
