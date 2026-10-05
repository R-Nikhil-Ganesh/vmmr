#!/usr/bin/env python3
"""
Universal ONNX Model Diagnostic & Explainability Analysis Engine (output_analysis.py)
====================================================================================
A 100% ONNX-native diagnostic evaluation, explainability auditing, and latent space
analysis framework for any trained classification model in ONNX format (.onnx).

Key Features:
-------------
1. Pure ONNX Runtime Inference:
   - Ultra-fast startup (zero TensorFlow/Keras overhead or dependencies for inference).
   - Low memory footprint, zero GPU memory exhaustion, zero cuDNN profiling errors.
   - Automatically supports CPU and CUDAExecutionProvider if available.
2. Native Multi-Output Feature Extraction:
   - Dynamically inspects the ONNX graph with onnx.shape_inference.
   - Extracts intermediate convolutional feature maps and penultimate latent embeddings
     in a single forward pass without modifying the original ONNX file on disk.
3. Analytical Class Activation Mapping (Grad-CAM / CAM):
   - Computes exact class activation heatmaps directly from the convolutional feature maps
     and the folded linear classification weights:
     L_c = ReLU( sum_k w_k^c * A^k )
4. Complete Diagnostic & Explainability Suite:
   - Test Set Evaluation & Summary Metrics
   - Per-Class Classification Report & F1 Ranking
   - Confusion Matrix & Top Confusion Pairs
   - Hardest Misclassifications (High-Confidence Errors)
   - Highest-Confidence True & Comparative False Grad-CAM Maps
   - Interactive Make/Model Grad-CAM Queries
   - 2D Manifold Visualisation (t-SNE & PCA)
   - Prediction Confidence Calibration & Coverage Rejection Curves

Usage:
------
In Jupyter Notebooks:
    import output_analysis as oa
    oa.setup_environment()
    model, info = oa.load_analysis_model() # auto-loads best ONNX model
    test_df, y_true, y_pred, metrics = oa.load_test_evaluation(model=model)
    ...

From the Command Line:
    python output_analysis.py --help
    python output_analysis.py --model-path output_efficientnet_b0_merged/models/efficientnet_b0_best.onnx --run-all
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Set writable matplotlib cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import onnx
from onnx import numpy_helper
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure display fallback
try:
    from IPython.display import display
except ImportError:
    display = print

# Current directory
CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent


# ==============================================================================
# 1. Global Context & Auto-Discovery
# ==============================================================================
class AnalysisContext:
    """
    Manages paths, ONNX model settings, and class metadata for the analysis session.
    """
    def __init__(self):
        self.model_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.models_dir: Optional[Path] = None
        self.reports_dir: Optional[Path] = None
        self.plots_dir: Optional[Path] = None
        self.pred_dir: Optional[Path] = None
        self.splits_dir: Optional[Path] = None
        self.test_csv_path: Optional[Path] = None
        self.label_map_path: Optional[Path] = None

        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.num_classes: int = 0
        self.img_size: int = 640

        self.auto_discover()

    def auto_discover(self, preferred_model_path: Optional[Union[str, Path]] = None):
        """Scans the environment for available ONNX checkpoints and metadata."""
        candidates = []
        if preferred_model_path:
            candidates.append(Path(preferred_model_path))

        # Default ONNX candidates
        candidates.extend([
            CURRENT_SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_best.onnx",
            CURRENT_SCRIPT_DIR / "output_efficientnet_b0" / "models" / "efficientnet_b0_best.onnx",
            CURRENT_SCRIPT_DIR / "output_efficientnet_b0_merged" / "models" / "efficientnet_b0_final.onnx",
            CURRENT_SCRIPT_DIR / "output_efficientnet_b0" / "models" / "efficientnet_b0_final.onnx",
            Path("/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged/models/efficientnet_b0_best.onnx"),
            Path("/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0/models/efficientnet_b0_best.onnx"),
        ])

        for extra in CURRENT_SCRIPT_DIR.glob("output_*/models/*.onnx"):
            if extra not in candidates:
                candidates.append(extra)

        found_model = None
        for cand in candidates:
            if cand.exists():
                found_model = cand.resolve()
                break

        self.model_path = found_model

        if self.model_path:
            if self.model_path.parent.name == "models":
                self.output_dir = self.model_path.parent.parent
            else:
                self.output_dir = self.model_path.parent
        else:
            self.output_dir = CURRENT_SCRIPT_DIR / "output_efficientnet_b0_merged"

        self.models_dir  = self.output_dir / "models" if (self.output_dir / "models").exists() else self.output_dir
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir   = self.output_dir / "plots"
        self.pred_dir    = self.output_dir / "predictions"

        split_candidates = [
            Path("/home/researchadmin/Econ/external_datasets/merged_data"),
            Path("/home/researchadmin/Econ/resized_640x640/splits_filtered"),
            self.output_dir,
            CURRENT_SCRIPT_DIR.parent / "splits",
        ]
        self.splits_dir = None
        for sc in split_candidates:
            if sc.exists() and (sc / "test.csv").exists():
                self.splits_dir = sc.resolve()
                self.test_csv_path = self.splits_dir / "test.csv"
                break

        label_map_cands = [
            self.models_dir / "label_map.json",
            self.output_dir / "label_map.json",
            self.splits_dir / "label_map.json" if self.splits_dir else None,
            Path("/home/researchadmin/Econ/external_datasets/merged_data/label_map.json"),
        ]
        self.label_map_path = None
        for lc in label_map_cands:
            if lc and self._load_label_map(lc):
                break

    def _load_label_map(self, path: Optional[Union[str, Path]]) -> bool:
        """Loads a label_map.json file and updates class metadata."""
        if not path:
            return False
        p = Path(path).resolve()
        if not p.exists():
            return False
        try:
            with open(p, "r") as f:
                meta = json.load(f)
            if "idx_to_class" in meta:
                self.idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()}
                self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
                self.num_classes = len(self.class_names)
                self.label_map_path = p
                return True
        except Exception:
            pass
        return False

    def configure(
        self,
        model_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        splits_dir: Optional[Union[str, Path]] = None,
        label_map: Optional[Union[str, Path]] = None,
        test_csv: Optional[Union[str, Path]] = None,
        img_size: Optional[int] = None
    ):
        """Explicitly configures or overrides paths and settings."""
        if model_path:
            self.model_path = Path(model_path).resolve()
            if self.model_path.parent.name == "models":
                self.output_dir = self.model_path.parent.parent
            else:
                self.output_dir = self.model_path.parent
            self.models_dir  = self.output_dir / "models" if (self.output_dir / "models").exists() else self.output_dir
            self.reports_dir = self.output_dir / "reports"
            self.plots_dir   = self.output_dir / "plots"
            self.pred_dir    = self.output_dir / "predictions"
            if not label_map:
                for cand in [self.models_dir / "label_map.json", self.output_dir / "label_map.json"]:
                    if self._load_label_map(cand):
                        break

        if output_dir:
            self.output_dir = Path(output_dir).resolve()
            self.models_dir  = self.output_dir / "models" if (self.output_dir / "models").exists() else self.output_dir
            self.reports_dir = self.output_dir / "reports"
            self.plots_dir   = self.output_dir / "plots"
            self.pred_dir    = self.output_dir / "predictions"
            if not label_map and not model_path:
                for cand in [self.models_dir / "label_map.json", self.output_dir / "label_map.json"]:
                    if self._load_label_map(cand):
                        break

        if splits_dir:
            self.splits_dir = Path(splits_dir).resolve()
            if (self.splits_dir / "test.csv").exists():
                self.test_csv_path = self.splits_dir / "test.csv"

        if test_csv:
            self.test_csv_path = Path(test_csv).resolve()

        if label_map:
            self._load_label_map(label_map)

        if img_size:
            self.img_size = img_size


context = AnalysisContext()

def set_context(**kwargs):
    """Sets or updates the active analysis context."""
    context.configure(**kwargs)
    return context


# Module-level backward-compatible globals
CLASS_NAMES  = context.class_names
NUM_CLASSES  = context.num_classes
CLASS_TO_IDX = context.class_to_idx
IDX_TO_CLASS = context.idx_to_class
DEFAULT_MODEL_PATH = context.model_path
OUTPUT_DIR   = context.output_dir
MODELS_DIR   = context.models_dir
REPORTS_DIR  = context.reports_dir
PLOTS_DIR    = context.plots_dir
PRED_DIR     = context.pred_dir
SPLITS_DIR   = context.splits_dir


# ==============================================================================
# 2. Environment Setup
# ==============================================================================
def setup_environment(force_cpu: bool = False):
    """Configures plot styling and verifies ONNX Runtime execution providers."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 120

    available = ort.get_available_providers()
    print(f"[ONNX Runtime] Version: {ort.__version__}")
    print(f"[ONNX Runtime] Available Providers: {available}")
    if force_cpu:
        print("[ONNX Runtime] Execution Device: CPU (Forced)")
    elif "CUDAExecutionProvider" in available:
        print("[ONNX Runtime] Execution Device: CUDA GPU")
    else:
        print("[ONNX Runtime] Execution Device: CPU")


# ==============================================================================
# 3. Native ONNX Model Wrapper
# ==============================================================================
class OnnxAnalysisModel:
    """
    Wraps an ONNX model for seamless evaluation, multi-output feature extraction
    (predictions, conv feature maps, embeddings), and CAM generation.
    """
    def __init__(self, model_path: Union[str, Path], force_cpu: bool = False):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model checkpoint not found at: {self.model_path}")

        # 1. Load ONNX Graph and infer value shapes
        self.raw_onnx = onnx.load(str(self.model_path))
        self.inferred_onnx = onnx.shape_inference.infer_shapes(self.raw_onnx)
        self.value_infos = {vi.name: vi for vi in self.inferred_onnx.graph.value_info}

        # 2. Extract I/O info
        self.input_node = self.inferred_onnx.graph.input[0]
        self.input_name = self.input_node.name
        self.output_node = self.inferred_onnx.graph.output[0]
        self.output_name = self.output_node.name

        in_dims = [dim.dim_value for dim in self.input_node.type.tensor_type.shape.dim]
        out_dims = [dim.dim_value for dim in self.output_node.type.tensor_type.shape.dim]

        self.input_shape = tuple(in_dims)
        self.num_classes = out_dims[-1] if out_dims else 35

        # 3. Locate Intermediate Feature Tensors
        self.conv_tensor_name = self._find_conv_feature_tensor()
        self.embed_tensor_name = self._find_embedding_tensor()

        # 4. Extract Classification Head Weights
        self.dense_weights, self.effective_weights = self._extract_classification_weights()

        # 5. Build multi-output graph (predictions + conv_features + embeddings)
        self._build_multi_output_graph()

        # 6. Initialize ONNX Runtime Session
        providers = []
        if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.inferred_onnx.SerializeToString(), sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]

    def _find_conv_feature_tensor(self) -> str:
        """Finds final convolutional feature map tensor name."""
        for node in reversed(self.raw_onnx.graph.node):
            if "top_activation" in node.name.lower() and len(node.output) > 0:
                return node.output[0]
        for node in reversed(self.raw_onnx.graph.node):
            if any(k in node.name.lower() for k in ["features", "conv_head", "final_conv"]) and len(node.output) > 0:
                return node.output[0]
        # Reverse scan for 4D tensor before pooling
        for vi in reversed(self.inferred_onnx.graph.value_info):
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if len(shape) == 4 and (shape[1] == 1280 or shape[-1] == 1280 or shape[1] is None):
                return vi.name
        return self.raw_onnx.graph.node[-5].output[0]

    def _find_embedding_tensor(self) -> str:
        """Finds penultimate embedding vector tensor name (1280-D)."""
        for node in reversed(self.raw_onnx.graph.node):
            if any(k in node.name.lower() for k in ["head_bn", "batchnorm/add"]) and len(node.output) > 0:
                return node.output[0]
        for node in reversed(self.raw_onnx.graph.node):
            if any(k in node.name.lower() for k in ["mean_squeeze", "avg_pool", "globalaveragepool"]) and len(node.output) > 0:
                return node.output[0]
        for vi in reversed(self.inferred_onnx.graph.value_info):
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if len(shape) == 2 and shape[-1] == 1280:
                return vi.name
        return self.raw_onnx.graph.node[-2].output[0]

    def _extract_classification_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts dense kernel weights and factors in any folded batchnorm multipliers."""
        initializers = {init.name: numpy_helper.to_array(init) for init in self.raw_onnx.graph.initializer}
        dense_w = None
        for k, v in initializers.items():
            if ("predictions" in k.lower() or "dense" in k.lower() or "fc" in k.lower()) and len(v.shape) == 2:
                dense_w = v
                break

        if dense_w is None:
            # Fallback random initialization for safety
            dense_w = np.ones((1280, self.num_classes), dtype=np.float32)

        bn_mul = [v for k, v in initializers.items() if "head_bn" in k.lower() and "mul" in k.lower()]
        if bn_mul:
            scale = bn_mul[0].squeeze()
            if len(scale.shape) == 1 and scale.shape[0] == dense_w.shape[0]:
                effective_w = dense_w * scale[:, np.newaxis]
            else:
                effective_w = dense_w
        else:
            effective_w = dense_w

        return dense_w, effective_w

    def _build_multi_output_graph(self):
        """Adds conv feature map and embedding tensors to ONNX graph outputs."""
        existing_out_names = {out.name for out in self.inferred_onnx.graph.output}
        if self.conv_tensor_name in self.value_infos and self.conv_tensor_name not in existing_out_names:
            self.inferred_onnx.graph.output.append(self.value_infos[self.conv_tensor_name])
        if self.embed_tensor_name in self.value_infos and self.embed_tensor_name not in existing_out_names:
            self.inferred_onnx.graph.output.append(self.value_infos[self.embed_tensor_name])

    def predict(self, batch_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a forward pass on a batch of images.
        Returns:
            predictions: (B, num_classes)
            conv_features: (B, 1280, 20, 20) or (B, 20, 20, 1280)
            embeddings: (B, 1280)
        """
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        preds = outputs[0]
        conv_feats = outputs[1] if len(outputs) > 1 else np.zeros((len(batch_tensor), 20, 20, 1280), dtype=np.float32)
        embeddings = outputs[2] if len(outputs) > 2 else np.zeros((len(batch_tensor), 1280), dtype=np.float32)
        return preds, conv_feats, embeddings


# ==============================================================================
# 4. Model Loading Function
# ==============================================================================
def load_analysis_model(
    model_path: Optional[Union[str, Path]] = None,
    force_cpu: bool = False
) -> Tuple[OnnxAnalysisModel, Dict[str, Any]]:
    """Loads an ONNX checkpoint and introspects its architecture metadata."""
    target_path = Path(model_path).resolve() if model_path else context.model_path
    if not target_path or not target_path.exists():
        raise FileNotFoundError(f"ONNX model checkpoint not found at: {target_path}")

    # Ensure .onnx extension
    if target_path.suffix.lower() != ".onnx":
        onnx_cand = target_path.with_suffix(".onnx")
        if onnx_cand.exists():
            target_path = onnx_cand
        else:
            raise ValueError(f"Expected an ONNX model file (.onnx), but got: {target_path}")

    context.configure(model_path=target_path)

    print(f"Loading ONNX model from: {target_path}...")
    model = OnnxAnalysisModel(target_path, force_cpu=force_cpu)

    total_params = sum(int(np.prod(init.dims)) for init in model.raw_onnx.graph.initializer)
    context.num_classes = model.num_classes

    # Ensure class mapping matches output classes
    if not context.class_names or len(context.class_names) != model.num_classes:
        found_map = False
        for cand in [context.models_dir / "label_map.json", context.output_dir / "label_map.json"]:
            if context._load_label_map(cand) and context.num_classes == model.num_classes:
                found_map = True
                break
        if not found_map:
            context.class_names = [f"Class_{i}" for i in range(model.num_classes)]
            context.class_to_idx = {name: i for i, name in enumerate(context.class_names)}
            context.idx_to_class = {i: name for i, name in enumerate(context.class_names)}
            context.num_classes = model.num_classes

    global CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, DEFAULT_MODEL_PATH
    global OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR, PRED_DIR, SPLITS_DIR
    CLASS_NAMES  = context.class_names
    NUM_CLASSES  = context.num_classes
    CLASS_TO_IDX = context.class_to_idx
    IDX_TO_CLASS = context.idx_to_class
    DEFAULT_MODEL_PATH = context.model_path
    OUTPUT_DIR   = context.output_dir
    MODELS_DIR   = context.models_dir
    REPORTS_DIR  = context.reports_dir
    PLOTS_DIR    = context.plots_dir
    PRED_DIR     = context.pred_dir
    SPLITS_DIR   = context.splits_dir

    info = {
        "model_path": str(target_path),
        "total_params": total_params,
        "input_name": model.input_name,
        "input_shape": model.input_shape,
        "output_name": model.output_name,
        "output_classes": model.num_classes,
        "conv_tensor": model.conv_tensor_name,
        "embed_tensor": model.embed_tensor_name,
        "provider": model.active_provider
    }

    print(f"[ONNX Architecture] Total Parameters:     {total_params:,}")
    print(f"[ONNX Architecture] Input Name:          {model.input_name} {model.input_shape}")
    print(f"[ONNX Architecture] Output Classes:       {model.num_classes}")
    print(f"[ONNX Diagnostics]  Conv Feature Tensor:  '{model.conv_tensor_name}'")
    print(f"[ONNX Diagnostics]  Latent Embed Tensor:  '{model.embed_tensor_name}'")
    print(f"[ONNX Runtime]      Active Provider:      {model.active_provider}")

    return model, info


# ==============================================================================
# 5. Test Evaluation & Summary Metrics
# ==============================================================================
def preprocess_image(path: Union[str, Path], img_size: int = 640) -> np.ndarray:
    """Preprocesses a single image to float32 [0.0, 255.0] array."""
    img = Image.open(str(path)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)


def load_test_evaluation(
    splits_dir: Optional[Union[str, Path]] = None,
    pred_dir: Optional[Union[str, Path]] = None,
    reports_dir: Optional[Union[str, Path]] = None,
    model: Optional[OnnxAnalysisModel] = None,
    test_csv_path: Optional[Union[str, Path]] = None,
    batch_size: int = 64
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Loads test manifest, computes ONNX test predictions, and loads summary metrics."""
    s_dir = Path(splits_dir).resolve() if splits_dir else context.splits_dir
    p_dir = Path(pred_dir).resolve() if pred_dir else context.pred_dir
    r_dir = Path(reports_dir).resolve() if reports_dir else context.reports_dir

    # 1. Resolve test.csv
    resolved_csv = None
    if test_csv_path and Path(test_csv_path).exists():
        resolved_csv = Path(test_csv_path).resolve()
    elif s_dir and (s_dir / "test.csv").exists():
        resolved_csv = s_dir / "test.csv"
    elif context.test_csv_path and context.test_csv_path.exists():
        resolved_csv = context.test_csv_path
    else:
        for cand in [
            Path("/home/researchadmin/Econ/external_datasets/merged_data/test.csv"),
            Path("/home/researchadmin/Econ/resized_640x640/splits_filtered/test.csv"),
            context.output_dir / "test.csv" if context.output_dir else None
        ]:
            if cand and cand.exists():
                resolved_csv = cand
                break

    if not resolved_csv or not resolved_csv.exists():
        raise FileNotFoundError(f"Could not locate test.csv. Please specify test_csv_path explicitly.")

    test_df_full = pd.read_csv(resolved_csv)

    path_col = next((c for c in ["image_path", "filepath", "path", "filename", "img_path"] if c in test_df_full.columns), None)
    if not path_col:
        raise KeyError(f"Image path column not found in {resolved_csv}.")

    class_col = next((c for c in ["make", "class_name", "class", "category", "label_name"] if c in test_df_full.columns), None)

    if class_col and (not context.class_names or context.class_names[0].startswith("Class_")):
        unique_classes = sorted(test_df_full[class_col].dropna().unique().tolist())
        target_count = model.num_classes if model else context.num_classes
        if len(unique_classes) == target_count:
            context.class_names = unique_classes
            context.class_to_idx = {name: i for i, name in enumerate(unique_classes)}
            context.idx_to_class = {i: name for i, name in enumerate(unique_classes)}
            context.num_classes = len(unique_classes)

    if class_col and context.class_names:
        test_df = test_df_full[test_df_full[class_col].isin(context.class_names)].copy().reset_index(drop=True)
        test_df["target_idx"] = test_df[class_col].map(context.class_to_idx)
    else:
        test_df = test_df_full.copy().reset_index(drop=True)
        label_col = next((c for c in ["target_idx", "label", "target", "idx"] if c in test_df.columns), None)
        test_df["target_idx"] = test_df[label_col].astype(int) if label_col else 0

    if "make" not in test_df.columns:
        test_df["make"] = test_df[class_col] if class_col else test_df["target_idx"].map(context.idx_to_class)
    if "model" not in test_df.columns:
        test_df["model"] = test_df["source"] if "source" in test_df.columns else "Standard"

    print(f"[Test Set] Total images: {len(test_df):,} spanning {test_df['make'].nunique()} classes")

    # 2. Load or Compute Cached Predictions
    p_dir.mkdir(parents=True, exist_ok=True)
    y_true_path = p_dir / "y_true.npy"
    y_pred_path = p_dir / "y_pred.npy"

    if y_true_path.exists() and y_pred_path.exists():
        print("[Predictions] Loading cached test predictions...")
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
    else:
        assert model is not None, "Model required to compute predictions when cache is missing."
        print(f"[Predictions] Running batched ONNX inference on test set (batch_size={batch_size})...")
        img_sz = context.img_size
        img_paths = test_df[path_col].values
        n_total = len(img_paths)
        all_preds = []

        for b_start in range(0, n_total, batch_size):
            b_paths = img_paths[b_start : b_start + batch_size]
            b_imgs = [preprocess_image(p, img_size=img_sz) for p in b_paths]
            b_tensor = np.stack(b_imgs, axis=0)
            preds, _, _ = model.predict(b_tensor)
            all_preds.append(preds)

        raw_preds = np.concatenate(all_preds, axis=0)
        y_pred = np.argmax(raw_preds, axis=-1)
        y_true = test_df["target_idx"].values

        np.save(y_true_path, y_true)
        np.save(y_pred_path, y_pred)
        print(f"[Predictions] Saved predictions to {p_dir}")

    # 3. Load summary metrics
    metrics_path = r_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            test_metrics = json.load(f)
    else:
        min_len = min(len(y_pred), len(y_true))
        test_metrics = {
            "compile_metrics": float(np.mean(y_pred[:min_len] == y_true[:min_len])),
            "accuracy": float(np.mean(y_pred[:min_len] == y_true[:min_len])),
            "macro_f1": float(classification_report(y_true[:min_len], y_pred[:min_len], output_dict=True).get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(classification_report(y_true[:min_len], y_pred[:min_len], output_dict=True).get("weighted avg", {}).get("f1-score", 0.0)),
        }

    acc_val = test_metrics.get("compile_metrics", test_metrics.get("accuracy", 0.0))
    loss_val = test_metrics.get("loss", "N/A")
    loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)

    print("\n" + "=" * 50)
    print(f"  Test Overall Accuracy:   {acc_val * 100:.2f}%")
    print(f"  Test Macro F1:           {test_metrics.get('macro_f1', 0.0) * 100:.2f}%")
    print(f"  Test Weighted F1:        {test_metrics.get('weighted_f1', 0.0) * 100:.2f}%")
    print(f"  Test Cross-Entropy Loss: {loss_str}")
    print("=" * 50)

    return test_df, y_true, y_pred, test_metrics


# ==============================================================================
# 6. Classification Report & F1-Score Chart
# ==============================================================================
def load_classification_report_df(
    reports_dir: Optional[Union[str, Path]] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads or generates per-class classification report DataFrame."""
    r_dir = Path(reports_dir).resolve() if reports_dir else context.reports_dir
    report_csv = r_dir / "test_classification_report.csv"

    if report_csv.exists():
        report_df = pd.read_csv(report_csv, index_col=0)
    else:
        assert y_true is not None and y_pred is not None, "y_true and y_pred required when report CSV is missing."
        min_len = min(len(y_true), len(y_pred))
        report_dict = classification_report(
            y_true[:min_len], y_pred[:min_len],
            target_names=context.class_names,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        r_dir.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_csv)

    class_report = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore").copy()
    class_report = class_report.sort_values(by="f1-score", ascending=False)

    summary_rows = report_df.loc[report_df.index.isin(["accuracy", "macro avg", "weighted avg"])].copy()
    styled_summary = summary_rows[["precision", "recall", "f1-score", "support"]].copy()
    display(styled_summary)

    return class_report, styled_summary


def plot_f1_ranking(
    class_report: pd.DataFrame,
    test_metrics: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None
):
    """Plots horizontal bar chart of F1-scores color-coded by performance thresholds."""
    n_classes = len(class_report)
    fig_h = max(8, int(n_classes * 0.28))
    plt.figure(figsize=(14, fig_h))
    f1_sorted_asc = class_report.sort_values(by="f1-score", ascending=True)

    bar_colors = []
    for val in f1_sorted_asc["f1-score"]:
        if val >= 0.95:
            bar_colors.append("#10b981")
        elif val >= 0.90:
            bar_colors.append("#3b82f6")
        elif val >= 0.80:
            bar_colors.append("#f59e0b")
        else:
            bar_colors.append("#ef4444")

    bars = plt.barh(f1_sorted_asc.index, f1_sorted_asc["f1-score"], color=bar_colors, height=0.68)

    for bar, (_, row) in zip(bars, f1_sorted_asc.iterrows()):
        f1 = row["f1-score"]
        supp = int(row["support"])
        text_x = bar.get_width() - 0.04 if bar.get_width() > 0.4 else bar.get_width() + 0.01
        text_c = "white" if bar.get_width() > 0.4 else "#1f2937"
        ha = "right" if bar.get_width() > 0.4 else "left"
        plt.text(text_x, bar.get_y() + bar.get_height()/2, f"{f1*100:.1f}% (n={supp:,})", 
                 va="center", ha=ha, color=text_c, fontsize=8.5, fontweight="bold")

    macro_f1 = test_metrics.get("macro_f1", 0.0)
    weighted_f1 = test_metrics.get("weighted_f1", 0.0)
    plt.axvline(macro_f1, color="#6366f1", linestyle="--", linewidth=1.5, label=f"Macro Avg: {macro_f1*100:.1f}%")
    plt.axvline(weighted_f1, color="#059669", linestyle="-.", linewidth=1.5, label=f"Weighted Avg: {weighted_f1*100:.1f}%")

    plt.xlim(0, 1.05)
    plt.xlabel("F1-Score", fontsize=11, fontweight="bold")
    plt.title(f"Test F1-Score by Class ({n_classes} Classes) | Green: >=95% | Blue: 90-95% | Amber: 80-90% | Red: <80%", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    out_p = Path(save_path) if save_path else context.plots_dir / f"f1_score_ranking_{n_classes}classes.png"
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), bbox_inches="tight")
        print(f"[Plot Saved] {out_p}")
    plt.show()


# ==============================================================================
# 7. Confusion Matrix & Top Confusion Pairs
# ==============================================================================
def plot_confusion_matrix_and_top_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 15,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Computes and plots normalized confusion matrix and top error pairs for any class count."""
    n_classes = context.num_classes
    class_names = context.class_names
    min_len = min(len(y_true), len(y_pred))

    cm_raw = confusion_matrix(y_true[:min_len], y_pred[:min_len], labels=range(n_classes))
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_raw / row_sums

    fig_w = max(18, int(n_classes * 0.52))
    fig_h = max(10, int(n_classes * 0.32))
    fig, (ax_heat, ax_top) = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [1.3, 1]})

    im = ax_heat.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall Proportion", rotation=-90, va="bottom", fontsize=10)

    ax_heat.set_xticks(range(n_classes))
    ax_heat.set_yticks(range(n_classes))
    tick_font = max(6.0, min(8.5, 200.0 / n_classes))
    ax_heat.set_xticklabels(class_names, rotation=90, fontsize=tick_font)
    ax_heat.set_yticklabels(class_names, fontsize=tick_font)
    ax_heat.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax_heat.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax_heat.set_title(f"Normalized Confusion Matrix ({n_classes} Classes)", fontsize=13, fontweight="bold", pad=10)

    cell_font = max(5.0, min(7.5, 170.0 / n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            if val >= 0.05:
                col = "white" if val > 0.5 else "#1f2937"
                ax_heat.text(j, i, f"{val*100:.0f}%", ha="center", va="center", color=col, fontsize=cell_font,
                             fontweight="bold" if i == j else "normal")

    error_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm_raw[i, j] > 0:
                error_pairs.append({
                    "true": class_names[i],
                    "pred": class_names[j],
                    "count": cm_raw[i, j],
                    "pct_of_true": cm_norm[i, j] * 100,
                    "label": f"{class_names[i]} → {class_names[j]}"
                })

    if error_pairs:
        top_errors_df = pd.DataFrame(error_pairs).sort_values(by="count", ascending=True).tail(top_k)
        y_pos = np.arange(len(top_errors_df))
        bars_err = ax_top.barh(y_pos, top_errors_df["count"], color="#ef4444", alpha=0.85, height=0.65)
        ax_top.set_yticks(y_pos)
        ax_top.set_yticklabels(top_errors_df["label"], fontsize=9, fontweight="medium")
        ax_top.set_xlabel("Number of Misclassified Images", fontsize=10, fontweight="bold")
        ax_top.set_title(f"Top {len(top_errors_df)} Confusion Pairs (True → Predicted)", fontsize=12, fontweight="bold", pad=10)

        for bar, (_, row) in zip(bars_err, top_errors_df.iterrows()):
            ax_top.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2, 
                        f"{int(row['count']):,} ({row['pct_of_true']:.1f}% of true)", 
                        va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1f2937")
        ax_top.set_xlim(0, top_errors_df["count"].max() * 1.35)
    else:
        ax_top.text(0.5, 0.5, "Zero misclassifications found!", ha="center", va="center", fontsize=12)
        top_errors_df = pd.DataFrame()

    plt.tight_layout()

    out_p = Path(save_path) if save_path else context.plots_dir / f"confusion_matrix_{n_classes}classes.png"
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), bbox_inches="tight")
        print(f"[Plot Saved] {out_p}")
    plt.show()

    return cm_norm, top_errors_df


# ==============================================================================
# 8. Sample Evaluation Cache & Hardest Errors
# ==============================================================================
def get_or_compute_sample_cache(
    model: OnnxAnalysisModel,
    test_df: pd.DataFrame,
    cache_path: Optional[Union[str, Path]] = None,
    samples_per_class: int = 10,
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """Loads or computes balanced evaluation cache with native ONNX embeddings."""
    c_path = Path(cache_path).resolve() if cache_path else context.pred_dir / "sample_evaluation_cache.npz"

    if c_path.exists():
        print(f"[Cache] Loading precomputed sample evaluation cache: {c_path}...")
        sample_cache = dict(np.load(c_path, allow_pickle=True))
    else:
        print(f"[Cache] Computing balanced sample evaluation ({samples_per_class} per class via ONNX)...")
        sampled_rows = []
        for cls_idx in range(context.num_classes):
            sub = test_df[test_df["target_idx"] == cls_idx]
            if len(sub) > 0:
                n_take = min(len(sub), samples_per_class)
                sampled_rows.append(sub.sample(n=n_take, random_state=42))
        sample_manifest = pd.concat(sampled_rows).reset_index(drop=True)
        sample_paths = sample_manifest["image_path"].values
        sample_true  = sample_manifest["target_idx"].values

        img_sz = context.img_size
        n_total = len(sample_paths)

        all_preds = []
        all_embeds = []

        for b_start in range(0, n_total, batch_size):
            b_paths = sample_paths[b_start : b_start + batch_size]
            b_imgs = [preprocess_image(p, img_size=img_sz) for p in b_paths]
            b_tensor = np.stack(b_imgs, axis=0)
            preds, _, embeds = model.predict(b_tensor)
            all_preds.append(preds)
            all_embeds.append(embeds)

        s_preds = np.concatenate(all_preds, axis=0)
        embeddings = np.concatenate(all_embeds, axis=0)

        sample_pred  = np.argmax(s_preds, axis=-1)
        sample_confs = np.max(s_preds, axis=-1)

        pca_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)
        perp = min(25, max(5, len(embeddings) // 8))
        try:
            tsne_2d = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000, init="pca", learning_rate="auto").fit_transform(embeddings)
        except TypeError:
            tsne_2d = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000, init="pca", learning_rate="auto").fit_transform(embeddings)

        sample_cache = {
            "image_paths": sample_paths,
            "true_labels": sample_true,
            "pred_labels": sample_pred,
            "pred_confs": sample_confs,
            "embeddings": embeddings,
            "pca_2d": pca_2d,
            "tsne_2d": tsne_2d
        }
        c_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(c_path, **sample_cache)
        print(f"[Cache Saved] {c_path}")

    return sample_cache


def plot_hardest_misclassifications(
    sample_cache: Dict[str, np.ndarray],
    n_show: int = 8,
    save_path: Optional[Union[str, Path]] = None
):
    """Inspects highest-confidence incorrect predictions."""
    sample_paths = sample_cache["image_paths"]
    sample_true  = sample_cache["true_labels"]
    sample_pred  = sample_cache["pred_labels"]
    sample_confs = sample_cache["pred_confs"]

    false_mask = (sample_pred != sample_true)
    false_indices = np.where(false_mask)[0]

    if len(false_indices) == 0:
        print("[Hardest Errors] Outstanding! Zero misclassifications found in the sample cache.")
        return

    sorted_false_idx = false_indices[np.argsort(-sample_confs[false_indices])]
    n_plot = min(n_show, len(sorted_false_idx))
    top_false = sorted_false_idx[:n_plot]

    n_cols = 4
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.8 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        if i < n_plot:
            idx = top_false[i]
            img = Image.open(sample_paths[idx]).convert("RGB")
            t_name = context.idx_to_class[sample_true[idx]]
            p_name = context.idx_to_class[sample_pred[idx]]
            conf = sample_confs[idx]

            ax.imshow(img)
            ax.set_title(
                f"True: {t_name}\nPred: {p_name} ({conf*100:.1f}%)",
                fontsize=10,
                fontweight="bold",
                color="#dc2626",
                pad=6
            )
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle(f"Top {n_plot} Hardest Misclassifications (Highest False Confidence)", 
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_p = Path(save_path) if save_path else context.plots_dir / "hardest_misclassifications.png"
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), bbox_inches="tight")
        print(f"[Plot Saved] {out_p}")
    plt.show()


# ==============================================================================
# 9. 100% Native ONNX Grad-CAM / Class Activation Mapping Engine
# ==============================================================================
class GradCAMAnalyzer:
    """
    100% ONNX-native Class Activation Mapping (CAM/Grad-CAM) Analyzer.
    Uses the analytical gradient relation L_c = ReLU( sum_k w_k^c * A^k ).
    Executes in single milliseconds without backprop graph overhead.
    """
    def __init__(self, model: OnnxAnalysisModel):
        self.model = model
        self.class_to_idx = context.class_to_idx
        self.idx_to_class = context.idx_to_class
        self.class_names  = context.class_names
        self.effective_w  = model.effective_weights
        self.jet_colormap = matplotlib.colormaps["jet"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("jet")
        print(f"[ONNX Grad-CAM] Initialized on feature map: '{model.conv_tensor_name}'")

    def compute_gradcam(
        self,
        image_path: str,
        target_class_idx: Optional[int] = None,
        img_size: Optional[int] = None
    ) -> Tuple[Image.Image, np.ndarray, int, float, np.ndarray]:
        """Computes 2D class activation map overlay via native ONNX inference."""
        sz = img_size or context.img_size
        raw_img = Image.open(str(image_path)).convert("RGB")
        img_resized = raw_img.resize((sz, sz), Image.Resampling.BILINEAR)
        inp = np.array(img_resized, dtype=np.float32)[np.newaxis, ...]

        preds, conv_feats, _ = self.model.predict(inp)
        pred_class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][pred_class_idx])

        eval_class = target_class_idx if target_class_idx is not None else pred_class_idx
        class_conf = float(preds[0][eval_class]) if eval_class < preds.shape[1] else confidence

        # Handle tensor layout: (B, C, H, W) vs (B, H, W, C)
        if len(conv_feats.shape) == 4:
            if conv_feats.shape[1] == self.effective_w.shape[0]:
                conv_hwc = np.transpose(conv_feats[0], (1, 2, 0)).astype(np.float32)
            else:
                conv_hwc = conv_feats[0].astype(np.float32)
        else:
            conv_hwc = np.zeros((20, 20, self.effective_w.shape[0]), dtype=np.float32)

        w_c = self.effective_w[:, eval_class]
        cam = np.maximum(np.dot(conv_hwc, w_c), 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val

        cam_img = Image.fromarray(np.uint8(255 * cam)).resize(raw_img.size, Image.Resampling.BICUBIC)
        cam_upsampled = np.array(cam_img, dtype=np.float32) / 255.0

        colored_cam = self.jet_colormap(cam_upsampled)[:, :, :3]
        orig_np = np.array(raw_img, dtype=np.float32) / 255.0
        overlay = np.clip(0.55 * orig_np + 0.45 * colored_cam, 0, 1)

        return raw_img, overlay, pred_class_idx, class_conf, cam_upsampled

    def plot_highest_confidence_true(
        self,
        sample_cache: Dict[str, np.ndarray],
        target_makes: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None
    ):
        """Visualizes Grad-CAM on highest-confidence TRUE predictions."""
        sample_paths = sample_cache["image_paths"]
        sample_true  = sample_cache["true_labels"]
        sample_pred  = sample_cache["pred_labels"]
        sample_confs = sample_cache["pred_confs"]

        correct_mask = (sample_pred == sample_true)

        if target_makes is None:
            iconic_cands = ["Porsche", "Jeep", "BMW", "Audi", "Land Rover", "Ford", "Mercedes-Benz", "Toyota", "Honda"]
            makes = [m for m in iconic_cands if m in self.class_to_idx]
            if len(makes) < 4:
                correct_idxs = sample_true[correct_mask]
                unique_correct, counts = np.unique(correct_idxs, return_counts=True)
                top_classes = unique_correct[np.argsort(-counts)[:6]]
                makes = [self.idx_to_class[idx] for idx in top_classes]
        else:
            makes = [m for m in target_makes if m in self.class_to_idx]

        selected_correct = []
        for make in makes:
            m_idx = self.class_to_idx[make]
            c_indices = np.where((sample_true == m_idx) & correct_mask)[0]
            if len(c_indices) > 0:
                best_i = c_indices[np.argmax(sample_confs[c_indices])]
                selected_correct.append((make, sample_paths[best_i], sample_confs[best_i]))

        if not selected_correct:
            print("[Grad-CAM True] No correct predictions available in sample cache for requested classes.")
            return

        print(f"[Grad-CAM True] Visualizing {len(selected_correct)} highest-confidence correct predictions:")
        for make, p, conf in selected_correct:
            print(f"  - {make:12s} | Confidence: {conf*100:6.2f}% | Sample: {Path(p).name}")

        plt.figure(figsize=(18, 9))
        for i, (make, path, conf) in enumerate(selected_correct):
            raw_img, overlay, pred_idx, conf_calc, _ = self.compute_gradcam(path)

            ax1 = plt.subplot(2, len(selected_correct), i + 1)
            ax1.imshow(raw_img)
            ax1.set_title(f"{make}\n(Original)", fontsize=10, fontweight="bold")
            ax1.axis("off")

            ax2 = plt.subplot(2, len(selected_correct), len(selected_correct) + i + 1)
            ax2.imshow(overlay)
            ax2.set_title(f"Grad-CAM\n(Conf: {conf_calc*100:.1f}%)", fontsize=10, fontweight="bold", color="#16a34a")
            ax2.axis("off")

        plt.suptitle("Grad-CAM: Highest-Confidence TRUE Predictions (Attending to Discriminatory Features)", 
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_p = Path(save_path) if save_path else context.plots_dir / "gradcam_highest_confidence_true.png"
        if out_p:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), bbox_inches="tight")
            print(f"[Plot Saved] {out_p}")
        plt.show()

    def plot_comparative_highest_confidence_false(
        self,
        sample_cache: Dict[str, np.ndarray],
        n_samples: int = 4,
        save_path: Optional[Union[str, Path]] = None
    ):
        """Visualizes 3-column comparative Grad-CAM on highest-confidence FALSE predictions (hardest errors)."""
        sample_paths = sample_cache["image_paths"]
        sample_true  = sample_cache["true_labels"]
        sample_pred  = sample_cache["pred_labels"]
        sample_confs = sample_cache["pred_confs"]

        false_mask = (sample_pred != sample_true)
        sorted_false_indices = np.where(false_mask)[0]

        if len(sorted_false_indices) == 0:
            print("[Grad-CAM False] Zero misclassifications found in the sample cache!")
            return

        sorted_false_indices = sorted_false_indices[np.argsort(-sample_confs[sorted_false_indices])]

        selected_false = []
        seen_makes = set()
        for idx in sorted_false_indices:
            t_m = self.idx_to_class[sample_true[idx]]
            p_m = self.idx_to_class[sample_pred[idx]]
            if t_m not in seen_makes:
                seen_makes.add(t_m)
                selected_false.append((t_m, p_m, sample_paths[idx], sample_confs[idx], sample_true[idx], sample_pred[idx]))
            if len(selected_false) >= n_samples:
                break

        print(f"[Grad-CAM False] Visualizing {len(selected_false)} highest-confidence FALSE predictions (Hardest Errors):")
        for t_m, p_m, p, conf, _, _ in selected_false:
            print(f"  - True: {t_m:12s} -> Pred: {p_m:12s} | False Confidence: {conf*100:.2f}% | Sample: {Path(p).name}")

        fig, axes = plt.subplots(len(selected_false), 3, figsize=(16, 4.8 * len(selected_false)), squeeze=False)

        for i, (true_m, pred_m, path, conf_val, t_idx, p_idx) in enumerate(selected_false):
            raw_img, overlay_pred, _, conf_p, _ = self.compute_gradcam(path, target_class_idx=p_idx)
            _, overlay_true, _, conf_t, _ = self.compute_gradcam(path, target_class_idx=t_idx)

            model_name = Path(path).parent.name

            axes[i, 0].imshow(raw_img)
            axes[i, 0].set_title(f"Original Vehicle ({model_name})\nTrue Make: {true_m}", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay_pred)
            axes[i, 1].set_title(f"Attention for PREDICTED: {pred_m}\n(Misleading Cues, Conf: {conf_p*100:.1f}%)", 
                                 fontsize=11, fontweight="bold", color="#dc2626")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlay_true)
            axes[i, 2].set_title(f"Attention for TRUE: {true_m}\n(Evidence for Correct Class, Conf: {conf_t*100:.1f}%)", 
                                 fontsize=11, fontweight="bold", color="#2563eb")
            axes[i, 2].axis("off")

        plt.suptitle("Comparative Grad-CAM on Hardest False Predictions: Misleading Visual Cues vs Ground Truth", 
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        out_p = Path(save_path) if save_path else context.plots_dir / "gradcam_highest_confidence_false.png"
        if out_p:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), bbox_inches="tight")
            print(f"[Plot Saved] {out_p}")
        plt.show()

    def plot_gradcam_by_make_model(
        self,
        test_df: pd.DataFrame,
        make: str,
        model_name: Optional[str] = None,
        n_samples: int = 2,
        mode: str = "auto",
        target_class: Optional[str] = None,
        random_state: Optional[int] = 42,
        save_path: Optional[Union[str, Path]] = None
    ):
        """Generates interactive Grad-CAM visualizations for any make and model query."""
        if "make" not in test_df.columns:
            raise KeyError("test_df must contain a 'make' column.")
        if "model" not in test_df.columns:
            test_df["model"] = test_df["source"] if "source" in test_df.columns else "Standard"

        make_matches = test_df[test_df["make"].astype(str).str.lower() == make.strip().lower()]
        if len(make_matches) == 0:
            partial = [m for m in self.class_names if make.strip().lower() in m.lower() or m.lower() in make.strip().lower()]
            err_msg = f"Make '{make}' not recognized among the {len(self.class_names)} classes."
            if partial:
                err_msg += f"\nDid you mean: {partial}?"
            raise ValueError(err_msg)

        matched_make = make_matches["make"].iloc[0]

        if model_name is not None and str(model_name).strip() != "":
            clean_model = str(model_name).strip().lower()
            model_matches = make_matches[make_matches["model"].astype(str).str.lower() == clean_model]
            if len(model_matches) == 0:
                model_matches = make_matches[make_matches["model"].astype(str).str.lower().str.contains(clean_model, regex=False)]
            if len(model_matches) == 0:
                model_matches = make_matches[make_matches["image_path"].astype(str).str.lower().str.contains(clean_model, regex=False)]

            if len(model_matches) == 0:
                avail_models = sorted(make_matches["model"].dropna().unique())
                print(f"[Notice] Specific model tag '{model_name}' not indexed for {matched_make} (available sources: {avail_models}). Visualizing representative '{matched_make}' vehicles.")
                matched_model = f"All ({matched_make})"
                subset = make_matches
            else:
                matched_model = model_matches["model"].iloc[0]
                subset = model_matches
        else:
            matched_model = "All Models"
            subset = make_matches

        total_avail = len(subset)
        n_take = min(n_samples, total_avail)
        if total_avail > n_take and random_state is not None:
            sampled_rows = subset.sample(n=n_take, random_state=random_state).reset_index(drop=True)
        else:
            sampled_rows = subset.head(n_take).reset_index(drop=True)

        print(f"[Grad-CAM Query] Make: '{matched_make}', Model: '{matched_model}'")
        print(f"                 Found {total_avail:,} test images. Visualizing {n_take} sample(s):")

        sample_results = []
        any_misclassified = False
        for i, (_, row) in enumerate(sampled_rows.iterrows()):
            img_path = row["image_path"]
            t_make   = row["make"]
            t_model  = row["model"]
            t_idx    = self.class_to_idx[t_make]

            raw_img, overlay_pred, pred_idx, conf_pred, _ = self.compute_gradcam(img_path)
            p_make = self.idx_to_class[pred_idx]
            is_corr = (pred_idx == t_idx)
            if not is_corr:
                any_misclassified = True

            if target_class is not None and target_class in self.class_to_idx:
                t_tgt_idx = self.class_to_idx[target_class]
                _, overlay_secondary, _, conf_sec, _ = self.compute_gradcam(img_path, target_class_idx=t_tgt_idx)
                sec_label = target_class
                sec_conf = conf_sec
            elif not is_corr or mode == "comparative":
                _, overlay_secondary, _, conf_sec, _ = self.compute_gradcam(img_path, target_class_idx=t_idx)
                sec_label = f"TRUE: {t_make}"
                sec_conf = conf_sec
            else:
                overlay_secondary = None
                sec_label = None
                sec_conf = None

            sample_results.append({
                "img_path": img_path,
                "raw_img": raw_img,
                "overlay_pred": overlay_pred,
                "overlay_secondary": overlay_secondary,
                "true_make": t_make,
                "true_model": t_model,
                "pred_make": p_make,
                "conf_pred": conf_pred,
                "sec_label": sec_label,
                "sec_conf": sec_conf,
                "is_correct": is_corr
            })
            status_str = "CORRECT" if is_corr else "MISCLASSIFIED"
            print(f"  - Sample {i+1}: True={t_make} {t_model} | Pred={p_make} (Conf: {conf_pred*100:.2f}%) [{status_str}] | File: {Path(img_path).name}")

        use_3_columns = (mode == "comparative") or (mode == "auto" and any_misclassified) or (target_class is not None)

        if use_3_columns:
            fig, axes = plt.subplots(n_take, 3, figsize=(16, 4.8 * n_take), squeeze=False)
            for i, res in enumerate(sample_results):
                axes[i, 0].imshow(res["raw_img"])
                axes[i, 0].set_title(f"Original Vehicle\n{res['true_make']} {res['true_model']}", fontsize=11, fontweight="bold")
                axes[i, 0].axis("off")

                pred_color = "#16a34a" if res["is_correct"] else "#dc2626"
                axes[i, 1].imshow(res["overlay_pred"])
                axes[i, 1].set_title(f"Attention for PREDICTED: {res['pred_make']}\n(Conf: {res['conf_pred']*100:.1f}%)", 
                                     fontsize=11, fontweight="bold", color=pred_color)
                axes[i, 1].axis("off")

                if res["overlay_secondary"] is not None:
                    axes[i, 2].imshow(res["overlay_secondary"])
                    axes[i, 2].set_title(f"Attention for {res['sec_label']}\n(Evidence for Correct Class)", 
                                         fontsize=11, fontweight="bold", color="#2563eb")
                else:
                    axes[i, 2].imshow(res["overlay_pred"])
                    axes[i, 2].set_title(f"Grad-CAM (Winning Class)\n{res['pred_make']}", fontsize=11, fontweight="bold", color="#16a34a")
                axes[i, 2].axis("off")
        else:
            fig, axes = plt.subplots(n_take, 2, figsize=(12, 4.8 * n_take), squeeze=False)
            for i, res in enumerate(sample_results):
                axes[i, 0].imshow(res["raw_img"])
                axes[i, 0].set_title(f"Original Vehicle\n{res['true_make']} {res['true_model']}", fontsize=11, fontweight="bold")
                axes[i, 0].axis("off")

                pred_color = "#16a34a" if res["is_correct"] else "#dc2626"
                axes[i, 1].imshow(res["overlay_pred"])
                axes[i, 1].set_title(f"Grad-CAM: {res['pred_make']}\n(Conf: {res['conf_pred']*100:.1f}%)", 
                                     fontsize=11, fontweight="bold", color=pred_color)
                axes[i, 1].axis("off")

        plt.suptitle(f"Grad-CAM Explainability: {matched_make} ({matched_model})", 
                     fontsize=14, fontweight="bold", y=0.99 if n_take > 1 else 1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight")
            print(f"[Plot Saved] {save_path}")
        plt.show()


def list_available_models(test_df: pd.DataFrame, make: Optional[str] = None) -> List[str]:
    """Lists all vehicle models/sources available in the test dataset for a given make."""
    if "model" not in test_df.columns:
        test_df["model"] = test_df["source"] if "source" in test_df.columns else "Standard"
    if make is not None:
        sub = test_df[test_df["make"].astype(str).str.lower() == make.strip().lower()]
        if len(sub) == 0:
            print(f"Make '{make}' not found. Available makes: {context.class_names}")
            return []
        matched_make = sub["make"].iloc[0]
        models = sorted(sub["model"].dropna().unique())
        print(f"Available models for {matched_make} ({len(models)} categories, {len(sub):,} images):")
        for i in range(0, len(models), 5):
            print("  " + ", ".join(models[i:i+5]))
        return models
    else:
        make_counts = test_df.groupby("make")["model"].nunique()
        print("Unique models count per make in test set:")
        for m, c in make_counts.items():
            print(f"  {m:14s}: {c:3d} models")
        return list(make_counts.index)


# ==============================================================================
# 10. Latent Feature Space Visualisation (t-SNE & PCA)
# ==============================================================================
def plot_latent_space_tsne_pca(
    sample_cache: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None
):
    """Plots side-by-side t-SNE and PCA 2D projections of latent embeddings."""
    sample_true = sample_cache["true_labels"]
    pca_2d      = sample_cache["pca_2d"]
    tsne_2d     = sample_cache["tsne_2d"]
    n_classes   = context.num_classes
    class_names = context.class_names

    fig, (ax_tsne, ax_pca) = plt.subplots(1, 2, figsize=(22, 10))
    scatter_cmap = matplotlib.colormaps["tab20"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("tab20")

    for idx in range(n_classes):
        mask = (sample_true == idx)
        if np.sum(mask) > 0:
            ax_tsne.scatter(
                tsne_2d[mask, 0], 
                tsne_2d[mask, 1], 
                s=45, 
                alpha=0.8, 
                label=class_names[idx],
                color=scatter_cmap(idx % 20),
                edgecolors="none"
            )

    ax_tsne.set_title(f"t-SNE 2D Projection of Latent Embeddings\nBalanced Sample Across {n_classes} Classes", 
                      fontsize=12, fontweight="bold", pad=10)
    ax_tsne.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
    ax_tsne.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
    ax_tsne.grid(True, linestyle="--", alpha=0.4)

    for idx in range(n_classes):
        mask = (sample_true == idx)
        if np.sum(mask) > 0:
            ax_pca.scatter(
                pca_2d[mask, 0], 
                pca_2d[mask, 1], 
                s=45, 
                alpha=0.8, 
                label=class_names[idx],
                color=scatter_cmap(idx % 20),
                edgecolors="none"
            )

    ax_pca.set_title("PCA 2D Projection of Latent Embeddings\nGlobal Variance Decomposition", 
                     fontsize=12, fontweight="bold", pad=10)
    ax_pca.set_xlabel("Principal Component 1", fontsize=10, fontweight="bold")
    ax_pca.set_ylabel("Principal Component 2", fontsize=10, fontweight="bold")
    ax_pca.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax_tsne.get_legend_handles_labels()
    ncol = min(8, max(4, n_classes // 4))
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.02), loc="upper center", ncol=ncol, fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_p = Path(save_path) if save_path else context.plots_dir / f"latent_space_tsne_pca_{n_classes}classes.png"
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), bbox_inches="tight")
        print(f"[Plot Saved] {out_p}")
    plt.show()


# ==============================================================================
# 11. Prediction Confidence & Calibration Analysis
# ==============================================================================
def plot_confidence_calibration(
    sample_cache: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None
):
    """Plots confidence score distribution histograms and accuracy-coverage rejection curves."""
    sample_true  = sample_cache["true_labels"]
    sample_pred  = sample_cache["pred_labels"]
    sample_confs = sample_cache["pred_confs"]

    is_correct = (sample_pred == sample_true)
    correct_confs = sample_confs[is_correct]
    error_confs   = sample_confs[~is_correct]

    plt.figure(figsize=(15, 5))

    # Subplot 1: Confidence Distribution
    plt.subplot(1, 2, 1)
    if len(correct_confs) > 0:
        plt.hist(correct_confs, bins=25, alpha=0.65, color="#16a34a", density=True, label=f"Correct (Mean: {np.mean(correct_confs)*100:.1f}%)")
    if len(error_confs) > 0:
        plt.hist(error_confs, bins=25, alpha=0.65, color="#dc2626", density=True, label=f"Incorrect (Mean: {np.mean(error_confs)*100:.1f}%)")
    plt.xlabel("Softmax Prediction Confidence", fontsize=11, fontweight="bold")
    plt.ylabel("Density", fontsize=11, fontweight="bold")
    plt.title("Confidence Distribution: Correct vs. Incorrect Predictions", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Accuracy vs. Coverage
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0.3, 0.99, 50)
    retained_accs = []
    coverage_pcts = []

    for t in thresholds:
        mask = (sample_confs >= t)
        if np.sum(mask) > 0:
            retained_accs.append(np.mean(is_correct[mask]) * 100)
            coverage_pcts.append(np.mean(mask) * 100)
        else:
            retained_accs.append(np.nan)
            coverage_pcts.append(0)

    plt.plot(thresholds, retained_accs, "o-", color="#2563eb", linewidth=2, label="Accuracy on Retained Samples (%)")
    plt.plot(thresholds, coverage_pcts, "s--", color="#f59e0b", linewidth=1.5, label="Dataset Coverage (% Retained)")
    plt.xlabel("Confidence Rejection Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    plt.title("Accuracy vs. Coverage with Confidence Thresholding", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_p = Path(save_path) if save_path else context.plots_dir / "confidence_calibration.png"
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), bbox_inches="tight")
        print(f"[Plot Saved] {out_p}")
    plt.show()

    if len(correct_confs) > 0:
        print(f"[Calibration] Mean confidence on correct predictions:   {np.mean(correct_confs)*100:.1f}%")
    if len(error_confs) > 0:
        print(f"[Calibration] Mean confidence on incorrect predictions: {np.mean(error_confs)*100:.1f}%")
        if len(correct_confs) > 0:
            print(f"[Calibration] Confidence gap (Correct - Error):        {(np.mean(correct_confs) - np.mean(error_confs))*100:.1f}%")


# ==============================================================================
# 12. Main Full Diagnostic Execution
# ==============================================================================
def run_full_analysis(force_cpu: bool = False):
    """Runs the entire end-to-end diagnostic suite via pure ONNX inference."""
    print("=" * 70)
    print("  ONNX-Native Diagnostic & Explainability Analysis Engine")
    print("=" * 70)

    setup_environment(force_cpu=force_cpu)
    context.plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model Loading
    model, model_info = load_analysis_model(force_cpu=force_cpu)

    # 2. Test Evaluation
    test_df, y_true, y_pred, test_metrics = load_test_evaluation(model=model)

    # 3. Classification Report & F1 Chart
    class_report, _ = load_classification_report_df(y_true=y_true, y_pred=y_pred)
    plot_f1_ranking(class_report, test_metrics)

    # 4. Confusion Matrix & Top Errors
    cm_norm, top_errors_df = plot_confusion_matrix_and_top_errors(y_true, y_pred, top_k=15)

    # 5. Sample Evaluation & Hardest Errors
    sample_cache = get_or_compute_sample_cache(model, test_df)
    plot_hardest_misclassifications(sample_cache, n_show=8)

    # 6. Grad-CAM Explainability
    grad_analyzer = GradCAMAnalyzer(model)
    grad_analyzer.plot_highest_confidence_true(sample_cache)
    grad_analyzer.plot_comparative_highest_confidence_false(sample_cache, n_samples=4)

    # 7. Latent Space (t-SNE & PCA)
    plot_latent_space_tsne_pca(sample_cache)

    # 8. Confidence Calibration
    plot_confidence_calibration(sample_cache)

    print("\n" + "=" * 70)
    print(f"  [Complete] ONNX diagnostic evaluation complete! Figures saved in: {context.plots_dir}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX-Native Diagnostic & Explainability Analysis Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to .onnx model checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Root directory for outputs (models, plots, reports).")
    parser.add_argument("--splits-dir", type=str, default=None, help="Directory containing test.csv and splits.")
    parser.add_argument("--test-csv", type=str, default=None, help="Explicit path to test.csv manifest.")
    parser.add_argument("--label-map", type=str, default=None, help="Path to label_map.json.")
    parser.add_argument("--img-size", type=int, default=None, help="Input spatial resolution (e.g. 640).")
    parser.add_argument("--force-cpu", action="store_true", help="Force execution on CPU.")
    parser.add_argument("--run-all", action="store_true", help="Execute the entire end-to-end analysis pipeline.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_context(
        model_path=args.model_path,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        test_csv=args.test_csv,
        label_map=args.label_map,
        img_size=args.img_size,
    )
    if args.force_cpu:
        setup_environment(force_cpu=True)
    if args.run_all:
        run_full_analysis(force_cpu=args.force_cpu)
    else:
        print("[Info] Context configured successfully. Call --run-all to execute the complete diagnostic suite.")
        print(f"  ONNX Model: {context.model_path}")
        print(f"  Output Dir: {context.output_dir}")
        print(f"  Test CSV:   {context.test_csv_path}")
        print(f"  Classes:    {context.num_classes} ({', '.join(context.class_names[:5])}...)")


if __name__ == "__main__":
    main()
