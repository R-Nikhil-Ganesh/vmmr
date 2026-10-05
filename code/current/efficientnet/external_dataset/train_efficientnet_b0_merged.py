#!/usr/bin/env python3
"""
EfficientNet-B0 Training Pipeline on Merged Vehicle Dataset (35 Makes)
======================================================================
Datasets Merged: BoxCars116k, Stanford Cars, CompCars CCTV (and Web supplement)
Splits Directory: /home/researchadmin/Econ/external_datasets/merged_data
Output Directory: /home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged
Environment: repo-clone/stanford-cars-model/.venv (TensorFlow 2.21 + CUDA)
"""

import os
import sys
import json
import argparse
import signal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("KERAS_HOME", "/home/researchadmin/Econ/.keras")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from tensorflow.keras.applications import EfficientNetB0


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Merged 35-Make Vehicle Dataset.")
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="/home/researchadmin/Econ/external_datasets/merged_data",
        help="Path containing train.csv, val.csv, test.csv, and label_map.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0_merged",
        help="Destination directory for checkpoints, metrics, and plots"
    )
    parser.add_argument("--img-size", type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam initial learning rate (default: 1e-4)")
    parser.add_argument("--unfreeze-layers", type=int, default=5, help="Number of backbone layers to unfreeze (default: 5)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Classifier head dropout rate (default: 0.4)")
    parser.add_argument("--no-mixed-precision", action="store_true", default=False)
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


def setup_environment(seed=42, use_mixed_precision=True):
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"[Warning] GPU memory growth error: {e}")
        gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
        print(f"[Device] GPU detected: {gpu_name}")
    else:
        print("[Device] No GPU found, running on CPU.")

    if use_mixed_precision and gpus:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print("[Precision] mixed_float16 enabled (compute=float16, variables=float32)")
    else:
        print("[Precision] float32")


def build_dataset(csv_path, img_size=640, batch_size=8, is_training=False):
    df = pd.read_csv(csv_path)
    paths  = df["image_path"].values
    labels = df["label"].values.astype("int32")

    def parse_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), 10000), reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(df)


def plot_training_curves(history_csv_path, plots_dir):
    """Reads CSVLogger output and saves accuracy and loss curves."""
    if not Path(history_csv_path).exists():
        print(f"[Plot] History CSV not found: {history_csv_path}, skipping plots.")
        return

    df = pd.read_csv(history_csv_path)
    epochs = df["epoch"] + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EfficientNet-B0 Training Curves (Merged 35-Make Dataset)", fontsize=15, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, df["accuracy"],     "o-",  label="Train Accuracy", linewidth=2, color="#2563EB")
    ax.plot(epochs, df["val_accuracy"], "s--", label="Val Accuracy",   linewidth=2, color="#16A34A")
    if "top_5_accuracy" in df.columns:
        ax.plot(epochs, df["top_5_accuracy"],     "^:",  label="Train Top-5 Acc", linewidth=1.5, color="#7C3AED", alpha=0.7)
        ax.plot(epochs, df["val_top_5_accuracy"], "D:",  label="Val Top-5 Acc",   linewidth=1.5, color="#D97706", alpha=0.7)
    ax.set_title("Accuracy", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Loss
    ax = axes[1]
    ax.plot(epochs, df["loss"],     "o-",  label="Train Loss", linewidth=2, color="#DC2626")
    ax.plot(epochs, df["val_loss"], "s--", label="Val Loss",   linewidth=2, color="#EA580C")
    ax.set_title("Loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = Path(plots_dir) / "training_curves.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved to: {out_path}")


def create_model(num_classes, img_size=640, dropout=0.4, unfreeze_layers=5):
    inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    n_frozen    = sum(1 for l in base_model.layers if not l.trainable)
    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[Model] EfficientNet-B0: {n_frozen} frozen, {n_trainable} trainable layers (last {unfreeze_layers} unfrozen)")

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=out, name="EfficientNetB0_Merged35Makes")
    return model



def export_model_to_onnx(model, output_path, img_size=640, opset=13):
    """Exports a Keras model to ONNX format and verifies with onnx.checker."""
    import tf2onnx
    import onnx
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = (tf.TensorSpec((None, img_size, img_size, 3), tf.float32, name="input_image"),)
    print(f"[ONNX Export] Converting model to ONNX (opset {opset})...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    onnx.checker.check_model(onnx_model)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX Export] Successfully saved: {output_path} ({size_mb:.2f} MB)")
    return str(output_path)


class OnnxCheckpointCallback(callbacks.Callback):
    """Callback that saves latest model weights and auto-exports to ONNX."""
    def __init__(self, models_dir, img_size=640, monitor="val_accuracy", mode="max", opset=13):
        super().__init__()
        self.models_dir = Path(models_dir)
        self.img_size = img_size
        self.monitor = monitor
        self.mode = mode
        self.opset = opset
        self.best_metric = -float("inf") if mode == "max" else float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        latest_keras = self.models_dir / "efficientnet_b0_latest.keras"
        latest_onnx  = self.models_dir / "efficientnet_b0_latest.onnx"
        self.model.save(str(latest_keras))
        try:
            export_model_to_onnx(self.model, latest_onnx, img_size=self.img_size, opset=self.opset)
        except Exception as e:
            print(f"[Warning] Failed to export latest ONNX at epoch {epoch+1}: {e}")

        current_metric = logs.get(self.monitor)
        if current_metric is not None:
            is_better = (current_metric > self.best_metric) if self.mode == "max" else (current_metric < self.best_metric)
            if is_better:
                self.best_metric = current_metric
                best_onnx = self.models_dir / "efficientnet_b0_best.onnx"
                try:
                    export_model_to_onnx(self.model, best_onnx, img_size=self.img_size, opset=self.opset)
                    print(f"[ONNX Checkpoint] New best {self.monitor}={current_metric:.4f} -> Updated {best_onnx.name}")
                except Exception as e:
                    print(f"[Warning] Failed to export best ONNX at epoch {epoch+1}: {e}")

def main():
    args = parse_args()
    setup_environment(seed=args.seed, use_mixed_precision=not args.no_mixed_precision)

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    models_dir  = output_dir / "models"
    plots_dir   = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for d in [models_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    label_map_file = splits_dir / "label_map.json"
    with open(label_map_file) as f:
        label_meta = json.load(f)
    idx_to_class = {int(k): v for k, v in label_meta["idx_to_class"].items()}
    num_classes  = len(idx_to_class)
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    print(f"\n{'='*70}")
    print(f"  EfficientNet-B0 | {num_classes} classes | {args.img_size}x{args.img_size} | batch={args.batch_size}")
    print(f"  Unfreeze last {args.unfreeze_layers} layers | LR={args.lr} | Epochs={args.epochs}")
    print(f"  Data:   {splits_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    val_ds,  n_val  = build_dataset(str(splits_dir / "val.csv"),  args.img_size, args.batch_size)
    test_ds, n_test = build_dataset(str(splits_dir / "test.csv"), args.img_size, args.batch_size)
    print(f"[Data] Val: {n_val:,} samples | Test: {n_test:,} samples")

    best_model_path  = models_dir / "efficientnet_b0_best.keras"
    final_model_path = models_dir / "efficientnet_b0_final.keras"

    if not args.evaluate_only:
        train_ds, n_train = build_dataset(str(splits_dir / "train.csv"), args.img_size, args.batch_size, is_training=True)
        print(f"[Data] Train: {n_train:,} samples\n")

        model = create_model(num_classes, img_size=args.img_size, dropout=args.dropout, unfreeze_layers=args.unfreeze_layers)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")],
        )

        cb_list = [
            callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
            callbacks.CSVLogger(str(reports_dir / "training_history.csv"), append=True),
        ]

        print(f">>> Training for up to {args.epochs} epochs...\n")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=cb_list,
            verbose=1,
        )

        model.save(str(final_model_path))
        print(f"\n[Saved] Final model: {final_model_path}")
        print(f"[Saved] Best model:  {best_model_path}")

        try:
            best_onnx = models_dir / "efficientnet_b0_best.onnx"
            final_onnx = models_dir / "efficientnet_b0_final.onnx"
            export_model_to_onnx(model, final_onnx, img_size=args.img_size, opset=args.opset)
            best_loaded = models.load_model(str(best_model_path))
            export_model_to_onnx(best_loaded, best_onnx, img_size=args.img_size, opset=args.opset)
        except Exception as e:
            print(f"[Warning] Automatic ONNX export failed: {e}")

        plot_training_curves(reports_dir / "training_history.csv", plots_dir)

    # Evaluation
    eval_path = Path(args.checkpoint_path) if args.checkpoint_path else best_model_path
    if not eval_path.exists():
        eval_path = final_model_path

    print(f"\n{'='*70}")
    print(f"  Evaluating: {eval_path.name}")
    print(f"{'='*70}")

    eval_model = models.load_model(str(eval_path))
    test_results = eval_model.evaluate(test_ds, verbose=1)
    metrics_summary = dict(zip(eval_model.metrics_names, [float(v) for v in test_results]))

    print("\n--- Test Metrics ---")
    for k, v in metrics_summary.items():
        print(f"  {k}: {v:.4f}")

    print("\nGenerating per-class classification report...")
    preds  = eval_model.predict(test_ds, verbose=1)
    y_pred = np.argmax(preds, axis=-1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(reports_dir / "test_classification_report.csv")

    metrics_summary["macro_f1"]    = float(report["macro avg"]["f1-score"])
    metrics_summary["weighted_f1"] = float(report["weighted avg"]["f1-score"])

    with open(reports_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n[Complete] All results saved in: {output_dir}")
    print(f"  - Metrics: {reports_dir / 'test_metrics.json'}")
    print(f"  - Report:  {reports_dir / 'test_classification_report.csv'}")
    print(f"  - History: {reports_dir / 'training_history.csv'}")
    print(f"  - Curves:  {plots_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
