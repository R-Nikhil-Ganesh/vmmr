#!/usr/bin/env python3
"""
EfficientNet-B0 Training Pipeline for Vehicle Classification using Keras and TensorFlow.
Dataset: /home/researchadmin/Econ/resized_640x640/splits_filtered/
Environment: tf-env
"""

import os
import json
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("KERAS_HOME", "/home/researchadmin/Econ/.keras")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for scripts/tmux
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from tensorflow.keras.applications import EfficientNetB0


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Vehicle Dataset using Keras.")
    parser.add_argument("--splits-dir", type=str,
        default="/home/researchadmin/Econ/resized_640x640/splits_filtered")
    parser.add_argument("--output-dir", type=str,
        default="/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/output_efficientnet_b0")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--no-mixed-precision", action="store_true", default=False)
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
                print(f"GPU memory growth error: {e}")
        gpu_name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
        print(f"[Device] GPU: {gpu_name}")
    else:
        print("[Device] No GPU found, running on CPU.")

    if use_mixed_precision and gpus:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        print(f"[Precision] mixed_float16 enabled (compute=float16, variables=float32)")
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
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(df)


def plot_training_curves(history_csv_path, plots_dir):
    """Reads CSVLogger output and saves accuracy and loss curve PNGs."""
    if not Path(history_csv_path).exists():
        print(f"[Plot] History CSV not found: {history_csv_path}, skipping plots.")
        return

    df = pd.read_csv(history_csv_path)
    epochs = df["epoch"] + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EfficientNet-B0 Training Curves", fontsize=15, fontweight="bold")

    # --- Accuracy ---
    ax = axes[0]
    ax.plot(epochs, df["accuracy"],     "o-",  label="Train Accuracy",     linewidth=2, color="#2563EB")
    ax.plot(epochs, df["val_accuracy"], "s--", label="Val Accuracy",       linewidth=2, color="#16A34A")
    if "top_5_accuracy" in df.columns:
        ax.plot(epochs, df["top_5_accuracy"],     "^:",  label="Train Top-5 Acc",  linewidth=1.5, color="#7C3AED", alpha=0.7)
        ax.plot(epochs, df["val_top_5_accuracy"], "D:",  label="Val Top-5 Acc",    linewidth=1.5, color="#D97706", alpha=0.7)
    ax.set_title("Accuracy", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # --- Loss ---
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

    # Freeze everything, then unfreeze the last N layers
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    n_frozen   = sum(1 for l in base_model.layers if not l.trainable)
    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[Model] EfficientNet-B0: {n_frozen} frozen, {n_trainable} trainable layers (last {unfreeze_layers})")

    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=out, name="EfficientNetB0_VehicleClassifier")
    return model


def main():
    args = parse_args()
    setup_environment(seed=args.seed, use_mixed_precision=not args.no_mixed_precision)

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for d in [models_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    label_map_file = splits_dir / "label_map.json"
    with open(label_map_file) as f:
        label_meta = json.load(f)
    idx_to_class = {int(k): v for k, v in label_meta["idx_to_class"].items()}
    num_classes  = len(idx_to_class)
    class_names  = [idx_to_class[i] for i in range(num_classes)]

    print(f"\n{'='*65}")
    print(f"  EfficientNet-B0 | {num_classes} classes | {args.img_size}x{args.img_size} | batch={args.batch_size}")
    print(f"  Unfreeze last {args.unfreeze_layers} layers | LR={args.lr} | Epochs={args.epochs}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}\n")

    val_ds,  n_val  = build_dataset(str(splits_dir / "val.csv"),  args.img_size, args.batch_size)
    test_ds, n_test = build_dataset(str(splits_dir / "test.csv"), args.img_size, args.batch_size)
    print(f"[Data] Val: {n_val:,} | Test: {n_test:,}")

    best_model_path  = models_dir / "efficientnet_b0_best.keras"
    final_model_path = models_dir / "efficientnet_b0_final.keras"

    if not args.evaluate_only:
        train_ds, n_train = build_dataset(str(splits_dir / "train.csv"), args.img_size, args.batch_size, is_training=True)
        print(f"[Data] Train: {n_train:,}\n")

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

        print(f">>> Training for {args.epochs} epochs...\n")
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

        # Plot and save training curves
        plot_training_curves(reports_dir / "training_history.csv", plots_dir)

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------
    eval_path = Path(args.checkpoint_path) if args.checkpoint_path else best_model_path
    if not eval_path.exists():
        eval_path = final_model_path

    print(f"\n{'='*65}")
    print(f"  Evaluating: {eval_path.name}")
    print(f"{'='*65}")

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

    print(f"\n[Done] Results in: {output_dir}")
    print(f"  - Metrics: {reports_dir / 'test_metrics.json'}")
    print(f"  - Report:  {reports_dir / 'test_classification_report.csv'}")
    print(f"  - History: {reports_dir / 'training_history.csv'}")
    print(f"  - Plots:   {plots_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
