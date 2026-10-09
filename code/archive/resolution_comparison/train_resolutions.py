#!/usr/bin/env python3
"""
Multi-Resolution Training & Evaluation Pipeline for EfficientNet-B0 on CompCars Showroom Dataset.
Retrains models independently at each target resolution:
[224x224, 256x256, 384x384, 512x512, 576x576, 640x640, 720x720]
Evaluates on the official test set and produces comparative metrics.
"""

import os
import sys
import time
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("KERAS_HOME", "/home/researchadmin/Econ/.keras")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, to_categorical

DEFAULT_DATA_DIR = "/home/researchadmin/Econ/dataset_Source"
DEFAULT_OUTPUT_DIR = "/home/researchadmin/Econ/repo-clone/stanford-cars-model/code/current/resolution_comparison"

BATCH_SIZES = {
    224: 32,
    256: 32,
    384: 32,
    512: 16,
    576: 16,
    640: 8,
    720: 8
}


def load_dataset_metadata(data_dir=DEFAULT_DATA_DIR):
    """
    Parses CompCars metadata, training lines, and test lines.
    Returns train_df, val_df, test_df, and meta_dict.
    """
    img_dir = os.path.join(data_dir, "image")
    label_dir = os.path.join(data_dir, "label")
    split_dir = os.path.join(data_dir, "train_test_split", "classification")
    misc_dir = os.path.join(data_dir, "misc")

    meta = loadmat(os.path.join(misc_dir, "make_model_name.mat"), simplify_cells=True)
    make_names = meta["make_names"]
    model_names = meta["model_names"]

    with open(os.path.join(split_dir, "train.txt"), "r") as f:
        train_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(split_dir, "test.txt"), "r") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    unique_models = sorted(list(set(int(l.split("/")[1]) for l in train_lines)))
    num_classes = len(unique_models)
    model_to_label = {model_id: idx for idx, model_id in enumerate(unique_models)}
    label_to_model = {idx: model_id for idx, model_id in enumerate(unique_models)}
    class_names = [model_names[m - 1] for m in unique_models]

    def parse_split(lines):
        rows = []
        for rel_path in lines:
            parts = rel_path.split("/")
            make_id = int(parts[0])
            model_id = int(parts[1])
            image_name = parts[3]

            lbl_file = os.path.join(label_dir, os.path.splitext(rel_path)[0] + ".txt")
            with open(lbl_file, "r") as lf:
                lbl_lines = lf.readlines()
            viewpoint = int(lbl_lines[0].strip())
            x1, y1, x2, y2 = [int(v) for v in lbl_lines[2].strip().split()]

            rows.append({
                "image": image_name,
                "make_id": make_id,
                "model_id": model_id,
                "make_name": make_names[make_id - 1],
                "class_name": model_names[model_id - 1],
                "label": model_to_label[model_id],
                "viewpoint": viewpoint,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "file_path": os.path.join(img_dir, rel_path)
            })
        return pd.DataFrame(rows)

    print("Parsing training lines and bounding boxes...", flush=True)
    raw_train_df = parse_split(train_lines)
    print("Parsing test lines and bounding boxes...", flush=True)
    test_df = parse_split(test_lines)

    train_df, val_df = train_test_split(
        raw_train_df,
        test_size=0.2,
        random_state=42,
        stratify=raw_train_df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Loaded dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test across {num_classes} classes.", flush=True)

    meta_dict = {
        "data_dir": data_dir,
        "img_dir": img_dir,
        "label_dir": label_dir,
        "make_names": make_names,
        "model_names": model_names,
        "unique_models": unique_models,
        "num_classes": num_classes,
        "model_to_label": model_to_label,
        "label_to_model": label_to_model,
        "class_names": class_names
    }
    return train_df, val_df, test_df, meta_dict


def _process_sample(args):
    file_path, bbox, target_size, datagen = args
    with Image.open(file_path) as img:
        cropped = img.crop(bbox).convert("RGB")
        resized = cropped.resize(target_size, Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

    if datagen is not None:
        arr = datagen.random_transform(arr)
        arr = datagen.preprocessing_function(arr)
    else:
        arr = preprocess_input(arr)
    return arr


class FastBBoxDataGenerator(Sequence):
    """
    High-throughput multi-threaded Sequence generator for bounding-box cropped and resized images.
    """
    def __init__(self, df, batch_size=32, target_size=(224, 224), num_classes=431, datagen=None, shuffle=True, workers=16, **kwargs):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.datagen = datagen
        self.shuffle = shuffle
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        items = [
            (row["file_path"], (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]), self.target_size, self.datagen)
            for _, row in batch_df.iterrows()
        ]
        batch_x = list(self.pool.map(_process_sample, items))
        batch_y = batch_df["label"].values.astype(int)

        return np.array(batch_x, dtype=np.float32), to_categorical(batch_y, num_classes=self.num_classes)

    def close(self):
        self.pool.shutdown(wait=False)


def build_model(resolution, num_classes=431):
    """
    Builds EfficientNetB0 for the given input resolution with the last 5 layers unfrozen.
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(resolution, resolution, 3)
    )
    base_model.trainable = True
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(num_classes, dtype="float32", activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_single_resolution(res, train_df, val_df, test_df, meta_dict, output_dir, epochs=20, workers=16, reduce_lr_patience=1):
    """
    Trains an EfficientNetB0 model from ImageNet weights at the given resolution,
    saves history, evaluates on test set, and saves class-wise metrics.
    """
    res_str = f"{res}x{res}"
    batch_size = BATCH_SIZES.get(res, 16)
    target_size = (res, res)
    num_classes = meta_dict["num_classes"]

    print(f"\n{'='*75}", flush=True)
    print(f"  STARTING TRAINING: Resolution {res_str} (Batch Size: {batch_size}, Max Epochs: {epochs})", flush=True)
    print(f"{'='*75}\n", flush=True)

    models_dir = os.path.join(output_dir, "models")
    histories_dir = os.path.join(output_dir, "histories")
    eval_dir = os.path.join(output_dir, "evaluations")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    model_save_path = os.path.join(models_dir, f"showroom_bbox_b0_{res_str}.keras")
    history_save_path = os.path.join(histories_dir, f"history_{res_str}.csv")
    classwise_save_path = os.path.join(eval_dir, f"classwise_acc_{res_str}.csv")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2)
    )

    train_gen = FastBBoxDataGenerator(train_df, batch_size=batch_size, target_size=target_size, num_classes=num_classes, datagen=train_datagen, shuffle=True, workers=workers)
    val_gen = FastBBoxDataGenerator(val_df, batch_size=batch_size, target_size=target_size, num_classes=num_classes, datagen=None, shuffle=False, workers=workers)

    model = build_model(res, num_classes=num_classes)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=reduce_lr_patience, min_lr=1e-6)

    t0_train = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    train_duration = time.time() - t0_train
    train_gen.close()
    val_gen.close()

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_save_path, index=False)
    print(f"\nTraining completed in {train_duration:.2f}s ({train_duration/60:.2f} min). Saved history to: {history_save_path}", flush=True)

    # Save trained model (.keras format)
    model.save(model_save_path)
    print(f"Saved model to: {model_save_path}", flush=True)

    # Save portable weights (.weights.h5 format for cross-version compatibility)
    weights_save_path = model_save_path.replace(".keras", ".weights.h5")
    model.save_weights(weights_save_path)
    print(f"Saved portable weights to: {weights_save_path}", flush=True)

    # Evaluate on test set
    print(f"\nEvaluating on full test set ({len(test_df)} samples) at {res_str}...", flush=True)
    test_gen = FastBBoxDataGenerator(test_df, batch_size=batch_size, target_size=target_size, num_classes=num_classes, datagen=None, shuffle=False, workers=workers)

    t0_eval = time.time()
    preds = model.predict(test_gen, verbose=1)
    eval_duration = time.time() - t0_eval
    test_gen.close()

    y_pred = np.argmax(preds, axis=1)
    y_true = np.array(test_df["label"].values, dtype=int)
    total_samples = len(test_df)
    total_correct = int(np.sum(y_pred == y_true))
    micro_acc = (total_correct / total_samples) * 100.0

    # Compute per-class accuracy
    label_to_model = meta_dict["label_to_model"]
    class_names = meta_dict["class_names"]
    img_dir = meta_dict["img_dir"]
    meta_lookup = test_df.drop_duplicates(subset=["label"]).set_index("label")
    label_to_make_id = meta_lookup["make_id"].to_dict()
    label_to_make_name = meta_lookup["make_name"].to_dict()
    label_to_sample = meta_lookup["file_path"].to_dict()

    class_records = []
    for c in range(num_classes):
        mask = (y_true == c)
        c_total = int(np.sum(mask))
        c_correct = int(np.sum(mask & (y_pred == c)))
        c_acc = (c_correct / c_total * 100.0) if c_total > 0 else 0.0

        m_id = label_to_model[c]
        mk_id = label_to_make_id.get(c, None)
        folder_rel = f"{mk_id}/{m_id}" if mk_id is not None else str(m_id)
        folder_full = os.path.join(img_dir, folder_rel)

        class_records.append({
            "class_index": c,
            "make_id": mk_id,
            "make_name": label_to_make_name.get(c, "Unknown"),
            "model_id": m_id,
            "class_name": class_names[c],
            "folder_path": folder_full,
            "sample_image": label_to_sample.get(c, ""),
            "total_samples": c_total,
            "correct_samples": c_correct,
            "accuracy_pct": round(c_acc, 2)
        })

    class_acc_df = pd.DataFrame(class_records)
    class_acc_df.to_csv(classwise_save_path, index=False)

    macro_mean = float(class_acc_df["accuracy_pct"].mean())
    macro_median = float(class_acc_df["accuracy_pct"].median())
    best_val_acc = float(history_df["val_accuracy"].max()) * 100.0
    epochs_trained = len(history_df)

    print(f"\n--- Results for Retrained Resolution {res_str} ---")
    print(f"  Epochs Trained       : {epochs_trained}")
    print(f"  Best Val Accuracy    : {best_val_acc:.2f}%")
    print(f"  Test Micro Accuracy  : {micro_acc:.2f}% ({total_correct}/{total_samples})")
    print(f"  Test Macro Mean Acc  : {macro_mean:.2f}%")
    print(f"  Test Macro Median Acc: {macro_median:.2f}%")
    print(f"  Evaluation Time      : {eval_duration:.2f}s ({total_samples/eval_duration:.1f} imgs/s)")

    # Clean up model from GPU memory to prevent memory fragmentation
    del model
    tf.keras.backend.clear_session()

    res_record = {
        "resolution": res_str,
        "res_dim": res,
        "batch_size": batch_size,
        "epochs_trained": epochs_trained,
        "best_val_acc_pct": round(best_val_acc, 2),
        "test_micro_acc_pct": round(micro_acc, 2),
        "test_macro_mean_pct": round(macro_mean, 2),
        "test_macro_median_pct": round(macro_median, 2),
        "total_correct": total_correct,
        "total_samples": total_samples,
        "training_time_min": round(train_duration / 60.0, 2),
        "eval_time_sec": round(eval_duration, 2)
    }

    json_path = os.path.join(eval_dir, f"summary_{res_str}.json")
    with open(json_path, "w") as jf:
        json.dump(res_record, jf, indent=2)
    print(f"Saved resolution summary JSON to: {json_path}", flush=True)

    return res_record


def generate_comparison_plots(summary_df, output_dir):
    """
    Renders accuracy vs resolution and training time curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

    # Plot 1: Micro and Macro Accuracy vs Resolution
    x = summary_df["res_dim"]
    ax1.plot(x, summary_df["test_micro_acc_pct"], marker="o", linewidth=2.5, color="#1d3557", label="Test Micro Accuracy (%)")
    ax1.plot(x, summary_df["test_macro_mean_pct"], marker="s", linewidth=2.0, color="#457b9d", linestyle="--", label="Test Macro Mean Accuracy (%)")
    ax1.plot(x, summary_df["best_val_acc_pct"], marker="^", linewidth=1.8, color="#e63946", linestyle=":", label="Best Val Accuracy (%)")

    best_idx = summary_df["test_micro_acc_pct"].idxmax()
    best_res = summary_df.loc[best_idx, "res_dim"]
    best_acc = summary_df.loc[best_idx, "test_micro_acc_pct"]
    ax1.scatter([best_res], [best_acc], color="#e63946", s=150, edgecolors="black", zorder=5)
    ax1.annotate(f"Optimal: {best_res}x{best_res}\n({best_acc:.2f}%)", xy=(best_res, best_acc),
                 xytext=(best_res, best_acc + 1.2), ha="center", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax1.set_title("CompCars Retrained Accuracy vs. Resolution (EfficientNet-B0)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Input Resolution Dimension (px)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Plot 2: Training Time vs Resolution
    ax2.plot(x, summary_df["training_time_min"], marker="o", linewidth=2.2, color="#2a9d8f")
    ax2.set_title("Training Duration vs. Input Resolution (RTX 4090)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Input Resolution Dimension (px)", fontsize=11)
    ax2.set_ylabel("Training Time (minutes)", fontsize=11)
    ax2.set_xticks(x)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "retrained_resolution_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison visualization to: {plot_path}", flush=True)


def compile_summary_and_plot(output_dir, all_resolutions):
    """
    Compiles all individual resolution summary JSONs into a master summary CSV and updates plots.
    """
    eval_dir = os.path.join(output_dir, "evaluations")
    summary_csv_path = os.path.join(output_dir, "summary_retrained_resolutions.csv")
    records = []
    for r in all_resolutions:
        json_path = os.path.join(eval_dir, f"summary_{r}x{r}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as jf:
                    records.append(json.load(jf))
            except Exception as e:
                print(f"Warning: could not read {json_path}: {e}")

    if records:
        df = pd.DataFrame(records).sort_values("res_dim").reset_index(drop=True)
        df.to_csv(summary_csv_path, index=False)
        print(f"\nUpdated master summary at: {summary_csv_path}", flush=True)
        print("\n" + "="*85)
        print("                    RETRAINING COMPARISON SUMMARY")
        print("="*85)
        cols = [c for c in ["resolution", "best_val_acc_pct", "test_micro_acc_pct", "test_macro_mean_pct", "training_time_min"] if c in df.columns]
        print(df[cols].to_string(index=False))
        print("="*85 + "\n")
        generate_comparison_plots(df, output_dir)
        return df
    return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Resolution Retraining Pipeline")
    parser.add_argument("--resolutions", type=str, default="224,256,384,512,576,640,720",
                        help="Comma-separated resolutions to retrain")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max epochs per resolution")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to CompCars dataset")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Path to output directory")
    parser.add_argument("--workers", type=int, default=16,
                        help="Data preloading worker threads")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of resolutions to train concurrently in parallel (e.g. 2)")
    parser.add_argument("--reduce_lr_patience", type=int, default=1,
                        help="Patience for ReduceLROnPlateau callback (default: 1)")

    args = parser.parse_args()
    resolutions = [int(r.strip()) for r in args.resolutions.split(",") if r.strip()]

    print(f"\n{'#'*80}")
    print(f"  MULTI-RESOLUTION RETRAINING EXPERIMENT")
    print(f"  Resolutions to train : {resolutions}")
    print(f"  Parallel Concurrency : {args.parallel}")
    print(f"  Dataset directory    : {args.data_dir}")
    print(f"  Output directory     : {args.output_dir}")
    print(f"{'#'*80}\n", flush=True)

    eval_dir = os.path.join(args.output_dir, "evaluations")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    done_resolutions = set()
    for res in resolutions:
        json_path = os.path.join(eval_dir, f"summary_{res}x{res}.json")
        model_path = os.path.join(models_dir, f"showroom_bbox_b0_{res}x{res}.keras")
        if os.path.exists(json_path) and os.path.exists(model_path):
            done_resolutions.add(res)

    pending_resolutions = [r for r in resolutions if r not in done_resolutions]

    if not pending_resolutions:
        print("All specified resolutions are already completed! Compiling summary...", flush=True)
        compile_summary_and_plot(args.output_dir, resolutions)
        return

    if args.parallel > 1:
        print(f"Running {len(pending_resolutions)} pending resolution(s) with concurrency = {args.parallel}...", flush=True)
        logs_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        def run_worker(res):
            log_path = os.path.join(logs_dir, f"train_{res}x{res}.log")
            worker_threads = max(4, args.workers // args.parallel)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--resolutions", str(res),
                "--epochs", str(args.epochs),
                "--data_dir", args.data_dir,
                "--output_dir", args.output_dir,
                "--workers", str(worker_threads),
                "--parallel", "1"
            ]
            print(f"[*] [Started] Resolution {res}x{res} (preload threads: {worker_threads}, log: {log_path})", flush=True)
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    lf.write(line)
                    lf.flush()
                    s_line = line.strip()
                    if s_line.startswith("Epoch ") or "Results for Retrained Resolution" in s_line or "Test Micro Accuracy" in s_line or "Best Val Accuracy" in s_line:
                        print(f"[{res}x{res}] {s_line}", flush=True)
                proc.wait()
            if proc.returncode != 0:
                print(f"[!] [Failed] Resolution {res}x{res} exited with code {proc.returncode}. See log: {log_path}", flush=True)
            else:
                print(f"[✓] [Completed] Resolution {res}x{res} successfully trained and evaluated!", flush=True)
                compile_summary_and_plot(args.output_dir, resolutions)
            return res, proc.returncode

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(run_worker, r) for r in pending_resolutions]
            for fut in as_completed(futures):
                fut.result()

        compile_summary_and_plot(args.output_dir, resolutions)

    else:
        # Sequential execution
        train_df, val_df, test_df, meta_dict = load_dataset_metadata(args.data_dir)
        for res in resolutions:
            if res in done_resolutions:
                print(f"Resolution {res}x{res} already completed. Skipping...", flush=True)
                continue

            train_single_resolution(
                res=res,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                meta_dict=meta_dict,
                output_dir=args.output_dir,
                epochs=args.epochs,
                workers=args.workers,
                reduce_lr_patience=args.reduce_lr_patience
            )
            compile_summary_and_plot(args.output_dir, resolutions)

        compile_summary_and_plot(args.output_dir, resolutions)


if __name__ == "__main__":
    main()
