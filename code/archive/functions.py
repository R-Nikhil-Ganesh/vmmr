import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.efficientnet import preprocess_input
from IPython.display import display

# --- 1. Model & Metadata Loader ---
def load_model_and_metadata(model_path, data_dir):
    """
    Loads the trained Keras model and extracts CompCars 431 benchmark metadata and label dictionaries.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at '{data_dir}'")

    print(f"Loading model from '{model_path}'...")
    model = load_model(model_path)
    
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
    
    meta_dict = {
        "data_dir": data_dir,
        "img_dir": img_dir,
        "label_dir": label_dir,
        "split_dir": split_dir,
        "make_names": make_names,
        "model_names": model_names,
        "train_lines": train_lines,
        "test_lines": test_lines,
        "unique_models": unique_models,
        "num_classes": num_classes,
        "model_to_label": model_to_label,
        "label_to_model": label_to_model,
        "class_names": class_names
    }
    print(f"Loaded model and metadata successfully. Total benchmark classes: {num_classes}")
    return model, meta_dict


def _load_crop_and_resize(args):
    file_path, box, target_size = args
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        cropped = img.crop(box)
        resized = cropped.resize(target_size, Image.BILINEAR)
        return np.array(resized, dtype=np.float32)


# --- 2. Bounding Box Data Generator ---
class BBoxDataGenerator(Sequence):
    """
    Custom Keras Sequence Generator that crops each car image according to its
    annotated bounding box coordinates [x1, y1, x2, y2] using high-throughput
    multithreading before resizing and feeding into the model for inference.
    """
    def __init__(self, df, batch_size=64, target_size=(224, 224), num_classes=431, workers=16, **kwargs):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        items = [
            (row["file_path"], (row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]), self.target_size)
            for _, row in batch_df.iterrows()
        ]
        batch_x = list(self.pool.map(_load_crop_and_resize, items))
        batch_x = preprocess_input(np.array(batch_x))
        return batch_x

    def close(self):
        self.pool.shutdown(wait=False)


# --- 3. Test DataFrame Preparation ---
def prepare_test_dataframe(data_dir, test_lines, meta_dict):
    """
    Parses test set paths and corresponding label files (.txt) to extract bounding box coordinates and viewpoints.
    """
    label_dir = os.path.join(data_dir, "label")
    img_dir = os.path.join(data_dir, "image")
    make_names = meta_dict["make_names"]
    model_names = meta_dict["model_names"]
    model_to_label = meta_dict["model_to_label"]
    
    test_rows = []
    for rel_path in test_lines:
        parts = rel_path.split("/")
        make_id = int(parts[0])
        model_id = int(parts[1])
        image_name = parts[3]

        label_file = os.path.join(label_dir, os.path.splitext(rel_path)[0] + ".txt")
        with open(label_file, "r") as lf:
            lbl_lines = lf.readlines()
        
        viewpoint = int(lbl_lines[0].strip())
        x1, y1, x2, y2 = [int(v) for v in lbl_lines[2].strip().split()]

        test_rows.append({
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

    test_df = pd.DataFrame(test_rows)
    print(f"Constructed test dataframe with {len(test_df)} samples across {test_df['label'].nunique()} classes.")
    return test_df


# --- 4. Class-wise Accuracy Computation ---
def compute_classwise_accuracy(model, test_df, meta_dict, csv_path="compcars_showroom_classwise_accuracy.csv", batch_size=64, target_size=(224, 224), force_recompute=False, data_dir=None, workers=16):
    """
    Loads cached class-wise accuracy metrics from CSV or executes model.predict() on test set.
    """
    if data_dir is None and meta_dict is not None:
        data_dir = meta_dict.get("data_dir")
    
    img_dir = meta_dict.get("img_dir") if meta_dict and "img_dir" in meta_dict else (os.path.join(data_dir, "image") if data_dir else None)
    num_classes = meta_dict["num_classes"] if meta_dict and "num_classes" in meta_dict else test_df["label"].nunique()
    label_to_model = meta_dict.get("label_to_model") if meta_dict else None
    class_names = meta_dict.get("class_names") if meta_dict else None

    if os.path.exists(csv_path) and not force_recompute:
        print(f"Loading precomputed class-wise accuracy metrics from '{csv_path}'...")
        class_acc_df = pd.read_csv(csv_path)
        # Dynamically re-anchor paths using current img_dir
        if img_dir is not None:
            class_acc_df["folder_path"] = class_acc_df.apply(
                lambda r: os.path.join(img_dir, f"{int(r['make_id'])}/{int(r['model_id'])}"),
                axis=1
            )
            if "sample_image" in class_acc_df.columns:
                class_acc_df["sample_image"] = class_acc_df["sample_image"].apply(
                    lambda p: os.path.join(img_dir, str(p).split("/image/")[-1]) if "/image/" in str(p) else p
                )
    else:
        print("Generating test set predictions with model.predict()...")
        test_generator = BBoxDataGenerator(test_df, batch_size=batch_size, target_size=target_size, num_classes=num_classes, workers=workers)
        test_pred_probs = model.predict(test_generator, verbose=1)
        test_generator.close()
        y_pred = np.argmax(test_pred_probs, axis=1)
        y_true = np.array(test_df["label"].values, dtype=int)

        meta_lookup = test_df.drop_duplicates(subset=["label"]).set_index("label")
        label_to_make_id = meta_lookup["make_id"].to_dict()
        label_to_make_name = meta_lookup["make_name"].to_dict()
        label_to_sample = meta_lookup["file_path"].to_dict()

        records = []
        for c in range(num_classes):
            mask = (y_true == c)
            total_samples = int(np.sum(mask))
            correct_samples = int(np.sum(mask & (y_pred == c)))
            acc = (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0
            
            m_id = label_to_model[c]
            mk_id = label_to_make_id.get(c, None)
            folder_rel = f"{mk_id}/{m_id}" if mk_id is not None else str(m_id)
            folder_full = os.path.join(img_dir, folder_rel)

            records.append({
                "class_index": c,
                "make_id": mk_id,
                "make_name": label_to_make_name.get(c, "Unknown"),
                "model_id": m_id,
                "class_name": class_names[c],
                "folder_path": folder_full,
                "sample_image": label_to_sample.get(c, ""),
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "accuracy_pct": round(acc, 2)
            })

        class_acc_df = pd.DataFrame(records)
        class_acc_df.to_csv(csv_path, index=False)
        print(f"Saved full 431-model metrics with folder paths to '{csv_path}'")

    return class_acc_df


# --- 5. Global Statistics & Tier Summary ---
def get_global_statistics(class_acc_df):
    """
    Computes global macro mean, median, standard deviation, and performance tier breakdown.
    """
    mean_acc = class_acc_df["accuracy_pct"].mean()
    median_acc = class_acc_df["accuracy_pct"].median()
    std_acc = class_acc_df["accuracy_pct"].std()
    min_acc = class_acc_df["accuracy_pct"].min()
    max_acc = class_acc_df["accuracy_pct"].max()
    num_classes = len(class_acc_df)

    print(f"\n================ Global Accuracy Metrics ({num_classes} Models) ================")
    print(f"  Macro Mean Accuracy   : {mean_acc:.2f}%")
    print(f"  Median Accuracy       : {median_acc:.2f}%")
    print(f"  Standard Deviation    : {std_acc:.2f}%")
    print(f"  Min Accuracy          : {min_acc:.2f}%")
    print(f"  Max Accuracy          : {max_acc:.2f}%")
    print(f"======================================================================\n")

    tier_bins = [0, 25, 50, 75, 90, 100]
    tier_labels = ["< 25% (Critical)", "25% - 50% (Poor)", "50% - 75% (Fair)", "75% - 90% (Good)", "90% - 100% (Excellent)"]
    class_acc_df["tier"] = pd.cut(class_acc_df["accuracy_pct"], bins=tier_bins, labels=tier_labels, include_lowest=True)

    tier_counts = class_acc_df["tier"].value_counts().reindex(tier_labels)
    tier_summary = pd.DataFrame({
        "Performance Tier": tier_labels,
        "Number of Models": tier_counts.values,
        "Share of All Models": [f"{v / num_classes * 100:.1f}%" for v in tier_counts.values]
    })
    
    stats_dict = {
        "mean": mean_acc,
        "median": median_acc,
        "std": std_acc,
        "min": min_acc,
        "max": max_acc
    }
    return stats_dict, tier_summary


# --- 6. Visualizations: Sorted Accuracy Bar Plot & Distribution ---
def plot_accuracy_spectrum(class_acc_df, figsize=(18, 9.5), dpi=100):
    """
    Renders a two-panel visualization:
    1. Sorted bar plot of all 431 models from lowest to highest accuracy with 50%, 70%, 90% threshold lines.
    2. Distribution histogram aligned with threshold colors.
    """
    sorted_df = class_acc_df.sort_values(by="accuracy_pct", ascending=True).reset_index(drop=True)
    sorted_df["rank"] = range(1, len(sorted_df) + 1)
    n = len(sorted_df)

    mean_acc = sorted_df["accuracy_pct"].mean()
    median_acc = sorted_df["accuracy_pct"].median()

    def get_tier_color(acc):
        if acc < 50.0: return "#d90429"      # Critical (<50%)
        elif acc < 70.0: return "#f77f00"    # Sub-baseline (50% - 70%)
        elif acc < 90.0: return "#06d6a0"    # Target (70% - 90%)
        else: return "#1b9aaa"               # High Precision (≥90%)

    tier_colors = [get_tier_color(a) for a in sorted_df["accuracy_pct"]]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # Panel 1: Sorted bar plot
    bars = ax0.bar(sorted_df["rank"], sorted_df["accuracy_pct"], color=tier_colors, width=0.88, alpha=0.9)

    ax0.axhline(50, color="#d90429", linestyle="--", linewidth=1.5, alpha=0.9)
    ax0.axhline(70, color="#f77f00", linestyle="--", linewidth=1.5, alpha=0.9)
    ax0.axhline(90, color="#1b9aaa", linestyle="--", linewidth=1.5, alpha=0.9)

    ax0.axhline(mean_acc, color="#3a0ca3", linestyle=":", linewidth=1.6)
    ax0.axhline(median_acc, color="#7209b7", linestyle="-.", linewidth=1.6)

    cnt_50 = int((sorted_df["accuracy_pct"] < 50.0).sum())
    cnt_70 = int((sorted_df["accuracy_pct"] < 70.0).sum())
    cnt_90 = int((sorted_df["accuracy_pct"] >= 90.0).sum())

    ax0.text(n + 3, 50, f"50% Threshold\n({cnt_50} models < 50%)", va="center", ha="left", color="#d90429", fontsize=8.5, fontweight="bold")
    ax0.text(n + 3, 70, f"70% Threshold\n({cnt_70} models < 70%)", va="center", ha="left", color="#f77f00", fontsize=8.5, fontweight="bold")
    ax0.text(n + 3, 90, f"90% Threshold\n({cnt_90} models ≥ 90%)", va="center", ha="left", color="#1b9aaa", fontsize=8.5, fontweight="bold")

    lowest_name = sorted_df.iloc[0]["class_name"].strip()
    lowest_acc = sorted_df.iloc[0]["accuracy_pct"]
    ax0.annotate(f"#1: {lowest_name} ({lowest_acc}%)",
                 xy=(1, lowest_acc), xytext=(12, 22),
                 arrowprops=dict(arrowstyle="->", color="#d90429", lw=1.5),
                 fontsize=9, fontweight="bold", color="#d90429",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d90429", alpha=0.9))

    magotan_sub = sorted_df[sorted_df["model_id"] == 504]
    if not magotan_sub.empty:
        m_idx = magotan_sub.index[0] + 1
        m_acc = magotan_sub["accuracy_pct"].values[0]
        ax0.annotate(f"#{m_idx}: Magotan ({m_acc}%)",
                 xy=(m_idx, m_acc), xytext=(m_idx + 22, m_acc + 15),
                 arrowprops=dict(arrowstyle="->", color="#d90429", lw=1.5),
                 fontsize=9, fontweight="bold", color="#d90429",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d90429", alpha=0.9))

    highest_name = sorted_df.iloc[-1]["class_name"].strip()
    highest_acc = sorted_df.iloc[-1]["accuracy_pct"]
    ax0.annotate(f"#{n}: {highest_name} ({highest_acc}%)",
                 xy=(n, highest_acc), xytext=(n - 120, 95),
                 arrowprops=dict(arrowstyle="->", color="#1b9aaa", lw=1.5),
                 fontsize=9, fontweight="bold", color="#1b9aaa",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1b9aaa", alpha=0.9))

    acc_200 = round(sorted_df.iloc[199]["accuracy_pct"]) if len(sorted_df) > 199 else 65
    ticks = [1, 50, 100, 150, 200, 250, 300, 350, 400, n]
    tick_labels = ["#1\n(Lowest)", "#50", "#100", "#150", f"#200\n(~{acc_200}%)", "#250", "#300", "#350", "#400", f"#{n}\n(Highest)"]
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(tick_labels, fontsize=9)
    ax0.set_xlim(-5, n + 48)
    ax0.set_ylim(0, 105)
    ax0.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax0.set_title(f"All {n} Car Models Sorted by Test Accuracy (Lowest → Highest)", fontsize=12.5, fontweight="bold", pad=10)
    ax0.grid(axis="y", alpha=0.3, linestyle="--")

    cnt_crit = int((sorted_df["accuracy_pct"] < 50.0).sum())
    cnt_sub = int(((sorted_df["accuracy_pct"] >= 50.0) & (sorted_df["accuracy_pct"] < 70.0)).sum())
    cnt_tgt = int(((sorted_df["accuracy_pct"] >= 70.0) & (sorted_df["accuracy_pct"] < 90.0)).sum())
    cnt_high = int((sorted_df["accuracy_pct"] >= 90.0).sum())

    pct_crit = (cnt_crit / n * 100.0) if n > 0 else 0.0
    pct_sub = (cnt_sub / n * 100.0) if n > 0 else 0.0
    pct_tgt = (cnt_tgt / n * 100.0) if n > 0 else 0.0
    pct_high = (cnt_high / n * 100.0) if n > 0 else 0.0

    legend_elements = [
        Patch(facecolor="#d90429", label=f"< 50% Critical ({cnt_crit} models, {pct_crit:.1f}%)"),
        Patch(facecolor="#f77f00", label=f"50% - 70% Sub-baseline ({cnt_sub} models, {pct_sub:.1f}%)"),
        Patch(facecolor="#06d6a0", label=f"70% - 90% Target ({cnt_tgt} models, {pct_tgt:.1f}%)"),
        Patch(facecolor="#1b9aaa", label=f"≥ 90% High Precision ({cnt_high} models, {pct_high:.1f}%)"),
        plt.Line2D([0], [0], color="#3a0ca3", linestyle=":", linewidth=1.6, label=f"Macro Mean ({mean_acc:.1f}%)"),
        plt.Line2D([0], [0], color="#7209b7", linestyle="-.", linewidth=1.6, label=f"Median ({median_acc:.1f}%)"),
    ]
    ax0.legend(handles=legend_elements, loc="upper left", framealpha=0.95, fontsize=9, ncol=2)

    # Panel 2: Distribution histogram
    n_bins, bins, patches_hist = ax1.hist(sorted_df["accuracy_pct"], bins=20, edgecolor="black", alpha=0.85)
    for p, bin_left in zip(patches_hist, bins[:-1]):
        if bin_left < 50:
            p.set_facecolor("#d90429")
        elif bin_left < 70:
            p.set_facecolor("#f77f00")
        elif bin_left < 90:
            p.set_facecolor("#06d6a0")
        else:
            p.set_facecolor("#1b9aaa")

    ax1.axvline(50, color="#d90429", linestyle="--", linewidth=1.5)
    ax1.axvline(70, color="#f77f00", linestyle="--", linewidth=1.5)
    ax1.axvline(90, color="#1b9aaa", linestyle="--", linewidth=1.5)
    ax1.axvline(mean_acc, color="#3a0ca3", linestyle=":", linewidth=1.6, label=f"Mean ({mean_acc:.1f}%)")
    ax1.axvline(median_acc, color="#7209b7", linestyle="-.", linewidth=1.6, label=f"Median ({median_acc:.1f}%)")

    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Models", fontsize=11, fontweight="bold")
    ax1.set_title("Distribution of Model Accuracies Across Threshold Tiers", fontsize=11.5, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("CompCars Showroom: Class-wise Accuracy Spectrum & Tier Breakdown", fontsize=14, y=1.002, fontweight="bold")
    plt.tight_layout()
    plt.show()


# --- 7. Search & Top/Bottom Models ---
def search_models(class_acc_df, query=None, min_acc=0.0, max_acc=100.0, limit=20, ascending=False):
    """
    Filters and searches across all 431 models by brand, model name, or accuracy range.
    """
    res = class_acc_df[
        (class_acc_df["accuracy_pct"] >= min_acc) &
        (class_acc_df["accuracy_pct"] <= max_acc)
    ]
    if query:
        cond = res["class_name"].str.contains(query, case=False, na=False)
        if "make_name" in res.columns:
            cond = cond | res["make_name"].str.contains(query, case=False, na=False)
        res = res[cond]
    display_cols = ["model_id", "make_id", "make_name", "class_name", "accuracy_pct", "correct_samples", "total_samples", "folder_path"]
    available_cols = [c for c in display_cols if c in res.columns]
    return res[available_cols].sort_values(by="accuracy_pct", ascending=ascending).head(limit)

def get_top_and_bottom_models(class_acc_df, n=10):
    """
    Returns top N and bottom N models by accuracy.
    """
    top_models = search_models(class_acc_df, limit=n, ascending=False)
    bottom_models = search_models(class_acc_df, limit=n, ascending=True)
    return top_models, bottom_models


# --- 8. Make-wise Accuracy & Visualizations ---
def compute_makewise_accuracy(class_acc_df, csv_path="compcars_showroom_makewise_accuracy.csv"):
    """
    Computes sample-weighted micro-accuracy and macro-average accuracy for all 75 brands.
    """
    make_records = []
    for (mk_id, mk_name), group in class_acc_df.groupby(["make_id", "make_name"]):
        tot_samples = group["total_samples"].sum()
        corr_samples = group["correct_samples"].sum()
        micro_acc = round(corr_samples / tot_samples * 100.0, 2) if tot_samples > 0 else 0.0
        macro_acc = round(group["accuracy_pct"].mean(), 2)
        
        best_row = group.loc[group["accuracy_pct"].idxmax()]
        worst_row = group.loc[group["accuracy_pct"].idxmin()]
        
        make_records.append({
            "make_id": int(mk_id),
            "make_name": mk_name,
            "num_models": len(group),
            "total_samples": int(tot_samples),
            "correct_samples": int(corr_samples),
            "accuracy_pct": micro_acc,
            "macro_acc_pct": macro_acc,
            "best_model": f"{best_row['class_name']} ({best_row['accuracy_pct']}%)",
            "worst_model": f"{worst_row['class_name']} ({worst_row['accuracy_pct']}%)"
        })

    make_acc_df = pd.DataFrame(make_records).sort_values(by="accuracy_pct", ascending=True).reset_index(drop=True)
    make_acc_df["rank"] = range(1, len(make_acc_df) + 1)
    n_makes = len(make_acc_df)

    if csv_path:
        make_acc_df.to_csv(csv_path, index=False)
        print(f"Saved full metrics for all {n_makes} brands to '{csv_path}'")

    tier_50 = make_acc_df[make_acc_df["accuracy_pct"] < 50.0]
    tier_70 = make_acc_df[(make_acc_df["accuracy_pct"] >= 50.0) & (make_acc_df["accuracy_pct"] < 70.0)]
    tier_90 = make_acc_df[(make_acc_df["accuracy_pct"] >= 70.0) & (make_acc_df["accuracy_pct"] < 90.0)]
    tier_top = make_acc_df[make_acc_df["accuracy_pct"] >= 90.0]

    def get_example_brands(df_tier, max_b=4):
        return ", ".join(df_tier["make_name"].head(max_b)) if not df_tier.empty else "None"

    tier_summary = pd.DataFrame([
        {"Tier": "< 50% (Critical)", "Brand Count": len(tier_50), "Share": f"{len(tier_50)/n_makes*100:.1f}%", "Example Brands": get_example_brands(tier_50, 3)},
        {"Tier": "50% - 70% (Sub-baseline)", "Brand Count": len(tier_70), "Share": f"{len(tier_70)/n_makes*100:.1f}%", "Example Brands": get_example_brands(tier_70, 5)},
        {"Tier": "70% - 90% (Target Range)", "Brand Count": len(tier_90), "Share": f"{len(tier_90)/n_makes*100:.1f}%", "Example Brands": get_example_brands(tier_90, 5)},
        {"Tier": "≥ 90% (High Precision)", "Brand Count": len(tier_top), "Share": f"{len(tier_top)/n_makes*100:.1f}%", "Example Brands": get_example_brands(tier_top, 5)}
    ])

    return make_acc_df, tier_summary

def plot_makewise_accuracy(make_acc_df, figsize=(18, 13.5), dpi=100):
    """
    Renders a two-panel horizontal bar chart ranking all 75 brands from lowest to highest accuracy.
    """
    n_makes = len(make_acc_df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    half = int(np.ceil(n_makes / 2))
    df_lower = make_acc_df.iloc[:half].reset_index(drop=True)
    df_upper = make_acc_df.iloc[half:].reset_index(drop=True)

    def get_color(acc):
        if acc < 50.0: return "#d90429"
        elif acc < 70.0: return "#f77f00"
        elif acc < 90.0: return "#06d6a0"
        else: return "#1b9aaa"

    panels = [
        (ax1, df_lower, f"Lower Half: Ranks #1 – #{half} ({df_lower.iloc[0]['accuracy_pct']}% → {df_lower.iloc[-1]['accuracy_pct']}%)"),
        (ax2, df_upper, f"Upper Half: Ranks #{half+1} – #{n_makes} ({df_upper.iloc[0]['accuracy_pct']}% → {df_upper.iloc[-1]['accuracy_pct']}%)")
    ]

    for ax, sub_df, title in panels:
        colors = [get_color(a) for a in sub_df["accuracy_pct"]]
        bars = ax.barh(sub_df["make_name"], sub_df["accuracy_pct"], color=colors, height=0.72, alpha=0.92)
        
        for bar, (_, r) in zip(bars, sub_df.iterrows()):
            w = bar.get_width()
            ax.text(
                w + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{r['accuracy_pct']}% ({r['num_models']}m, {r['total_samples']}i)",
                va="center", ha="left", fontsize=8, fontweight="bold", color="#222222"
            )
        
        ax.axvline(50, color="#d90429", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.axvline(70, color="#f77f00", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.axvline(90, color="#1b9aaa", linestyle="--", linewidth=1.2, alpha=0.85)
        
        ax.set_xlim(0, 116)
        ax.set_xlabel("Sample-Weighted Test Accuracy (%)", fontsize=10, fontweight="bold", labelpad=6)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    tier_50_count = (make_acc_df["accuracy_pct"] < 50.0).sum()
    tier_70_count = ((make_acc_df["accuracy_pct"] >= 50.0) & (make_acc_df["accuracy_pct"] < 70.0)).sum()
    tier_90_count = ((make_acc_df["accuracy_pct"] >= 70.0) & (make_acc_df["accuracy_pct"] < 90.0)).sum()
    tier_top_count = (make_acc_df["accuracy_pct"] >= 90.0).sum()

    legend_elements = [
        Patch(facecolor="#d90429", label=f"< 50% Critical ({tier_50_count} brands)"),
        Patch(facecolor="#f77f00", label=f"50% - 70% Sub-baseline ({tier_70_count} brands)"),
        Patch(facecolor="#06d6a0", label=f"70% - 90% Target ({tier_90_count} brands)"),
        Patch(facecolor="#1b9aaa", label=f"≥ 90% High Precision ({tier_top_count} brands)")
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=9.5, framealpha=0.95)

    plt.suptitle("CompCars Showroom: Accuracy Breakdown Across ALL 75 Car Brands (Sorted Lowest → Highest)", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()

def view_make_summary(class_acc_df, make_query):
    """
    Queries any brand and displays all its individual models sorted from lowest to highest accuracy.
    """
    match = class_acc_df[class_acc_df["make_name"].str.contains(make_query, case=False, na=False)]
    if match.empty:
        print(f"No brand found matching '{make_query}'.")
        return None
    
    brand_name = match.iloc[0]["make_name"]
    tot_m = len(match)
    tot_s = match["total_samples"].sum()
    tot_c = match["correct_samples"].sum()
    brand_acc = round(tot_c / tot_s * 100.0, 2)
    
    print(f"=== Brand Summary: {brand_name} ({tot_m} models, {tot_s} test images, Overall Acc: {brand_acc}%) ===")
    display_cols = ["model_id", "class_name", "accuracy_pct", "correct_samples", "total_samples", "folder_path"]
    summary_df = match[display_cols].sort_values(by="accuracy_pct", ascending=True).reset_index(drop=True)
    display(summary_df)
    return summary_df


# --- 9. CCTV Overlap Analysis ---
def analyze_cctv_overlap(class_acc_df, mappings_file="compcars_431_model_mappings.csv"):
    """
    Compares accuracy of showroom models that overlap with the CompCars CCTV surveillance dataset.
    """
    if not os.path.exists(mappings_file):
        if os.path.exists(os.path.join("../", mappings_file)):
            mappings_file = os.path.join("../", mappings_file)
    
    if os.path.exists(mappings_file):
        mappings_df = pd.read_csv(mappings_file)
        cctv_merged = class_acc_df.merge(
            mappings_df[["class_index", "in_surveillance", "sv_model_id", "body_type"]],
            on="class_index",
            how="left"
        )
        
        cctv_shared = cctv_merged[cctv_merged["in_surveillance"] == True]
        showroom_only = cctv_merged[cctv_merged["in_surveillance"] == False]
        
        print(f"Total Benchmark Models               : {len(cctv_merged)}")
        print(f"  - Shared with CompCars CCTV        : {len(cctv_shared)} (Macro Mean Acc: {cctv_shared['accuracy_pct'].mean():.2f}%)")
        print(f"  - Showroom Only                    : {len(showroom_only)} (Macro Mean Acc: {showroom_only['accuracy_pct'].mean():.2f}%)")
        
        low_acc_cctv = cctv_shared[cctv_shared["accuracy_pct"] < 70.0]
        print(f"\nCCTV-Shared Models with < 70% Accuracy : {len(low_acc_cctv)} / {len(cctv_shared)} ({len(low_acc_cctv)/len(cctv_shared)*100:.1f}%)")
        
        print("\nTop 10 Lowest Accuracy CCTV-Shared Models:")
        low_cctv_display = low_acc_cctv[["model_id", "sv_model_id", "make_name", "class_name", "body_type", "accuracy_pct", "total_samples"]].sort_values("accuracy_pct").head(10)
        display(low_cctv_display)
        return cctv_merged, low_cctv_display
    else:
        print(f"Mappings file '{mappings_file}' not found.")
        return None, None


# --- 10. Interactive Filesystem Image Inspection with Bounding Boxes ---
def view_model_images(class_acc_df, query, max_images=None, cols=4, show_bbox=True, year=None, img_dir=None, data_dir=None):
    """
    Look up any car model and display its actual images loaded directly from the filesystem folder,
    with official bounding box annotations overlaid in a clean grid layout.
    """
    viewpoint_map = {
        -1: "Uncertain",
        1: "Front",
        2: "Rear",
        3: "Side",
        4: "Front-side",
        5: "Rear-side"
    }

    if isinstance(query, int):
        match = class_acc_df[class_acc_df["model_id"] == query]
    else:
        match = class_acc_df[class_acc_df["class_name"].str.contains(str(query), case=False, na=False)]
    
    if match.empty:
        print(f"No model found matching '{query}'.")
        return
    
    row = match.iloc[0]
    
    if img_dir is None and data_dir is not None:
        img_dir = os.path.join(data_dir, "image")

    # Dynamically resolve folder path
    folder = None
    if img_dir is not None:
        cand = os.path.join(img_dir, f"{int(row['make_id'])}/{int(row['model_id'])}")
        if os.path.exists(cand):
            folder = cand
    
    if folder is None and "folder_path" in row and os.path.exists(str(row["folder_path"])):
        folder = str(row["folder_path"])

    if folder is None:
        folder = os.path.join(img_dir, f"{int(row['make_id'])}/{int(row['model_id'])}") if img_dir else str(row.get("folder_path", ""))

    print(f"Model: {row['class_name']} (model_id: {row['model_id']}, make_id: {row.get('make_id', 'N/A')})")
    print(f"Accuracy: {row['accuracy_pct']}% ({row['correct_samples']}/{row['total_samples']} correct)")
    print(f"Filesystem Folder: {folder}")
    
    found_files = []
    if os.path.exists(folder):
        for root, _, files in os.walk(folder):
            rel_year = os.path.basename(root)
            if year is not None and str(year) != str(rel_year):
                continue
            for f in sorted(files):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    found_files.append(os.path.join(root, f))
                    if max_images is not None and len(found_files) >= max_images:
                        break
            if max_images is not None and len(found_files) >= max_images:
                break
    
    total_found = len(found_files)
    if total_found == 0:
        print(f"No image files found in folder for year={year}.")
        return
        
    print(f"Displaying {total_found} image(s) in a {cols}-column grid...")
    
    num_rows = int(np.ceil(total_found / cols))
    fig, axes = plt.subplots(num_rows, cols, figsize=(4.2 * cols, 3.4 * num_rows))
    
    if num_rows == 1 and cols == 1:
        axes_flat = [axes]
    elif num_rows == 1 or cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = list(axes.flatten())

    for idx, p in enumerate(found_files):
        ax = axes_flat[idx]
        
        lbl_p = p.replace("/image/", "/label/").rsplit(".", 1)[0] + ".txt"
        bbox = None
        vp_name = "Unknown"
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as lf:
                lbl_lines = lf.readlines()
            if len(lbl_lines) >= 3:
                vp = int(lbl_lines[0].strip())
                vp_name = viewpoint_map.get(vp, f"VP {vp}")
                x1, y1, x2, y2 = [int(v) for v in lbl_lines[2].strip().split()]
                bbox = (x1, y1, x2, y2)

        with Image.open(p) as im:
            im_rgb = im.convert("RGB")
            ax.imshow(im_rgb)
            img_year = os.path.basename(os.path.dirname(p))
            fname = os.path.basename(p)

            if bbox and show_bbox:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), w, h,
                    linewidth=2.2, edgecolor="crimson", facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(
                    bbox[0] + 5, bbox[1] + 25, f"{vp_name} [{w}x{h}]",
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="crimson", alpha=0.85, edgecolor="none")
                )
                ax.set_title(f"#{idx+1} [Year {img_year}] {fname}\n{vp_name} BBox: {bbox}", fontsize=8.5)
            else:
                ax.set_title(f"#{idx+1} [Year {img_year}] {fname}", fontsize=8.5)
            ax.axis("off")

    for idx in range(total_found, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle(
        f"{row['class_name']} ({row.get('make_name', '')}, model_id: {row['model_id']}) - All {total_found} Images with Bounding Boxes",
        fontsize=14, y=1.002
    )
    plt.tight_layout()
    plt.show()
