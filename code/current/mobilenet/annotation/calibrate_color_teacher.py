#!/usr/bin/env python3
"""
Learn a correction from a color teacher's 13-color probabilities to the CompCars human labels.

Qwen's errors against humans are systematic (white -> silver, brown -> black/gray, champagne -> gray), so a
multinomial logistic regression on log-probabilities, fit on human-labeled train rows, can undo part of them.
The output is in the CompCars 10-color taxonomy (gray inside silver, beige/gold inside champagne), because that
is what the human labels distinguish.

Inputs are eval_color_teacher.py output folders produced with the SAME head options on two splits:
  python annotation/eval_color_teacher.py --tag cfg_train --split train ...
  python annotation/eval_color_teacher.py --tag cfg_val   --split val   ...
  python annotation/calibrate_color_teacher.py --train-dir .../teacher_eval/cfg_train --val-dir .../teacher_eval/cfg_val

Writes calibrator.pkl, calibration_metrics.json and val_predictions_calibrated.csv into --out-dir
(default <val-dir>/calibrated). Selection of C is by 5-fold CV on train; val is only reported.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "color_distillation"))

from color_heads import COLORS  # noqa: E402
from evaluate_compcars_color_gt import COMPCARS_ID2COLOR, QWEN_TO_COMPCARS_GROUPED  # noqa: E402

PROB_COLS = [f"p_{c}" for c in COLORS]


def features(df: pd.DataFrame) -> np.ndarray:
    logp = np.log(df[PROB_COLS].to_numpy(np.float64).clip(1e-6, 1.0))
    mass = np.log(df["option_mass"].to_numpy(np.float64).clip(1e-6, 1.0))[:, None]
    return np.hstack([logp, mass])


def load(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "predictions.csv")
    df = df[(df["gt_source"] == "compcars") & df[PROB_COLS].notna().all(1)]
    if df.empty:
        raise SystemExit(f"{d}: no CompCars rows with probabilities")
    return df.reset_index(drop=True)


def report(gt: np.ndarray, pred: np.ndarray) -> dict:
    labels = [c for c in COMPCARS_ID2COLOR.values() if (gt == c).any()]
    return {"n": int(len(gt)), "grouped_acc_pct": round(float((gt == pred).mean() * 100), 2),
            "grouped_recall_pct": {c: round(float((pred[gt == c] == c).mean() * 100), 2) for c in labels},
            "macro_recall_pct": round(float(np.mean([(pred[gt == c] == c).mean() for c in labels]) * 100), 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="'balanced' trades overall accuracy for recall on rare colors")
    args = p.parse_args()

    tr, va = load(Path(args.train_dir)), load(Path(args.val_dir))
    out_dir = Path(args.out_dir or Path(args.val_dir) / "calibrated")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", tr), ("val", va)):
        print(f"[calib] {name}: {len(d):,} rows, splits {d['split'].value_counts().to_dict()}")

    lr = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], cv=5, max_iter=2000, scoring="accuracy",
                              class_weight=None if args.class_weight == "none" else "balanced")
    clf = make_pipeline(StandardScaler(), lr)   # log-probs reach -13.8; unscaled, lbfgs does not converge
    clf.fit(features(tr), tr["gt_color"].to_numpy())
    print(f"[calib] chosen C = {float(np.atleast_1d(lr.C_)[0])}")

    gt = va["gt_color"].to_numpy()
    before = va["color"].map(QWEN_TO_COMPCARS_GROUPED).to_numpy()
    after = clf.predict(features(va))
    res = {"train_dir": args.train_dir, "val_dir": args.val_dir, "class_weight": args.class_weight,
           "C": float(np.atleast_1d(lr.C_)[0]),
           "val_before": report(gt, before), "val_after": report(gt, after),
           "train_after": report(tr["gt_color"].to_numpy(), clf.predict(features(tr)))}
    (out_dir / "calibration_metrics.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump({"model": clf, "prob_cols": PROB_COLS, "classes": list(clf.classes_)}, f)
    va.assign(calibrated_color=after, calibrated_conf=clf.predict_proba(features(va)).max(1)).to_csv(
        out_dir / "val_predictions_calibrated.csv", index=False)

    b, a = res["val_before"], res["val_after"]
    print(f"[calib] val grouped: {b['grouped_acc_pct']}% -> {a['grouped_acc_pct']}%   "
          f"macro recall {b['macro_recall_pct']}% -> {a['macro_recall_pct']}%")
    for c in a["grouped_recall_pct"]:
        print(f"         {c:10s} {b['grouped_recall_pct'][c]:6.2f} -> {a['grouped_recall_pct'][c]:6.2f}")
    print(f"[calib] wrote {out_dir}")


if __name__ == "__main__":
    main()
