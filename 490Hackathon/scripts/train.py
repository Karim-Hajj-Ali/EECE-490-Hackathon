#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import json
import sys

# Ensure we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankruptcy.data import (
    load_csv, build_labels, select_target, split_by_time, arrays_from_frames,
    DEFAULT_FEATURES, NAME_COL, YEAR_COL, STATUS_COL
)
from bankruptcy.model import train_catboost, evaluate_thresholds, save_artifacts

def main():
    ap = argparse.ArgumentParser(description="Train bankruptcy next-year predictor")
    ap.add_argument("--csv", required=True, help="Path to american_bankruptcy.csv")
    ap.add_argument("--outdir", default="models", help="Where to save model & meta")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--cost_fn", type=float, default=5.0, help="Cost weight for FN")
    ap.add_argument("--cost_fp", type=float, default=1.0, help="Cost weight for FP")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = str(outdir / "catboost_bankruptcy.cbm")
    meta_path  = str(outdir / "metadata.json")

    # 1) Load & label
    df = load_csv(args.csv)
    df = build_labels(df, feature_cols=DEFAULT_FEATURES)
    target_col, task_desc, train_filter = select_target(df)

    # 2) Time split
    train_df, val_df, test_df = split_by_time(df)

    # For onset training, keep only rows that were alive at t
    if train_filter is not None:
        train_df = train_df[train_filter.loc[train_df.index]].copy()

    # 3) Arrays
    X_train, y_train, X_val, y_val, X_test, y_test = arrays_from_frames(
        train_df, val_df, test_df, DEFAULT_FEATURES, target_col
    )

    # Safety checks
    if X_train.size == 0:
        raise RuntimeError("No training rows after preprocessing. Check CSV and feature columns.")
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    print(f"Train rows: {len(y_train)}  (pos={n_pos}, neg={n_neg})")
    print(f"Val rows:   {len(y_val)}")
    print(f"Test rows:  {len(y_test)}")
    print(f"Target used: {target_col} | {task_desc}")

    # 4) Train
    clf = train_catboost(X_train, y_train, X_val, y_val, random_state=args.random_state)

    # 5) Thresholds & validation metrics
    thr_f1, thr_cost, metrics = evaluate_thresholds(
        clf, X_val, y_val, cost_fn=args.cost_fn, cost_fp=args.cost_fp
    )

    # 6) Save artifacts
    train_medians = pd.DataFrame(X_train, columns=DEFAULT_FEATURES).median().to_dict()
    meta = {
        "feature_cols": DEFAULT_FEATURES,
        "year_col": YEAR_COL,
        "name_col": NAME_COL,
        "status_col": STATUS_COL,
        "target_col": target_col,
        "task_desc": task_desc,
        "threshold_f1": float(thr_f1),
        "threshold_cost": float(thr_cost),
        "train_feature_medians": train_medians,
        "notes": {"val_metrics": metrics},
    }
    save_artifacts(clf, model_path, meta_path, meta)

    print("Saved model to:", model_path)
    print("Saved meta to:  ", meta_path)
    print("Val ROC-AUC:", meta["notes"]["val_metrics"]["val_roc_auc"])
    print("Val PR-AUC:", meta["notes"]["val_metrics"]["val_pr_auc"])
    print("Chosen thresholds | F1:", round(thr_f1, 3), " | Cost-sensitive:", round(thr_cost, 3))

if __name__ == "__main__":
    main()
