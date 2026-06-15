"""
Baseline detection of the ZigBee Beacon Request Flood attack.

This script reproduces a simple baseline on the public ZigBee Beacon dataset.
It uses standard classifiers with their default parameters and reports
accuracy, precision, recall and weighted F1 with stratified cross-validation.

Dataset: https://figshare.com/s/548457fa650f1179a98a  (file: zigbee_beacon.csv)

Usage:
    pip install -r requirements.txt
    python baseline.py --data zigbee_beacon.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# Predictor variables, as described in the paper (no temporal context, C = 0).
PREDICTORS = [
    "frame.time_delta",
    "frame.len",
    "wpan.fcf",
    "wpan.frame_type",
    "wpan.security",
    "wpan.pending",
    "wpan.ack_request",
    "wpan.pan_id_compression",
    "wpan.dst_addr_mode",
    "wpan.src_addr_mode",
    "wpan.seq_no",
    "wpan.cmd",
    "wpan.fcs_ok",
    # "n_protocols" is derived below from "frame.protocols".
]

LABEL = "type"


def to_number(value):
    """Convert a raw field to a number: booleans, hex (0x..) and decimals."""
    if isinstance(value, float) and np.isnan(value):
        return -1.0
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return -1.0
    low = text.lower()
    if low == "true":
        return 1.0
    if low == "false":
        return 0.0
    try:
        if low.startswith("0x"):
            return float(int(low, 16))
        return float(text)
    except ValueError:
        return -1.0


def load_dataset(path):
    if not os.path.exists(path):
        sys.exit(
            f"Dataset not found: {path}\n"
            "Download 'zigbee_beacon.csv' from https://figshare.com/s/548457fa650f1179a98a "
            "and pass its path with --data."
        )
    df = pd.read_csv(path, low_memory=False)

    # Derived feature: number of protocol layers in the frame.
    df["n_protocols"] = (
        df["frame.protocols"].fillna("").astype(str).apply(
            lambda s: 0 if s == "" else len(s.split(":"))
        )
    )

    features = PREDICTORS + ["n_protocols"]
    X = df[features].apply(lambda col: col.map(to_number))
    y = (df[LABEL].astype(str).str.strip().str.lower() == "attack").astype(int)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="ZigBee Beacon Flood baseline")
    parser.add_argument("--data", default="zigbee_beacon.csv", help="Path to the CSV dataset")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    print(f"Loaded {len(X)} frames | normal: {(y == 0).sum()} | attack: {(y == 1).sum()}")
    print(f"Features used: {X.shape[1]} | Context: C = 0 | CV folds: {args.folds}\n")

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    header = f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}"
    print(header)
    print("-" * len(header))
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        acc = scores["test_accuracy"].mean() * 100
        prec = scores["test_precision"].mean() * 100
        rec = scores["test_recall"].mean() * 100
        f1 = scores["test_f1"].mean() * 100
        print(f"{name:<20}{acc:>9.2f}%{prec:>10.2f}%{rec:>8.2f}%{f1:>8.2f}%")

    print("\nAll classifiers use scikit-learn default parameters (standard baseline).")


if __name__ == "__main__":
    main()
