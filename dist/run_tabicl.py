"""Run with Python 3.11+ and tabicl==2.2.0, pandas and scikit-learn.

From the repository root: python dist/run_tabicl.py work/datasets
TabICL downloads the published checkpoint on first use.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabicl import TabICLClassifier

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def load_task(directory, dataset):
    if dataset == "absence":
        frame = pd.read_csv(directory / "Absenteeism_at_work.csv", sep=";")
        target = "Absenteeism time in hours"
        return frame.drop(columns=[target, "ID"]), (frame[target] >= 8).astype(int)
    if dataset == "productivity":
        frame = pd.read_csv(directory / "garments_worker_productivity.csv")
        target = (frame["actual_productivity"] >= frame["targeted_productivity"]).astype(int)
        return frame.drop(columns=["actual_productivity", "targeted_productivity", "date"]), target

    path = directory / "jobchange.arff"
    lines = path.read_text().splitlines()
    columns = [line.split()[1] for line in lines if line.upper().startswith("@ATTRIBUTE ")]
    start = next(i for i, line in enumerate(lines) if line.strip().upper() == "@DATA") + 1
    frame = pd.read_csv(path, skiprows=start, names=columns, quotechar="'", na_values=["nan", "?"])
    label = "LookingForJobChange"
    _, sample = train_test_split(frame, test_size=1500, stratify=frame[label], random_state=42)
    return sample.drop(columns=[label, "gender"]), sample[label].eq("Yes").astype(int)


def benchmark(directory, dataset):
    X, y = load_task(directory, dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    numeric = X.select_dtypes(include=np.number).columns
    categorical = X.columns.difference(numeric, sort=False)
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), categorical),
    ])
    model = TabICLClassifier(
        checkpoint_version=CHECKPOINT, n_estimators=1,
        device="cpu", n_jobs=2, random_state=42,
    )
    train = preprocess.fit_transform(X_train)
    test = preprocess.transform(X_test)
    start = perf_counter()
    model.fit(train, y_train)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probabilities = model.predict_proba(test)[:, 1]
    return {
        "model": "TabICLv2",
        "status": "EXECUTED",
        "checkpoint": CHECKPOINT,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "fit_seconds": fit_seconds,
        "test_inference_seconds": perf_counter() - start,
        "metrics": {
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "brier": brier_score_loss(y_test, probabilities),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--dataset", choices=["jobchange", "absence", "productivity"])
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else ["jobchange", "absence", "productivity"]
    print(json.dumps({key: benchmark(args.directory, key) for key in datasets}, indent=2))
