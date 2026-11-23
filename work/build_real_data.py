from __future__ import annotations

import hashlib
import json
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "datasets"
OUT = ROOT / "dist"


def clean(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    return str(value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arff(path: Path) -> pd.DataFrame:
    names = []
    lines = path.read_text(encoding="utf-8").splitlines()
    data_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            match = re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", stripped, re.I)
            if match:
                names.append(next(group for group in match.groups() if group is not None))
        elif stripped.lower() == "@data":
            data_start = index + 1
            break
    if data_start is None:
        raise ValueError("ARFF @data marker not found")
    rows = csv.reader(line for line in lines[data_start:] if line and not line.startswith("%"))
    return pd.DataFrame(rows, columns=names).replace("?", np.nan)


job = read_arff(DATA / "jobchange.arff")
absence = pd.read_csv(DATA / "Absenteeism_at_work.csv", sep=";")
productivity = pd.read_csv(DATA / "garments_worker_productivity.csv")

configs = {
    "jobchange": {
        "frame": job,
        "target": job["LookingForJobChange"].map({"No": 0, "Yes": 1}),
        "drop": ["LookingForJobChange"],
        "columns": ["city_development_index", "gender", "relevent_experience", "education_level", "experience", "training_hours", "LookingForJobChange"],
        "source": "https://www.openml.org/d/46935",
        "source_label": "OpenML 46935 · TabArena-curated job-change intention",
        "license": "See OpenML dataset card",
        "question": "Reported job-change search intention",
        "definition": "Original target supplied by the source; intention is not a verified exit.",
        "file": DATA / "jobchange.arff",
    },
    "absence": {
        "frame": absence,
        "target": (absence["Absenteeism time in hours"] >= 8).astype(int),
        "drop": ["Absenteeism time in hours", "ID"],
        "columns": ["Reason for absence", "Month of absence", "Day of the week", "Service time", "Age", "Work load Average/day ", "Absenteeism time in hours"],
        "source": "https://archive.ics.uci.edu/dataset/445/absenteeism+at+work",
        "source_label": "UCI 445 · Brazilian courier absence records",
        "license": "CC BY 4.0",
        "question": "Absence episode of at least eight hours",
        "definition": "PeopleFM-derived binary target: source absence hours ≥ 8. This threshold is a demonstration choice, not a clinical or employment rule.",
        "file": DATA / "Absenteeism_at_work.csv",
    },
    "productivity": {
        "frame": productivity,
        "target": (productivity["actual_productivity"] >= productivity["targeted_productivity"]).astype(int),
        "drop": ["actual_productivity", "targeted_productivity", "date"],
        "columns": ["quarter", "department", "day", "team", "targeted_productivity", "wip", "actual_productivity"],
        "source": "https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees",
        "source_label": "UCI 597 · Garment-team productivity records",
        "license": "CC BY 4.0",
        "question": "Team met its recorded productivity target",
        "definition": "PeopleFM-derived binary target: actual productivity ≥ recorded target. Rows describe teams and dates, not individual employees.",
        "file": DATA / "garments_worker_productivity.csv",
    },
}

previews = {}
benchmarks = {}
for key, cfg in configs.items():
    frame = cfg["frame"].copy()
    subset = frame[cfg["columns"]].head(5)
    previews[key] = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "headers": list(subset.columns),
        "sample_rows": [[clean(v) for v in row] for row in subset.to_numpy()],
        "source": cfg["source"],
        "source_label": cfg["source_label"],
        "license": cfg["license"],
        "question": cfg["question"],
        "definition": cfg["definition"],
        "sha256": digest(cfg["file"]),
    }

    if key == "jobchange":
        continue
    X = frame.drop(columns=cfg["drop"])
    y = cfg["target"]
    categorical = list(X.select_dtypes(exclude=np.number).columns)
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    models = {
        "logistic": LogisticRegression(max_iter=2000, C=1),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=3, random_state=42),
        "hist_gradient": HistGradientBoostingClassifier(max_iter=150, learning_rate=.06, random_state=42),
    }
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        probability = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "status": "EXECUTED",
            "metrics": {
                "roc_auc": float(roc_auc_score(y_test, probability)),
                "average_precision": float(average_precision_score(y_test, probability)),
                "brier": float(brier_score_loss(y_test, probability)),
            },
        })
    rows.append({"model": "TabICLv2", "status": "NOT_RUN", "reason": "No foundation-model run was executed for this extension."})
    benchmarks[key] = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_share_test": float(y_test.mean()),
        "split": "Stratified 80/20 holdout; random_state=42",
        "models": rows,
    }

(OUT / "real-dataset-previews.json").write_text(json.dumps(previews, indent=2), encoding="utf-8")
(OUT / "real-benchmark-extension.json").write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
aggregate_path = OUT / "aggregate-results.json"
aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
aggregate.pop("ibm", None)
aggregate["description"] = "Executed job-change benchmark metrics. IBM synthetic results were removed from the public article; see the website for the separate real-data extensions."
aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
print(json.dumps({"previews": list(previews), "benchmarks": list(benchmarks)}, indent=2))
