from sklearn.model_selection import StratifiedKFold
import polars as pl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score)
import numpy as np
import yaml
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
from datetime import datetime
from src.utils.io import save_json, copy_file

SEED = 42

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to experiment config yaml"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        help="Choose data type (original/binned)"
    )
    parser
    return parser.parse_args()

MODEL_MAP = {
    "logistic_regression":      LogisticRegression,
    "knn":                      KNeighborsClassifier,
    "decision_tree_classifier": DecisionTreeClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "catboost_classifier":      CatBoostClassifier,
    "xgb_classifier":           XGBClassifier,
    "lgbm_classifier":          LGBMClassifier,
}

SCALER_MAP = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler
}

def make_folds(
        X: pl.DataFrame,
        y: pl.Series,
        n_split: int = 5,
        shuffle: bool = True,
        random_state: int = SEED
        ) -> list[tuple]:
    skf =  StratifiedKFold(
        n_splits=n_split,
        shuffle=shuffle,
        random_state=random_state
        )
    
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds

def calculate_metrics(
        y_true,
        y_pred,
        y_proba=None,
        metric_names: list[str] | None = None) -> dict[str, float]:
    if metric_names is None:
        metric_names = ["accuracy"]

    results = {}

    for metric_name in metric_names:
        if metric_name == "accuracy":
            results["accuracy"] = accuracy_score(y_true, y_pred)
        elif metric_name == "precision":
            results["precision"] = precision_score(y_true, y_pred, zero_division=0)

        elif metric_name == "recall":
            results["recall"] = recall_score(y_true, y_pred, zero_division=0)

        elif metric_name == "f1":
            results["f1"] = f1_score(y_true, y_pred, zero_division=0)

        elif metric_name == "roc_auc":
            if y_proba is None:
                results["roc_auc"] = None
            else:
                results["roc_auc"] = roc_auc_score(y_true, y_proba)

        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    return results

# Config loading
args = parse_args()
CONFIG_PATH = args.config
with open(file=CONFIG_PATH, mode="r") as f:
    config = yaml.safe_load(f)

# Data config
data_config = config["data"][args.dataset_type]
target_col = config["data"]["target"]
train_path = data_config["train_path"]
test_path = data_config["test_path"]

# Metric names
metric_names = config["metrics"]["log"]

# Scaler name
scaler_name = config["preprocessing"]["scaler"]

# Model config
model_config = config["model"]
model_name = model_config["name"]
model_params = model_config["params"]


train_dataset = pl.read_parquet(train_path)
test_dataset = pl.read_parquet(test_path)

X = train_dataset.drop(target_col)
y = train_dataset.get_column(target_col)

folds = make_folds(
    X,
    y,
    n_split=config["validation"]["n_splits"],
    shuffle=config["validation"]["shuffle"],
    random_state= config["validation"]["random_state"])

scaler = SCALER_MAP[scaler_name]()
fold_rows = []

for i, (train_idx, val_idx) in enumerate(folds, start=1):
    X_train_fold = X[train_idx]
    X_val_fold = X[val_idx]

    scaler.fit(X_train_fold)

    X_train_fold = scaler.transform(X_train_fold)
    X_val_fold = scaler.transform(X_val_fold)

    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]

    model_class = MODEL_MAP[model_name]
    model = model_class(**model_params)
    model.fit(X_train_fold, y_train_fold)

    val_pred = model.predict(X_val_fold)

    if hasattr(model, "predict_proba"):
        val_proba = model.predict_proba(X_val_fold)[:, 1]
    else:
        val_proba = None

    fold_metrics = calculate_metrics(
        y_true=y_val_fold,
        y_pred=val_pred,
        y_proba=val_proba,
        metric_names=metric_names
        )
    fold_row = {
        "fold": i,
        **fold_metrics
    }
    fold_rows.append(fold_row)

fold_metrics_df = pl.DataFrame(fold_rows)
print(fold_metrics_df)
summary = {}
for metric_name in metric_names:
    values = fold_metrics_df.get_column(metric_name).drop_nulls()
    summary[f"mean_{metric_name}"] = float(values.mean())
    summary[f"std_{metric_name}"] = float(values.std())
    print("=="*30)
    print(f"Mean {metric_name}: {summary[f"mean_{metric_name}"]:.04f}")
    print(f"Std {metric_name}: {summary[f"std_{metric_name}"]:.04f}")

run_name = (
    f"{config['experiment']['name']}_"
    f"{args.dataset_type}_"
    f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
)
run_dir = Path("logs")/"classic"/run_name
run_dir.mkdir(parents=True, exist_ok=True)

fold_metrics_df.write_csv(run_dir / "fold_metrics_df.csv")
copy_file(src=CONFIG_PATH, dst=run_dir/"config.yaml")

summary = {
    "experiment_name": config["experiment"]["name"],
    "model_name": model_name,
    "model_params": model_params,
    "preprocessing": scaler_name,
    "dataset_type": args.dataset_type,
    "primary_metric": config["metrics"]["primary"],
    "config_path": str(CONFIG_PATH),
    **summary,
}

save_json(data=summary, path=run_dir/"summary.json")