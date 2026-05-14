import argparse
from pathlib import Path

import joblib
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils.config import load_config
from src.utils.io import copy_file, make_run_dirs, make_run_name, save_json
from src.utils.metrics import calculate_metrics, summarize_metrics
from src.utils.seed import set_seed

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config.yaml"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices= ["original", "binned"],
        required=True,
        help="Choose data type (original/binned)"
    )
    return parser.parse_args()

def make_folds(
        X: pl.DataFrame,
        y: pl.Series,
        n_split: int = 5,
        shuffle: bool = True,
        random_state: int = 42
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

def build_scaler(config: dict):
    scaler_name = config["preprocessing"]["scaler"]
    scaler_class = SCALER_MAP[scaler_name]
    return scaler_class()

def build_model(config: dict):
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]

    model_class = MODEL_MAP[model_name]
    return model_class(**model_params)

def run_cv(
        X: pl.DataFrame,
        y: pl.Series,
        config: dict,
        metric_names: list[str]
        ) -> pl.DataFrame:
    folds = make_folds(
        X,
        y,
        n_split=config["validation"]["n_splits"],
        shuffle=config["validation"]["shuffle"],
        random_state= config["validation"]["random_state"])

    fold_rows = []

    for i, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]

        scaler = build_scaler(config=config)
        scaler.fit(X_train_fold)

        X_train_fold = scaler.transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        model = build_model(config=config)
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
    return pl.DataFrame(fold_rows)

def train_final_model(
        X: pl.DataFrame,
        y: pl.Series,
        config: dict,
        ):
    scaler = build_scaler(config=config)
    X_scaled = scaler.fit_transform(X)

    model = build_model(config=config)
    model.fit(X_scaled, y)

    return model, scaler

def save_experiment(
        run_dir: Path,
        artifact_dir: Path,
        config_path: Path,
        config: dict,
        args: argparse.Namespace,
        fold_metrics_df: pl.DataFrame,
        summary: dict,
        model,
        scaler,
        features: list[str],
        ) -> None:
    model_path = artifact_dir / "model.joblib"
    scaler_path = artifact_dir / "scaler.joblib"
    metadata_path = artifact_dir / "metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    fold_metrics_df.write_csv(run_dir / "fold_metrics.csv")
    copy_file(src=config_path, dst=run_dir / "config.yaml")

    save_json(
        data={
            "features": features,
            "target": config["data"]["target"],
            "dataset_type": args.dataset_type,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
        },
        path=metadata_path,
    )

    save_json(
        data={
            "experiment_name": config["experiment"]["name"],
            "model_name": config["model"]["name"],
            "model_params": config["model"]["params"],
            "preprocessing": config["preprocessing"]["scaler"],
            "dataset_type": args.dataset_type,
            "primary_metric": config["metrics"]["primary"],
            "config_path": str(config_path),
            "artifact_dir": str(artifact_dir),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            **summary,
        },
        path=run_dir / "summary.json",
    )
    print("=="*25)
    print(f"Metadata saved to: {metadata_path}")
    print(f"Summary saved to: {run_dir}/summary.json")
    print("=="*25)

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    # Data config
    data_config = config["data"][args.dataset_type]
    target_col = config["data"]["target"]
    metric_names = config["metrics"]["log"]
    seed = config["experiment"]["seed"]
    set_seed(seed=seed)

    train_dataset = pl.read_parquet(data_config["train_path"])

    X = train_dataset.drop(target_col)
    y = train_dataset.get_column(target_col)

    fold_metrics_df = run_cv(
        X=X,
        y=y,
        config=config,
        metric_names=metric_names
        )
    
    summary = summarize_metrics(
        fold_metrics_df=fold_metrics_df,
        metric_names=metric_names,
        )
    
    run_name = make_run_name(config=config, dataset_type=args.dataset_type)
    run_dir, artifact_dir = make_run_dirs(run_name=run_name, model_type="classic")

    final_model, final_scaler = train_final_model(
        X=X,
        y=y,
        config=config
        )
    
    save_experiment(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        config_path=args.config,
        config=config,
        args=args,
        fold_metrics_df=fold_metrics_df,
        summary=summary,
        model=final_model,
        scaler=final_scaler,
        features=X.columns
        )

if __name__ == "__main__":
    main()
