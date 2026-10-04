"""Train a House Prices regressor with fold-local preprocessing."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.house_price.preprocessing import build_preprocessor
from src.utils.config import load_config
from src.utils.io import copy_file, make_run_dirs, make_run_name, save_json
from src.utils.metrics import summarize_metrics
from src.utils.seed import set_seed

MODEL_MAP = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "catboost": CatBoostRegressor,
    "xgboost": XGBRegressor,
}


def build_model(X: pd.DataFrame, name: str, config: dict):
    """Keep imputers and encoders inside the estimator fitted for each fold."""
    preprocessing = build_preprocessor(X)
    params = dict(config["models"][name])
    if name in {"random_forest", "xgboost"}:
        params["random_state"] = config["experiment"]["seed"]
    elif name == "catboost":
        params["random_seed"] = config["experiment"]["seed"]
    regressor = Pipeline([
        ("preprocessing", preprocessing),
        ("model", MODEL_MAP[name](**params)),
    ])
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def run_cv(X: pd.DataFrame, y: pd.Series, model, config: dict) -> pl.DataFrame:
    """Evaluate cloned regressors on the configured folds in price units."""
    validation = config["validation"]
    folds = KFold(
        n_splits=validation["n_splits"],
        shuffle=validation["shuffle"],
        random_state=config["experiment"]["seed"] if validation["shuffle"] else None,
    )
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds.split(X), start=1):
        fitted = clone(model).fit(X.iloc[train_index], y.iloc[train_index])
        predicted = np.maximum(fitted.predict(X.iloc[valid_index]), 0)
        actual = y.iloc[valid_index]
        rows.append({
            "fold": fold,
            "rmsle": root_mean_squared_log_error(actual, predicted),
            "rmse": mean_squared_error(actual, predicted) ** 0.5,
            "mae": mean_absolute_error(actual, predicted),
        })
    return pl.DataFrame(rows)


def main() -> None:
    """Train and save the selected classic House Prices model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/house_price/classic.yaml"))
    parser.add_argument("--model", choices=MODEL_MAP, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    data = pd.read_csv(config["data"]["train_path"], keep_default_na=False, na_values=[""])
    target = config["data"]["target"]
    id_column = config["data"]["id"]
    X = data.drop(columns=[target, id_column])
    y = data[target]
    model = build_model(X, args.model, config)
    fold_metrics = run_cv(X, y, model, config)
    summary = summarize_metrics(fold_metrics, config["metrics"]["log"])

    run_config = {"experiment": {"name": args.model}}
    run_name = make_run_name(run_config, "raw")
    run_dir, artifact_dir = make_run_dirs(run_name, "house_price/classic")
    model.fit(X, y)
    joblib.dump(model, artifact_dir / "model.joblib")
    fold_metrics.write_csv(run_dir / "fold_metrics.csv")
    copy_file(args.config, run_dir / "config.yaml")
    save_json({
        "features": X.columns.tolist(),
        "id": id_column,
        "target": target,
        "test_path": config["data"]["test_path"],
    }, artifact_dir / "metadata.json")
    save_json({
        "experiment_name": args.model,
        "model_name": args.model,
        "primary_metric": config["metrics"]["primary"],
        "config_path": str(args.config),
        "artifact_dir": str(artifact_dir),
        **summary,
    }, run_dir / "summary.json")
    print(f"{args.model}: mean RMSLE {summary['mean_rmsle']:.5f}; saved to {artifact_dir}")


if __name__ == "__main__":
    main()
