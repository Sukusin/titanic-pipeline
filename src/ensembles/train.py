"""Compare ensembles and their base models on identical outer CV folds."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from src.ensembles.model import (
    METHODS,
    OOFStackingClassifier,
    build_ensemble,
    prediction_scores,
)
from src.ensembles.submission_predictions import create_submission
from src.utils.config import load_config
from src.utils.io import copy_file, make_run_dirs, make_run_name, save_json
from src.utils.metrics import calculate_metrics, summarize_metrics
from src.utils.seed import set_seed


def validate_config(config: dict, base_configs: list[dict], y) -> None:
    """Reject invalid methods, voting weights and insufficient stratified fold sizes."""
    methods = config["methods"]
    if not methods or len(methods) != len(set(methods)) or set(methods).difference(METHODS):
        raise ValueError(f"Choose unique ensemble methods from {METHODS}.")
    if len(base_configs) < 2:
        raise ValueError("An ensemble needs at least two base models.")
    weights = config["voting"].get("weights")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != (len(base_configs),)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError("Voting weights must match the base models and have a positive sum.")
    classes, counts = np.unique(y, return_counts=True)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError("Ensembles require binary target labels 0 and 1.")
    n_splits = config["validation"]["n_splits"]
    if n_splits < 2 or counts.min() < n_splits:
        raise ValueError("Each class must contain at least n_splits rows for outer CV.")


def save_inner_oof(model: OOFStackingClassifier, y, row_ids, path: Path) -> None:
    """Save auditable meta-training features with their original row and inner fold IDs."""
    data = {
        "row_id": row_ids,
        "inner_fold": model.oof_fold_,
        "target": y,
        **{
            name: model.oof_predictions_[:, index]
            for index, (name, _) in enumerate(model.estimators)
        },
    }
    pl.DataFrame(data).write_csv(path)


def run_cv(X, y, base_configs: list[dict], config: dict, run_dir: Path) -> pl.DataFrame:
    """Evaluate stacking with inner OOF training confined to each outer train fold."""
    validation = config["validation"]
    splitter = StratifiedKFold(
        n_splits=validation["n_splits"],
        shuffle=validation["shuffle"],
        random_state=validation["random_state"] if validation["shuffle"] else None,
    )
    rows, predictions = [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        for method_index, method in enumerate(config["methods"]):
            print(f"Fold {fold}/{validation['n_splits']}: {method}", flush=True)
            model = build_ensemble(method, base_configs, config).fit(X[train_idx], y[train_idx])
            scores = prediction_scores(model, X[val_idx])
            predicted = model.predict(X[val_idx])
            rows.append(
                {
                    "model_name": method,
                    "model_type": "ensemble",
                    "fold": fold,
                    **calculate_metrics(y[val_idx], predicted, config["metrics"]["log"], scores),
                }
            )
            predictions.append(
                pl.DataFrame(
                    {
                        "model_name": method,
                        "fold": fold,
                        "row_id": val_idx,
                        "target": y[val_idx],
                        "prediction": predicted,
                        "score": scores,
                    }
                )
            )
            if isinstance(model, OOFStackingClassifier):
                save_inner_oof(
                    model, y[train_idx], train_idx, run_dir / f"{method}_fold_{fold}_inner_oof.csv"
                )

            # The fitted base models use the same outer training rows as the ensemble.
            if method_index == 0:
                for index, estimator in enumerate(model.estimators_):
                    name = f"base_{index + 1}_{base_configs[index]['experiment']['name']}"
                    rows.append(
                        {
                            "model_name": name,
                            "model_type": "base",
                            "fold": fold,
                            **calculate_metrics(
                                y[val_idx],
                                estimator.predict(X[val_idx]),
                                config["metrics"]["log"],
                                estimator.predict_proba(X[val_idx])[:, 1],
                            ),
                        }
                    )

    pl.concat(predictions).sort(["model_name", "row_id"]).write_csv(run_dir / "oof_predictions.csv")
    return pl.DataFrame(rows)


def main() -> None:
    """Run nested CV, save all ensembles, compare scores and create the best submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ensemble_config/ensembles.yaml")
    )
    parser.add_argument("--dataset-type", choices=["original", "binned"], default="original")
    args = parser.parse_args()
    config = load_config(args.config)
    config["dataset_type"] = args.dataset_type
    base_configs = [load_config(path) for path in config["base_configs"]]
    set_seed(config["experiment"]["seed"])

    data_config = config["data"][args.dataset_type]
    target = config["data"]["target"]
    train = pl.read_parquet(data_config["train_path"])
    features = [column for column in train.columns if column not in (target, "PassengerId")]
    X, y = train.select(features).to_numpy(), train.get_column(target).to_numpy()
    validate_config(config, base_configs, y)

    run_name = make_run_name(config, args.dataset_type)
    run_dir, artifact_dir = make_run_dirs(run_name, "ensembles")
    copy_file(args.config, run_dir / "config.yaml")
    for index, path in enumerate(config["base_configs"], start=1):
        copy_file(path, run_dir / "base_configs" / f"base_{index}.yaml")

    fold_metrics = run_cv(X, y, base_configs, config, run_dir)
    fold_metrics.write_csv(run_dir / "fold_metrics.csv")
    results = []
    for name in fold_metrics.get_column("model_name").unique(maintain_order=True):
        subset = fold_metrics.filter(pl.col("model_name") == name)
        results.append(
            {
                "experiment_name": config["experiment"]["name"],
                "model_name": name,
                "model_type": subset.get_column("model_type")[0],
                "dataset_type": args.dataset_type,
                "primary_metric": config["metrics"]["primary"],
                "run_dir": str(run_dir),
                "artifact_dir": str(artifact_dir / name) if name in config["methods"] else None,
                **summarize_metrics(subset, config["metrics"]["log"]),
            }
        )

    for method in config["methods"]:
        print(f"Final training: {method}", flush=True)
        model = build_ensemble(method, base_configs, config).fit(X, y)
        model_dir = artifact_dir / method
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        save_json(
            {
                "model_name": method,
                "features": features,
                "target": target,
                "dataset_type": args.dataset_type,
                "test_path": data_config["test_path"],
                "base_configs": base_configs,
                "ensemble_config": config,
            },
            model_dir / "metadata.json",
        )
        if isinstance(model, OOFStackingClassifier):
            save_inner_oof(model, y, np.arange(len(y)), run_dir / f"{method}_final_inner_oof.csv")

    primary = config["metrics"]["primary"]
    comparison = pl.DataFrame(results).sort(f"mean_{primary}", descending=True)
    comparison.write_csv(run_dir / "comparison.csv")
    comparison.write_csv(Path("logs/ensembles/leaderboard.csv"))
    best = comparison.filter(pl.col("model_type") == "ensemble").row(0, named=True)
    save_json({"best_ensemble": best, "results": comparison.to_dicts()}, run_dir / "summary.json")


    output = Path("submissions/submission_ensemble.csv")
    create_submission(Path(best["artifact_dir"]), output)
    print(comparison.select("model_name", "mean_accuracy", "mean_f1", "mean_roc_auc"))
    print(f"Comparison: {run_dir / 'comparison.csv'}")
    print(f"Best ensemble: {best['model_name']}; submission: {output}")


if __name__ == "__main__":
    main()
