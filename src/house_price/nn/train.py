"""Train and cross-validate the House Prices neural network."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.model_selection import KFold

from src.house_price.nn.data_setup import prepare_final, prepare_fold
from src.house_price.nn.engine import fit
from src.house_price.nn.model import MODEL_MAP
from src.utils.config import load_config
from src.utils.io import copy_file, make_run_dirs, make_run_name, save_json
from src.utils.metrics import summarize_metrics
from src.utils.nn import OPTIMIZER_MAP, build_scheduler
from src.utils.seed import set_seed


def train(config: dict, config_path: Path) -> tuple[Path, Path]:
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device_name = config["experiment"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    data = pd.read_csv(config["data"]["train_path"], keep_default_na=False, na_values=[""])
    target = config["data"]["target"]
    id_column = config["data"]["id"]
    X = data.drop(columns=[target, id_column])
    y_log = np.log1p(data[target].to_numpy(dtype=float)).astype(np.float32).reshape(-1, 1)
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    model_params = config["model"]["params"]
    model_class = MODEL_MAP[config["model"]["name"]]
    optimizer_class = OPTIMIZER_MAP[config["optimizer"]["name"]]

    validation = config["validation"]
    folds = KFold(
        n_splits=validation["n_splits"],
        shuffle=validation["shuffle"],
        random_state=seed if validation["shuffle"] else None,
    )
    fold_rows = []
    for number, (train_index, valid_index) in enumerate(folds.split(X), start=1):
        train_loader, valid_loader, _, in_features = prepare_fold(
            X, y_log, train_index, valid_index, batch_size
        )
        model = model_class(in_features=in_features, **model_params)
        optimizer = optimizer_class(model.parameters(), **config["optimizer"]["params"])
        scheduler = build_scheduler(optimizer, config.get("scheduler"))
        history = fit(model, train_loader, optimizer, scheduler, device, epochs, valid_loader)
        last = history[-1]
        fold_rows.append({"fold": number, **{
            name: last[f"val_{name}"] for name in config["metrics"]["log"]
        }})
        print(f"Fold {number}: RMSLE {last['val_rmsle']:.5f}", flush=True)

    final_loader, preprocessor, in_features = prepare_final(X, y_log, batch_size)
    final_model = model_class(in_features=in_features, **model_params)
    final_optimizer = optimizer_class(final_model.parameters(), **config["optimizer"]["params"])
    final_scheduler = build_scheduler(final_optimizer, config.get("scheduler"))
    final_history = fit(final_model, final_loader, final_optimizer,
                        final_scheduler, device, epochs)

    run_name = make_run_name(config, "raw")
    run_dir, artifact_dir = make_run_dirs(run_name, "house_price/deepnn")
    torch.save(final_model.cpu().state_dict(), artifact_dir / "model.pt")
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    pl.DataFrame(fold_rows).write_csv(run_dir / "fold_metrics.csv")
    pl.DataFrame(final_history).write_csv(run_dir / "final_history.csv")
    copy_file(config_path, run_dir / "config.yaml")
    save_json({
        "features": X.columns.tolist(),
        "in_features": in_features,
        "model_name": config["model"]["name"],
        "model_params": model_params,
        "id": id_column,
        "target": target,
        "test_path": config["data"]["test_path"],
    }, artifact_dir / "metadata.json")
    summary = summarize_metrics(pl.DataFrame(fold_rows), config["metrics"]["log"])
    save_json({
        "experiment_name": config["experiment"]["name"],
        "model_name": config["model"]["name"],
        "primary_metric": config["metrics"]["primary"],
        "config_path": str(config_path),
        "artifact_dir": str(artifact_dir),
        "run_dir": str(run_dir),
        **summary,
    }, run_dir / "summary.json")
    print(f"Mean RMSLE {summary['mean_rmsle']:.5f}; saved to {artifact_dir}")
    return run_dir, artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/house_price/deepnn/model_one.yaml"))
    args = parser.parse_args()
    train(load_config(args.config), args.config)


if __name__ == "__main__":
    main()
