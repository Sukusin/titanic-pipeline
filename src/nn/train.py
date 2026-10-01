import argparse
from pathlib import Path

import joblib
import polars as pl
import torch
import torch.nn as nn
import yaml

import src.nn.engine as engine
import src.nn.model as models
from src.nn.checkpoint import save_model
from src.nn.data_setup import build_final_dataloader, build_kfold_dataloader
from src.nn.embedding_model import EmbeddingModel
from src.nn.preprocessing import model_input_params
from src.utils.io import copy_file, make_run_dirs, make_run_name, save_json
from src.utils.metrics import summarize_metrics
from src.utils.seed import set_seed

MODEL_MAP = {
    "custom_model": models.CustomModel,
    "embedding_model": EmbeddingModel,
    "model_one": models.ModelOne,
    "model_two": models.ModelTwo,
    "model_batch_norm": models.ModelBatchNorm,
}

CRITERION_MAP = {
    "bce_with_logits": nn.BCEWithLogitsLoss,
}

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

SCHEDULER_MAP = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict | None,
) -> engine.Scheduler:
    """Create a scheduler from config, or return None when it is disabled."""
    if not scheduler_config:
        return None

    scheduler_name = scheduler_config.get("name")
    if scheduler_name is None:
        return None
    if scheduler_name not in SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    scheduler_class = SCHEDULER_MAP[scheduler_name]
    scheduler_params = scheduler_config.get("params", {})
    return scheduler_class(optimizer, **scheduler_params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to expriment config.yaml",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["original", "binned"],
        required=True,
        help="Choose data type (original/binned)",
    )
    return parser.parse_args()


def save_experiment(
    run_dir: Path,
    artifact_dir: Path,
    config_path: Path,
    config: dict,
    args: argparse.Namespace,
    fold_metrics_df: pl.DataFrame,
    final_history_df: pl.DataFrame,
    model: torch.nn.Module,
    scaler,
    features: list[str],
) -> None:
    model_path = artifact_dir / "model.pt"
    scaler_path = artifact_dir / "scaler.joblib"
    metadata_path = artifact_dir / "metadata.json"

    save_model(
        model=model,
        artifact_dir=artifact_dir,
        model_name=model_path.name,
    )
    joblib.dump(scaler, scaler_path)

    fold_metrics_df.write_csv(run_dir / "fold_metrics.csv")
    final_history_df.write_csv(run_dir / "final_history.csv")
    copy_file(src=config_path, dst=run_dir / "config.yaml")

    save_json(
        data={
            "features": features,
            "target": config["data"]["target"],
            "dataset_type": args.dataset_type,
            "model_input_params": model_input_params(scaler),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
        },
        path=metadata_path,
    )

    summary = summarize_metrics(
        fold_metrics_df=fold_metrics_df,
        metric_names=config["metrics"]["log"],
    )

    save_json(
        data={
            "experiment_name": config["experiment"]["name"],
            "model_name": config["model"]["name"],
            "model_params": config["model"]["params"],
            "criterion": config["criterion"]["name"],
            "optimizer_name": config["optimizer"]["name"],
            "optimizer_params": config["optimizer"]["params"],
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

    print("==" * 25)
    print(f"Metadata saved to: {metadata_path}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")
    print("==" * 25)


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = config["experiment"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # data config
    data_config = config["data"]
    train_path = data_config[args.dataset_type]["train_path"]
    target_col = data_config["target"]

    # validation config
    validation_config = config["validation"]
    # val_type = validation_config["type"]
    n_splits = validation_config["n_splits"]
    shuffle = validation_config["shuffle"]

    # preprocessing config
    preprocessing_config = config["preprocessing"]
    scaler_name = preprocessing_config["scaler"]

    # metrics config
    metrics_config = config["metrics"]
    # primary_metric = metrics_config["primary"]
    metric_names = metrics_config["log"]

    # model config
    model = config["model"]["name"]
    model_class = MODEL_MAP[model]
    model_params = config["model"]["params"]
    categorical_features = None
    if model == "embedding_model":
        categorical_features = preprocessing_config.get("categorical_features", {}).get(
            args.dataset_type
        )
        if not categorical_features:
            raise ValueError("Embedding models require categorical_features for the dataset type.")

    # criterion config
    criterion_name = config["criterion"]["name"]

    # optimizer config
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]

    # training config
    training_epochs = config["training"]["epochs"]
    training_batch_size = config["training"]["batch_size"]

    # An omitted scheduler key and `scheduler: null` both disable scheduling.
    scheduler_config = config.get("scheduler")

    fold_dataloaders, in_features = build_kfold_dataloader(
        train_path=train_path,
        target_col=target_col,
        scaler_name=scaler_name,
        batch_size=training_batch_size,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed,
        categorical_features=categorical_features,
        dataset_type=args.dataset_type,
    )

    criterion = CRITERION_MAP[criterion_name]()
    fold_rows = []

    for fold, (train_loader, val_loader, fold_preprocessor) in enumerate(fold_dataloaders, start=1):
        print(f"Fold {fold}/{n_splits}")
        model = model_class(
            in_features=in_features,
            **model_params,
            **model_input_params(fold_preprocessor),
        )
        optimizer_class = OPTIMIZER_MAP[optimizer_name]
        optimizer = optimizer_class(params=model.parameters(), **optimizer_params)
        scheduler = build_scheduler(optimizer, scheduler_config)

        results = engine.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=training_epochs,
            metric_names=metric_names,
        )
        fold_row = {"fold": fold}
        for metric_name, metric_values in results.items():
            if not metric_values:
                continue

            output_metric_name = metric_name
            if metric_name.startswith("val_"):
                output_metric_name = metric_name.removeprefix("val_")

            fold_row[output_metric_name] = metric_values[-1]
        fold_rows.append(fold_row)

    print("Training final model on whole dataset...")
    final_train_loader, in_features, scaler = build_final_dataloader(
        train_path=train_path,
        target_col=target_col,
        scaler_name=scaler_name,
        batch_size=training_batch_size,
        random_state=seed,
        categorical_features=categorical_features,
        dataset_type=args.dataset_type,
    )

    final_model = model_class(
        in_features=in_features,
        **model_params,
        **model_input_params(scaler),
    )

    final_optimizer = optimizer_class(params=final_model.parameters(), **optimizer_params)
    final_scheduler = build_scheduler(final_optimizer, scheduler_config)

    final_results = engine.fit_final(
        model=final_model,
        train_loader=final_train_loader,
        criterion=criterion,
        optimizer=final_optimizer,
        scheduler=final_scheduler,
        device=device,
        epochs=training_epochs,
    )

    run_name = make_run_name(config=config, dataset_type=args.dataset_type)
    run_dir, artifact_dir = make_run_dirs(run_name=run_name, model_type="deepnn")

    train_features = pl.read_parquet(train_path).drop(target_col).columns

    save_experiment(
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        config_path=args.config,
        config=config,
        args=args,
        fold_metrics_df=pl.DataFrame(fold_rows),
        final_history_df=pl.DataFrame(final_results),
        model=final_model,
        scaler=scaler,
        features=train_features,
    )


if __name__ == "__main__":
    main()
