import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path

import src.nn.engine as engine
import src.nn.model as models
from src.nn.data_setup import build_kfold_dataloader
from src.nn.checkpoint import save_model
from src.utils.io import make_run_name, make_run_dirs

CRITERION_MAP = {
    "bce_with_logits": nn.BCEWithLogitsLoss,
}

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

MODEL_MAP = {
    "model_one": models.ModelOne,
    "model_two": models.ModelTwo,
    "model_batch_norm": models.ModelBatchNorm,
}


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
        choices= ["original", "binned"],
        required=True,
        help="Choose data type (original/binned)"
    )
    return parser.parse_args()

def train_final_model():
    pass

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    device = config["experiment"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # data config
    data_config = config["data"]
    train_path = data_config[args.dataset_type]["train_path"]
    target_col = data_config["target"]

    # validation config
    validation_config = config["validation"]
    val_type = validation_config["type"]
    n_splits = validation_config["n_splits"]
    shuffle = validation_config["shuffle"]

    #preprocessing config
    preprocessing_config = config["preprocessing"]
    scaler_name = preprocessing_config["scaler"]

    # metrics config
    metrics_config = config["metrics"]
    primary_metric = metrics_config["primary"]
    metric_names = metrics_config["log"]

    # model config
    model = config["model"]["name"]
    model_class = MODEL_MAP[model]
    model_params = config["model"]["params"]

    # criterion config
    criterion_name = config["criterion"]["name"]

    # optimizer config
    optimizer_name = config["optimizer"]["name"]
    optimizer_params = config["optimizer"]["params"]

    # training config
    training_epochs = config["training"]["epochs"]
    training_batch_size = config["training"]["batch_size"]


    fold_dataloaders, in_features = build_kfold_dataloader(
        train_path=train_path,
        target_col=target_col,
        scaler_name=scaler_name,
        batch_size=training_batch_size,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed
        )

    criterion = CRITERION_MAP[criterion_name]()

    for fold, (train_loader, val_loader) in enumerate(fold_dataloaders, start=1):
        print(f"Fold {fold}/{n_splits}")
        model = model_class(
            in_features=in_features,
            **model_params
            )
        optimizer_class = OPTIMIZER_MAP[optimizer_name]
        optimizer = optimizer_class(params=model.parameters(), **optimizer_params)

        results = engine.fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=training_epochs,
            metric_names=metric_names
            )
        print(results)
if __name__ == "__main__":
    main()
# save_model(model, target_dir=".", model_name="baseline.pt")
