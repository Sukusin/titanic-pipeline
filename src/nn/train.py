import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path

import src.nn.engine as engine
import src.nn.model as models
from src.nn.data_setup import build_kfold_dataloader
from src.nn.checkpoint import save_model

CONFIG_PATH = "configs/deepnn_config/model_one.yaml"

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

args = parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

seed = config["experiment"]["seed"]
device = config["experiment"]["device"]
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"

# data config
data_config = config["data"]
train_path = data_config["original"]["train_path"]
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
optimizer_step = config["optimizer"]["step"] #learning rate

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

model = model_class(
    in_features=in_features,
    **model_params
    )

criterion = CRITERION_MAP[criterion_name]()
optimizer_class = OPTIMIZER_MAP[optimizer_name]
optimizer = optimizer_class(params=model.parameters(), lr=optimizer_step)

results = engine.fit(
    model=model,
    fold_loaders=fold_dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=training_epochs,
    metric_names=metric_names
    )

save_model(model, target_dir=".", model_name="baseline.pt")
