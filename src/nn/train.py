import torch
import torch.nn as nn
import yaml

import src.nn.engine as engine
import src.nn.model as models
from src.nn.data_setup import build_kfold_dataloader
from src.nn.checkpoint import save_model

CONFIG_PATH = "configs/deepnn_config/baseline.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

seed = config["experiment"]["seed"]
device = config["experiment"]["device"]

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
scaler = preprocessing_config["scaler"]

# metrics config
metrics_config = config["metrics"]
primary_metric = metrics_config["primary"]
metrics_dict = metrics_config["log"]

# model config
model = config["model"]["name"]

# optimizer config
optimizer_name = config["optimizer"]["name"]
optimizer_step = config["optimizer"]["step"] #learning rate

# training config
training_epochs = config["training"]["epochs"]
training_batch_size = config["training"]["batch_size"]


fold_dataloaders, in_features = build_kfold_dataloader(
    train_path=train_path,
    target_col=target_col,
    batch_size=training_batch_size,
    n_splits=n_splits,
    shuffle=shuffle,
    random_state=seed
    )

model = models.BaselineModel(
    in_features=in_features,
    hidden_feature=16,
    out_features=1
    )

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=optimizer_step)

engine.fit(
    model=model,
    fold_loaders=fold_dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=training_epochs
    )

save_model(model, target_dir=".", model_name="baseline.pt")
