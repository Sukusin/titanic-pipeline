from pathlib import Path

import polars as pl
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SCALER_MAP = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler
}

# def build_scaler(config: dict):
#     scaler_name = config["preprocessing"]["scaler"]
#     scaler_class = SCALER_MAP[scaler_name]
#     return scaler_class()

# def make_folds(train_idx: list[int], val_idx: list[int]):
        

def build_kfold_dataloader(
        train_path: str | Path,
        target_col: str,
        batch_size: int,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        ):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    df = pl.read_parquet(train_path)

    X = df.drop(target_col).to_numpy()
    y = df.get_column(target_col).to_numpy().reshape(-1, 1)

    fold_loaders = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        scaler = StandardScaler()

        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        train_dataset = TensorDataset(
            torch.tensor(X_train_fold_scaled, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_fold_scaled, dtype=torch.float32),
            torch.tensor(y_val_fold, dtype=torch.float32),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        fold_loaders.append((train_loader, val_loader))

    in_features = X.shape[1]
    return fold_loaders, in_features
