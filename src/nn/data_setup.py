from pathlib import Path

import polars as pl
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.nn.preprocessing import EmbeddingPreprocessor

SCALER_MAP = {"minmax": MinMaxScaler, "standard": StandardScaler}


def build_preprocessor(
    scaler_name: str,
    feature_names: list[str],
    categorical_features: list[str] | None,
):
    """Create numeric scaling, or fold-local category encoding plus numeric scaling."""
    scaler = SCALER_MAP[scaler_name]()
    if categorical_features is not None:
        return EmbeddingPreprocessor(scaler, feature_names, categorical_features)
    return scaler


def build_kfold_dataloader(
    train_path: str | Path,
    target_col: str,
    scaler_name: str,
    batch_size: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    categorical_features: list[str] | None = None,
):
    """Return train/validation/preprocessor triples fitted exclusively on each train fold."""
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None
    )

    df = pl.read_parquet(train_path)

    X = df.drop(target_col).to_numpy()
    y = df.get_column(target_col).to_numpy().reshape(-1, 1)
    feature_names = df.drop(target_col).columns

    fold_loaders = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]

        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        scaler = build_preprocessor(scaler_name, feature_names, categorical_features)
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

        fold_loaders.append((train_loader, val_loader, scaler))

    in_features = X.shape[1]
    return fold_loaders, in_features


def build_final_dataloader(
    train_path: str | Path,
    target_col: str,
    scaler_name: str,
    batch_size: int,
    random_state: int = 42,
    categorical_features: list[str] | None = None,
):
    """Fit and return final preprocessing alongside the full-data training loader."""
    df = pl.read_parquet(train_path)

    X = df.drop(target_col).to_numpy()
    y = df.get_column(target_col).to_numpy().reshape(-1, 1)
    in_features = X.shape[1]

    scaler = build_preprocessor(scaler_name, df.drop(target_col).columns, categorical_features)
    X_scaled = scaler.fit_transform(X)
    train_dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )

    final_train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return final_train_loader, in_features, scaler
