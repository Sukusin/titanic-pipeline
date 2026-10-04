"""Prepare House Prices tensors with preprocessing fitted on training rows."""

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset

from src.house_price.preprocessing import build_preprocessor


def make_loader(
    X: np.ndarray, y_log: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_log, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_fold(
    X: pd.DataFrame, y_log: np.ndarray, train_index: np.ndarray,
    valid_index: np.ndarray, batch_size: int,
) -> tuple[DataLoader, DataLoader, ColumnTransformer, int]:
    """Fit preprocessing only on the current training fold."""
    preprocessor = build_preprocessor(X.iloc[train_index])
    train_features = preprocessor.fit_transform(X.iloc[train_index])
    valid_features = preprocessor.transform(X.iloc[valid_index])
    train_loader = make_loader(train_features, y_log[train_index], batch_size, True)
    valid_loader = make_loader(valid_features, y_log[valid_index], batch_size, False)
    return train_loader, valid_loader, preprocessor, train_features.shape[1]


def prepare_final(
    X: pd.DataFrame, y_log: np.ndarray, batch_size: int
) -> tuple[DataLoader, ColumnTransformer, int]:
    preprocessor = build_preprocessor(X)
    features = preprocessor.fit_transform(X)
    return make_loader(features, y_log, batch_size, True), preprocessor, features.shape[1]
