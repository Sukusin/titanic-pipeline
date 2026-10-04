"""Train a regression MLP on log prices and evaluate in price units."""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_log_error
from torch.utils.data import DataLoader

from src.utils.nn import Scheduler, step_scheduler


def train_step(
    model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(features)
        total_count += len(features)
    return total_loss / total_count


def validate(
    model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    actual_log, predicted_log = [], []
    with torch.inference_mode():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            prediction = model(features)
            total_loss += criterion(prediction, target).item() * len(features)
            total_count += len(features)
            actual_log.extend(target.cpu().numpy().ravel())
            predicted_log.extend(prediction.cpu().numpy().ravel())
    actual = np.expm1(actual_log)
    predicted = np.maximum(np.expm1(predicted_log), 0)
    return {
        "loss": total_loss / total_count,
        "rmsle": root_mean_squared_log_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mae": mean_absolute_error(actual, predicted),
    }


def fit(
    model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: Scheduler, device: torch.device, epochs: int,
    valid_loader: DataLoader | None = None,
) -> list[dict[str, float]]:
    """Train for a fixed number of epochs, mirroring Titanic's DNN loop."""
    model.to(device)
    criterion = torch.nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss}
        if valid_loader is not None:
            metrics = validate(model, valid_loader, criterion, device)
            row.update({f"val_{name}": value for name, value in metrics.items()})
            scheduler_metric = metrics["loss"]
        else:
            scheduler_metric = train_loss
        step_scheduler(scheduler, scheduler_metric)
        history.append(row)
    return history
