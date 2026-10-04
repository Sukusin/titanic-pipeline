import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import calculate_metrics
from src.utils.nn import Scheduler, step_scheduler


def train_step(
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
        ) -> tuple[float, float]:
    model.train().to(device)

    train_loss = 0.0
    train_correct = 0
    train_total = 0


    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        train_pred = model(X)

        loss = criterion(train_pred, y)
        train_loss += loss.item() * X.size(0)

        train_pred_class = (torch.sigmoid(train_pred) >= 0.5).float()
        train_correct += (train_pred_class == y).sum().item()
        train_total += y.numel()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= train_total
    train_acc = train_correct / train_total
    return train_loss, train_acc


def val_step(
        model: torch.nn.Module,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        metric_names: list[str],
        ) -> dict[str, float | None]:
    model.eval().to(device)

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_proba: list[float] = []

    with torch.inference_mode():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            val_logits = model(X)
            loss = criterion(val_logits, y)

            val_proba = torch.sigmoid(val_logits)
            val_pred_class = (val_proba >= 0.5).float()

            val_loss += loss.item() * X.size(0)
            val_correct += (val_pred_class == y).sum().item()
            val_total += y.numel()

            all_y_true.extend(y.int().cpu().numpy().ravel())
            all_y_pred.extend(val_pred_class.int().cpu().numpy().ravel())
            all_y_proba.extend(val_proba.cpu().numpy().ravel())

    metrics = calculate_metrics(
        y_true=all_y_true,
        y_pred=all_y_pred,
        metric_names=metric_names,
        y_proba=all_y_proba,
    )
    metrics["loss"] = val_loss / val_total
    return metrics


def fit(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler,
        device: torch.device,
        epochs: int,
        metric_names: list[str],
        ) -> dict[str, list[float | None]]:
    results: dict[str, list[float | None]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    
    for _epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
            )
        val_metrics = val_step(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            metric_names=metric_names
        )
        val_loss = val_metrics["loss"]
        if val_loss is None:
            raise RuntimeError("Validation loss is required to update the scheduler.")
        step_scheduler(scheduler, val_loss)

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        for metric_name, metric_value in val_metrics.items():
            results.setdefault(f"val_{metric_name}", [])
            results[f"val_{metric_name}"].append(metric_value)
    print(
        # f"Epoch {epoch+1}/{epochs} | ",
        f"Train loss: {train_loss:.04f} | "
        f"Train accuracy: {train_accuracy:.04f} | "
        f"Val loss: {val_metrics["loss"]:.04f} | "
        f"Val accuracy: {val_metrics["accuracy"]:.04f} | \n"
    )
    return results

def fit_final(
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Scheduler,
        device: torch.device,
        epochs: int,
        ) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
    }
    for epoch in tqdm(range(1, epochs+1),):
        train_loss, train_accuracy = train_step(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
            )
        step_scheduler(scheduler, train_loss)
        if epoch % 10 == 0 :
            print(
                f"Final epoch {epoch}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train accuracy: {train_accuracy:.4f}"
            )
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
    return results
