"""Neural-network pieces shared by both competitions."""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

Scheduler = LRScheduler | ReduceLROnPlateau | None

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


class ModelOne(nn.Module):
    """Two-layer MLP used as a simple baseline."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict | None
) -> Scheduler:
    """Build the scheduler configured for a training run."""
    if not config or config.get("name") is None:
        return None
    scheduler_map = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    return scheduler_map[config["name"]](optimizer, **config.get("params", {}))


def step_scheduler(scheduler: Scheduler, metric: float) -> None:
    """Advance a scheduler after each epoch."""
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()
