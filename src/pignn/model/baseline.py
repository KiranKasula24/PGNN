"""Phase 4 graph-free MLP baseline and compact training helpers."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import Data


def pooled_summary(graph: Data) -> torch.Tensor:
    """Pool [node, time, feature] into graph-level mean/std/max/min features."""
    x = graph.x.float()
    mask = graph.feature_mask.float()
    count = mask.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (x * mask).sum(dim=(0, 1)) / count
    centered_sq = ((x - mean) ** 2 * mask).sum(dim=(0, 1)) / count
    # Invalid values are excluded before extrema are calculated.
    maximum = x.masked_fill(mask == 0, float("-inf")).amax(dim=(0, 1)).nan_to_num(0.0, neginf=0.0)
    minimum = x.masked_fill(mask == 0, float("inf")).amin(dim=(0, 1)).nan_to_num(0.0, posinf=0.0)
    return torch.cat((mean, centered_sq.sqrt(), maximum, minimum))


class BaselineMLP(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count * 4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
        )

    def forward(self, graph: Data) -> torch.Tensor:
        return self.forward_features(pooled_summary(graph)).squeeze(-1)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class TrainingResult:
    initial_loss: float
    final_loss: float


def overfit_tiny_batch(graphs: list[Data], epochs: int = 500, learning_rate: float = 0.02) -> tuple[BaselineMLP, TrainingResult]:
    """Development safeguard: 10-ish examples must be learnable before full training."""
    if not graphs:
        raise ValueError("at least one graph is required")
    torch.manual_seed(17)
    model = BaselineMLP(graphs[0].x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    targets = torch.stack([graph.y.squeeze() for graph in graphs])
    def loss_value() -> torch.Tensor:
        return loss_fn(torch.stack([model(graph) for graph in graphs]), targets)
    initial = float(loss_value().detach())
    for _ in range(epochs):
        optimizer.zero_grad(); loss = loss_value(); loss.backward(); optimizer.step()
    return model, TrainingResult(initial_loss=initial, final_loss=float(loss_value().detach()))


def train_validation_split(graphs: list[Data], epochs: int = 100, learning_rate: float = 0.001, validation_fraction: float = 0.2) -> tuple[BaselineMLP, dict]:
    """Train the Phase 4 baseline on synthetic graphs with a fixed reproducible split."""
    if len(graphs) < 2:
        raise ValueError("at least two graphs are required")
    torch.manual_seed(17)
    summaries = torch.stack([pooled_summary(graph) for graph in graphs])
    labels = torch.stack([graph.y.squeeze() for graph in graphs])
    order = torch.randperm(len(graphs))
    split = max(1, int(len(graphs) * (1 - validation_fraction)))
    train_idx, validation_idx = order[:split], order[split:]
    mean, std = summaries[train_idx].mean(0), summaries[train_idx].std(0).clamp_min(1e-6)
    features = (summaries - mean) / std
    model = BaselineMLP(graphs[0].x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive = labels[train_idx].sum().clamp_min(1)
    negative = len(train_idx) - positive
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(negative / positive).clamp_min(1.0))
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model.forward_features(features[train_idx]).squeeze(-1), labels[train_idx])
        loss.backward(); optimizer.step()
    with torch.no_grad():
        train_loss = float(loss_fn(model.forward_features(features[train_idx]).squeeze(-1), labels[train_idx]))
        validation_loss = float(loss_fn(model.forward_features(features[validation_idx]).squeeze(-1), labels[validation_idx])) if len(validation_idx) else train_loss
    return model, {"train_loss": train_loss, "validation_loss": validation_loss, "normalization_mean": mean, "normalization_std": std}
