"""Phase 5 physics-informed spatio-temporal GNN (GCN -> GRU equivalent)."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_max_pool


class PIGNN(nn.Module):
    """Fixed PIM-weighted spatial messages followed by per-node temporal GRU."""
    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.spatial = GCNConv(feature_count, hidden_size, add_self_loops=True, normalize=True)
        self.temporal = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.10)
        self.node_head = nn.Linear(hidden_size, 1)
        self.rate_head = nn.Linear(hidden_size, 1)
        self.rate_scale_mm_per_day = 10.0

    def forward(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [all nodes, window, feature], generated per scenario or PyG Batch.
        x, mask = graph.x.float(), graph.feature_mask.float()
        x = x * mask
        encoded_steps = [torch.relu(self.spatial(x[:, step, :], graph.edge_index, graph.edge_attr.squeeze(-1))) for step in range(x.shape[1])]
        temporal_input = torch.stack(encoded_steps, dim=1)
        temporal_output, _ = self.temporal(temporal_input)
        hidden = self.dropout(temporal_output)
        node_logits = self.node_head(hidden[:, -1, :]).squeeze(-1)
        # Unitless normalized rate. Multiply by rate_scale_mm_per_day before
        # applying Fukuzono; normalization keeps multi-task training stable.
        normalized_rate = self.rate_head(hidden).squeeze(-1)
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(node_logits.shape[0], dtype=torch.long, device=node_logits.device)
        return node_logits, global_max_pool(node_logits.unsqueeze(-1), batch).squeeze(-1), normalized_rate

    def predicted_rate_mm_per_day(self, graph: Data) -> torch.Tensor:
        return self(graph)[2] * self.rate_scale_mm_per_day


@dataclass(frozen=True)
class PIGNNTrainingResult:
    initial_loss: float
    final_loss: float


def _node_severity_loss(node_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Synthetic per-node targets are continuous severity, not binary classes."""
    return nn.functional.smooth_l1_loss(torch.sigmoid(node_logits), targets)


def overfit_tiny_batch(graphs: list[Data], epochs: int = 500, learning_rate: float = 0.02) -> tuple[PIGNN, PIGNNTrainingResult]:
    """Mandatory Phase 5 test: a tiny labelled batch must be learnable."""
    if not graphs:
        raise ValueError("at least one graph is required")
    torch.manual_seed(29)
    model = PIGNN(graphs[0].x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    batch = Batch.from_data_list(graphs)
    targets = batch.y.flatten()
    def loss_value() -> torch.Tensor:
        node_logits, graph_logits, _ = model(batch)
        return loss_fn(graph_logits, targets) + _node_severity_loss(node_logits, batch.node_y)
    initial = float(loss_value().detach())
    for _ in range(epochs):
        optimizer.zero_grad(); loss = loss_value(); loss.backward(); optimizer.step()
    return model, PIGNNTrainingResult(initial, float(loss_value().detach()))


def train_validation_split(graphs: list[Data], epochs: int = 20, learning_rate: float = 0.001, batch_size: int = 32, validation_fraction: float = 0.2, hidden_size: int = 32) -> tuple[PIGNN, dict]:
    """Pretrain on synthetic graphs, preserving scenario-specific PIM edges in batches."""
    if len(graphs) < 2:
        raise ValueError("at least two graphs are required")
    torch.manual_seed(29)
    order = torch.randperm(len(graphs)).tolist()
    split = max(1, int(len(graphs) * (1 - validation_fraction)))
    train_graphs = [graphs[index] for index in order[:split]]
    validation_graphs = [graphs[index] for index in order[split:]]
    model = PIGNN(graphs[0].x.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = sum(float(graph.y.item()) for graph in train_graphs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(1.0, (len(train_graphs) - positives) / max(positives, 1))))
    def batch_loss(batch: Batch) -> torch.Tensor:
        node_logits, graph_logits, normalized_rate = model(batch)
        # Index 3 is displacement_filt in the frozen default registry.
        displacement = batch.x[:, :, 3]
        target_rate = torch.diff(displacement, dim=1, prepend=displacement[:, :1]) / model.rate_scale_mm_per_day
        valid = batch.feature_mask[:, :, 3]
        rate_loss = (((normalized_rate - target_rate) ** 2) * valid).sum() / valid.sum().clamp_min(1)
        node_loss = _node_severity_loss(node_logits, batch.node_y)
        return loss_fn(graph_logits, batch.y.flatten()) + node_loss + 0.1 * rate_loss

    def evaluate(items: list[Data]) -> float:
        model.eval(); losses = []
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch = Batch.from_data_list(items[start:start + batch_size])
                losses.append(float(batch_loss(batch)))
        return sum(losses) / len(losses)

    initial_train_loss = evaluate(train_graphs)
    training_loss_by_epoch: list[float] = []
    for _ in range(epochs):
        model.train()
        for start in range(0, len(train_graphs), batch_size):
            batch = Batch.from_data_list(train_graphs[start:start + batch_size])
            optimizer.zero_grad()
            loss = batch_loss(batch)
            loss.backward(); optimizer.step()
        training_loss_by_epoch.append(evaluate(train_graphs))
    return model, {
        "initial_train_loss": initial_train_loss,
        "train_loss": evaluate(train_graphs),
        "validation_loss": evaluate(validation_graphs) if validation_graphs else evaluate(train_graphs),
        "training_loss_by_epoch": training_loss_by_epoch,
    }
