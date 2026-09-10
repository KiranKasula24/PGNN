"""Phase 5 synthetic pretraining entrypoint for the GCN -> GRU PIGNN."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pignn.datagen.dataset_writer import load_parquet
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.pignn import train_validation_split

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--window", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--output", type=Path, default=Path("checkpoints/pignn-0.1.0.pt"))
args = parser.parse_args()
scenarios = load_parquet(args.dataset)
if {scenario["source"] for scenario in scenarios} != {"synthetic"}:
    raise ValueError("Phase 5 pretraining accepts synthetic-only datasets")
graphs = [scenario_to_pyg(scenario, args.window) for scenario in scenarios]
model, metrics = train_validation_split(graphs, epochs=args.epochs, batch_size=args.batch_size)
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state_dict": model.state_dict(), "feature_count": graphs[0].x.shape[-1], "window_size": args.window, "model_version": "pignn-0.1.0", "metrics": metrics, "architecture": "PIM-weighted GCNConv -> GRU"}, args.output)
print({"checkpoint": str(args.output), "scenarios": len(graphs), **metrics})
