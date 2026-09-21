"""Train PIGNN and graph-free baseline on one synthetic dataset and compare loss."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pignn.datagen.dataset_writer import load_parquet
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model import baseline, pignn

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--window", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--output", type=Path, default=Path("data/evaluations/pignn_vs_baseline.json"))
args = parser.parse_args()
scenarios = load_parquet(args.dataset)
graphs = [scenario_to_pyg(scenario, args.window) for scenario in scenarios]
_, pignn_metrics = pignn.train_validation_split(graphs, epochs=args.epochs, batch_size=args.batch_size)
_, baseline_metrics = baseline.train_validation_split(graphs, epochs=args.epochs)
report = {
    "dataset": str(args.dataset), "scenarios": len(graphs), "epochs": args.epochs,
    "pignn": {key: value for key, value in pignn_metrics.items() if key != "training_loss_by_epoch"},
    "baseline": {key: value for key, value in baseline_metrics.items() if key not in {"normalization_mean", "normalization_std"}},
    "interpretation": "Lower validation loss is better; synthetic-only comparison is not real-world accuracy evidence.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
