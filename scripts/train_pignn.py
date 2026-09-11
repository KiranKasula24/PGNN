"""Phase 5 synthetic pretraining entrypoint for the GCN -> GRU PIGNN."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from pignn.datagen.dataset_writer import load_parquet
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.pignn import train_validation_split
from pignn.config import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
parser.add_argument("--model-config", type=Path, default=Path("config/model.yaml"))
parser.add_argument("--epochs", type=int)
parser.add_argument("--window", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--output", type=Path, default=Path("checkpoints/pignn-0.1.0.pt"))
parser.add_argument("--model-version", type=str)
args = parser.parse_args()
settings = load_yaml(args.config)
model_settings = load_yaml(args.model_config)
args.epochs = args.epochs if args.epochs is not None else settings["epochs"]
args.window = args.window if args.window is not None else settings["window_size"]
args.batch_size = args.batch_size if args.batch_size is not None else settings["batch_size"]
model_version = args.model_version or settings["model_version"]
if settings["mode"] != "pretrain":
    raise ValueError("train_pignn.py supports mode: pretrain only; Phase 8 owns fine-tuning")
scenarios = load_parquet(args.dataset)
if {scenario["source"] for scenario in scenarios} != {"synthetic"}:
    raise ValueError("Phase 5 pretraining accepts synthetic-only datasets")
graphs = [scenario_to_pyg(scenario, args.window) for scenario in scenarios]
if model_settings["model"] != "pignn_gcn_gru":
    raise ValueError("train_pignn.py requires model: pignn_gcn_gru")
model, metrics = train_validation_split(graphs, epochs=args.epochs, learning_rate=settings["learning_rate"], batch_size=args.batch_size, hidden_size=model_settings["hidden_size"])
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model_state_dict": model.state_dict(), "feature_count": graphs[0].x.shape[-1], "hidden_size": model_settings["hidden_size"], "window_size": args.window, "model_version": model_version, "metrics": metrics, "architecture": "PIM-weighted GCNConv -> GRU"}, args.output)
print({"checkpoint": str(args.output), "scenarios": len(graphs), **metrics})
