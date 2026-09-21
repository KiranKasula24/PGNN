"""Evaluate one trained PIGNN checkpoint over several synthetic node counts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.pignn import PIGNN

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pignn-0.1.1.pt"))
parser.add_argument("--node-counts", type=int, nargs="+", default=[2, 4, 6, 8, 12])
parser.add_argument("--scenarios", type=int, default=100)
parser.add_argument("--seed", type=int, default=20260921)
parser.add_argument("--output", type=Path, default=Path("data/evaluations/node_count_sweep.json"))
args = parser.parse_args()
saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = PIGNN(int(saved["feature_count"]), int(saved.get("hidden_size", 32)))
model.load_state_dict(saved["model_state_dict"]); model.eval()
rng = np.random.default_rng(args.seed); results = []
for node_count in args.node_counts:
    correct = 0; absolute_errors = []
    for _ in range(args.scenarios):
        graph = scenario_to_pyg(generate_scenario(rng, GenerationConfig(node_count=node_count, time_steps=max(20, saved["window_size"]), dangerous_fraction=0.5)), saved["window_size"])
        with torch.no_grad():
            node_logits, graph_logits, _ = model(graph)
        correct += int((torch.sigmoid(graph_logits).item() >= 0.5) == bool(graph.y.item()))
        absolute_errors.extend(torch.abs(torch.sigmoid(node_logits) - graph.node_y).tolist())
    results.append({"node_count": node_count, "scenarios": args.scenarios, "graph_accuracy": correct / args.scenarios, "node_severity_mae": sum(absolute_errors) / len(absolute_errors)})
report = {"checkpoint": saved["model_version"], "results": results}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
