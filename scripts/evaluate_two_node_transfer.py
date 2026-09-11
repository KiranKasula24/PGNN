"""Option-B transfer check: evaluate a pretrained model on 2-node deployment topology.

This is synthetic topology-transfer evaluation, not a substitute for Phase 8
fine-tuning on real labelled readings.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.graph.build_graph import scenario_to_pyg
from pignn.model.pignn import PIGNN

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/pignn-0.1.0.pt")
parser.add_argument("--scenarios", type=int, default=100)
parser.add_argument("--spacing-m", type=float, default=100.0)
parser.add_argument("--seed", type=int, default=20260911)
args = parser.parse_args()
saved = torch.load(args.checkpoint, weights_only=False)
model = PIGNN(saved["feature_count"], saved.get("hidden_size", 32)); model.load_state_dict(saved["model_state_dict"]); model.eval()
rng = np.random.default_rng(args.seed)
correct = 0; node_absolute_errors = []
for index in range(args.scenarios):
    config = GenerationConfig(node_count=2, node_spacing_m=args.spacing_m, time_steps=max(saved["window_size"], 20), dangerous_fraction=0.5)
    graph = scenario_to_pyg(generate_scenario(rng, config), saved["window_size"])
    with torch.no_grad():
        node_logits, graph_logit, _ = model(graph)
    correct += int((torch.sigmoid(graph_logit).item() >= 0.5) == bool(graph.y.item()))
    node_absolute_errors.extend(torch.abs(torch.sigmoid(node_logits) - graph.node_y).tolist())
print({"evaluation": "synthetic_two_node_transfer_only", "scenarios": args.scenarios, "spacing_m": args.spacing_m, "graph_accuracy": correct / args.scenarios, "node_severity_mae": sum(node_absolute_errors) / len(node_absolute_errors), "checkpoint": saved["model_version"]})
