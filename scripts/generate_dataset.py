from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from pignn.datagen.dataset_writer import write_parquet
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario
from pignn.datagen.sensor_simulator import SensorNoise
from pignn.config import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=3000)
parser.add_argument("--seed", type=int, default=20260910)
parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
parser.add_argument("--config", type=Path, default=Path("config/data_gen.yaml"))
args = parser.parse_args()
settings = load_yaml(args.config)
noise = settings["noise"]
config = GenerationConfig(node_count=settings["node_count"], time_steps=settings["time_steps"], time_step_days=settings["time_step_days"], dangerous_fraction=1 - settings["safe_fraction"], noise=SensorNoise(**noise))
if args.scenarios == parser.get_default("scenarios"):
    args.scenarios = settings["scenario_count"]
if args.seed == parser.get_default("seed"):
    args.seed = settings["seed"]
rng = np.random.default_rng(args.seed)
scenarios = [generate_scenario(rng, config) for _ in range(args.scenarios)]
path = write_parquet(scenarios, args.output, {"seed": args.seed, "scenarios": args.scenarios, "generation": config})
print(f"Wrote {len(scenarios)} scenarios to {path}")
