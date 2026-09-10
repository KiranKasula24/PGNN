from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from pignn.datagen.dataset_writer import write_parquet
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=3000)
parser.add_argument("--seed", type=int, default=20260910)
parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
args = parser.parse_args()
config = GenerationConfig()
rng = np.random.default_rng(args.seed)
scenarios = [generate_scenario(rng, config) for _ in range(args.scenarios)]
path = write_parquet(scenarios, args.output, {"seed": args.seed, "scenarios": args.scenarios, "generation": config})
print(f"Wrote {len(scenarios)} scenarios to {path}")
