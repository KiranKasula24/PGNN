import numpy as np
import pyarrow.parquet as pq
from pignn.datagen.dataset_writer import validate_scenario, write_parquet
from pignn.datagen.scenario_generator import GenerationConfig, generate_scenario


def test_generated_schema_and_source():
    scenario = generate_scenario(np.random.default_rng(7), GenerationConfig(node_count=4, time_steps=8))
    validate_scenario(scenario)
    assert scenario["source"] == "synthetic"
    assert len(scenario["time_series"]) == 4


def test_class_imbalance_is_deliberate_and_within_bounds():
    config = GenerationConfig(node_count=3, time_steps=12, dangerous_fraction=0.10)
    rng = np.random.default_rng(44)
    labels = [generate_scenario(rng, config)["label"]["subsidence_occurred"] for _ in range(250)]
    ratio = sum(labels) / len(labels)
    assert 0.04 <= ratio <= 0.17, ratio


def test_parquet_writer_preserves_locked_fields(tmp_path):
    scenario = generate_scenario(np.random.default_rng(9), GenerationConfig(node_count=3, time_steps=6))
    path = write_parquet([scenario], tmp_path, {"case": "test"})
    row = pq.read_table(path).to_pylist()[0]
    assert row["source"] == "synthetic"
    assert row["scenario_id"] == scenario["scenario_id"]
    assert row["dataset_version"]
