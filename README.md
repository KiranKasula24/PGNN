# PIGNN ML service

FastAPI service and synthetic-data foundation for the mine-subsidence PIGNN.
Phases 0–6 are implemented: a frozen `/predict` contract stub, validated
physics simulator, registry-based sensor simulation, versioned synthetic
Parquet generation, feature/graph conversion, an MLP baseline, a PIM-weighted
GCN→GRU PIGNN, and uncertainty-aware inverse-velocity post-processing.
No Next.js integration is required at this stage.

## Quick start

```powershell
uv sync --extra dev
uv run pytest
uv run uvicorn pignn.service.main:app --reload
uv run python scripts/generate_dataset.py --scenarios 1000
uv run python scripts/visualize_physics.py
uv run python scripts/train_baseline.py data/synthetic/<dataset>.parquet
uv run python scripts/train_pignn.py data/synthetic/<dataset>.parquet --epochs 3 --window 8
```

See the final hand-off notes for the complete test sequence. Synthetic outputs
are for pipeline development only; they are not real-world accuracy evidence.

## Phase 3 and 4 boundaries

`pignn.features` is registry-based: each source supplies its own extractor and
validity mask. Pre-processed InSAR is accepted only as numeric LOS and
coherence fields; low-coherence values become unavailable, never zero.
`pignn.graph.scenario_to_pyg()` emits a PyTorch Geometric `Data` object with
PIM-derived fixed edge weights. The Phase 4 MLP intentionally pools node/window
features and does not use those edges; Phase 5 replaces it with the full GNN.

The Phase 5 PIGNN applies those fixed physics edges in each temporal slice,
then uses a GRU for time evolution. It has severity and deformation-rate heads.
Phase 6 uses MC-dropout rate trajectories with guarded Fukuzono extrapolation;
an unavailable estimate remains `null`, never an invented date.

## Known prototype boundaries

Kalman Q/R and virtual sensor noise are provisional until hardware stillness
tests yield measured R values. `risk_score` is a clearly named temporary rate
heuristic, not the firmware's calibrated Fuzzy Risk Index. Tilt proxies also
need replacement with a calibrated spatial/IMU forward model. Phase 7 serving,
Phase 8 real-data fine-tuning, and Phase 9 InSAR calibration are intentionally
not implemented. Use `scripts/evaluate_two_node_transfer.py` to assess the
larger-array pretrained model on a synthetic 2-node, realistic-spacing topology.
