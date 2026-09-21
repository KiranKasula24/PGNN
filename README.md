# PIGNN ML service

FastAPI service and synthetic-data foundation for the mine-subsidence PIGNN.
Phases 0–7 are implemented: a checkpoint-backed `/predict` endpoint, validated
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
uv run python scripts/train_pignn.py data/synthetic/<dataset>.parquet --window 8
```

See the final hand-off notes for the complete test sequence. Synthetic outputs
are for pipeline development only; they are not real-world accuracy evidence.
The training command writes `checkpoints/pignn-0.1.1.pt`, the same default file
loaded by the service. The configured 60 epochs are the meaningful synthetic
pretraining default; use `--epochs 3` only for a CPU smoke test and compare `initial_train_loss`, `train_loss`,
and `validation_loss` before deploying a checkpoint.

After a trained checkpoint is available, run both validation artifacts on the
same generated dataset:

```powershell
uv run python scripts/run_ablation.py data/synthetic/<dataset>.parquet --epochs 60
uv run python scripts/evaluate_node_count_sweep.py --checkpoint checkpoints/pignn-0.1.1.pt
```

They write JSON reports under `data/evaluations/`. The Crack/Anomaly Score is
now the eighth registered GNN input. New sensor rows may supply it as
`crack_anomaly_score`; absent real Tier-2 values remain masked.

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
tests yield measured R values. `fuzzy_risk_index` is a clearly named temporary rate
heuristic, not the firmware's calibrated Fuzzy Risk Index. Tilt proxies also
need replacement with a calibrated spatial/IMU forward model. Phase 8 real-data
fine-tuning and Phase 9 InSAR calibration are intentionally
not implemented. Use `scripts/evaluate_two_node_transfer.py` to assess the
larger-array pretrained model on a synthetic 2-node, realistic-spacing topology.

## Scheduler / Supabase contract

Set `PIGNN_SUPABASE_URL` and `PIGNN_SUPABASE_SERVICE_ROLE_KEY` to enable the
one-minute worker. It reads `nodes`, `readings`, and `insar_node_features`, then
inserts directly into `predictions`; it requires no job or queue table. The
database column `insar_los_velocity_mm` is treated as the agreed LOS displacement
for an SLC pair and is passed unchanged into the ML displacement feature. Table
names can be overridden with `PIGNN_SUPABASE_*_TABLE` environment variables.

Run one persistent service instance to enable the periodic pass:

```powershell
uv run uvicorn --env-file .env pignn.service.main:app --host 0.0.0.0 --port 8000
```

It runs immediately at startup and then every 60 seconds. Deploy this command
as a long-running worker/web service with exactly one process; multiple Uvicorn
workers each start a scheduler and would create duplicate prediction rows.
`GET /health` reports whether the scheduler is running and its last pass status.

## Physics Expected-State Model

The product calls this component the Digital Twin; code and API names use
"Expected-State Model" to keep it distinct from the ML model. The active mine
type is selected once in `config/mine_geometry.json`.

- `POST /expected-state/state` — Longwall Monitoring Mode. It compares the
  independent PIM/Knothe expected field with baseline-relative cumulative
  displacement supplied by the caller.
- `POST /expected-state/whatif` — Longwall Planning Mode. Supply node local
  metre positions, `future_time_days`, and optionally `geometry_override` for a
  different hypothetical panel. Its response is always `is_hypothetical: true`.
- `POST /expected-state/bord-and-pillar/state` — deterministic CPHSR/pillar
  structural state using the configured prototype or surveyed B&P geometry.
- `POST /expected-state/bord-and-pillar/whatif` — B&P susceptibility projection
  for a hypothetical extraction percentage; it is not an mm-deformation field.
- `POST /risk-synthesis` — combines exactly two 0–1 inputs into one Risk Score
  and returns a separate agreement badge.
- `POST /expected-state/panel-aggregation` — conservatively rolls node scores
  into a Longwall panel score.
- `GET /expected-state/context` — currently returns explicitly simulated
  geology priors. It is not Earth Engine data.

The scheduler writes Longwall Monitoring Mode to `twin_state` only after all
of the following exist: the proposed SQL schema, a node baseline, `mine_x_m` /
`mine_y_m` coordinates in the configured local CRS, and a displacement reading.
It skips incomplete registrations rather than inventing values. Pairwise InSAR
LOS displacement is retained as a side observation; it is not cumulative and
does not enter the Physics Deviation Index.

The current database source of truth is
`config/supabase_locked_schema.sql`. It aligns node registration with
`nodes.latitude`, `nodes.longitude`, and `baseline_displacement_mm`; aligns
InSAR with `insar_grid_features.grid_id` and `los_displacement_mm`; and aligns
Twin output with `twin_state.mode` and `observed_value_mm`. The service owns the
nearest-node-to-grid lookup. Low-coherence grid values are masked, never read
as zero movement.

Before applying database changes, review the locked schema against the live
Supabase project and set RLS so only backend/service roles write the configured
tables.
Every current development placeholder is recorded in `ASSUMPTIONS.md`.

## Backend authentication and deployment

Every endpoint, including `/health`, requires this header:

```text
X-GEOARGUS-INTERNAL-KEY: <PIGNN_INTERNAL_API_KEY>
```

Set `PIGNN_INTERNAL_API_KEY` only in the FastAPI/Cloud Run service and in the
Next.js backend's server-side environment. Never expose it to the browser.

`Dockerfile` targets Cloud Run on port 8080. The trained checkpoint remains a
non-versioned binary: deployment must copy the approved
`pignn-0.1.1.pt` into `checkpoints/` before `docker build`, or mount/download it
to `PIGNN_CHECKPOINT` before starting Uvicorn. The tracked `.gitkeep` only
ensures a clean clone can build; it is not a usable model artifact.
