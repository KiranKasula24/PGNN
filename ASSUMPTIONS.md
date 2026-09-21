# Prototype Assumptions

This file records every non-measured value used by the Physics Expected-State
Model. These values are for development and demonstration only. They must be
replaced by surveyed mine plans, production logs, GNSS registration records,
geology priors, and physics calibration before operational use.

## Coordinate system

- **Assumption:** all prototype panel and node coordinates use a local planar
  coordinate system in metres, centred on the prototype panel at `(0, 0)`.
- **Why:** no surveyed mine coordinate reference system or GNSS node positions
  have been supplied yet.
- **Required replacement:** mine CRS/projection and one-time GNSS registration
  coordinates for every installed node.

## Longwall prototype geometry and extraction state

| Field | Assumed value | Basis / replacement required |
| --- | ---: | --- |
| Panel ID | `LW-PROTOTYPE-A` | Development identifier; replace with a mine-plan panel ID. |
| Panel width | 200 m | Illustrative prototype geometry; replace with surveyed boundary polygon. |
| Panel length | 300 m | Illustrative prototype geometry; replace with surveyed boundary polygon. |
| Depth | 200 m | Illustrative prototype depth; replace with surveyed seam depth. |
| Extraction thickness | 3.0 m | Illustrative mining height; replace with mine production data. |
| Status | `active` | Prototype state; replace with current production state. |
| Active face position | 120 m / 40% | Manually chosen prototype progress; replace with production-log face position and timestamp. |
| Extraction start | 2025-01-01 UTC | Prototype date; replace with actual extraction start date. |
| Orientation | 0° | Local-coordinate simplification; replace with panel azimuth. |

## Longwall physics parameters

| Parameter | Assumed value | Basis / replacement required |
| --- | ---: | --- |
| Subsidence factor `a` | 0.70 | Illustrative Indian coal-measure PIM default. Replace with geology-derived prior and InSAR physics calibration. |
| Major influence angle `β` | 55° | Illustrative Indian PIM default. Replace with site-specific prior/calibration. |
| Knothe coefficient `c` | 0.015/day | Illustrative temporal coefficient. Replace through historical InSAR curve fitting. |
| Uncertainty band | 100 mm | Temporary constant. Replace with uncertainty propagated from SoilGrids/geology priors and calibration residuals. |

## Bord-and-pillar placeholders

The `bord_and_pillar` object in `config/mine_geometry.json` is only a schema
prototype. Its width, depth, seam, geology, extraction, goaf, and three-pillar
adjacency fields are all assumed values.

- **CPHSR placeholder:** the implementation uses an equal 0.25 weight for
  normalized rock-to-soil ratio, depth-to-extraction-height ratio, brittleness,
  and density. The provisional normalization ranges are respectively `0–4`,
  `20–100`, `0–1`, and `1800–3000 kg/m³`. Its output is marked
  `coefficients_verified: false`; it is not a validated CPHSR score.
- **Pillar unit weight:** `0.024525 MPa/m`, derived from an assumed rock density
  of `2500 kg/m³` and standard gravity. It must be replaced by a geological
  survey value.
- **Intact pillar strength:** `10 MPa`, a prototype-only value. It must be
  replaced with site/laboratory strength data.

Until the source paper and site data arrive, we will use the provisional values
above for: (1) CPHSR equal weights and normalization ranges, (2) pillar unit
weight and intact strength, and (3) Longwall PIM/Knothe parameters. Each output
continues to carry an explicit `coefficients_verified: false` or
`source: simulated_prototype` marker as applicable.

## Simulated Earth Engine / geology context

- **Assumption:** Earth Engine is unavailable during prototype development, so
  a deterministic simulated context is used: sandy-loam soil, bulk density
  `1.45 g/cm³`, NDVI `0.55`, slope `3°`, and parameter priors `a=0.70`,
  `β=55°`, `c=0.015/day`.
- **Uncertainty:** a provisional `100 mm` expected-deformation band is used;
  it is not SoilGrids uncertainty propagation.
- **Replacement:** a real Earth Engine/SoilGrids/OpenLandMap query and the
  provider's published per-pixel uncertainty must replace this simulator.

## Supabase expected-state schema

**Superseded:** the supplied locked DDL is now mirrored at
`config/supabase_locked_schema.sql`. It uses `nodes.baseline_displacement_mm`
and `insar_grid_features/grid_id/los_displacement_mm`; the old provisional
`node_baseline` and `insar_node_features` names are not part of the live
contract.

**Temporary spatial assumption:** registered latitude/longitude is projected
to local metres around the node centroid for the prototype panel. Replace this
with the surveyed AOI/panel origin after `site_config.panel_boundary` has a
locked coordinate representation.

**Open schema mapping:** `site_config` is the future live geometry source, but
the locked DDL does not define the JSON shape of `panel_boundary` or
`current_face_position`, nor columns for extraction-start time and calibrated
Longwall `a`, `beta`, and `c`. The service therefore continues to use the
explicitly-assumed local `mine_geometry.json` for those physics inputs until
that mapping is agreed. This avoids silently inventing operational geometry.

- **Assumption:** the project currently has no `node_baseline` or `twin_state`
  tables. The repository therefore contains a proposed, **not yet applied** SQL
  migration at `config/supabase_expected_state_schema.sql`.
- **Assumption:** `nodes.node_id` is a bigint, matching the current Python
  service contract. Confirm this against the live schema before running it.
- **Replacement:** apply the migration through the backend team's normal
  Supabase migration process, then confirm RLS and service-role permissions.

## Deployment checkpoint and internal authentication

- **Checkpoint delivery:** `pignn-0.1.1.pt` is intentionally gitignored. The
  deployment pipeline must inject the approved artifact into `checkpoints/`
  before building the container, or make it available at `PIGNN_CHECKPOINT` at
  startup. A clean clone contains only `checkpoints/.gitkeep` and cannot serve
  `/predict` until the artifact is supplied.
- **Internal API key:** all Python service routes require the server-side
  `PIGNN_INTERNAL_API_KEY` through `X-GEOARGUS-INTERNAL-KEY`. It is a
  prototype shared-secret mechanism; migrate to Cloud Run IAM/service identity
  when the GCP deployment and backend identity are available.

- **Scheduler coordinates:** automatic Longwall monitoring requires
  `nodes.mine_x_m` and `nodes.mine_y_m` in the same local-metre coordinate
  system as `mine_geometry.json`. The existing mock latitude/longitude fields
  are deliberately not converted silently, because a mine CRS/origin has not
  been supplied. Nodes without these fields, a baseline, or a displacement
  reading are skipped rather than generating a misleading Twin row.

## InSAR database contract needed by the Expected-State Model

The InSAR team may write through the backend; no direct call to this service is
needed. The stored `insar_node_features` row needs: `node_id` (the exact same
identifier as `nodes.node_id`), `raster_date` (UTC acquisition timestamp),
`insar_los_displacement_mm` (numeric LOS displacement, with its reference
epoch/baseline documented), `insar_coherence` (0–1), and preferably
`cell_id` (the source grid cell). A unique `(node_id, raster_date)` is required
to make imports idempotent. The crucial unresolved semantic is whether LOS
displacement is pairwise or cumulative relative to a named reference; the Twin
can only compare cumulative quantities after that reference is known.

**Current confirmed semantic:** `insar_los_displacement_mm` is LOS displacement
for one raster/SLC pair. It is therefore a pairwise side observation and is
not used as `observed_cumulative_displacement_mm` or in the Physics Deviation
Index. Baseline-relative displacement sensor data supplies that cumulative
comparison until the InSAR team publishes a common-reference cumulative series.

## Deliberate modelling simplifications

- Multi-panel output will be the sum of independently calculated panel fields.
  This is a linear-independence approximation, not a 3D non-linear multi-seam
  geomechanical model.
- The expected-state calculation does not consume live readings. Observed
  cumulative displacement is compared only after the independent physics field
  is computed.
- Node baseline displacement and cumulative displacement are not yet connected
  to the database because GNSS registration records and the agreed storage
  schema have not yet been supplied.

## Risk Synthesis prototype policy

- **Fusion rule:** `1 - (1 - signal_a) × (1 - signal_b)` (a bounded noisy-OR).
  This is a provisional implementation of the required no-single-channel-veto
  principle: one strong signal can raise risk even when the other is low.
- **Risk-action bands:** `<0.30 = monitor`, `0.30–<0.60 = inspect`,
  `0.60–<0.80 = restrict_access`, and `≥0.80 = emergency_review`. These are
  product-demo thresholds, not DGMS-referenced operational thresholds.
- **Agreement bands:** absolute input difference `≤0.20 = agreeing`,
  `≤0.50 = mixed`, otherwise `disagreeing`. This badge is intentionally
  separate from risk colour and must not alter the Risk Score.

## Bord-and-Pillar safety-factor risk calibration

- **Safe safety factor:** `SF >= 1.50` maps to a pillar contribution of `0`.
  A failed pillar (`SF <= 1`) maps to `1`. Between those values, the prototype
  uses a linear margin mapping: `1 - (SF - 1) / (1.5 - 1)`.
- **Why:** raw `1/SF` over-alarmed the supplied prototype geometry: an intact
  `SF=1.248` became `0.801`, falsely crossing the emergency action threshold.
- **Replacement:** confirm the mine's accepted design safety factor and its
  action thresholds with a qualified geotechnical/mining engineer.

- **Bord-and-Pillar planning:** the hypothetical extraction-percent endpoint
  linearly scales the current structural risk relative to the assumed current
  `55%` extraction. This is only a placeholder until verified CPHSR/mining-state
  relationships are available.

## Panel aggregation prototype policy

- Node-to-panel association uses the rectangular panel boundary expanded by
  `r = depth / tan(β)`, the PIM major-influence radius. It requires real node
  coordinates and surveyed panel boundaries before live use.
- Panel severity is the **maximum** severity among associated nodes. This is a
  conservative prototype policy: a panel score must not hide its worst node.
  It may be replaced by a validated exposure-aware or weighted policy later.

## Planning Mode geometry overrides

- A what-if request may override any Longwall panel geometry/physics field and
  runs only in memory. It is never written to `twin_state` or live risk tables.
- Any number supplied by a user in such a request is a scenario input, not a
  surveyed fact. The API response remains `is_hypothetical: true`.

## Crack/Anomaly Score prototype

- **Assumption:** synthetic pretraining derives `crack_anomaly_score` from
  simulated resistivity decrease, moisture increase, temperature drift, and
  simulated geology susceptibility. It is an interpretable 0–1 proxy, not
  evidence of a physical crack.
- **Replacement:** firmware Tier-2 readings, calibrated geology context, and
  agreed rule breakpoints must replace this simulation before deployment.
- **Model consequence:** it is a registered GNN feature. Adding it changes the
  feature count from seven to eight, so old checkpoints/datasets are
  incompatible and a full retrain is required.
