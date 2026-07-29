# Aegis-V2X Dataset — Version History

Per configs/dataset.yaml `versioning` (current_version: "v0.0.0-unreleased").
Semantic versioning per 09_Dataset_Design_and_Annotation_Guide, Chapter 20.

## [Unreleased] — v0.0.0-unreleased — Phase 2 pipeline implemented

Code complete, not yet run against real CARLA/Sionna RT (requires GPU
workstation). Added under `simulation/`:

- `traffic_generation/` — vehicle/pedestrian/RSU spawning, density-aware
  (dense/sparse per configs/simulation.yaml)
- `carla_scenarios/` — CARLA world connection, scenario building, sensor
  rig (LiDAR/GPS/IMU/camera/speed), per-frame physical + geometry export,
  5 scenario configs covering all 5 town_types x all 4 weather categories
- `sionna_configs/` — CARLA-to-Sionna geometry adapter, ray-traced channel
  simulator (CSI/SNR/RSSI/path loss/beam index/LOS-NLOS)
- `synchronization/` — <=10ms multi-stream timestamp synchronizer
  (configs/simulation.yaml `synchronization.tolerance_ms`)
- `annotation/` — trust/criticality labeler (weights/temperature exactly
  matching configs/model.yaml), future-channel labeler, bootstrap
  TAHS/FSDP for validating the boundedness/monotonicity/determinism
  invariants in docs/interfaces.md
- `dataset_pipeline/` — schema (matches configs/dataset.yaml schema_fields
  and naming_convention exactly), writer, generator, validator (Chapter 15),
  splitter (scene-disjoint, ratios from configs/dataset.yaml), preprocessor,
  statistics
- `scripts/` — CLI entry points for the full pipeline
- `tests/unit/test_synchronizer.py`, `test_annotation_labelers.py`,
  `test_schema_validator.py`, `test_splitter.py` — 100% passing against
  synthetic data (no CARLA/Sionna required)

### Known open items before v1.0

- Static Mitsuba scenes per CARLA map not yet exported (one-time offline
  step; see `simulation/sionna_configs/geometry_adapter.py` docstring)
- No real CARLA/Sionna RT run executed yet
- `sample` naming convention interpretation (scenario-prefixed scene_id vs.
  bare 2-digit integer implied by configs/dataset.yaml) needs Vaishnavi's
  sign-off — see `simulation/dataset_pipeline/schema.py` docstring
- `simulation/annotation/decision_labeler.py`'s FSDP policy table is a
  hand-authored bootstrap; Phase 5 must produce the real Pareto-optimal
  policy table (`models/fsdp_policy_table.json` per configs/model.yaml)

## v1.0 — (not yet released)

*(fill in once the first full dataset generation + validation + split run completes)*
