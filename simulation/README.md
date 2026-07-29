# simulation/

**Phase:** Phase 2
**Owner:** Haridharani
**Status:** Code complete, not yet run on a GPU workstation. See
`datasets/metadata/CHANGELOG.md` for what's implemented vs. outstanding.

## Purpose
CARLA scenario definitions, Sionna RT configuration scripts, and traffic
generation logic. Produces the synchronized multimodal dataset in
`datasets/` (see `configs/dataset.yaml` for schema/paths).

## Layout

```
simulation/
├── carla_scenarios/     CARLA world setup, sensors, scene export, scenario_configs/*.yaml
├── sionna_configs/      CARLA -> Sionna geometry adapter, ray-traced channel simulator
├── traffic_generation/  Vehicle/pedestrian/RSU spawning (density-aware)
├── synchronization/     <=10ms multi-stream timestamp synchronizer
├── annotation/          Trust/criticality/future-channel/bootstrap-TAHS-FSDP labelers
├── dataset_pipeline/    Schema, writer, generator, validator, splitter, preprocessor, stats
└── scripts/             CLI entry points
```

## Prerequisites (GPU workstation, not this repo checkout alone)

CARLA 0.9.15 + NVIDIA Sionna RT per `requirements/simulation.txt` — see that
file for the exact pinned versions and install notes.

## Running the pipeline (from repo root)

```bash
# Full pipeline for one scenario (requires CARLA + Sionna RT installed and running)
python -m simulation.scripts.generate_dataset --scenario simulation/carla_scenarios/scenario_configs/urban_dense_clear_day.yaml

# All 5 scenarios (100 scenes total)
python -m simulation.scripts.generate_dataset --scenario-dir simulation/carla_scenarios/scenario_configs

# Annotation stage only, if raw CARLA/Sionna data already exists (no GPU needed)
python -m simulation.scripts.generate_dataset --scenario-dir simulation/carla_scenarios/scenario_configs --skip-simulation

python -m simulation.scripts.validate_dataset
python -m simulation.scripts.split_dataset
python -m simulation.scripts.compute_statistics
```

## Running the tests (no CARLA/Sionna required)

```bash
pytest tests/unit/test_synchronizer.py tests/unit/test_annotation_labelers.py \
       tests/unit/test_schema_validator.py tests/unit/test_splitter.py -v
```

This folder is part of the fixed Aegis-V2X repository structure
(Master Project Instructions). Do not add files here outside of the
phase listed above without updating docs/module_decomposition.md.
