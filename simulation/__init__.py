"""Aegis-V2X Phase 2 — Simulation & Dataset (Owner: Haridharani).

Subpackages:
    carla_scenarios   Physical environment simulation (CARLA)
    sionna_configs     Wireless channel simulation (NVIDIA Sionna RT)
    traffic_generation Vehicle/pedestrian/RSU population logic
    synchronization    Multi-stream timestamp alignment (<=10ms tolerance)
    annotation         Ground-truth label generation (trust/criticality/TAHS/FSDP)
    dataset_pipeline   Schema, writer, generator, validator, splitter, stats
    scripts            CLI entry points

See configs/simulation.yaml, configs/dataset.yaml, configs/model.yaml (Phase 1,
owner Vaishnavi) for the frozen parameter contracts this package implements.
"""
