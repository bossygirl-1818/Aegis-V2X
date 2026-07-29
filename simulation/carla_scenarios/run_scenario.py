"""CLI entry point: run a CARLA scenario end-to-end and export raw frames.

Reads global parameters from the root, Phase-1-owned configs/simulation.yaml
(sensor frequencies, vehicles_per_scene range) and per-scenario definitions
from simulation/carla_scenarios/scenario_configs/*.yaml.

Usage (run from repo root, per pyproject.toml pythonpath="."):
    python -m simulation.carla_scenarios.run_scenario \\
        --scenario simulation/carla_scenarios/scenario_configs/urban_dense_clear_day.yaml \\
        --output datasets/raw/carla
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .client import CarlaClientManager, CarlaServerConfig
from .scenario_builder import ScenarioBuilder
from .scene_exporter import SceneExporter
from .sensors import SensorRig
from .town_maps import resolve_map

logger = logging.getLogger("aegis_v2x.simulation.carla_scenarios.run_scenario")


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_single_scene(scene_index: int, global_cfg: dict, scenario_cfg: dict, output_dir: Path) -> None:
    scene_id = f"{scenario_cfg['scenario_id']}_Scene{scene_index:02d}"
    map_name = resolve_map(scenario_cfg["town_type"])
    server_cfg = CarlaServerConfig(traffic_manager_seed=int(scenario_cfg.get("random_seed", 42)))

    vehicles_range = tuple(global_cfg["carla"]["vehicles_per_scene"])
    sensor_freq = {
        "lidar_hz": global_cfg["sensors"]["lidar_hz"], "gps_hz": global_cfg["sensors"]["gps_hz"],
        "imu_hz": global_cfg["sensors"]["imu_hz"], "camera_fps": global_cfg["sensors"]["camera_fps"],
        "vehicle_speed_hz": global_cfg["sensors"]["vehicle_speed_hz"],
    }

    with CarlaClientManager(server_cfg, map_name) as manager:
        builder = ScenarioBuilder(manager.world, manager.traffic_manager, vehicles_range,
                                   seed=int(scenario_cfg.get("random_seed", 42)))
        actors = builder.build(scenario_cfg)

        sensor_rigs = {}
        for vehicle in actors.vehicles:
            rig = SensorRig(manager.world, vehicle, sensor_freq)
            rig.attach_all()
            sensor_rigs[vehicle.id] = rig

        exporter = SceneExporter(manager.world, actors, sensor_rigs, scene_id, output_dir)

        duration_sec = float(scenario_cfg["duration_sec"])
        dt = server_cfg.fixed_delta_seconds
        frame_rate_hz = float(scenario_cfg.get("frame_rate_hz", 10))
        export_every_n_ticks = max(1, round((1.0 / frame_rate_hz) / dt))
        total_ticks = int(duration_sec / dt)

        logger.info("Scene %s: %d ticks, exporting every %d ticks (%.1f Hz), map=%s",
                    scene_id, total_ticks, export_every_n_ticks, frame_rate_hz, map_name)

        for tick_idx in range(total_ticks):
            frame = manager.tick()
            timestamp = tick_idx * dt
            if tick_idx % export_every_n_ticks == 0:
                exporter.export_frame(frame, timestamp)

        for rig in sensor_rigs.values():
            rig.destroy()

    logger.info("Scene %s complete.", scene_id)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a CARLA scenario and export raw frames.")
    parser.add_argument("--scenario", required=True, type=Path, help="Path to a scenario_configs/*.yaml file")
    parser.add_argument("--global-config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--output", type=Path, default=Path("datasets/raw/carla"))
    parser.add_argument("--num-scenes", type=int, default=None, help="Override num_scenes from the scenario config")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    global_cfg = _load_yaml(args.global_config)
    scenario_cfg = _load_yaml(args.scenario)

    num_scenes = args.num_scenes or int(scenario_cfg.get("num_scenes", 1))
    for scene_index in range(num_scenes):
        try:
            run_single_scene(scene_index, global_cfg, scenario_cfg, args.output)
        except Exception:
            logger.exception("Scene %d of scenario '%s' failed; continuing with next scene.",
                              scene_index, scenario_cfg["scenario_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
