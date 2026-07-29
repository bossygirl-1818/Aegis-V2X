"""Top-level CLI: run the full Phase 2 pipeline for one or all scenario configs.

Stage 1 (requires CARLA + Sionna RT, GPU workstation): physical + wireless
simulation, producing datasets/raw/carla and datasets/raw/sionna.
Stage 2 (pure Python, runs anywhere): synchronize + annotate + write.

Usage (from repo root):
    python -m simulation.scripts.generate_dataset \\
        --scenario-dir simulation/carla_scenarios/scenario_configs
    python -m simulation.scripts.generate_dataset \\
        --scenario-dir simulation/carla_scenarios/scenario_configs --skip-simulation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from simulation.carla_scenarios.town_maps import resolve_map
from simulation.dataset_pipeline.generator import DatasetGenerator
from simulation.dataset_pipeline.schema import TrafficDensity, WeatherCondition

logger = logging.getLogger("aegis_v2x.simulation.scripts.generate_dataset")


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_simulation_stage(scenario_cfg: dict, global_config_path: Path,
                          raw_carla_dir: Path, raw_sionna_dir: Path, static_scene_dir: Path) -> None:
    # Imported lazily: requires 'carla'/'sionna' packages, only present on the
    # GPU workstation, not on the annotation-only code path.
    from simulation.carla_scenarios.run_scenario import main as run_carla_main
    from simulation.sionna_configs.run_channel_sim import main as run_sionna_main

    map_name = resolve_map(scenario_cfg["town_type"])
    num_scenes = int(scenario_cfg.get("num_scenes", 1))

    run_carla_main(["--scenario", str(scenario_cfg["_source_path"]), "--global-config", str(global_config_path),
                    "--output", str(raw_carla_dir), "--num-scenes", str(num_scenes)])

    for scene_index in range(num_scenes):
        scene_dir = raw_carla_dir / f"{scenario_cfg['scenario_id']}_Scene{scene_index:02d}"
        run_sionna_main(["--scene-dir", str(scene_dir), "--map-name", map_name,
                          "--static-scene-dir", str(static_scene_dir), "--output", str(raw_sionna_dir)])


def run_annotation_stage(scenario_cfg: dict, raw_carla_dir: Path, raw_sionna_dir: Path,
                          repo_root: Path, sync_tolerance_ms: float) -> int:
    generator = DatasetGenerator(raw_carla_dir, raw_sionna_dir, repo_root, sync_tolerance_ms)
    weather = WeatherCondition(scenario_cfg["weather"])
    density = TrafficDensity(scenario_cfg["traffic_density"])
    scenario_metadata = {
        "scenario_id": scenario_cfg["scenario_id"], "town_type": scenario_cfg["town_type"],
        "map": resolve_map(scenario_cfg["town_type"]), "duration_sec": scenario_cfg.get("duration_sec"),
        "frame_rate_hz": scenario_cfg.get("frame_rate_hz", 10), "random_seed": scenario_cfg.get("random_seed"),
    }

    total = 0
    num_scenes = int(scenario_cfg.get("num_scenes", 1))
    for scene_index in range(num_scenes):
        scene_id = f"{scenario_cfg['scenario_id']}_Scene{scene_index:02d}"
        try:
            total += len(generator.process_scene(scene_id, weather, density, scenario_metadata))
        except FileNotFoundError as exc:
            logger.warning("Skipping scene %s: %s", scene_id, exc)
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Aegis-V2X Phase 2 dataset generation pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", type=Path, help="Single scenario YAML config")
    group.add_argument("--scenario-dir", type=Path, help="Directory of scenario YAML configs")

    parser.add_argument("--global-config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--static-scene-dir", type=Path, default=Path("simulation/sionna_configs/static_scenes"))
    parser.add_argument("--raw-carla-dir", type=Path, default=Path("datasets/raw/carla"))
    parser.add_argument("--raw-sionna-dir", type=Path, default=Path("datasets/raw/sionna"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sync-tolerance-ms", type=float, default=10.0)
    parser.add_argument("--skip-simulation", action="store_true",
                         help="Skip CARLA/Sionna generation; only run synchronize+annotate+write.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    scenario_paths = [args.scenario] if args.scenario else sorted(args.scenario_dir.glob("*.yaml"))
    if not scenario_paths:
        logger.error("No scenario configs found.")
        return 1

    grand_total = 0
    for scenario_path in scenario_paths:
        scenario_cfg = _load_yaml(scenario_path)
        scenario_cfg["_source_path"] = scenario_path
        logger.info("=== Scenario: %s ===", scenario_cfg["scenario_id"])

        if not args.skip_simulation:
            run_simulation_stage(scenario_cfg, args.global_config, args.raw_carla_dir, args.raw_sionna_dir,
                                  args.static_scene_dir)

        grand_total += run_annotation_stage(scenario_cfg, args.raw_carla_dir, args.raw_sionna_dir,
                                             args.repo_root, args.sync_tolerance_ms)

    logger.info("Pipeline complete. Total annotated samples: %d", grand_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
