"""CLI entry point: run Sionna RT channel simulation over a CARLA raw scene export.

Reads datasets/raw/carla/<scene_id>/frame_*/{vehicle_states,geometry_snapshot}.json
and writes datasets/raw/sionna/<scene_id>/frame_*/link_<rsu>_<vehicle>.{npz,json}

Usage (run from repo root):
    python -m simulation.sionna_configs.run_channel_sim \\
        --scene-dir datasets/raw/carla/urban_dense_clear_day_Scene00 \\
        --map-name Town10HD --output datasets/raw/sionna
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .channel_simulator import ChannelSimulationResult, ChannelSimulator
from .geometry_adapter import GeometryAdapter

logger = logging.getLogger("aegis_v2x.simulation.sionna_configs.run_channel_sim")


def _extract_positions(geometry_snapshot: list) -> Tuple[Dict[int, tuple], Dict[int, tuple]]:
    rsu_positions, vehicle_positions = {}, {}
    for entry in geometry_snapshot:
        pos = tuple(entry["center_xyz"])
        if entry.get("role") == "rsu":
            rsu_positions[entry["actor_id"]] = pos
        elif entry.get("material") == "vehicle":
            vehicle_positions[entry["actor_id"]] = pos
    return rsu_positions, vehicle_positions


def process_scene(scene_dir: Path, map_name: str, static_scene_dir: Path, output_dir: Path) -> int:
    adapter = GeometryAdapter(static_scene_dir)
    simulator = ChannelSimulator()

    scene_id = scene_dir.name
    out_scene_dir = Path(output_dir) / scene_id
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    frame_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("frame_"))
    if not frame_dirs:
        logger.warning("No frame_* directories found under %s", scene_dir)
        return 0

    processed = 0
    for frame_dir in frame_dirs:
        geometry_path = frame_dir / "geometry_snapshot.json"
        if not geometry_path.exists():
            logger.warning("Missing geometry_snapshot.json in %s, skipping frame.", frame_dir)
            continue

        with open(geometry_path) as fh:
            geometry_snapshot = json.load(fh)

        frame = int(frame_dir.name.split("_")[-1])
        timestamp = _read_timestamp(frame_dir)
        rsu_positions, vehicle_positions = _extract_positions(geometry_snapshot)
        if not rsu_positions or not vehicle_positions:
            continue

        scene = adapter.build_frame_scene(map_name, geometry_snapshot)
        results = simulator.simulate_frame(scene, rsu_positions, vehicle_positions, frame, timestamp)

        out_frame_dir = out_scene_dir / frame_dir.name
        out_frame_dir.mkdir(exist_ok=True)
        for result in results:
            _write_result(out_frame_dir, result)
        processed += 1

    logger.info("Sionna RT: processed %d frames for scene %s", processed, scene_id)
    return processed


def _read_timestamp(frame_dir: Path) -> float:
    vehicle_states_path = frame_dir / "vehicle_states.json"
    if not vehicle_states_path.exists():
        return 0.0
    with open(vehicle_states_path) as fh:
        states = json.load(fh)
    return float(states[0]["timestamp"]) if states else 0.0


def _write_result(out_frame_dir: Path, result: ChannelSimulationResult) -> None:
    prefix = out_frame_dir / f"link_{result.link_id}"
    np.savez_compressed(f"{prefix}.npz", csi=result.csi)
    metadata = {
        "link_id": result.link_id, "vehicle_id": result.vehicle_id, "frame": result.frame,
        "wireless_timestamp": result.wireless_timestamp, "snr_db": result.snr_db,
        "rssi_dbm": result.rssi_dbm, "path_loss_db": result.path_loss_db,
        "delay_spread_s": result.delay_spread_s, "propagation_delay_s": result.propagation_delay_s,
        "beam_index": result.beam_index, "los": result.los,
        "num_multipath_components": result.num_multipath_components,
    }
    with open(f"{prefix}.json", "w") as fh:
        json.dump(metadata, fh, indent=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Sionna RT channel simulation over a CARLA scene export.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--map-name", required=True, help="CARLA map name, e.g. Town10HD")
    parser.add_argument("--static-scene-dir", type=Path, default=Path("simulation/sionna_configs/static_scenes"))
    parser.add_argument("--output", type=Path, default=Path("datasets/raw/sionna"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    process_scene(args.scene_dir, args.map_name, args.static_scene_dir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
