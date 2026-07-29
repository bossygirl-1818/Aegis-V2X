"""Orchestrates the annotate+synchronize+write stage of the pipeline:

  datasets/raw/carla/<scene_id>/frame_*/{vehicle_states,*_gps,*_imu,*_vehicle_speed}.json
  datasets/raw/sionna/<scene_id>/frame_*/link_<rsu>_<vehicle>.json (+ .npz CSI)
        |
        v  (this module)
  datasets/metadata/<scene_id>_Metadata.json
  datasets/annotations/<scene_id>_Annotations.json

Deliberately does NOT import carla or sionna — it only reads the JSON/NPZ
artifacts those tools already wrote (via simulation.carla_scenarios.run_scenario
and simulation.sionna_configs.run_channel_sim), so this stage is runnable and
testable without a GPU workstation.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulation.annotation.future_channel_labeler import ChannelSample, FutureChannelLabeler
from simulation.annotation.trust_criticality_labeler import TrustCriticalityLabeler
from simulation.synchronization.buffer import TimestampedBuffer
from simulation.synchronization.synchronizer import StreamSynchronizer

from .schema import DatasetSample, TrafficDensity, WeatherCondition, build_sample_id
from .writer import DatasetWriter

logger = logging.getLogger("aegis_v2x.simulation.dataset_pipeline.generator")


class DatasetGenerator:
    """Turns one scene's raw CARLA + Sionna exports into DatasetSample records."""

    def __init__(self, raw_carla_dir: Path, raw_sionna_dir: Path, repo_root: Path = Path("."),
                 sync_tolerance_ms: float = 10.0, prediction_horizon_frames: int = 3):
        self._carla_dir = Path(raw_carla_dir)
        self._sionna_dir = Path(raw_sionna_dir)
        self._writer = DatasetWriter(repo_root)
        self._sync_tolerance_ms = sync_tolerance_ms
        self._horizon_frames = prediction_horizon_frames
        self._trust_criticality_labeler = TrustCriticalityLabeler.from_config_file() \
            if (Path(repo_root) / "configs" / "model.yaml").exists() else TrustCriticalityLabeler()
        self._future_labeler = FutureChannelLabeler()

    def process_scene(self, scene_id: str, weather: WeatherCondition, traffic_density: TrafficDensity,
                       scenario_metadata: Optional[dict] = None) -> List[DatasetSample]:
        carla_scene_dir = self._carla_dir / scene_id
        sionna_scene_dir = self._sionna_dir / scene_id
        if not carla_scene_dir.exists() or not sionna_scene_dir.exists():
            raise FileNotFoundError(f"Missing raw data for scene '{scene_id}' under {carla_scene_dir} / {sionna_scene_dir}")

        per_vehicle_channel = self._load_channel_sequences(sionna_scene_dir)
        synced_streams = self._synchronize(scene_id, carla_scene_dir, sionna_scene_dir)

        samples: List[DatasetSample] = []
        for vehicle_id, channel_seq in per_vehicle_channel.items():
            future_labels = self._future_labeler.label_sequence(channel_seq, self._horizon_frames)
            samples.extend(self._build_samples(scene_id, vehicle_id, channel_seq, future_labels,
                                                synced_streams.get(vehicle_id, {}), weather, traffic_density))

        self._writer.write_scene_annotations(scene_id, samples)
        self._write_metadata(scene_id, sionna_scene_dir, per_vehicle_channel, weather, traffic_density,
                              scenario_metadata or {})
        logger.info("Scene %s: generated %d annotated samples across %d vehicles",
                    scene_id, len(samples), len(per_vehicle_channel))
        return samples

    def _write_metadata(self, scene_id: str, sionna_scene_dir: Path,
                         per_vehicle_channel: Dict[int, List[ChannelSample]],
                         weather: WeatherCondition, traffic_density: TrafficDensity, scenario_metadata: dict) -> None:
        rsu_ids = set()
        for frame_dir in sionna_scene_dir.glob("frame_*"):
            for link_json in frame_dir.glob("link_*.json"):
                rsu_ids.add(link_json.stem.split("_")[1])

        metadata = {
            "scene_id": scene_id,
            "scenario_id": scenario_metadata.get("scenario_id", scene_id.rsplit("_Scene", 1)[0]),
            "town_type": scenario_metadata.get("town_type"), "map": scenario_metadata.get("map"),
            "weather": weather.value, "traffic_density": traffic_density.value,
            "num_vehicles": len(per_vehicle_channel), "num_roadside_units": len(rsu_ids),
            "duration_sec": scenario_metadata.get("duration_sec"),
            "frame_rate_hz": scenario_metadata.get("frame_rate_hz", 10),
            "random_seed": scenario_metadata.get("random_seed"),
            "software_versions": self._collect_software_versions(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generated_by": "Haridharani",
        }
        self._writer.write_scene_metadata(scene_id, metadata)

    @staticmethod
    def _collect_software_versions() -> dict:
        def _safe_version(pkg: str) -> str:
            try:
                return pkg_version(pkg)
            except PackageNotFoundError:
                return "not_installed"
        return {"python": platform.python_version(), "carla": _safe_version("carla"),
                "sionna": _safe_version("sionna")}

    def _load_channel_sequences(self, sionna_scene_dir: Path) -> Dict[int, List[ChannelSample]]:
        per_vehicle: Dict[int, List[ChannelSample]] = {}
        for frame_dir in sorted(sionna_scene_dir.glob("frame_*")):
            for link_json in sorted(frame_dir.glob("link_*.json")):
                with open(link_json) as fh:
                    meta = json.load(fh)
                csi_magnitude = self._read_csi_magnitude(link_json.with_suffix(".npz"))
                sample = ChannelSample(timestamp=meta["wireless_timestamp"], csi_magnitude=csi_magnitude,
                                        snr_db=meta["snr_db"], path_loss_db=meta["path_loss_db"],
                                        beam_index=meta["beam_index"])
                per_vehicle.setdefault(meta["vehicle_id"], []).append(sample)

        for vehicle_id in per_vehicle:
            per_vehicle[vehicle_id].sort(key=lambda s: s.timestamp)
        return per_vehicle

    def _read_csi_magnitude(self, csi_npz_path: Path) -> float:
        if not csi_npz_path.exists():
            return 0.0
        with np.load(csi_npz_path) as data:
            csi = data["csi"]
        return float(np.mean(np.abs(csi))) if csi.size else 0.0

    def _synchronize(self, scene_id: str, carla_scene_dir: Path, sionna_scene_dir: Path) -> Dict[int, Dict[int, dict]]:
        result: Dict[int, Dict[int, dict]] = {}
        for vehicle_id in self._discover_vehicle_ids(carla_scene_dir):
            synchronizer = StreamSynchronizer(tolerance_ms=self._sync_tolerance_ms)
            csi_buffer = self._csi_anchor_buffer(sionna_scene_dir, vehicle_id)
            if len(csi_buffer) == 0:
                continue
            synchronizer.register_stream("csi", csi_buffer, is_anchor=True)
            synchronizer.register_stream("gps", self._sensor_buffer(carla_scene_dir, vehicle_id, "gps"))
            synchronizer.register_stream("imu", self._sensor_buffer(carla_scene_dir, vehicle_id, "imu"))
            synchronizer.register_stream("vehicle_speed", self._sensor_buffer(carla_scene_dir, vehicle_id, "vehicle_speed"))

            sync_results, violations = StreamSynchronizer.split_results(synchronizer.synchronize(scene_id, vehicle_id))
            if violations:
                logger.debug("Scene %s vehicle %d: %d frames failed sync tolerance", scene_id, vehicle_id, len(violations))
            result[vehicle_id] = {r.frame_id: {**r.streams, "max_offset_ms": r.max_offset_ms} for r in sync_results}
        return result

    def _discover_vehicle_ids(self, carla_scene_dir: Path) -> List[int]:
        ids = set()
        for frame_dir in carla_scene_dir.glob("frame_*"):
            for f in frame_dir.glob("vehicle*_gps.json"):
                ids.add(int(f.name.split("_")[0].replace("vehicle", "")))
        return sorted(ids)

    def _csi_anchor_buffer(self, sionna_scene_dir: Path, vehicle_id: int) -> TimestampedBuffer:
        buffer: TimestampedBuffer = TimestampedBuffer()
        for frame_dir in sorted(sionna_scene_dir.glob("frame_*")):
            for link_json in frame_dir.glob(f"link_*_{vehicle_id}.json"):
                with open(link_json) as fh:
                    meta = json.load(fh)
                buffer.add(meta["wireless_timestamp"], meta)
        return buffer

    def _sensor_buffer(self, carla_scene_dir: Path, vehicle_id: int, sensor_type: str) -> TimestampedBuffer:
        buffer: TimestampedBuffer = TimestampedBuffer()
        for frame_dir in sorted(carla_scene_dir.glob("frame_*")):
            f = frame_dir / f"vehicle{vehicle_id}_{sensor_type}.json"
            if not f.exists():
                continue
            with open(f) as fh:
                payload = json.load(fh)
            buffer.add(payload["timestamp"], payload)
        return buffer

    def _build_samples(self, scene_id: str, vehicle_id: int, channel_seq: List[ChannelSample], future_labels,
                        synced_frames: Dict[int, dict], weather: WeatherCondition,
                        traffic_density: TrafficDensity) -> List[DatasetSample]:
        samples: List[DatasetSample] = []
        max_sync_age_s = 1.0

        for frame_id, (channel, future) in enumerate(zip(channel_seq, future_labels)):
            synced = synced_frames.get(frame_id)
            if synced is None:
                continue

            gps_data = synced.get("gps", {})
            imu_data = synced.get("imu", {})
            speed_data = synced.get("vehicle_speed", {})
            sync_age_normalized = min(synced["max_offset_ms"] / 1000.0 / max_sync_age_s, 1.0)
            comm_quality_normalized = float(np.clip((channel.snr_db + 10.0) / 40.0, 0.0, 1.0))

            tc_label = self._trust_criticality_labeler.label(
                prediction_error=future.prediction_error, prediction_uncertainty=future.prediction_uncertainty,
                sync_age_normalized=sync_age_normalized, comm_quality_normalized=comm_quality_normalized,
                relative_speed_normalized=float(np.clip(speed_data.get("speed_mps", 0.0) / 40.0, 0.0, 1.0)),
                blockage_prob=float(np.clip(1.0 - comm_quality_normalized, 0.0, 1.0)),
                channel_degradation_normalized=float(np.clip(channel.path_loss_db / 160.0, 0.0, 1.0)),
                traffic_density_normalized={"sparse": 0.25, "dense": 0.85}[traffic_density.value],
            )

            sample_id = build_sample_id(scene_id, vehicle_id, frame_id)
            samples.append(DatasetSample(
                sample=sample_id, scene_id=scene_id, frame_id=frame_id, vehicle_id=vehicle_id,
                timestamp=channel.timestamp, lidar=f"synchronized/{scene_id}/{sample_id}.npz",
                gps=(gps_data.get("lat", 0.0), gps_data.get("lon", 0.0), gps_data.get("alt", 0.0)),
                imu=(tuple(imu_data.get("accel", (0.0, 0.0, 0.0))), tuple(imu_data.get("gyro", (0.0, 0.0, 0.0)))),
                speed=speed_data.get("speed_mps", 0.0), csi=f"synchronized/{scene_id}/{sample_id}.npz",
                snr=channel.snr_db, rssi=0.0, path_loss=max(channel.path_loss_db, 0.0), beam_index=channel.beam_index,
                traffic_density=traffic_density, weather=weather,
                ground_truth_future_csi=future.future_csi_magnitude, ground_truth_future_beam=future.future_beam_index,
                ground_truth_trust=tc_label.trust, ground_truth_criticality=tc_label.criticality,
                sync_offset_ms=synced["max_offset_ms"],
            ))
        return samples
