"""Dataset statistics, per 09_Dataset_Design_and_Annotation_Guide Chapter 18."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .schema import DatasetSample


@dataclass
class DistributionSummary:
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float

    @classmethod
    def from_values(cls, values: np.ndarray) -> "DistributionSummary":
        if len(values) == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(mean=float(np.mean(values)), std=float(np.std(values)), min=float(np.min(values)),
                   max=float(np.max(values)), p50=float(np.percentile(values, 50)), p90=float(np.percentile(values, 90)))


@dataclass
class DatasetStatistics:
    num_scenes: int
    num_vehicles: int
    num_frames: int
    frames_per_scene: DistributionSummary
    avg_snr_db: float
    weather_distribution: Dict[str, int]
    traffic_density_distribution: Dict[str, int]
    beam_index_distribution: Dict[int, int]
    trust_distribution: DistributionSummary
    criticality_distribution: DistributionSummary

    def to_dict(self) -> dict:
        return {
            "num_scenes": self.num_scenes, "num_vehicles": self.num_vehicles, "num_frames": self.num_frames,
            "frames_per_scene": self.frames_per_scene.__dict__, "avg_snr_db": self.avg_snr_db,
            "weather_distribution": self.weather_distribution,
            "traffic_density_distribution": self.traffic_density_distribution,
            "beam_index_distribution": {str(k): v for k, v in self.beam_index_distribution.items()},
            "trust_distribution": self.trust_distribution.__dict__,
            "criticality_distribution": self.criticality_distribution.__dict__,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def compute_statistics(samples: List[DatasetSample]) -> DatasetStatistics:
    if not samples:
        empty = DistributionSummary.from_values(np.array([]))
        return DatasetStatistics(0, 0, 0, empty, 0.0, {}, {}, {}, empty, empty)

    scene_ids = {s.scene_id for s in samples}
    vehicle_ids = {(s.scene_id, s.vehicle_id) for s in samples}
    frames_per_scene = DistributionSummary.from_values(
        np.array(list(Counter(s.scene_id for s in samples).values())))

    return DatasetStatistics(
        num_scenes=len(scene_ids), num_vehicles=len(vehicle_ids), num_frames=len(samples),
        frames_per_scene=frames_per_scene, avg_snr_db=float(np.mean([s.snr for s in samples])),
        weather_distribution=dict(Counter(s.weather.value for s in samples)),
        traffic_density_distribution=dict(Counter(s.traffic_density.value for s in samples)),
        beam_index_distribution=dict(Counter(s.beam_index for s in samples)),
        trust_distribution=DistributionSummary.from_values(np.array([s.ground_truth_trust for s in samples])),
        criticality_distribution=DistributionSummary.from_values(np.array([s.ground_truth_criticality for s in samples])),
    )
