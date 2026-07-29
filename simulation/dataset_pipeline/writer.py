"""Writes DatasetSample records and their heavy array payloads to disk,
using the paths frozen in configs/dataset.yaml `paths:` (all relative to the
repo root — datasets/ lives at the top level, not nested under simulation/).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from .schema import DatasetSample, build_sample_id

logger = logging.getLogger("aegis_v2x.simulation.dataset_pipeline.writer")

_DEFAULT_PATHS = {
    "raw_carla": "datasets/raw/carla", "raw_sionna": "datasets/raw/sionna",
    "synchronized": "datasets/synchronized", "processed": "datasets/processed",
    "train": "datasets/train", "validation": "datasets/validation", "test": "datasets/test",
    "annotations": "datasets/annotations", "metadata": "datasets/metadata", "statistics": "datasets/statistics",
}


def load_dataset_paths(dataset_yaml_path: Path = Path("configs/dataset.yaml")) -> Dict[str, str]:
    if not dataset_yaml_path.exists():
        return dict(_DEFAULT_PATHS)
    with open(dataset_yaml_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("paths", _DEFAULT_PATHS)


class DatasetWriter:
    def __init__(self, repo_root: Path = Path("."), dataset_yaml_path: Path = Path("configs/dataset.yaml")):
        paths = load_dataset_paths(dataset_yaml_path)
        self.synchronized_dir = Path(repo_root) / paths["synchronized"]
        self.metadata_dir = Path(repo_root) / paths["metadata"]
        self.annotations_dir = Path(repo_root) / paths["annotations"]
        for d in (self.synchronized_dir, self.metadata_dir, self.annotations_dir):
            d.mkdir(parents=True, exist_ok=True)

    def sample_filename(self, sample: DatasetSample) -> str:
        return f"{sample.sample}.npz"

    def write_sample_payload(self, sample: DatasetSample, lidar_points: np.ndarray, csi: np.ndarray) -> None:
        scene_dir = self.synchronized_dir / sample.scene_id
        scene_dir.mkdir(exist_ok=True)
        out_path = scene_dir / self.sample_filename(sample)
        np.savez_compressed(
            out_path, lidar=lidar_points.astype(np.float32), csi=csi.astype(np.complex64),
            sample_metadata=np.frombuffer(sample.model_dump_json().encode("utf-8"), dtype=np.uint8),
        )
        logger.debug("Wrote sample payload: %s", out_path)

    def write_scene_metadata(self, scene_id: str, metadata: Dict) -> Path:
        out_path = self.metadata_dir / f"{scene_id}_Metadata.json"
        with open(out_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        return out_path

    def write_scene_annotations(self, scene_id: str, samples: List[DatasetSample]) -> Path:
        out_path = self.annotations_dir / f"{scene_id}_Annotations.json"
        payload = [json.loads(s.model_dump_json()) for s in samples]
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return out_path

    def load_scene_annotations(self, scene_id: str) -> List[DatasetSample]:
        path = self.annotations_dir / f"{scene_id}_Annotations.json"
        with open(path) as fh:
            payload = json.load(fh)
        return [DatasetSample(**entry) for entry in payload]
