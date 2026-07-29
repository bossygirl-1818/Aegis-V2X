"""Scene-disjoint train/validation/test splitting, per
09_Dataset_Design_and_Annotation_Guide Chapter 16. Default ratios come from
configs/dataset.yaml `dataset_targets` (train_split=0.70, validation_split=0.15,
test_split=0.15) — frozen, Phase 1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

from .schema import DatasetSample


@dataclass
class SplitAssignment:
    train_scenes: List[str]
    validation_scenes: List[str]
    test_scenes: List[str]

    def split_of(self, scene_id: str) -> str:
        if scene_id in self.train_scenes:
            return "train"
        if scene_id in self.validation_scenes:
            return "validation"
        if scene_id in self.test_scenes:
            return "test"
        raise KeyError(f"scene_id '{scene_id}' was not assigned to any split")


def _load_default_ratios(dataset_yaml_path: Path = Path("configs/dataset.yaml")) -> tuple[float, float, float]:
    if not dataset_yaml_path.exists():
        return 0.70, 0.15, 0.15
    with open(dataset_yaml_path) as fh:
        cfg = yaml.safe_load(fh)
    targets = cfg.get("dataset_targets", {})
    return (targets.get("train_split", 0.70), targets.get("validation_split", 0.15), targets.get("test_split", 0.15))


class DatasetSplitter:
    def __init__(self, train_ratio: float = None, val_ratio: float = None, test_ratio: float = None,
                 seed: int = 42):
        default_train, default_val, default_test = _load_default_ratios()
        train_ratio = default_train if train_ratio is None else train_ratio
        val_ratio = default_val if val_ratio is None else val_ratio
        test_ratio = default_test if test_ratio is None else test_ratio

        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self._train_ratio, self._val_ratio, self._test_ratio = train_ratio, val_ratio, test_ratio
        self._seed = seed

    def assign(self, scene_ids: Sequence[str]) -> SplitAssignment:
        unique_scenes = sorted(set(scene_ids))
        rng = random.Random(self._seed)
        shuffled = unique_scenes[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = round(n * self._train_ratio)
        n_val = round(n * self._val_ratio)
        n_test = n - n_train - n_val

        train, val, test = shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]
        assert len(train) + len(val) + len(test) == n
        return SplitAssignment(train_scenes=train, validation_scenes=val, test_scenes=test)

    def split_samples(self, samples: List[DatasetSample]) -> Dict[str, List[DatasetSample]]:
        assignment = self.assign([s.scene_id for s in samples])
        result: Dict[str, List[DatasetSample]] = {"train": [], "validation": [], "test": []}
        for sample in samples:
            result[assignment.split_of(sample.scene_id)].append(sample)
        return result

    @staticmethod
    def verify_no_leakage(splits: Dict[str, List[DatasetSample]]) -> bool:
        scene_sets = {name: {s.scene_id for s in group} for name, group in splits.items()}
        names = list(scene_sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if scene_sets[names[i]] & scene_sets[names[j]]:
                    return False
        return True
