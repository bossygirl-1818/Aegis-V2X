"""Dataset preprocessing, per 09_Dataset_Design_and_Annotation_Guide Chapter 17:
normalization, coordinate transformation, missing-value removal, outlier
detection, feature scaling, timestamp alignment, and dataset indexing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import DatasetSample

_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (self.std if self.std > 1e-9 else 1.0)


class DatasetPreprocessor:
    def __init__(self, outlier_zscore_threshold: float = 4.0):
        self._outlier_threshold = outlier_zscore_threshold

    def drop_invalid(self, samples: List[DatasetSample], invalid_sample_ids: Sequence[str]) -> List[DatasetSample]:
        invalid = set(invalid_sample_ids)
        return [s for s in samples if s.sample not in invalid]

    def gps_to_local_enu(self, lat: float, lon: float, alt: float,
                          origin_lat: float, origin_lon: float, origin_alt: float) -> Tuple[float, float, float]:
        lat_rad = math.radians(lat)
        origin_lat_rad = math.radians(origin_lat)
        north = math.radians(lat - origin_lat) * _EARTH_RADIUS_M
        east = math.radians(lon - origin_lon) * _EARTH_RADIUS_M * math.cos(origin_lat_rad)
        up = alt - origin_alt
        return east, north, up

    def batch_gps_to_local_enu(self, samples: List[DatasetSample]) -> np.ndarray:
        if not samples:
            return np.empty((0, 3), dtype=np.float64)
        origin_lat, origin_lon, origin_alt = samples[0].gps
        return np.array([self.gps_to_local_enu(*s.gps, origin_lat, origin_lon, origin_alt) for s in samples])

    def detect_outliers_zscore(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        mean, std = float(np.mean(values)), float(np.std(values))
        if std < 1e-9:
            return np.zeros_like(values, dtype=bool)
        return np.abs((values - mean) / std) > self._outlier_threshold

    def fit_normalization(self, values: np.ndarray) -> NormalizationStats:
        values = np.asarray(values, dtype=np.float64)
        return NormalizationStats(mean=float(np.mean(values)), std=float(np.std(values)))

    def min_max_scale(self, values: np.ndarray, feature_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo < 1e-9:
            return np.full_like(values, feature_range[0])
        return (values - lo) / (hi - lo) * (feature_range[1] - feature_range[0]) + feature_range[0]

    def align_to_grid(self, timestamps: np.ndarray, frame_rate_hz: float) -> np.ndarray:
        dt = 1.0 / frame_rate_hz
        return np.round(np.asarray(timestamps, dtype=np.float64) / dt) * dt

    def build_index(self, samples: List[DatasetSample]) -> pd.DataFrame:
        rows = [{
            "sample": s.sample, "scene_id": s.scene_id, "vehicle_id": s.vehicle_id, "frame_id": s.frame_id,
            "timestamp": s.timestamp, "ground_truth_trust": s.ground_truth_trust,
            "ground_truth_criticality": s.ground_truth_criticality, "weather": s.weather.value,
            "traffic_density": s.traffic_density.value,
        } for s in samples]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["scene_id", "vehicle_id", "frame_id"]).reset_index(drop=True)
        return df
