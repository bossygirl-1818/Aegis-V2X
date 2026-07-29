"""Dataset quality validation, per 09_Dataset_Design_and_Annotation_Guide
Chapter 15: missing values, sensor synchronization, timestamp consistency,
CSI integrity, LiDAR completeness, GPS validity, duplicate frames, corrupted
files. Produces a JSON validation report.

Operates on in-memory DatasetSample lists (plus optional array payloads) so
it is fully unit-testable without touching real CARLA/Sionna output.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import DatasetSample


@dataclass
class ValidationIssue:
    sample: str
    category: str
    message: str
    severity: str = "error"


@dataclass
class ValidationReport:
    total_samples: int
    issue_count: int
    issues_by_category: Dict[str, int]
    passed: bool
    issues: List[ValidationIssue]

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples, "issue_count": self.issue_count,
            "issues_by_category": self.issues_by_category, "passed": self.passed,
            "issues": [asdict(i) for i in self.issues],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


class DatasetValidator:
    def __init__(self, sync_tolerance_ms: float = 10.0, min_lidar_points: int = 1000,
                 gps_lat_range: Tuple[float, float] = (-90.0, 90.0),
                 gps_lon_range: Tuple[float, float] = (-180.0, 180.0)):
        self._sync_tolerance_ms = sync_tolerance_ms
        self._min_lidar_points = min_lidar_points
        self._gps_lat_range = gps_lat_range
        self._gps_lon_range = gps_lon_range

    def validate_samples(self, samples: List[DatasetSample]) -> ValidationReport:
        issues: List[ValidationIssue] = []
        seen_keys: Counter = Counter()

        for sample in samples:
            issues.extend(self._check_missing_and_nan(sample))
            issues.extend(self._check_sync_tolerance(sample))
            issues.extend(self._check_gps_validity(sample))
            seen_keys[(sample.scene_id, sample.vehicle_id, sample.frame_id)] += 1

        issues.extend(self._check_duplicates(seen_keys))
        issues.extend(self._check_timestamp_consistency(samples))
        return self._build_report(len(samples), issues)

    def validate_array_payload(self, sample_id: str, lidar_points: Optional[np.ndarray],
                                csi: Optional[np.ndarray]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        if lidar_points is None:
            issues.append(ValidationIssue(sample_id, "corrupted_file", "LiDAR array missing/unreadable"))
        elif lidar_points.shape[0] < self._min_lidar_points:
            issues.append(ValidationIssue(sample_id, "lidar_completeness",
                          f"LiDAR point count {lidar_points.shape[0]} below minimum {self._min_lidar_points}",
                          severity="warning"))
        elif not np.all(np.isfinite(lidar_points)):
            issues.append(ValidationIssue(sample_id, "corrupted_file", "LiDAR array contains NaN/Inf"))

        if csi is None:
            issues.append(ValidationIssue(sample_id, "corrupted_file", "CSI array missing/unreadable"))
        elif csi.size == 0:
            issues.append(ValidationIssue(sample_id, "csi_integrity", "CSI array is empty"))
        elif not np.all(np.isfinite(csi)):
            issues.append(ValidationIssue(sample_id, "csi_integrity", "CSI array contains NaN/Inf"))
        elif np.allclose(csi, 0.0):
            issues.append(ValidationIssue(sample_id, "csi_integrity", "CSI array is all-zero (likely blocked link)",
                                           severity="warning"))
        return issues

    def _check_missing_and_nan(self, sample: DatasetSample) -> List[ValidationIssue]:
        issues = []
        numeric_fields = {
            "snr": sample.snr, "rssi": sample.rssi, "path_loss": sample.path_loss,
            "ground_truth_trust": sample.ground_truth_trust, "ground_truth_criticality": sample.ground_truth_criticality,
        }
        for field_name, value in numeric_fields.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                issues.append(ValidationIssue(sample.sample, "missing_value", f"{field_name} is missing/NaN"))
        return issues

    def _check_sync_tolerance(self, sample: DatasetSample) -> List[ValidationIssue]:
        if sample.sync_offset_ms > self._sync_tolerance_ms:
            return [ValidationIssue(sample.sample, "sync_tolerance",
                    f"sync_offset_ms={sample.sync_offset_ms:.2f} exceeds tolerance {self._sync_tolerance_ms}ms")]
        return []

    def _check_gps_validity(self, sample: DatasetSample) -> List[ValidationIssue]:
        issues = []
        lat, lon, _alt = sample.gps
        lat_lo, lat_hi = self._gps_lat_range
        lon_lo, lon_hi = self._gps_lon_range
        if not (lat_lo <= lat <= lat_hi):
            issues.append(ValidationIssue(sample.sample, "gps_validity", f"gps lat={lat} out of range"))
        if not (lon_lo <= lon <= lon_hi):
            issues.append(ValidationIssue(sample.sample, "gps_validity", f"gps lon={lon} out of range"))
        return issues

    def _check_duplicates(self, seen_keys: Counter) -> List[ValidationIssue]:
        issues = []
        for (scene_id, vehicle_id, frame_id), count in seen_keys.items():
            if count > 1:
                issues.append(ValidationIssue(f"{scene_id}_Vehicle{vehicle_id:02d}_Frame{frame_id:06d}",
                                               "duplicate_frame", f"appears {count} times"))
        return issues

    def _check_timestamp_consistency(self, samples: List[DatasetSample]) -> List[ValidationIssue]:
        issues = []
        by_vehicle: Dict[Tuple[str, int], List[DatasetSample]] = {}
        for sample in samples:
            by_vehicle.setdefault((sample.scene_id, sample.vehicle_id), []).append(sample)

        for (scene_id, vehicle_id), group in by_vehicle.items():
            ordered = sorted(group, key=lambda s: s.frame_id)
            timestamps = [s.timestamp for s in ordered]
            if any(t2 < t1 for t1, t2 in zip(timestamps, timestamps[1:])):
                issues.append(ValidationIssue(f"{scene_id}_Vehicle{vehicle_id:02d}", "timestamp_consistency",
                                               "timestamps are non-monotonic across frames"))
        return issues

    def _build_report(self, total: int, issues: List[ValidationIssue]) -> ValidationReport:
        errors = [i for i in issues if i.severity == "error"]
        counts = Counter(i.category for i in issues)
        return ValidationReport(total_samples=total, issue_count=len(issues), issues_by_category=dict(counts),
                                 passed=(len(errors) == 0), issues=issues)
