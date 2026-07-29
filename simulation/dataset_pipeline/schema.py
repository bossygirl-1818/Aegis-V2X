"""Dataset sample schema — matches configs/dataset.yaml `schema_fields` and
`naming_convention` exactly (Phase 1, frozen, owner Vaishnavi):

    schema_fields: sample, scene_id, frame_id, vehicle_id, timestamp, lidar,
        gps, imu, speed, csi, snr, rssi, path_loss, beam_index,
        traffic_density, weather, ground_truth_future_csi,
        ground_truth_future_beam, ground_truth_trust, ground_truth_criticality

    naming_convention: "Scene{scene_id:02d}_Vehicle{vehicle_id:02d}_Frame{frame_id:06d}.npz"

Naming convention note (documented interpretation): `scene_id` in this
pipeline is a descriptive, scenario-prefixed string (e.g.
"urban_dense_clear_day_Scene00") rather than a bare integer, because scenes
must remain traceable to their scenario for scene-disjoint splitting
(Chapter 16) and statistics (Chapter 18). The `sample` field is therefore
built as `f"{scene_id}_Vehicle{vehicle_id:02d}_Frame{frame_id:06d}"`, which
preserves the frozen convention's zero-padding for vehicle_id (2 digits)
and frame_id (6 digits) without discarding scenario traceability. Flagged
for Vaishnavi to confirm.

Uses only the frozen `traffic_density` (sparse, dense) and `weather`
(clear_day, rain, fog, night) categories from configs/simulation.yaml — no
"medium" density or compound weather variants.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


class WeatherCondition(str, Enum):
    CLEAR_DAY = "clear_day"
    RAIN = "rain"
    FOG = "fog"
    NIGHT = "night"


class TrafficDensity(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"


def build_sample_id(scene_id: str, vehicle_id: int, frame_id: int) -> str:
    """Implements configs/dataset.yaml's naming_convention (see module docstring)."""
    return f"{scene_id}_Vehicle{vehicle_id:02d}_Frame{frame_id:06d}"


class DatasetSample(BaseModel):
    """One synchronized, annotated (Scene, Vehicle, Frame) sample.

    Field names are exactly configs/dataset.yaml's schema_fields list.
    Heavy array data (LiDAR point cloud, raw CSI) is referenced by relative
    file path (`lidar`, `csi` fields) rather than embedded here.

    `sync_offset_ms` is an additive QA field (not in the frozen schema_fields
    list) needed to record how close to the 10ms tolerance boundary a sample
    was — useful for the validation report, harmless as an extra field.
    """

    model_config = {"frozen": True}

    schema_version: str = SCHEMA_VERSION
    sample: str
    scene_id: str
    frame_id: int = Field(ge=0)
    vehicle_id: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)

    lidar: str
    gps: Tuple[float, float, float]      # (lat, lon, alt)
    imu: Tuple[Tuple[float, float, float], Tuple[float, float, float]]  # (accel_xyz, gyro_xyz)
    speed: float = Field(ge=0.0)

    csi: str
    snr: float
    rssi: float
    path_loss: float = Field(ge=0.0)
    beam_index: int = Field(ge=0)

    traffic_density: TrafficDensity
    weather: WeatherCondition

    ground_truth_future_csi: float = Field(ge=0.0)
    ground_truth_future_beam: int = Field(ge=0)
    ground_truth_trust: float = Field(ge=0.0, le=1.0)
    ground_truth_criticality: float = Field(ge=0.0, le=1.0)

    sync_offset_ms: float = Field(ge=0.0)

    @field_validator("beam_index", "ground_truth_future_beam")
    @classmethod
    def _beam_index_plausible(cls, value: int) -> int:
        if value >= 256:
            raise ValueError(f"beam_index {value} exceeds plausible codebook size")
        return value
