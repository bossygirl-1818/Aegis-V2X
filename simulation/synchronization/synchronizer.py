"""Aligns multiple named timestamp streams onto an anchor stream's timeline
and enforces configs/simulation.yaml's `synchronization.tolerance_ms` (10ms).

CSI (from Sionna RT, 10Hz) is the anchor stream. For each CSI timestamp, the
nearest reading from every other registered stream is located; if all are
within tolerance, a `SyncResult` is emitted carrying exactly the fields
configs/simulation.yaml `synchronization.required_fields` lists: scene_id,
frame_id, vehicle_id, simulation_timestamp, wireless_timestamp,
synchronization_timestamp. Otherwise a `SyncViolation` is emitted — flagged,
not silently dropped, per Chapter 9 of the Dataset Design Guide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .buffer import TimestampedBuffer

logger = logging.getLogger("aegis_v2x.simulation.synchronization.synchronizer")


@dataclass
class SyncResult:
    scene_id: str
    vehicle_id: int
    frame_id: int
    simulation_timestamp: float
    wireless_timestamp: float
    synchronization_timestamp: float
    streams: Dict[str, Any]
    max_offset_ms: float


@dataclass
class SyncViolation:
    scene_id: str
    vehicle_id: int
    frame_id: int
    wireless_timestamp: float
    offending_stream: str
    offset_ms: float
    tolerance_ms: float


class StreamSynchronizer:
    """Synchronizes N named streams to one anchor stream (CSI), per (scene, vehicle)."""

    def __init__(self, tolerance_ms: float = 10.0, simulation_timestamp_stream: str = "gps"):
        if tolerance_ms <= 0:
            raise ValueError("tolerance_ms must be positive")
        self._tolerance_ms = tolerance_ms
        self._simulation_timestamp_stream = simulation_timestamp_stream
        self._streams: Dict[str, TimestampedBuffer] = {}
        self._anchor_stream: Optional[str] = None

    def register_stream(self, name: str, buffer: TimestampedBuffer, is_anchor: bool = False) -> None:
        self._streams[name] = buffer
        if is_anchor:
            self._anchor_stream = name

    @property
    def tolerance_ms(self) -> float:
        return self._tolerance_ms

    def synchronize(self, scene_id: str, vehicle_id: int) -> List[Any]:
        if self._anchor_stream is None:
            raise RuntimeError("No anchor stream registered. Call register_stream(..., is_anchor=True).")

        anchor_buffer = self._streams[self._anchor_stream]
        other_streams = {name: buf for name, buf in self._streams.items() if name != self._anchor_stream}

        outputs: List[Any] = []
        for frame_id, wireless_timestamp in enumerate(anchor_buffer.timestamps()):
            matched: Dict[str, Any] = {self._anchor_stream: anchor_buffer.nearest(wireless_timestamp).data}
            max_offset_ms = 0.0
            violation: Optional[SyncViolation] = None

            for name, buf in other_streams.items():
                match = buf.nearest(wireless_timestamp)
                if match is None:
                    violation = SyncViolation(scene_id, vehicle_id, frame_id, wireless_timestamp, name,
                                               float("inf"), self._tolerance_ms)
                    break

                offset_ms = abs(match.delta_seconds) * 1000.0
                max_offset_ms = max(max_offset_ms, offset_ms)
                if offset_ms > self._tolerance_ms:
                    violation = SyncViolation(scene_id, vehicle_id, frame_id, wireless_timestamp, name,
                                               offset_ms, self._tolerance_ms)
                    break
                matched[name] = match.data

            if violation is not None:
                outputs.append(violation)
                continue

            simulation_timestamp = self._resolve_simulation_timestamp(matched, wireless_timestamp)
            outputs.append(SyncResult(
                scene_id=scene_id, vehicle_id=vehicle_id, frame_id=frame_id,
                simulation_timestamp=simulation_timestamp, wireless_timestamp=wireless_timestamp,
                synchronization_timestamp=wireless_timestamp, streams=matched, max_offset_ms=max_offset_ms,
            ))
        return outputs

    def _resolve_simulation_timestamp(self, matched: Dict[str, Any], fallback: float) -> float:
        stream_data = matched.get(self._simulation_timestamp_stream)
        if isinstance(stream_data, dict) and "timestamp" in stream_data:
            return float(stream_data["timestamp"])
        return fallback

    @staticmethod
    def split_results(outputs: List[Any]):
        results = [o for o in outputs if isinstance(o, SyncResult)]
        violations = [o for o in outputs if isinstance(o, SyncViolation)]
        return results, violations
