"""Timestamp-indexed buffer supporting efficient nearest-neighbor lookup.

Used to hold one sensor/channel stream's readings and answer "what reading
is closest to time t, and how far away is it?" in O(log n).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class NearestMatch(Generic[T]):
    data: T
    timestamp: float
    delta_seconds: float


class TimestampedBuffer(Generic[T]):
    """Keeps (timestamp, data) pairs sorted by timestamp for nearest lookups."""

    def __init__(self) -> None:
        self._timestamps: List[float] = []
        self._data: List[T] = []

    def add(self, timestamp: float, data: T) -> None:
        idx = bisect.bisect_left(self._timestamps, timestamp)
        self._timestamps.insert(idx, timestamp)
        self._data.insert(idx, data)

    def extend(self, readings: List[Tuple[float, T]]) -> None:
        for ts, data in readings:
            self.add(ts, data)

    def __len__(self) -> int:
        return len(self._timestamps)

    def nearest(self, query_timestamp: float) -> Optional[NearestMatch[T]]:
        if not self._timestamps:
            return None

        idx = bisect.bisect_left(self._timestamps, query_timestamp)
        candidates = []
        if idx < len(self._timestamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        best_idx = min(candidates, key=lambda i: abs(self._timestamps[i] - query_timestamp))
        delta = self._timestamps[best_idx] - query_timestamp
        return NearestMatch(data=self._data[best_idx], timestamp=self._timestamps[best_idx], delta_seconds=delta)

    def timestamps(self) -> List[float]:
        return list(self._timestamps)
