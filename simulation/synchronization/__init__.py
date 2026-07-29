"""Multi-stream timestamp synchronization for Aegis-V2X Phase 2.

Aligns CARLA sensor streams with Sionna RT wireless streams to a common
per-vehicle sample timeline, enforcing configs/simulation.yaml's
`synchronization.tolerance_ms` (frozen, Phase 1: 10ms).

Pure Python/NumPy — no CARLA or Sionna dependency, fully testable (see
tests/unit/test_synchronizer.py).
"""

from .buffer import TimestampedBuffer
from .synchronizer import StreamSynchronizer, SyncResult, SyncViolation

__all__ = ["TimestampedBuffer", "StreamSynchronizer", "SyncResult", "SyncViolation"]
