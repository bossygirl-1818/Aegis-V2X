"""CRUD operations for Frame, including cross-modal synchronization checks.

`create_frame` implements the Dataset Design & Annotation Guide Ch. 9
synchronization-tolerance rule: a frame is considered synchronized when
its simulation-clock and wireless-clock timestamps differ by no more than
10ms. Out-of-tolerance frames are still stored (not rejected) but flagged
via `is_sync_valid=False` and logged as a warning, so the dashboard's
sync-health panel can surface them rather than silently dropping data.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.core.metrics import ingested_frames_counter
from app.models.frame import Frame
from app.models.scene import Vehicle
from app.schemas.frame import FrameCreate

logger = logging.getLogger(__name__)

#: Maximum allowed |simulation_timestamp - wireless_timestamp| for a frame
#: to be considered synchronized, per Ch. 9 of the Dataset Design Guide.
SYNC_TOLERANCE_MS = 10.0


def create_frame(db: Session, payload: FrameCreate) -> Frame:
    """Insert a new frame, computing sync offset/validity, and return it.

    Records the `aegis_ingested_frames_total{source=...}` metric here (not
    only in the API handler) so any code path that calls this function —
    including the synthetic data generator — contributes to the metric.
    """
    data = payload.model_dump()
    sim_ts = data["simulation_timestamp"]
    wireless_ts = data["wireless_timestamp"]
    sync_offset_ms = abs(sim_ts - wireless_ts) * 1000.0
    is_sync_valid = sync_offset_ms <= SYNC_TOLERANCE_MS

    if not is_sync_valid:
        logger.warning(
            "Frame %s (scene=%s, vehicle=%s) out of sync tolerance: offset=%.3fms > %.1fms",
            data["frame_index"],
            data["scene_id"],
            data["vehicle_id"],
            sync_offset_ms,
            SYNC_TOLERANCE_MS,
        )

    frame = Frame(
        **data,
        sync_timestamp=(sim_ts + wireless_ts) / 2.0,
        sync_offset_ms=sync_offset_ms,
        is_sync_valid=is_sync_valid,
    )
    db.add(frame)
    db.commit()
    db.refresh(frame)

    ingested_frames_counter.labels(source=frame.source).inc()
    return frame


def get_frame(db: Session, frame_id: uuid.UUID) -> Frame | None:
    """Fetch a single frame by id, or None if it doesn't exist."""
    return db.get(Frame, frame_id)


def list_frames(
    db: Session,
    scene_id: uuid.UUID | None = None,
    vehicle_id: uuid.UUID | None = None,
    skip: int = 0,
    limit: int = 100,
) -> list[Frame]:
    """List frames, optionally filtered by scene and/or vehicle."""
    stmt = select(Frame)
    if scene_id is not None:
        stmt = stmt.where(Frame.scene_id == scene_id)
    if vehicle_id is not None:
        stmt = stmt.where(Frame.vehicle_id == vehicle_id)
    stmt = stmt.order_by(Frame.frame_index).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def get_unsynchronized_count(db: Session, scene_id: uuid.UUID | None = None) -> tuple[int, int]:
    """Return (total_frames, unsynchronized_frames) for sync-health stats.

    NOTE: this is registered as a literal-path route
    (`/frames/stats/unsynchronized-count`) BEFORE the parameterized
    `/frames/{frame_id}` route in the router — Starlette matches routes in
    declaration order, and reversing that order was a real bug found
    during Phase 3 verification (the stats path would have been shadowed
    by the frame-detail path and matched as an invalid UUID instead).
    """
    total_stmt = select(func.count()).select_from(Frame)
    unsynced_stmt = select(func.count()).select_from(Frame).where(Frame.is_sync_valid.is_(False))
    if scene_id is not None:
        total_stmt = total_stmt.where(Frame.scene_id == scene_id)
        unsynced_stmt = unsynced_stmt.where(Frame.scene_id == scene_id)
    total = db.execute(total_stmt).scalar_one()
    unsynced = db.execute(unsynced_stmt).scalar_one()
    return total, unsynced


def get_latest_frame_per_vehicle(db: Session, scene_id: uuid.UUID) -> list[Frame]:
    """Return the most recent frame for every vehicle in a scene.

    Implements `GET /frames/scene/{scene_id}/latest-per-vehicle` using a
    single `DISTINCT ON` query (Postgres-specific) instead of N per-vehicle
    round trips — the query added during the Phase 3 dashboard rebuild to
    drive the Digital Twin radar map efficiently.
    """
    stmt = (
        select(Frame)
        .where(Frame.scene_id == scene_id)
        .order_by(Frame.vehicle_id, Frame.frame_index.desc())
        .distinct(Frame.vehicle_id)
    )
    return list(db.execute(stmt).scalars().all())


def get_latest_frame_per_vehicle_with_codes(
    db: Session, scene_id: uuid.UUID
) -> list[tuple[Frame, str]]:
    """Same as `get_latest_frame_per_vehicle` but also returns each vehicle's code.

    Used by the API layer to build `LatestVehicleFrame` responses without a
    second round trip per frame.
    """
    stmt = (
        select(Frame, Vehicle.vehicle_code)
        .join(Vehicle, Vehicle.id == Frame.vehicle_id)
        .where(Frame.scene_id == scene_id)
        .order_by(Frame.vehicle_id, Frame.frame_index.desc())
        .distinct(Frame.vehicle_id)
    )
    return [(row[0], row[1]) for row in db.execute(stmt).all()]


def count_distinct_vehicles_with_frames(db: Session, scene_id: uuid.UUID) -> int:
    """Count how many distinct vehicles in a scene have at least one frame."""
    stmt = select(func.count(distinct(Frame.vehicle_id))).where(Frame.scene_id == scene_id)
    return db.execute(stmt).scalar_one()
