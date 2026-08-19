"""Synthetic data generation endpoint (development/demo only).

Stands in for the real CARLA + NVIDIA Sionna RT pipeline (Phase 2, not yet
built) so the backend, dashboard, and downstream phases can be developed
and demoed end-to-end against realistic data and the *exact* production
schema, without blocking on Phase 2. See `app/services/synthetic_data_service.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.synthetic import SyntheticSceneRequest, SyntheticSceneResponse
from app.services.synthetic_data_service import generate_synthetic_scene

router = APIRouter(prefix="/synthetic", tags=["synthetic"])


@router.post("/scenes", response_model=SyntheticSceneResponse, status_code=status.HTTP_201_CREATED)
def create_synthetic_scene(
    payload: SyntheticSceneRequest, db: Session = Depends(get_db)
) -> SyntheticSceneResponse:
    """Generate a full synthetic scene: vehicles, frames, trust/criticality/decisions."""
    return generate_synthetic_scene(db, payload)
