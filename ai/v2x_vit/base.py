"""
Abstract contract for V2X-ViT cooperative multimodal feature fusion.

Role (02_System_Architecture, Layer 4): fuses spatial features across
connected vehicles into a unified cooperative perception representation,
feeding the GRU temporal predictor.

Implemented in: Phase 4 (ai/v2x_vit/model.py, PyTorch)
Config: configs/model.yaml -> perception.v2x_vit
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseV2XViT(ABC):
    """Contract every concrete V2X-ViT implementation must satisfy."""

    @abstractmethod
    def fuse(self, per_vehicle_features: list[Any]) -> Any:
        """
        Fuse per-vehicle spatial features into a single cooperative
        representation via transformer-based attention.

        Parameters
        ----------
        per_vehicle_features : list[Any]
            One feature tensor per connected vehicle (from PointPillars).

        Returns
        -------
        Any
            Unified multimodal feature representation.
        """
        raise NotImplementedError
