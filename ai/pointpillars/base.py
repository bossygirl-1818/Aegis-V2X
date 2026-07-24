"""
Abstract contract for the PointPillars spatial feature extractor.

Role (02_System_Architecture, Layer 4): extracts 3D object / spatial
features from LiDAR point clouds, feeding V2X-ViT.

Implemented in: Phase 4 (ai/pointpillars/model.py, PyTorch)
Config: configs/model.yaml -> perception.pointpillars
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePointPillars(ABC):
    """Contract every concrete PointPillars implementation must satisfy."""

    @abstractmethod
    def extract_features(self, lidar_point_cloud: Any) -> Any:
        """
        Extract spatial features from a raw LiDAR point cloud.

        Parameters
        ----------
        lidar_point_cloud : Any
            Raw point cloud, shape (N, 4) — (x, y, z, intensity) — prior
            to voxelization. Concrete type finalized in Phase 4.

        Returns
        -------
        Any
            Spatial feature tensor consumed by V2X-ViT.
        """
        raise NotImplementedError
