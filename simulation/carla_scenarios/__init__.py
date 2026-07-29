"""CARLA physical-environment simulation for Aegis-V2X Phase 2.

Implements System Architecture Layer 1 (Physical Environment): vehicle
dynamics, road geometry, traffic, weather, GPS, and sensor generation.
Parameters are sourced from configs/simulation.yaml (Phase 1, frozen).
"""

from .client import CarlaClientManager, CarlaServerConfig
from .scene_exporter import SceneExporter
from .scenario_builder import ScenarioBuilder

__all__ = ["CarlaClientManager", "CarlaServerConfig", "ScenarioBuilder", "SceneExporter"]
