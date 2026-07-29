"""CARLA server connection and world/traffic-manager lifecycle management.

Wraps the raw `carla` client API so the rest of the pipeline never touches
connection setup, synchronous-mode bookkeeping, or teardown directly.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

try:
    import carla
except ImportError as exc:  # pragma: no cover - only raised on non-workstation envs
    raise ImportError(
        "The 'carla' package is required. Install CARLA 0.9.15 per "
        "requirements/simulation.txt on a GPU workstation with a running CARLA server."
    ) from exc

logger = logging.getLogger("aegis_v2x.simulation.carla_scenarios.client")


class CarlaConnectionError(RuntimeError):
    """Raised when the CARLA server cannot be reached or configured."""


@dataclass(frozen=True)
class CarlaServerConfig:
    """Connection and world settings.

    `carla.version` is pinned in configs/simulation.yaml ("0.9.15"); the rest
    are runtime/world settings not covered by that file and default to
    values appropriate for deterministic dataset generation.
    """

    host: str = "127.0.0.1"
    port: int = 2000
    timeout_sec: float = 30.0
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.01  # 100Hz physics tick, matches IMU rate
    no_rendering_mode: bool = False
    traffic_manager_port: int = 8000
    traffic_manager_seed: int = 42


class CarlaClientManager:
    """Owns the CARLA client/world connection and synchronous-mode lifecycle.

    Example:
        with CarlaClientManager(config, map_name="Town10HD") as manager:
            world = manager.world
    """

    def __init__(self, config: CarlaServerConfig, map_name: str):
        self._config = config
        self._map_name = map_name
        self._client: Optional["carla.Client"] = None
        self._world: Optional["carla.World"] = None
        self._traffic_manager: Optional["carla.TrafficManager"] = None
        self._original_settings: Optional["carla.WorldSettings"] = None

    @property
    def world(self) -> "carla.World":
        if self._world is None:
            raise CarlaConnectionError("World not initialized. Use CarlaClientManager as a context manager.")
        return self._world

    @property
    def traffic_manager(self) -> "carla.TrafficManager":
        if self._traffic_manager is None:
            raise CarlaConnectionError("Traffic manager not initialized.")
        return self._traffic_manager

    def connect(self) -> "CarlaClientManager":
        logger.info("Connecting to CARLA server at %s:%d", self._config.host, self._config.port)
        try:
            self._client = carla.Client(self._config.host, self._config.port)
            self._client.set_timeout(self._config.timeout_sec)
            self._world = self._client.load_world(self._map_name)
        except RuntimeError as exc:
            raise CarlaConnectionError(
                f"Could not connect to CARLA at {self._config.host}:{self._config.port}. "
                "Is the CARLA server running (./CarlaUE4.sh -RenderOffScreen)?"
            ) from exc

        self._original_settings = self._world.get_settings()
        self._apply_synchronous_settings()
        self._configure_traffic_manager()
        logger.info("Connected. Map=%s synchronous_mode=%s dt=%.4fs",
                    self._map_name, self._config.synchronous_mode, self._config.fixed_delta_seconds)
        return self

    def _apply_synchronous_settings(self) -> None:
        settings = self._world.get_settings()
        settings.synchronous_mode = self._config.synchronous_mode
        settings.fixed_delta_seconds = self._config.fixed_delta_seconds
        settings.no_rendering_mode = self._config.no_rendering_mode
        self._world.apply_settings(settings)

    def _configure_traffic_manager(self) -> None:
        self._traffic_manager = self._client.get_trafficmanager(self._config.traffic_manager_port)
        self._traffic_manager.set_synchronous_mode(self._config.synchronous_mode)
        self._traffic_manager.set_random_device_seed(self._config.traffic_manager_seed)

    def tick(self) -> int:
        """Advance the simulation by one fixed timestep. Returns the new frame id."""
        if not self._config.synchronous_mode:
            raise CarlaConnectionError("tick() requires synchronous_mode=True")
        return self._world.tick()

    def set_weather(self, weather: "carla.WeatherParameters") -> None:
        self.world.set_weather(weather)

    def cleanup(self) -> None:
        """Destroy all actors and restore the original world settings."""
        if self._world is None:
            return
        try:
            actors = self._world.get_actors().filter("vehicle.*") + self._world.get_actors().filter("sensor.*") \
                + self._world.get_actors().filter("walker.*")
            for actor in actors:
                try:
                    actor.destroy()
                except RuntimeError:
                    logger.warning("Failed to destroy actor id=%s", actor.id)
        finally:
            if self._original_settings is not None:
                self._world.apply_settings(self._original_settings)
            logger.info("CARLA world cleaned up and settings restored.")

    def __enter__(self) -> "CarlaClientManager":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


@contextmanager
def carla_session(config: CarlaServerConfig, map_name: str) -> Iterator[CarlaClientManager]:
    """Functional convenience wrapper around CarlaClientManager."""
    manager = CarlaClientManager(config, map_name)
    try:
        manager.connect()
        yield manager
    finally:
        manager.cleanup()
