"""Builds one CARLA scene (weather + populated actors) from a scenario definition.

Weather/density/town-type categories are the frozen enums from
configs/simulation.yaml: town_types (urban, highway, junction, roundabout,
straight_road), traffic_density (dense, sparse), weather (clear_day, rain,
fog, night). Actor spawning itself is delegated to
traffic_generation.TrafficGenerator so that responsibility lives in its
dedicated Phase 2 subpackage rather than being duplicated here.
"""

from __future__ import annotations

import logging

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise ImportError("carla_scenarios.scenario_builder requires the 'carla' package.") from exc

from simulation.traffic_generation.traffic_generator import SpawnedActors, TrafficGenerator

logger = logging.getLogger("aegis_v2x.simulation.carla_scenarios.scenario_builder")

# Maps configs/simulation.yaml's 4 weather categories to CARLA WeatherParameters.
# "night" and "fog" are expressed as parameter dicts (CARLA has no built-in
# night/fog-only preset that isolates just one factor); "clear_day" and "rain"
# use CARLA's built-in presets directly.
_WEATHER_MAP = {
    "clear_day": "ClearNoon",
    "rain": "HardRainNoon",
    "fog": {"cloudiness": 60.0, "precipitation": 0.0, "fog_density": 70.0,
            "fog_distance": 20.0, "sun_altitude_angle": 30.0},
    "night": {"cloudiness": 10.0, "precipitation": 0.0, "sun_altitude_angle": -30.0},
}

_WEATHER_ATTRS = (
    "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
    "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance", "wetness",
)


class ScenarioBuilder:
    """Applies weather and populates one scene given a scenario definition dict."""

    def __init__(self, world: "carla.World", traffic_manager: "carla.TrafficManager",
                 vehicles_per_scene_range: tuple[int, int], seed: int = 42):
        self._world = world
        self._generator = TrafficGenerator(world, traffic_manager, vehicles_per_scene_range, seed)

    def build(self, scenario: dict) -> SpawnedActors:
        """`scenario` keys: weather, traffic_density, num_roadside_units, autopilot (optional)."""
        self._apply_weather(scenario["weather"])
        actors = self._generator.build_scene(
            density=scenario["traffic_density"],
            num_roadside_units=int(scenario.get("num_roadside_units", 2)),
            autopilot=bool(scenario.get("autopilot", True)),
        )
        logger.info("Scenario '%s' built on map with weather='%s'",
                    scenario.get("scenario_id", "unnamed"), scenario["weather"])
        return actors

    def _apply_weather(self, weather_key: str) -> None:
        if weather_key not in _WEATHER_MAP:
            raise ValueError(
                f"weather '{weather_key}' not in the frozen configs/simulation.yaml weather list "
                f"{list(_WEATHER_MAP)}"
            )
        preset = _WEATHER_MAP[weather_key]
        if isinstance(preset, str):
            weather = getattr(carla.WeatherParameters, preset)
        else:
            weather = carla.WeatherParameters(**{k: v for k, v in preset.items() if k in _WEATHER_ATTRS})
        self._world.set_weather(weather)
