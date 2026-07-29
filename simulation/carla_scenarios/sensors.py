"""Attaches and manages the per-vehicle sensor rig (LiDAR, GPS, IMU, camera, speed).

Frequencies come from configs/simulation.yaml `sensors:` section (frozen,
Phase 1): lidar_hz, gps_hz, imu_hz, camera_fps, vehicle_speed_hz, csi_hz.
Per-sensor physical parameters not covered by that file (LiDAR channel
count/range, noise stddevs, camera resolution) are internal, documented
defaults — see `DEFAULT_SENSOR_PARAMS` below.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Any, Dict, List

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise ImportError("carla_scenarios.sensors requires the 'carla' package.") from exc

import numpy as np

logger = logging.getLogger("aegis_v2x.simulation.carla_scenarios.sensors")

# Physical sensor parameters not specified in configs/simulation.yaml (which
# only pins frequencies). Kept here, documented, rather than silently
# hardcoded inline.
DEFAULT_SENSOR_PARAMS = {
    "lidar": {"channels": 64, "range_m": 120.0, "points_per_second": 1_300_000,
              "upper_fov_deg": 10.0, "lower_fov_deg": -30.0, "noise_stddev_m": 0.0},
    "gps": {"noise_lat_stddev": 0.0000005, "noise_lon_stddev": 0.0000005, "noise_alt_stddev": 0.05},
    "imu": {"accel_noise_stddev": 0.05, "gyro_noise_stddev": 0.01, "gyro_bias_stddev": 0.001},
    "camera": {"width": 1280, "height": 720, "fov_deg": 90},
    "mount_offset_xyz_m": (0.0, 0.0, 1.8),
}


@dataclass
class SensorReading:
    sensor_type: str
    vehicle_id: int
    frame: int
    timestamp: float
    data: Any


class SensorRig:
    """Owns all sensors attached to a single vehicle actor."""

    def __init__(self, world: "carla.World", vehicle: "carla.Actor", sensor_frequencies: Dict[str, float],
                 sensor_params: Dict[str, dict] = None, enable_camera: bool = True):
        self._world = world
        self._vehicle = vehicle
        self._freq = sensor_frequencies  # {"lidar_hz":10, "gps_hz":10, "imu_hz":100, "camera_fps":30, "vehicle_speed_hz":10}
        self._params = sensor_params or DEFAULT_SENSOR_PARAMS
        self._enable_camera = enable_camera
        self._blueprint_library = world.get_blueprint_library()
        self._sensors: List["carla.Actor"] = []
        self._queue: "queue.Queue[SensorReading]" = queue.Queue()

    @property
    def vehicle_id(self) -> int:
        return self._vehicle.id

    def attach_all(self) -> None:
        mount = self._mount_transform()
        self._attach_lidar(mount)
        self._attach_gnss(mount)
        self._attach_imu(mount)
        if self._enable_camera:
            self._attach_camera(mount)
        logger.debug("Attached %d sensors to vehicle %d", len(self._sensors), self._vehicle.id)

    def _mount_transform(self) -> "carla.Transform":
        x, y, z = self._params["mount_offset_xyz_m"]
        return carla.Transform(carla.Location(x=x, y=y, z=z))

    def _attach_lidar(self, mount: "carla.Transform") -> None:
        p = self._params["lidar"]
        bp = self._blueprint_library.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", str(p["channels"]))
        bp.set_attribute("range", str(p["range_m"]))
        bp.set_attribute("points_per_second", str(p["points_per_second"]))
        bp.set_attribute("rotation_frequency", str(self._freq["lidar_hz"]))
        bp.set_attribute("upper_fov", str(p["upper_fov_deg"]))
        bp.set_attribute("lower_fov", str(p["lower_fov_deg"]))
        bp.set_attribute("noise_stddev", str(p.get("noise_stddev_m", 0.0)))

        sensor = self._world.spawn_actor(bp, mount, attach_to=self._vehicle)
        sensor.listen(self._on_lidar)
        self._sensors.append(sensor)

    def _on_lidar(self, measurement: "carla.LidarMeasurement") -> None:
        points = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
        self._queue.put(SensorReading("lidar", self._vehicle.id, measurement.frame, measurement.timestamp, points))

    def _attach_gnss(self, mount: "carla.Transform") -> None:
        p = self._params["gps"]
        bp = self._blueprint_library.find("sensor.other.gnss")
        bp.set_attribute("sensor_tick", str(1.0 / self._freq["gps_hz"]))
        bp.set_attribute("noise_lat_stddev", str(p["noise_lat_stddev"]))
        bp.set_attribute("noise_lon_stddev", str(p["noise_lon_stddev"]))
        bp.set_attribute("noise_alt_stddev", str(p["noise_alt_stddev"]))

        sensor = self._world.spawn_actor(bp, mount, attach_to=self._vehicle)
        sensor.listen(self._on_gnss)
        self._sensors.append(sensor)

    def _on_gnss(self, measurement: "carla.GnssMeasurement") -> None:
        self._queue.put(SensorReading(
            "gps", self._vehicle.id, measurement.frame, measurement.timestamp,
            {"lat": measurement.latitude, "lon": measurement.longitude, "alt": measurement.altitude},
        ))

    def _attach_imu(self, mount: "carla.Transform") -> None:
        p = self._params["imu"]
        bp = self._blueprint_library.find("sensor.other.imu")
        bp.set_attribute("sensor_tick", str(1.0 / self._freq["imu_hz"]))
        for axis in ("x", "y", "z"):
            bp.set_attribute(f"noise_accel_stddev_{axis}", str(p["accel_noise_stddev"]))
            bp.set_attribute(f"noise_gyro_stddev_{axis}", str(p["gyro_noise_stddev"]))
        bp.set_attribute("noise_gyro_bias_x", str(p["gyro_bias_stddev"]))

        sensor = self._world.spawn_actor(bp, mount, attach_to=self._vehicle)
        sensor.listen(self._on_imu)
        self._sensors.append(sensor)

    def _on_imu(self, measurement: "carla.IMUMeasurement") -> None:
        self._queue.put(SensorReading(
            "imu", self._vehicle.id, measurement.frame, measurement.timestamp,
            {"accel": (measurement.accelerometer.x, measurement.accelerometer.y, measurement.accelerometer.z),
             "gyro": (measurement.gyroscope.x, measurement.gyroscope.y, measurement.gyroscope.z),
             "compass": measurement.compass},
        ))

    def _attach_camera(self, mount: "carla.Transform") -> None:
        p = self._params["camera"]
        bp = self._blueprint_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(p["width"]))
        bp.set_attribute("image_size_y", str(p["height"]))
        bp.set_attribute("fov", str(p["fov_deg"]))
        bp.set_attribute("sensor_tick", str(1.0 / self._freq["camera_fps"]))

        sensor = self._world.spawn_actor(bp, mount, attach_to=self._vehicle)
        sensor.listen(self._on_camera)
        self._sensors.append(sensor)

    def _on_camera(self, image: "carla.Image") -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        self._queue.put(SensorReading("camera", self._vehicle.id, image.frame, image.timestamp, array))

    def read_speed(self, frame: int, timestamp: float) -> SensorReading:
        velocity = self._vehicle.get_velocity()
        speed_mps = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))
        reading = SensorReading("vehicle_speed", self._vehicle.id, frame, timestamp, {"speed_mps": speed_mps})
        self._queue.put(reading)
        return reading

    def drain(self, timeout_sec: float = 0.05) -> List[SensorReading]:
        readings: List[SensorReading] = []
        try:
            while True:
                readings.append(self._queue.get(timeout=timeout_sec))
        except queue.Empty:
            pass
        return readings

    def destroy(self) -> None:
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                logger.warning("Failed to destroy sensor %s on vehicle %d", sensor.type_id, self._vehicle.id)
        self._sensors.clear()
