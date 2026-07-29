"""Ray-traced wireless channel simulation between roadside units (Tx) and
vehicles (Rx) using Sionna RT.

Which quantities to compute is driven by configs/simulation.yaml's
`sionna_rt.outputs` list (frozen, Phase 1): csi, snr, rssi, path_loss,
delay_spread, beam_index, los_nlos, multipath_components, reflection_paths,
propagation_delay. Radio/antenna parameters (carrier frequency, array size,
codebook size, ray-tracing depth) are not specified in that file and are
documented internal defaults below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from sionna.rt import PlanarArray, Receiver, Scene, Transmitter
except ImportError as exc:  # pragma: no cover
    raise ImportError("sionna_configs.channel_simulator requires the 'sionna' package (Sionna RT).") from exc

logger = logging.getLogger("aegis_v2x.simulation.sionna_configs.channel_simulator")

_BOLTZMANN_DBM_HZ_K = -228.6
_REFERENCE_TEMP_K = 290.0

# Documented internal defaults (configs/simulation.yaml only specifies
# ray_tracing: true and the outputs list, not these physical-layer params).
DEFAULT_RADIO_PARAMS = {
    "carrier_frequency_hz": 28.0e9, "bandwidth_hz": 100.0e6, "num_subcarriers": 128,
    "tx_array": {"rows": 8, "cols": 8, "pattern": "38.901", "polarization": "V"},
    "rx_array": {"rows": 4, "cols": 4, "pattern": "38.901", "polarization": "V"},
    "num_beams": 64, "tx_power_dbm": 23.0, "noise_figure_db": 7.0,
    "ray_tracing": {"max_depth": 5, "method": "fibonacci", "num_samples": 1_000_000,
                    "los": True, "reflection": True, "diffraction": True,
                    "scattering": True, "scat_keep_prob": 0.001, "edge_diffraction": True},
}


@dataclass
class ChannelSimulationResult:
    link_id: str
    vehicle_id: int
    frame: int
    wireless_timestamp: float
    csi: np.ndarray
    snr_db: float
    rssi_dbm: float
    path_loss_db: float
    delay_spread_s: float
    propagation_delay_s: float
    beam_index: int
    los: bool
    num_multipath_components: int


class ChannelSimulator:
    """Runs Sionna RT ray tracing for a single frame's RSU-vehicle links."""

    def __init__(self, radio_params: dict = None):
        self._p = radio_params or DEFAULT_RADIO_PARAMS
        self._noise_power_dbm = self._compute_noise_power_dbm()
        self._beam_codebook = np.deg2rad(np.linspace(-60.0, 60.0, self._p["num_beams"]))

    def _compute_noise_power_dbm(self) -> float:
        bw_hz = float(self._p["bandwidth_hz"])
        nf_db = float(self._p["noise_figure_db"])
        thermal_noise_dbm = _BOLTZMANN_DBM_HZ_K + 10.0 * np.log10(_REFERENCE_TEMP_K) + 10.0 * np.log10(bw_hz)
        return thermal_noise_dbm + nf_db

    def configure_scene(self, scene: Scene) -> None:
        scene.tx_array = PlanarArray(num_rows=self._p["tx_array"]["rows"], num_cols=self._p["tx_array"]["cols"],
                                      pattern=self._p["tx_array"]["pattern"], polarization=self._p["tx_array"]["polarization"])
        scene.rx_array = PlanarArray(num_rows=self._p["rx_array"]["rows"], num_cols=self._p["rx_array"]["cols"],
                                      pattern=self._p["rx_array"]["pattern"], polarization=self._p["rx_array"]["polarization"])
        scene.frequency = float(self._p["carrier_frequency_hz"])

    def simulate_frame(self, scene: Scene, rsu_positions: Dict[int, Tuple[float, float, float]],
                        vehicle_positions: Dict[int, Tuple[float, float, float]],
                        frame: int, wireless_timestamp: float) -> List[ChannelSimulationResult]:
        self.configure_scene(scene)
        scene.remove(list(scene.transmitters.keys()) + list(scene.receivers.keys()))

        for rsu_id, pos in rsu_positions.items():
            scene.add(Transmitter(name=f"tx_{rsu_id}", position=pos, power_dbm=float(self._p["tx_power_dbm"])))
        for vehicle_id, pos in vehicle_positions.items():
            scene.add(Receiver(name=f"rx_{vehicle_id}", position=pos))

        rt = self._p["ray_tracing"]
        paths = scene.compute_paths(
            max_depth=rt["max_depth"], method=rt.get("method", "fibonacci"),
            num_samples=rt.get("num_samples", 1_000_000), los=rt.get("los", True),
            reflection=rt.get("reflection", True), diffraction=rt.get("diffraction", True),
            scattering=rt.get("scattering", True), scat_keep_prob=rt.get("scat_keep_prob", 0.001),
            edge_diffraction=rt.get("edge_diffraction", True),
        )

        num_sc = int(self._p["num_subcarriers"])
        bw_hz = float(self._p["bandwidth_hz"])
        frequencies = np.linspace(-bw_hz / 2, bw_hz / 2, num_sc)

        results: List[ChannelSimulationResult] = []
        for rsu_id in rsu_positions:
            for vehicle_id in vehicle_positions:
                link_id = f"{rsu_id}_{vehicle_id}"
                a, tau = paths.cir(tx=f"tx_{rsu_id}", rx=f"rx_{vehicle_id}")
                if a is None or len(a) == 0:
                    logger.debug("No propagation path for link %s at frame %d (blocked)", link_id, frame)
                    continue
                results.append(self._to_result(link_id, vehicle_id, frame, wireless_timestamp, a, tau, frequencies))
        return results

    def _to_result(self, link_id: str, vehicle_id: int, frame: int, wireless_timestamp: float,
                    path_gains: np.ndarray, path_delays: np.ndarray, frequencies: np.ndarray) -> ChannelSimulationResult:
        csi = np.sum(path_gains[:, None] * np.exp(-2j * np.pi * frequencies[None, :] * path_delays[:, None]), axis=0)
        received_power_lin = np.mean(np.abs(csi) ** 2)
        received_power_dbm = 10.0 * np.log10(max(received_power_lin, 1e-15)) + float(self._p["tx_power_dbm"])
        snr_db = received_power_dbm - self._noise_power_dbm
        path_loss_db = float(self._p["tx_power_dbm"]) - received_power_dbm

        weights = np.abs(path_gains) ** 2
        mean_delay = float(np.average(path_delays, weights=weights)) if weights.sum() > 0 else 0.0
        delay_spread = float(np.sqrt(np.average((path_delays - mean_delay) ** 2, weights=weights))) if weights.sum() > 0 else 0.0

        beam_index = self._select_best_beam(csi)
        los = bool(np.min(path_delays) <= (np.mean(path_delays) * 0.5 + 1e-12)) if len(path_delays) else False

        return ChannelSimulationResult(
            link_id=link_id, vehicle_id=vehicle_id, frame=frame, wireless_timestamp=wireless_timestamp,
            csi=csi.astype(np.complex64), snr_db=float(snr_db), rssi_dbm=float(received_power_dbm),
            path_loss_db=float(path_loss_db), delay_spread_s=delay_spread,
            propagation_delay_s=float(np.min(path_delays)) if len(path_delays) else 0.0,
            beam_index=beam_index, los=los, num_multipath_components=int(len(path_gains)),
        )

    def _select_best_beam(self, csi: np.ndarray) -> int:
        num_sc = len(csi)
        best_idx, best_gain = 0, -np.inf
        for idx, angle_rad in enumerate(self._beam_codebook):
            steering = np.exp(1j * np.pi * np.sin(angle_rad) * np.arange(num_sc))
            gain = float(np.abs(np.vdot(steering, csi)) ** 2)
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        return best_idx
