"""
Digital Twin state representation.

Defines DT_t = {CSI_t, SNR_t, B_t, M_t, E_t, U_t, Age_t} exactly as
specified in 03_Mathematical_Formulation.docx, Section 4.

This module contains ONLY the data contract (a frozen, typed representation
of the Digital Twin state). No estimation, synchronization, or decision
logic lives here — see ai/trust_estimator, ai/criticality, and
ai/twintrust_ap for those responsibilities (Phases 4-5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChannelState:
    """
    Wireless channel characteristics at a single time step.

    Parameters
    ----------
    csi : Any
        Channel State Information (complex tensor / array, shape defined
        in Phase 4 by the GRU predictor's expected input format).
    snr : float
        Signal-to-Noise Ratio in dB.
    beam_index : int
        Active beam index selected by the current beam configuration.
    path_loss : float
        Path loss in dB, as produced by NVIDIA Sionna RT.

    Example
    -------
    >>> ChannelState(csi=None, snr=18.4, beam_index=12, path_loss=98.2)
    """

    csi: Any
    snr: float
    beam_index: int
    path_loss: float


@dataclass(frozen=True)
class MobilityState:
    """
    Vehicle mobility information contributing to M_t.

    Parameters
    ----------
    position : tuple[float, float, float]
        (x, y, z) position in the simulation/world frame.
    velocity : tuple[float, float, float]
        (vx, vy, vz) velocity vector.
    heading : float
        Heading angle in radians.
    relative_speed : float
        Relative speed with respect to the nearest communication peer,
        used as a Criticality Estimator feature.
    """

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    heading: float
    relative_speed: float


@dataclass(frozen=True)
class EnvironmentalContext:
    """
    Environmental context contributing to E_t.

    Parameters
    ----------
    weather : str
        One of the categories defined in configs/simulation.yaml.
    traffic_density : str
        "sparse" or "dense", per configs/simulation.yaml.
    blockage_probability : float
        Predicted probability of LOS blockage, in [0, 1].
    """

    weather: str
    traffic_density: str
    blockage_probability: float


@dataclass(frozen=True)
class DigitalTwinState:
    """
    Complete Digital Twin state DT_t, as formalized in
    03_Mathematical_Formulation.docx, Section 4:

        DT_t = {CSI_t, SNR_t, B_t, M_t, E_t, U_t, Age_t}

    Parameters
    ----------
    timestamp : float
        Simulation time (seconds) at which this state was constructed.
    channel : ChannelState
        Current wireless channel state (CSI_t, SNR_t, B_t).
    mobility : MobilityState
        Current vehicle mobility state (M_t).
    environment : EnvironmentalContext
        Current environmental context (E_t).
    prediction_uncertainty : float
        U_t — uncertainty reported by the GRU channel predictor, in [0, 1].
    sync_age_seconds : float
        Age_t — elapsed time since the Digital Twin was last synchronized
        with ground-truth simulator state.
    metadata : dict
        Free-form metadata (scene_id, vehicle_id, frame_id) for dataset
        traceability. Not part of the mathematical state vector itself.

    Example
    -------
    >>> state = DigitalTwinState(
    ...     timestamp=12.345,
    ...     channel=ChannelState(csi=None, snr=15.0, beam_index=4, path_loss=95.0),
    ...     mobility=MobilityState((0, 0, 0), (10, 0, 0), 0.0, 5.0),
    ...     environment=EnvironmentalContext("clear_day", "sparse", 0.1),
    ...     prediction_uncertainty=0.2,
    ...     sync_age_seconds=0.05,
    ... )
    >>> state.sync_age_seconds
    0.05
    """

    timestamp: float
    channel: ChannelState
    mobility: MobilityState
    environment: EnvironmentalContext
    prediction_uncertainty: float
    sync_age_seconds: float
    metadata: dict = field(default_factory=dict)
