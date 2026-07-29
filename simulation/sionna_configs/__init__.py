"""NVIDIA Sionna RT wireless-channel simulation for Aegis-V2X Phase 2.

Implements System Architecture Layer 2 (Wireless Environment). Which
outputs to compute is driven by configs/simulation.yaml's `sionna_rt.outputs`
list (frozen, Phase 1); radio/antenna/ray-tracing parameters not covered by
that file are documented internal defaults (see channel_simulator.py).
"""

from .channel_simulator import ChannelSimulationResult, ChannelSimulator
from .geometry_adapter import GeometryAdapter

__all__ = ["ChannelSimulator", "ChannelSimulationResult", "GeometryAdapter"]
