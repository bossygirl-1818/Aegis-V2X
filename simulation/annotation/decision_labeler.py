"""Bootstrap TAHS/FSDP implementations, matching the frozen abstract
contracts in ai/twintrust_ap/tahs.py (BaseTAHS) and ai/twintrust_ap/fsdp.py
(BaseFSDP) exactly — including reusing that module's TrustBin, CriticalityBin,
and CommunicationAction enums directly, rather than redefining equivalents.

Important scope note: configs/dataset.yaml's schema_fields does NOT include
a stored prediction-horizon or communication-action column — trust and
criticality are the only ground-truth labels persisted per sample. TAHS/FSDP
outputs are meant to be derived on demand downstream. These classes exist so
that:

  1. Phase 2 can sanity-check its generated trust/criticality distribution
     against the boundedness/monotonicity/determinism invariants in
     docs/interfaces.md before handing the dataset off.
  2. Phase 5 (Vaishnavi, ai/twintrust_ap/) has a working, tested reference
     bootstrap policy to compare against or start from — NOT a substitute
     for the offline Pareto-optimized policy table that Phase 5 must
     ultimately produce (System Architecture, Section 5).

This file lives in simulation/annotation/, not ai/twintrust_ap/, and must
not be imported by Phase 5's runtime code as if it were the final policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ai.twintrust_ap.fsdp import BaseFSDP, CommunicationAction, CriticalityBin, TrustBin
from ai.twintrust_ap.tahs import BaseTAHS

_DISCRETE_HORIZONS = np.array([1, 2, 3, 5, 8, 10])  # configs/model.yaml tahs.horizon_discretization


@dataclass(frozen=True)
class TAHSParams:
    """Defaults match configs/model.yaml `tahs:` exactly."""

    horizon_min: int = 1
    horizon_max: int = 10
    beta: float = 1.0
    gamma: float = 1.0

    @classmethod
    def from_model_config(cls, model_cfg: dict) -> "TAHSParams":
        t = model_cfg["tahs"]
        return cls(horizon_min=t["horizon_min"], horizon_max=t["horizon_max"], beta=t["beta"], gamma=t["gamma"])


class BootstrapTAHS(BaseTAHS):
    """Reference TAHS: H_t = H_min + (H_max - H_min) * sigmoid(beta*T - gamma*C)."""

    def __init__(self, params: TAHSParams = None):
        self._p = params or TAHSParams()

    def select_horizon(self, trust: float, criticality: float) -> int:
        if not (0.0 <= trust <= 1.0) or not (0.0 <= criticality <= 1.0):
            raise ValueError("trust and criticality must be in [0, 1]")
        sigmoid_arg = self._p.beta * trust - self._p.gamma * criticality
        continuous_h = self._p.horizon_min + (self._p.horizon_max - self._p.horizon_min) * self._sigmoid(sigmoid_arg)
        idx = int(np.argmin(np.abs(_DISCRETE_HORIZONS - continuous_h)))
        return int(_DISCRETE_HORIZONS[idx])

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))


# (trust_bin, criticality_bin) -> single CommunicationAction, per ai/twintrust_ap/fsdp.py's
# frozen 6-value action space. Bootstrap heuristic — see module docstring.
_POLICY_TABLE: Dict[Tuple[TrustBin, CriticalityBin], CommunicationAction] = {
    (TrustBin.HIGH, CriticalityBin.LOW): CommunicationAction.MAINTAIN_STATE,
    (TrustBin.HIGH, CriticalityBin.MEDIUM): CommunicationAction.INCREASE_HORIZON,
    (TrustBin.HIGH, CriticalityBin.HIGH): CommunicationAction.BEAM_SWITCH,

    (TrustBin.MEDIUM, CriticalityBin.LOW): CommunicationAction.PREDICT,
    (TrustBin.MEDIUM, CriticalityBin.MEDIUM): CommunicationAction.SYNCHRONIZE,
    (TrustBin.MEDIUM, CriticalityBin.HIGH): CommunicationAction.BEAM_SWITCH,

    (TrustBin.LOW, CriticalityBin.LOW): CommunicationAction.SYNCHRONIZE,
    (TrustBin.LOW, CriticalityBin.MEDIUM): CommunicationAction.REDUCE_HORIZON,
    (TrustBin.LOW, CriticalityBin.HIGH): CommunicationAction.SYNCHRONIZE,
}


class BootstrapFSDP(BaseFSDP):
    """Reference FSDP implementation satisfying BaseFSDP.discretize/lookup_action/decide."""

    def __init__(self, trust_thresholds: Tuple[float, float] = (0.4, 0.7),
                 criticality_thresholds: Tuple[float, float] = (0.34, 0.67)):
        self._trust_thresholds = trust_thresholds
        self._criticality_thresholds = criticality_thresholds

    def discretize(self, trust: float, criticality: float) -> Tuple[TrustBin, CriticalityBin]:
        if not (0.0 <= trust <= 1.0) or not (0.0 <= criticality <= 1.0):
            raise ValueError("trust and criticality must be in [0, 1]")
        return self._bin_trust(trust), self._bin_criticality(criticality)

    def lookup_action(self, trust_bin: TrustBin, criticality_bin: CriticalityBin) -> CommunicationAction:
        return _POLICY_TABLE[(trust_bin, criticality_bin)]

    def _bin_trust(self, trust: float) -> TrustBin:
        low_hi, hi_lo = self._trust_thresholds
        if trust < low_hi:
            return TrustBin.LOW
        if trust < hi_lo:
            return TrustBin.MEDIUM
        return TrustBin.HIGH

    def _bin_criticality(self, criticality: float) -> CriticalityBin:
        low_hi, hi_lo = self._criticality_thresholds
        if criticality < low_hi:
            return CriticalityBin.LOW
        if criticality < hi_lo:
            return CriticalityBin.MEDIUM
        return CriticalityBin.HIGH
