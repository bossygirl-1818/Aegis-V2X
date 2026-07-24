"""
TwinTrust-AP: the unified policy coordinating TAHS and FSDP.

This is the top-level entry point described in 04_Novel_Algorithm_Design,
Section 4. It does not introduce new decision logic of its own — it
composes a BaseTAHS and a BaseFSDP implementation into the single
joint-adaptive-decision interface consumed by the runtime execution layer.

Implemented in: Phase 5
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ai.twintrust_ap.fsdp import CommunicationAction


@dataclass(frozen=True)
class JointAdaptiveDecision:
    """
    The coordinated output of TwinTrust-AP for a single runtime cycle.

    Parameters
    ----------
    horizon : int
        Selected prediction horizon (from TAHS).
    action : CommunicationAction
        Selected communication action (from FSDP).
    trust : float
        Trust value the decision was conditioned on (for logging/audit).
    criticality : float
        Criticality value the decision was conditioned on.
    """

    horizon: int
    action: CommunicationAction
    trust: float
    criticality: float


class BaseTwinTrustAP(ABC):
    """Contract for the composed TwinTrust-AP decision engine."""

    @abstractmethod
    def decide(self, trust: float, criticality: float) -> JointAdaptiveDecision:
        """
        Produce a joint adaptive decision from current trust and
        criticality values.

        Parameters
        ----------
        trust : float
            Calibrated trust probability, in [0, 1].
        criticality : float
            Criticality score, in [0, 1].

        Returns
        -------
        JointAdaptiveDecision
        """
        raise NotImplementedError
