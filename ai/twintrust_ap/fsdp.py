"""
Abstract contract for the Finite-State Decision Policy (FSDP).

Mathematical basis (03_Mathematical_Formulation.docx, Section 8):
    Trust, Criticality each discretized into {Low, Medium, High}
    => 3 x 3 = 9 runtime states
    A_t = pi(T, C)   (offline-generated policy table, O(1) lookup)

Determinism invariant (Section 11): for a fixed state pair (T, C),
FSDP always returns the same action.

Implemented in: Phase 5 (ai/twintrust_ap/fsdp.py — this file, extended)
Depends on: an offline-generated policy table (configs/model.yaml ->
            fsdp.policy_table_path), produced by the Phase 5 offline
            Pareto optimization pipeline described in Document 2, Section 5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class TrustBin(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CriticalityBin(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommunicationAction(str, Enum):
    SYNCHRONIZE = "synchronize"
    PREDICT = "predict"
    BEAM_SWITCH = "beam_switch"
    REDUCE_HORIZON = "reduce_horizon"
    INCREASE_HORIZON = "increase_horizon"
    MAINTAIN_STATE = "maintain_state"


class BaseFSDP(ABC):
    """Contract every concrete FSDP implementation must satisfy."""

    @abstractmethod
    def discretize(self, trust: float, criticality: float) -> tuple[TrustBin, CriticalityBin]:
        """
        Map continuous (trust, criticality) values to their discrete bins.

        Parameters
        ----------
        trust : float
            Calibrated trust probability, in [0, 1].
        criticality : float
            Criticality score, in [0, 1].

        Returns
        -------
        tuple[TrustBin, CriticalityBin]
            The discretized (trust_bin, criticality_bin) pair.
        """
        raise NotImplementedError

    @abstractmethod
    def lookup_action(
        self, trust_bin: TrustBin, criticality_bin: CriticalityBin
    ) -> CommunicationAction:
        """
        Perform the constant-time policy-table lookup for a discretized
        state pair.

        Parameters
        ----------
        trust_bin : TrustBin
        criticality_bin : CriticalityBin

        Returns
        -------
        CommunicationAction
            The single coordinated action selected by the offline policy.
        """
        raise NotImplementedError

    def decide(self, trust: float, criticality: float) -> CommunicationAction:
        """
        Convenience method combining discretize() + lookup_action().
        Concrete subclasses generally do not need to override this.

        Parameters
        ----------
        trust : float
        criticality : float

        Returns
        -------
        CommunicationAction
        """
        trust_bin, criticality_bin = self.discretize(trust, criticality)
        return self.lookup_action(trust_bin, criticality_bin)
