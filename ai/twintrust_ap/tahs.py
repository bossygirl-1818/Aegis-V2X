"""
Abstract contract for Trust-Adaptive Horizon Selection (TAHS).

Mathematical basis (03_Mathematical_Formulation.docx, Section 7):
    H_t = H_min + (H_max - H_min) * sigmoid(beta*T_t - gamma*C_t)
    H in {1, 2, 3, 5, 8, 10}   (discretized, see configs/model.yaml)

Monotonicity invariant (Section 11): for fixed criticality,
    T1 > T2  =>  H1 >= H2
i.e. higher trust never reduces the prediction horizon. Phase 5
implementations MUST satisfy this and it should be covered by a
dedicated property-based unit test.

Implemented in: Phase 5 (ai/twintrust_ap/tahs.py — this file, extended)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTAHS(ABC):
    """Contract every concrete TAHS implementation must satisfy."""

    @abstractmethod
    def select_horizon(self, trust: float, criticality: float) -> int:
        """
        Select the prediction horizon given current trust and criticality.

        Parameters
        ----------
        trust : float
            Calibrated trust probability T_t, in [0, 1].
        criticality : float
            Criticality score C_t, in [0, 1].

        Returns
        -------
        int
            Selected horizon, one of the discretized values in
            configs/model.yaml -> tahs.horizon_discretization.
        """
        raise NotImplementedError
