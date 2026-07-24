"""
Abstract contract for the Context-Aware Criticality Estimator.

Mathematical basis (03_Mathematical_Formulation.docx, Section 6):
    C_t = sum_i(alpha_i * f_i),   sum(alpha_i) = 1

Implemented in: Phase 4 (ai/criticality/estimator.py)
Consumes: digital_twin.state.DigitalTwinState
Produces: criticality score C in [0, 1]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from digital_twin.state import DigitalTwinState


class BaseCriticalityEstimator(ABC):
    """Contract every concrete criticality estimator implementation must satisfy."""

    @abstractmethod
    def estimate(self, state: DigitalTwinState) -> float:
        """
        Compute the normalized criticality score for a given Digital
        Twin state.

        Parameters
        ----------
        state : DigitalTwinState
            Current Digital Twin state (DT_t).

        Returns
        -------
        float
            Criticality score C_t, in the closed interval [0, 1]. Higher
            values indicate more urgent communication requirements.
        """
        raise NotImplementedError
