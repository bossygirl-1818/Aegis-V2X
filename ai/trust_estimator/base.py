"""
Abstract contract for the Calibrated Twin Trust Estimator.

Mathematical basis (03_Mathematical_Formulation.docx, Section 5):
    z_t = [w1*e_t, w2*u_t, w3*a_t, w4*q_t]
    S_t = sum_i(w_i * z_i)
    T_t = sigmoid(S_t / tau)

Implemented in: Phase 4 (ai/trust_estimator/estimator.py)
Consumes: digital_twin.state.DigitalTwinState
Produces: calibrated trust probability T in [0, 1]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from digital_twin.state import DigitalTwinState


class BaseTrustEstimator(ABC):
    """Contract every concrete trust estimator implementation must satisfy."""

    @abstractmethod
    def estimate(self, state: DigitalTwinState) -> float:
        """
        Compute the calibrated trust probability for a given Digital
        Twin state.

        Parameters
        ----------
        state : DigitalTwinState
            Current Digital Twin state (DT_t).

        Returns
        -------
        float
            Calibrated trust probability T_t, in the closed interval [0, 1].

        Raises
        ------
        ValueError
            If the resulting value falls outside [0, 1] (calibration
            invariant violation) — Phase 4 implementations must guard this.
        """
        raise NotImplementedError

    @abstractmethod
    def calibration_error(self, predictions: list[float], outcomes: list[bool]) -> float:
        """
        Compute Expected Calibration Error (ECE) over a validation batch,
        used to verify the "calibrated probability, not arbitrary
        confidence score" claim central to the project's novelty.

        Parameters
        ----------
        predictions : list[float]
            Predicted trust probabilities.
        outcomes : list[bool]
            Whether the Digital Twin was, in fact, reliable at that step
            (ground truth from evaluation harness).

        Returns
        -------
        float
            Expected Calibration Error, lower is better.
        """
        raise NotImplementedError
