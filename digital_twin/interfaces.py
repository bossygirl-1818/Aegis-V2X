"""
Abstract contracts for Digital Twin runtime management.

These interfaces define WHAT the Phase 5 StateManager must do, without
committing to HOW. Concrete implementations are out of scope for Phase 1
per the Master Project Instructions ("do not implement AI models yet").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from digital_twin.state import DigitalTwinState


class BaseDigitalTwinManager(ABC):
    """
    Contract for the runtime component that owns the current
    DigitalTwinState, applies synchronization decisions, and exposes the
    state to the Trust Estimator, Criticality Estimator, and TwinTrust-AP.

    Implemented in: Phase 5 (digital_twin/manager.py)
    Consumed by: ai/trust_estimator, ai/criticality, ai/twintrust_ap
    """

    @abstractmethod
    def get_current_state(self) -> DigitalTwinState:
        """
        Return the most recently committed Digital Twin state.

        Returns
        -------
        DigitalTwinState
            The current DT_t.
        """
        raise NotImplementedError

    @abstractmethod
    def synchronize(self, ground_truth_state: DigitalTwinState) -> DigitalTwinState:
        """
        Overwrite the Digital Twin's internal state with a fresh
        ground-truth observation, resetting sync_age_seconds to 0.

        Parameters
        ----------
        ground_truth_state : DigitalTwinState
            The latest observation from CARLA / Sionna RT.

        Returns
        -------
        DigitalTwinState
            The newly synchronized state.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self, dt_seconds: float) -> DigitalTwinState:
        """
        Advance the Digital Twin by dt_seconds without a fresh
        synchronization (i.e., age the twin and apply any predicted
        state update produced by the GRU channel predictor).

        Parameters
        ----------
        dt_seconds : float
            Elapsed time since the last runtime cycle.

        Returns
        -------
        DigitalTwinState
            The updated (aged) state.
        """
        raise NotImplementedError
