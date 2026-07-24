"""
Abstract contract for the GRU temporal channel predictor.

Role (02_System_Architecture, Layer 4): predicts future wireless channel
characteristics (CSI, SNR, beam quality) from historical sequences.

Implemented in: Phase 4 (ai/gru/model.py, PyTorch)
Config: configs/model.yaml -> perception.gru
Horizon driven by: ai/twintrust_ap/tahs.py (Phase 5)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGRUPredictor(ABC):
    """Contract every concrete GRU predictor implementation must satisfy."""

    @abstractmethod
    def predict(self, historical_sequence: Any, horizon: int) -> Any:
        """
        Predict future channel state tau steps ahead.

        Parameters
        ----------
        historical_sequence : Any
            Historical channel/mobility feature sequence.
        horizon : int
            Number of future steps to predict, selected by TAHS
            (one of {1, 2, 3, 5, 8, 10}).

        Returns
        -------
        Any
            Predicted future channel state Y_hat_{t+tau}, along with an
            associated prediction_uncertainty consumed by the Trust
            Estimator (U_t in the DigitalTwinState).
        """
        raise NotImplementedError
