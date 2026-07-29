"""Ground-truth future-channel labels and trust-input proxies, computed by
looking ahead in the simulated (offline, fully-known) channel time series.

Because dataset generation has access to the entire simulated trajectory,
"future" CSI/SNR/beam/path-loss at a horizon is simply the value that many
frames ahead in the same link's time series — the supervised target for the
GRU predictor trained in Phase 4 (ai/gru). Matches configs/dataset.yaml
schema fields: ground_truth_future_csi, ground_truth_future_beam.

Also computes the `prediction_error` / `prediction_uncertainty` proxies
consumed by trust_criticality_labeler.TrustCriticalityLabeler, using a naive
persistence baseline since the real GRU predictor doesn't exist yet in
Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class ChannelSample:
    timestamp: float
    csi_magnitude: float
    snr_db: float
    path_loss_db: float
    beam_index: int


@dataclass
class FutureChannelLabel:
    horizon_frames: int
    future_csi_magnitude: float
    future_snr_db: float
    future_path_loss_db: float
    future_beam_index: int
    prediction_error: float
    prediction_uncertainty: float


class FutureChannelLabeler:
    def __init__(self, uncertainty_window: int = 5, error_norm_scale: float = 1.0):
        if uncertainty_window < 2:
            raise ValueError("uncertainty_window must be >= 2")
        self._window = uncertainty_window
        self._error_norm_scale = error_norm_scale

    def label_sequence(self, samples: Sequence[ChannelSample], horizon_frames: int) -> List[FutureChannelLabel]:
        if horizon_frames < 1:
            raise ValueError("horizon_frames must be >= 1")
        n = len(samples)
        labels: List[FutureChannelLabel] = []
        magnitudes = np.array([s.csi_magnitude for s in samples], dtype=np.float64)

        for i in range(n - horizon_frames):
            future = samples[i + horizon_frames]
            naive_pred = samples[i].csi_magnitude
            error = self._normalized_error(naive_pred, future.csi_magnitude)
            uncertainty = self._rolling_uncertainty(magnitudes, i)

            labels.append(FutureChannelLabel(
                horizon_frames=horizon_frames, future_csi_magnitude=future.csi_magnitude,
                future_snr_db=future.snr_db, future_path_loss_db=future.path_loss_db,
                future_beam_index=future.beam_index, prediction_error=error, prediction_uncertainty=uncertainty,
            ))
        return labels

    def _normalized_error(self, predicted: float, actual: float) -> float:
        raw_error = abs(predicted - actual) / max(self._error_norm_scale, 1e-9)
        return float(np.clip(raw_error, 0.0, 1.0))

    def _rolling_uncertainty(self, magnitudes: np.ndarray, index: int) -> float:
        start = max(0, index - self._window + 1)
        window = magnitudes[start:index + 1]
        if len(window) < 2:
            return 0.0
        std = float(np.std(window))
        mean = float(np.mean(np.abs(window))) + 1e-9
        return float(np.clip(std / mean, 0.0, 1.0))
