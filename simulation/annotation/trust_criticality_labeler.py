"""Computes ground-truth Trust (T_t) and Criticality (C_t) labels per
03_Mathematical_Formulation, Sections 5-6, using the exact weights and
calibration temperature frozen in configs/model.yaml (Phase 1, owner
Vaishnavi):

    trust_estimator.feature_weights: prediction_error=0.35, prediction_uncertainty=0.25,
        sync_age=0.20, comm_quality=0.20; calibration_temperature=0.8
    criticality_estimator.feature_weights: relative_speed=0.25, blockage_probability=0.25,
        sync_age=0.15, channel_degradation=0.20, traffic_density=0.15

    T_t = sigmoid(S_t / tau),  S_t = sum_i(w_i * f_i)
    C_t = sum_i(alpha_i * f_i)

Because the real Trust/Criticality Estimator networks are trained in Phase 4
(ai/trust_estimator, ai/criticality), Phase 2 labels are computed from
simulation-observable proxies for the four trust features (prediction
error/uncertainty from annotation.future_channel_labeler, sync age from
synchronization.synchronizer, comm quality from measured SNR) — documented,
deterministic, reproducible, and a legitimate bootstrap target for
supervised training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class TrustWeights:
    w_error: float = 0.35
    w_uncertainty: float = 0.25
    w_sync_age: float = 0.20
    w_comm_quality: float = 0.20
    temperature: float = 0.8  # tau; configs/model.yaml trust_estimator.calibration_temperature

    def __post_init__(self) -> None:
        total = self.w_error + self.w_uncertainty + self.w_sync_age + self.w_comm_quality
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Trust weights must sum to 1.0, got {total}")
        if self.temperature <= 0:
            raise ValueError("temperature (tau) must be positive")

    @classmethod
    def from_model_config(cls, model_cfg: dict) -> "TrustWeights":
        fw = model_cfg["trust_estimator"]["feature_weights"]
        return cls(w_error=fw["prediction_error"], w_uncertainty=fw["prediction_uncertainty"],
                   w_sync_age=fw["sync_age"], w_comm_quality=fw["comm_quality"],
                   temperature=model_cfg["trust_estimator"]["calibration_temperature"])


@dataclass(frozen=True)
class CriticalityWeights:
    alpha_relative_speed: float = 0.25
    alpha_blockage_prob: float = 0.25
    alpha_sync_age: float = 0.15
    alpha_channel_degradation: float = 0.20
    alpha_traffic_density: float = 0.15

    def __post_init__(self) -> None:
        total = (self.alpha_relative_speed + self.alpha_blockage_prob + self.alpha_sync_age
                 + self.alpha_channel_degradation + self.alpha_traffic_density)
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Criticality weights must sum to 1.0, got {total}")

    @classmethod
    def from_model_config(cls, model_cfg: dict) -> "CriticalityWeights":
        fw = model_cfg["criticality_estimator"]["feature_weights"]
        return cls(alpha_relative_speed=fw["relative_speed"], alpha_blockage_prob=fw["blockage_probability"],
                   alpha_sync_age=fw["sync_age"], alpha_channel_degradation=fw["channel_degradation"],
                   alpha_traffic_density=fw["traffic_density"])


@dataclass
class TrustCriticalityLabel:
    trust: float
    criticality: float
    trust_score_raw: float
    features: dict


class TrustCriticalityLabeler:
    def __init__(self, trust_weights: Optional[TrustWeights] = None,
                 criticality_weights: Optional[CriticalityWeights] = None):
        self._tw = trust_weights or TrustWeights()
        self._cw = criticality_weights or CriticalityWeights()

    @classmethod
    def from_config_file(cls, model_yaml_path: Path = Path("configs/model.yaml")) -> "TrustCriticalityLabeler":
        with open(model_yaml_path) as fh:
            model_cfg = yaml.safe_load(fh)
        return cls(TrustWeights.from_model_config(model_cfg), CriticalityWeights.from_model_config(model_cfg))

    def compute_trust(self, prediction_error: float, prediction_uncertainty: float,
                       sync_age_normalized: float, comm_quality_normalized: float) -> tuple[float, float]:
        for name, value in (("prediction_error", prediction_error), ("prediction_uncertainty", prediction_uncertainty),
                             ("sync_age_normalized", sync_age_normalized), ("comm_quality_normalized", comm_quality_normalized)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be normalized to [0, 1], got {value}")

        s_t = (
            self._tw.w_error * (1.0 - prediction_error)
            + self._tw.w_uncertainty * (1.0 - prediction_uncertainty)
            + self._tw.w_sync_age * (1.0 - sync_age_normalized)
            + self._tw.w_comm_quality * comm_quality_normalized
        )
        centered = (s_t - 0.5) * 2.0
        trust = float(_sigmoid(centered / self._tw.temperature))
        return trust, float(s_t)

    def compute_criticality(self, relative_speed_normalized: float, blockage_prob: float,
                             sync_age_normalized: float, channel_degradation_normalized: float,
                             traffic_density_normalized: float) -> float:
        for name, value in (("relative_speed_normalized", relative_speed_normalized), ("blockage_prob", blockage_prob),
                             ("sync_age_normalized", sync_age_normalized),
                             ("channel_degradation_normalized", channel_degradation_normalized),
                             ("traffic_density_normalized", traffic_density_normalized)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be normalized to [0, 1], got {value}")

        criticality = (
            self._cw.alpha_relative_speed * relative_speed_normalized
            + self._cw.alpha_blockage_prob * blockage_prob
            + self._cw.alpha_sync_age * sync_age_normalized
            + self._cw.alpha_channel_degradation * channel_degradation_normalized
            + self._cw.alpha_traffic_density * traffic_density_normalized
        )
        return float(np.clip(criticality, 0.0, 1.0))

    def label(self, *, prediction_error: float, prediction_uncertainty: float, sync_age_normalized: float,
              comm_quality_normalized: float, relative_speed_normalized: float, blockage_prob: float,
              channel_degradation_normalized: float, traffic_density_normalized: float) -> TrustCriticalityLabel:
        trust, s_t = self.compute_trust(prediction_error, prediction_uncertainty, sync_age_normalized, comm_quality_normalized)
        criticality = self.compute_criticality(relative_speed_normalized, blockage_prob, sync_age_normalized,
                                                channel_degradation_normalized, traffic_density_normalized)
        return TrustCriticalityLabel(
            trust=trust, criticality=criticality, trust_score_raw=s_t,
            features={"prediction_error": prediction_error, "prediction_uncertainty": prediction_uncertainty,
                      "sync_age_normalized": sync_age_normalized, "comm_quality_normalized": comm_quality_normalized,
                      "relative_speed_normalized": relative_speed_normalized, "blockage_prob": blockage_prob,
                      "channel_degradation_normalized": channel_degradation_normalized,
                      "traffic_density_normalized": traffic_density_normalized},
        )
