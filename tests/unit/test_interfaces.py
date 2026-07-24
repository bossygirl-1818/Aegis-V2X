"""
Phase 1 smoke tests: verify the interface scaffolding imports cleanly,
data contracts are constructible, and abstract classes cannot be
instantiated directly (i.e., the interface boundary is real).

These are NOT tests of estimator/policy *logic* — that logic doesn't
exist yet (by design, per Phase 1 scope). They exist to catch broken
imports, typos, and accidental concrete-method omissions early.
"""

import pytest

from ai.criticality.base import BaseCriticalityEstimator
from ai.gru.base import BaseGRUPredictor
from ai.pointpillars.base import BasePointPillars
from ai.trust_estimator.base import BaseTrustEstimator
from ai.twintrust_ap.fsdp import BaseFSDP, CommunicationAction, CriticalityBin, TrustBin
from ai.twintrust_ap.policy import BaseTwinTrustAP, JointAdaptiveDecision
from ai.twintrust_ap.tahs import BaseTAHS
from ai.v2x_vit.base import BaseV2XViT
from digital_twin.interfaces import BaseDigitalTwinManager
from digital_twin.state import DigitalTwinState


@pytest.mark.parametrize(
    "abstract_cls",
    [
        BaseDigitalTwinManager,
        BaseTrustEstimator,
        BaseCriticalityEstimator,
        BaseTAHS,
        BaseFSDP,
        BaseTwinTrustAP,
        BasePointPillars,
        BaseV2XViT,
        BaseGRUPredictor,
    ],
)
def test_abstract_classes_cannot_be_instantiated(abstract_cls):
    """Every module contract must remain a true interface in Phase 1."""
    with pytest.raises(TypeError):
        abstract_cls()


def test_digital_twin_state_is_constructible(sample_dt_state):
    """DigitalTwinState is a usable data contract, not just a stub."""
    assert isinstance(sample_dt_state, DigitalTwinState)
    assert 0.0 <= sample_dt_state.prediction_uncertainty <= 1.0
    assert sample_dt_state.sync_age_seconds >= 0.0


def test_digital_twin_state_is_frozen(sample_dt_state):
    """State must be immutable — mutations should go through the manager."""
    with pytest.raises(Exception):
        sample_dt_state.timestamp = 999.0


def test_fsdp_enums_cover_expected_values():
    assert {b.value for b in TrustBin} == {"low", "medium", "high"}
    assert {b.value for b in CriticalityBin} == {"low", "medium", "high"}
    assert {a.value for a in CommunicationAction} == {
        "synchronize",
        "predict",
        "beam_switch",
        "reduce_horizon",
        "increase_horizon",
        "maintain_state",
    }


def test_joint_adaptive_decision_is_constructible():
    decision = JointAdaptiveDecision(
        horizon=5, action=CommunicationAction.PREDICT, trust=0.7, criticality=0.3
    )
    assert decision.horizon == 5
    assert decision.action == CommunicationAction.PREDICT
