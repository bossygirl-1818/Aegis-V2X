"""Phase 1 tests verifying every configs/*.yaml file is well-formed and
contains the parameters the mathematical formulation depends on."""

from ai.utils.config_loader import load_config


def test_model_config_loads_and_matches_math_formulation():
    cfg = load_config("model")
    assert cfg["trust_estimator"]["calibration_temperature"] == 0.8
    assert cfg["tahs"]["horizon_min"] == 1
    assert cfg["tahs"]["horizon_max"] == 10
    assert cfg["tahs"]["horizon_discretization"] == [1, 2, 3, 5, 8, 10]
    assert cfg["fsdp"]["num_states"] == 9
    assert len(cfg["fsdp"]["trust_bins"]) == 3
    assert len(cfg["fsdp"]["criticality_bins"]) == 3
    assert abs(sum(cfg["criticality_estimator"]["feature_weights"].values()) - 1.0) < 1e-6


def test_simulation_config_loads():
    cfg = load_config("simulation")
    assert cfg["synchronization"]["tolerance_ms"] == 10
    assert cfg["dataset_targets"]["train_split"] == 0.70


def test_deployment_config_loads():
    cfg = load_config("deployment")
    assert cfg["device"] == "jetson_orin"


def test_dataset_config_loads():
    cfg = load_config("dataset")
    assert "csi" in cfg["schema_fields"]


def test_logging_config_loads():
    cfg = load_config("logging")
    assert cfg["version"] == 1
