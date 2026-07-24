#!/usr/bin/env python3
"""
Verify the Phase 1 development environment is correctly set up.

Checks:
  - Python version >= 3.10
  - Core Phase-1 dependencies importable (pyyaml, pydantic)
  - All configs/*.yaml parse successfully
  - All ai/ and digital_twin/ interfaces import without error

Usage
-----
    python scripts/verify_env.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def check_python_version() -> bool:
    ok = sys.version_info >= (3, 10)
    print(f"[{'OK' if ok else 'FAIL'}] Python version: {sys.version.split()[0]} (need >= 3.10)")
    return ok


def check_core_imports() -> bool:
    try:
        import pydantic  # noqa: F401
        import yaml  # noqa: F401

        print("[OK] Core dependencies (pyyaml, pydantic) importable")
        return True
    except ImportError as e:
        print(f"[FAIL] Missing core dependency: {e}")
        return False


def check_configs() -> bool:
    from ai.utils.config_loader import load_config

    all_ok = True
    for name in ["model", "simulation", "deployment", "dataset", "logging"]:
        try:
            load_config(name)
            print(f"[OK] configs/{name}.yaml parses correctly")
        except Exception as e:
            print(f"[FAIL] configs/{name}.yaml: {e}")
            all_ok = False
    return all_ok


def check_interfaces() -> bool:
    try:
        from ai.criticality.base import BaseCriticalityEstimator  # noqa: F401
        from ai.gru.base import BaseGRUPredictor  # noqa: F401
        from ai.pointpillars.base import BasePointPillars  # noqa: F401
        from ai.trust_estimator.base import BaseTrustEstimator  # noqa: F401
        from ai.twintrust_ap.fsdp import BaseFSDP  # noqa: F401
        from ai.twintrust_ap.policy import BaseTwinTrustAP  # noqa: F401
        from ai.twintrust_ap.tahs import BaseTAHS  # noqa: F401
        from ai.v2x_vit.base import BaseV2XViT  # noqa: F401
        from digital_twin.interfaces import BaseDigitalTwinManager  # noqa: F401
        from digital_twin.state import DigitalTwinState  # noqa: F401

        print("[OK] All Phase 1 module interfaces import cleanly")
        return True
    except ImportError as e:
        print(f"[FAIL] Interface import error: {e}")
        return False


if __name__ == "__main__":
    results = [
        check_python_version(),
        check_core_imports(),
        check_configs(),
        check_interfaces(),
    ]
    print()
    if all(results):
        print("Phase 1 environment verification: PASSED")
        sys.exit(0)
    else:
        print("Phase 1 environment verification: FAILED")
        sys.exit(1)
