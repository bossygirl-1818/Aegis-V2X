"""
Shared YAML configuration loader used across all ai/ sub-packages.

Centralizing this avoids every module re-implementing yaml.safe_load and
keeps config-path resolution consistent (relative to repository root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """
    Load a YAML configuration file from configs/ by name.

    Parameters
    ----------
    name : str
        Config file name without extension, e.g. "model", "simulation".

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If configs/{name}.yaml does not exist.

    Example
    -------
    >>> cfg = load_config("model")
    >>> cfg["trust_estimator"]["calibration_temperature"]
    0.8
    """
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
