"""CLI: validate all scenes' annotated samples and write a quality report.

Usage (from repo root):
    python -m simulation.scripts.validate_dataset --output datasets/logs/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import yaml

from simulation.dataset_pipeline.schema import DatasetSample
from simulation.dataset_pipeline.validator import DatasetValidator

logger = logging.getLogger("aegis_v2x.simulation.scripts.validate_dataset")


def _annotations_dir(dataset_yaml_path: Path) -> Path:
    if dataset_yaml_path.exists():
        with open(dataset_yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        return Path(cfg["paths"]["annotations"])
    return Path("datasets/annotations")


def load_all_samples(annotations_dir: Path) -> List[DatasetSample]:
    samples: List[DatasetSample] = []
    for path in sorted(annotations_dir.glob("*_Annotations.json")):
        with open(path) as fh:
            payload = json.load(fh)
        samples.extend(DatasetSample(**entry) for entry in payload)
    return samples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Aegis-V2X dataset.")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("datasets/logs/validation_report.json"))
    parser.add_argument("--sync-tolerance-ms", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    annotations_dir = args.annotations_dir or _annotations_dir(args.dataset_config)
    samples = load_all_samples(annotations_dir)
    if not samples:
        logger.error("No annotated samples found under %s", annotations_dir)
        return 1

    validator = DatasetValidator(sync_tolerance_ms=args.sync_tolerance_ms)
    report = validator.validate_samples(samples)
    report.write(args.output)

    logger.info("Validated %d samples: %d issues (%d categories). Passed=%s. Report: %s",
                report.total_samples, report.issue_count, len(report.issues_by_category), report.passed, args.output)
    return 0 if report.passed else 2


if __name__ == "__main__":
    sys.exit(main())
