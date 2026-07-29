"""CLI: compute dataset-wide statistics and write to configs/dataset.yaml's
`paths.statistics` location.

Usage (from repo root):
    python -m simulation.scripts.compute_statistics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from simulation.dataset_pipeline.statistics import compute_statistics
from simulation.scripts.validate_dataset import _annotations_dir, load_all_samples

logger = logging.getLogger("aegis_v2x.simulation.scripts.compute_statistics")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute Aegis-V2X dataset statistics.")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("datasets/statistics/dataset_statistics.json"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    annotations_dir = args.annotations_dir or _annotations_dir(args.dataset_config)
    samples = load_all_samples(annotations_dir)
    if not samples:
        logger.error("No annotated samples found under %s", annotations_dir)
        return 1

    stats = compute_statistics(samples)
    stats.write(args.output)
    logger.info("Statistics written to %s: %d scenes, %d vehicles, %d frames",
                args.output, stats.num_scenes, stats.num_vehicles, stats.num_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
