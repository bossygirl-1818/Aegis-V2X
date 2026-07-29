"""CLI: split annotated samples into scene-disjoint train/validation/test sets.

Writes to the paths frozen in configs/dataset.yaml (datasets/train,
datasets/validation, datasets/test).

Usage (from repo root):
    python -m simulation.scripts.split_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from simulation.dataset_pipeline.splitter import DatasetSplitter
from simulation.scripts.validate_dataset import _annotations_dir, load_all_samples

logger = logging.getLogger("aegis_v2x.simulation.scripts.split_dataset")


def _split_paths(dataset_yaml_path: Path) -> dict:
    if dataset_yaml_path.exists():
        with open(dataset_yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        paths = cfg["paths"]
        return {"train": Path(paths["train"]), "validation": Path(paths["validation"]), "test": Path(paths["test"])}
    return {"train": Path("datasets/train"), "validation": Path("datasets/validation"), "test": Path("datasets/test")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Split the Aegis-V2X dataset (scene-disjoint).")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--train", type=float, default=None)
    parser.add_argument("--val", type=float, default=None)
    parser.add_argument("--test", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    annotations_dir = args.annotations_dir or _annotations_dir(args.dataset_config)
    samples = load_all_samples(annotations_dir)
    if not samples:
        logger.error("No annotated samples found under %s", annotations_dir)
        return 1

    splitter = DatasetSplitter(train_ratio=args.train, val_ratio=args.val, test_ratio=args.test, seed=args.seed)
    splits = splitter.split_samples(samples)

    if not DatasetSplitter.verify_no_leakage(splits):
        logger.error("Scene leakage detected between splits — aborting write.")
        return 2

    out_paths = _split_paths(args.dataset_config)
    for split_name, split_samples in splits.items():
        out_dir = out_paths[split_name]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "samples.json"
        with open(out_path, "w") as fh:
            json.dump([json.loads(s.model_dump_json()) for s in split_samples], fh, indent=2)
        logger.info("%s: %d samples -> %s", split_name, len(split_samples), out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
