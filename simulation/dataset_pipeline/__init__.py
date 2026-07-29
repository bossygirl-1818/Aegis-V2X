"""Dataset assembly, validation, splitting, preprocessing, and statistics
for the Aegis-V2X Phase 2 multimodal dataset.

Schema and paths are frozen in configs/dataset.yaml (Phase 1, owner
Vaishnavi); this package implements against that contract exactly rather
than inventing its own field names.
"""

from .schema import DatasetSample, SCHEMA_VERSION, TrafficDensity, WeatherCondition
from .splitter import DatasetSplitter, SplitAssignment
from .statistics import DatasetStatistics, compute_statistics
from .validator import DatasetValidator, ValidationIssue, ValidationReport

__all__ = [
    "DatasetSample", "SCHEMA_VERSION", "TrafficDensity", "WeatherCondition",
    "DatasetSplitter", "SplitAssignment",
    "DatasetStatistics", "compute_statistics",
    "DatasetValidator", "ValidationIssue", "ValidationReport",
]
