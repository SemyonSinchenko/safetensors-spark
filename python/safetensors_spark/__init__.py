"""
safetensors-spark Python utilities.

This package is pure Python with no dependency on Apache Spark or the
Scala safetensors-spark connector. It provides lightweight utilities for
working with datasets written by the Scala connector.

Modules:
    mlflow   - MLflow dataset lineage integration (log_dataset)
    dataset  - Distributed dataset utilities (DistributedSafetensorsDataset)
"""

from safetensors_spark.dataset import (
    DistributedSafetensorsDataset,
    ShardInfo,
)

__version__ = "0.1.0"

__all__ = [
    "DistributedSafetensorsDataset",
    "ShardInfo",
]
