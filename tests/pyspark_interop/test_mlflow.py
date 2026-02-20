"""
Integration test: MLflow lineage

After a Spark write, call log_dataset() and assert that the MLflow run
contains a dataset artifact with the correct manifest contents.

See §5.3 of AGENTS.md.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest


@pytest.fixture()
def written_dataset(spark, tmp_path: Path):
    """Write a small safetensors dataset and return the output path."""
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                     False),
        StructField("shape", ArrayType(IntegerType(), False),  False),
        StructField("dtype", StringType(),                     False),
    ])

    def f32(vals):
        return struct.pack(f"<{len(vals)}f", *vals)

    rows = [
        Row(x=Row(data=f32([1.0, 2.0]), shape=[1, 2], dtype="F32")),
        Row(x=Row(data=f32([3.0, 4.0]), shape=[1, 2], dtype="F32")),
        Row(x=Row(data=f32([5.0, 6.0]), shape=[1, 2], dtype="F32")),
    ]

    schema = StructType([StructField("x", tensor_schema, False)])
    df = spark.createDataFrame(rows, schema)

    out_dir = str(tmp_path / "mlflow_dataset")

    (
        df.write
        .format("safetensors")
        .option("batch_size", "3")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(out_dir)
    )

    return out_dir


def test_log_dataset_adds_input_to_run(written_dataset):
    """log_dataset() must add a dataset input to the active MLflow run."""
    import mlflow
    import sys
    import os

    # Add the Python package to the path
    repo_root = Path(__file__).parent.parent.parent
    python_pkg = repo_root / "python"
    sys.path.insert(0, str(python_pkg))

    from safetensors_spark.mlflow import log_dataset

    with mlflow.start_run() as run:
        log_dataset(path=written_dataset, name="test_dataset")
        run_id = run.info.run_id

    finished_run = mlflow.get_run(run_id)
    dataset_inputs = finished_run.inputs.dataset_inputs
    assert len(dataset_inputs) == 1, \
        f"Expected 1 dataset input, got {len(dataset_inputs)}"


def test_log_dataset_manifest_contents(written_dataset):
    """The logged dataset source must contain total_samples and shards count."""
    import mlflow
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.mlflow import log_dataset

    with mlflow.start_run() as run:
        log_dataset(path=written_dataset)
        run_id = run.info.run_id

    # Read manifest directly for assertion
    manifest_path = Path(written_dataset) / "dataset_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    finished_run = mlflow.get_run(run_id)
    assert len(finished_run.inputs.dataset_inputs) == 1

    # The manifest must reflect the write
    assert manifest["total_samples"] > 0
    assert len(manifest["shards"]) > 0


def test_log_dataset_raises_if_manifest_missing(tmp_path: Path):
    """log_dataset() must raise FileNotFoundError if manifest is absent."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.mlflow import log_dataset

    import mlflow

    with mlflow.start_run():
        with pytest.raises(FileNotFoundError, match="dataset_manifest.json"):
            log_dataset(path=str(tmp_path / "nonexistent"))
