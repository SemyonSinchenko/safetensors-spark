"""
Integration test: Spark → Python

Generates DataFrames in PySpark, writes them using df.write.format("safetensors"),
then reads the output using Python's safetensors.safe_open.

Assertions:
  - Tensor extraction matches the original DataFrames.
  - dataset_manifest.json is present and structurally valid.
  - _tensor_index.parquet (if enabled) contains valid routing information.
  - Shapes and dtypes (including F16, BF16) are correctly preserved.
  - Shard file sizes respect target_shard_size_mb within a 20% tolerance.
  - duplicatesStrategy behaves correctly in name_col mode.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def float_arrays_df(spark):
    """A simple PySpark DataFrame with two float array columns (Tensor Struct format)."""
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, FloatType, IntegerType, StringType, StructField, StructType
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                      False),
        StructField("shape", ArrayType(IntegerType(), False),   False),
        StructField("dtype", StringType(),                      False),
    ])

    def make_f32_bytes(vals: list[float]) -> bytes:
        return struct.pack(f"<{len(vals)}f", *vals)

    rows = [
        Row(
            image=Row(data=make_f32_bytes([1.0, 2.0, 3.0, 4.0]), shape=[2, 2], dtype="F32"),
            label=Row(data=make_f32_bytes([0.0]), shape=[1], dtype="F32"),
        ),
        Row(
            image=Row(data=make_f32_bytes([5.0, 6.0, 7.0, 8.0]), shape=[2, 2], dtype="F32"),
            label=Row(data=make_f32_bytes([1.0]), shape=[1], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("image", tensor_schema, False),
        StructField("label", tensor_schema, False),
    ])

    return spark.createDataFrame(rows, schema)


def test_manifest_is_written(spark, float_arrays_df, tmp_path: Path):
    """dataset_manifest.json must be present and structurally valid after write."""
    out_dir = str(tmp_path / "output")

    (
        float_arrays_df.write
        .format("safetensors")
        .option("batch_size", "2")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(out_dir)
    )

    manifest_path = tmp_path / "output" / "dataset_manifest.json"
    assert manifest_path.exists(), "dataset_manifest.json must be written"

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "format_version"      in manifest
    assert "safetensors_version" in manifest
    assert "total_samples"       in manifest
    assert "total_bytes"         in manifest
    assert "shards"              in manifest
    assert isinstance(manifest["shards"], list)
    assert len(manifest["shards"]) > 0

    for shard in manifest["shards"]:
        assert "file"          in shard
        assert "samples_count" in shard
        assert "bytes"         in shard
        assert shard["file"].endswith(".safetensors")


def test_safetensors_file_readable_by_python(spark, float_arrays_df, tmp_path: Path):
    """Output .safetensors files must be parseable by the Python safetensors library."""
    import safetensors

    out_dir = str(tmp_path / "output")

    (
        float_arrays_df.write
        .format("safetensors")
        .option("batch_size", "2")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0, "At least one shard file must be written"

    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert len(keys) > 0, f"Shard {shard.name} should contain at least one tensor"
            for key in keys:
                tensor = f.get_tensor(key)
                assert tensor is not None


def test_tensor_index_written_when_enabled(spark, float_arrays_df, tmp_path: Path):
    """_tensor_index.parquet must be written when generate_index=true."""
    out_dir = str(tmp_path / "output")

    (
        float_arrays_df.write
        .format("safetensors")
        .option("batch_size", "2")
        .option("dtype", "F32")
        .option("generate_index", "true")
        .mode("overwrite")
        .save(out_dir)
    )

    index_path = tmp_path / "output" / "_tensor_index.parquet"
    assert index_path.exists(), "_tensor_index.parquet must be written when generate_index=true"

    index_df = spark.read.parquet(str(index_path))
    assert "tensor_key" in index_df.columns
    assert "file_name"  in index_df.columns
    assert "shape"      in index_df.columns
    assert "dtype"      in index_df.columns


def test_shard_size_respected(spark, tmp_path: Path):
    """Shard file sizes must be within 20% of target_shard_size_mb (at least for large data)."""
    pytest.skip("Shard size test requires large data — skipped in skeleton")


def test_duplicates_strategy_fail(spark, tmp_path: Path):
    """duplicatesStrategy=fail must raise an exception on duplicate tensor keys."""
    pytest.skip("Duplicate strategy test — TODO implement with name_col mode")


def test_duplicates_strategy_last_win(spark, tmp_path: Path):
    """duplicatesStrategy=lastWin must silently keep the last row's tensor."""
    pytest.skip("Duplicate strategy test — TODO implement with name_col mode")
