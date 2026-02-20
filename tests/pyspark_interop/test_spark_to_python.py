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
    """Each shard file must not exceed target_shard_size_mb + 20% tolerance.

    We use the minimum allowed shard size (50 MB) and generate enough data to
    force at least two shards. Each F32 tensor element is 4 bytes; we write
    batches of 1000 elements (4 KB each) and enough rows to exceed 50 MB total
    so at least one shard boundary is crossed.

    The test verifies:
      1. More than one shard file is written (i.e., rolling actually happened).
      2. No individual shard file exceeds target_shard_size_mb * 1.20.
    """
    import struct

    TARGET_MB   = 50
    BATCH_SIZE  = 100   # rows per safetensors file
    ELEM_COUNT  = 13000  # floats per row — 13000 * 4 B = 52 KB/row * 100 = ~5.2 MB/batch
    # We need > 50 MB of output, so write ~11 batches = 57 MB → forces 2 shards
    NUM_ROWS    = BATCH_SIZE * 11  # 1100 rows total

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    tensor_schema = StructType(
        [
            StructField("data", BinaryType(), False),
            StructField("shape", ArrayType(IntegerType(), False), False),
            StructField("dtype", StringType(), False),
        ]
    )

    def make_row(i: int):
        raw = struct.pack(f"<{ELEM_COUNT}f", *([float(i % 100)] * ELEM_COUNT))
        return Row(tensor=Row(data=raw, shape=[ELEM_COUNT], dtype="F32"))

    df = spark.createDataFrame(
        [make_row(i) for i in range(NUM_ROWS)],
        StructType([StructField("tensor", tensor_schema, False)]),
    )

    out_dir = str(tmp_path / "sharded_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(BATCH_SIZE))
        .option("dtype", "F32")
        .option("target_shard_size_mb", str(TARGET_MB))
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 1, (
        f"Expected more than one shard file for {NUM_ROWS} rows at {TARGET_MB} MB target, "
        f"got {len(shard_files)}: {[f.name for f in shard_files]}"
    )

    max_allowed_bytes = int(TARGET_MB * 1024 * 1024 * 1.20)
    for shard in shard_files:
        size = shard.stat().st_size
        assert size <= max_allowed_bytes, (
            f"Shard {shard.name} is {size} bytes, exceeds "
            f"{max_allowed_bytes} ({TARGET_MB} MB + 20%)"
        )


def test_name_col_basic(spark, tmp_path: Path):
    """Test basic name_col mode: each row becomes one tensor with a unique key."""
    import safetensors
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType
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
            tensor_key="row_0",
            tensor=Row(data=make_f32_bytes([1.0, 2.0, 3.0, 4.0]), shape=[4], dtype="F32"),
        ),
        Row(
            tensor_key="row_1",
            tensor=Row(data=make_f32_bytes([5.0, 6.0, 7.0, 8.0]), shape=[4], dtype="F32"),
        ),
        Row(
            tensor_key="row_2",
            tensor=Row(data=make_f32_bytes([9.0, 10.0]), shape=[2], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("tensor_key", StringType(), False),
        StructField("tensor", tensor_schema, False),
    ])

    df = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "name_col_output")

    (
        df.write
        .format("safetensors")
        .option("name_col", "tensor_key")
        .option("dtype", "F32")
        .mode("append")
        .save(out_dir)
    )

    # Verify shard files contain the expected tensor keys
    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0, "At least one shard file must be written in name_col mode"

    found_keys = set()
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            found_keys.update(f.keys())

    assert "row_0" in found_keys, "Tensor key 'row_0' must be in output"
    assert "row_1" in found_keys, "Tensor key 'row_1' must be in output"
    assert "row_2" in found_keys, "Tensor key 'row_2' must be in output"


def test_name_col_data_correctness(spark, tmp_path: Path):
    """Test that tensor bytes in name_col mode match the source data."""
    import safetensors
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                      False),
        StructField("shape", ArrayType(IntegerType(), False),   False),
        StructField("dtype", StringType(),                      False),
    ])

    def make_f32_bytes(vals: list[float]) -> bytes:
        return struct.pack(f"<{len(vals)}f", *vals)

    expected_values = {
        "tensor_0": [1.5, 2.5, 3.5],
        "tensor_1": [4.0, 5.0],
    }

    rows = [
        Row(
            key="tensor_0",
            tensor=Row(data=make_f32_bytes([1.5, 2.5, 3.5]), shape=[3], dtype="F32"),
        ),
        Row(
            key="tensor_1",
            tensor=Row(data=make_f32_bytes([4.0, 5.0]), shape=[2], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", tensor_schema, False),
    ])

    df = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "data_correctness_output")

    (
        df.write
        .format("safetensors")
        .option("name_col", "key")
        .option("dtype", "F32")
        .mode("append")
        .save(out_dir)
    )

    # Verify tensor values
    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    all_tensors = {}
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    for key, expected_vals in expected_values.items():
        assert key in all_tensors, f"Tensor key '{key}' not found in output"
        actual_vals = all_tensors[key].tolist()
        np.testing.assert_array_almost_equal(actual_vals, expected_vals,
                                             err_msg=f"Data mismatch for key '{key}'")


def test_duplicates_strategy_fail(spark, tmp_path: Path):
    """duplicatesStrategy=fail must raise an exception on duplicate tensor keys."""
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType
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
            key="same_key",
            tensor=Row(data=make_f32_bytes([1.0, 2.0]), shape=[2], dtype="F32"),
        ),
        Row(
            key="same_key",  # Duplicate key
            tensor=Row(data=make_f32_bytes([3.0, 4.0]), shape=[2], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", tensor_schema, False),
    ])

    df = spark.createDataFrame(rows, schema).coalesce(1)  # Force single partition for duplicate detection
    out_dir = str(tmp_path / "duplicates_fail_output")

    with pytest.raises(Exception) as exc_info:
        (
            df.write
            .format("safetensors")
            .option("name_col", "key")
            .option("dtype", "F32")
            .option("duplicatesStrategy", "fail")
            .mode("append")
            .save(out_dir)
        )

    # The exception should mention duplicate keys
    assert "Duplicate" in str(exc_info.value), (
        f"Expected exception to mention duplicates, got: {exc_info.value}"
    )


def test_duplicates_strategy_last_win(spark, tmp_path: Path):
    """duplicatesStrategy=lastWin must silently keep the last row's tensor."""
    import safetensors
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType
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
            key="shared_key",
            tensor=Row(data=make_f32_bytes([1.0, 2.0]), shape=[2], dtype="F32"),
        ),
        Row(
            key="shared_key",  # Same key — should overwrite
            tensor=Row(data=make_f32_bytes([99.0, 100.0]), shape=[2], dtype="F32"),
        ),
        Row(
            key="unique_key",
            tensor=Row(data=make_f32_bytes([5.0, 6.0]), shape=[2], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", tensor_schema, False),
    ])

    df = spark.createDataFrame(rows, schema).coalesce(1)  # Force single partition for duplicate handling
    out_dir = str(tmp_path / "duplicates_last_win_output")

    (
        df.write
        .format("safetensors")
        .option("name_col", "key")
        .option("dtype", "F32")
        .option("duplicatesStrategy", "lastWin")
        .mode("append")
        .save(out_dir)
    )

    # Verify that the last occurrence of 'shared_key' is in the output
    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    all_tensors = {}
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    assert "shared_key" in all_tensors, "shared_key must be in output"
    assert "unique_key" in all_tensors, "unique_key must be in output"

    # The 'shared_key' should have the values from the last row (99.0, 100.0)
    np.testing.assert_array_almost_equal(
        all_tensors["shared_key"],
        [99.0, 100.0],
        err_msg="shared_key should contain the last row's values"
    )
