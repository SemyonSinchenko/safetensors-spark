"""
Integration test: Spark → Python

Generates DataFrames in PySpark, writes them using df.write.format("safetensors"),
then reads the output using Python's safetensors.safe_open.

Assertions:
  - Tensor extraction matches the original DataFrames.
  - dataset_manifest.json is present and structurally valid.
  - _tensor_index.parquet (if enabled) contains valid routing information.
  - Shapes and dtypes (including F16, BF16) are correctly preserved.
  - Each batch-mode output file is a valid standalone safetensors file.
  - tail_strategy (drop/pad/write) is correctly applied to the last incomplete batch.
  - KV mode shard files respect target_shard_size_mb within a 20% tolerance.
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
        assert "shard_path"    in shard
        assert shard["shard_path"].endswith(".safetensors")


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


def test_batch_mode_one_file_per_batch(spark, tmp_path: Path):
    """Each batch produces exactly one valid, standalone safetensors file.

    Writes 6 rows with batch_size=2 → expects 3 shard files, each readable
    by the Python safetensors library with the correct stacked tensor shape.
    """
    import safetensors

    BATCH_SIZE = 2
    NUM_ROWS   = 6
    ELEM_COUNT = 4  # floats per sample

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType,
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                    False),
        StructField("shape", ArrayType(IntegerType(), False), False),
        StructField("dtype", StringType(),                    False),
    ])

    rows = [
        Row(tensor=Row(
            data=struct.pack(f"<{ELEM_COUNT}f", *[float(i)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", tensor_schema, False)])
    ).coalesce(1)  # single partition so batch boundaries are deterministic

    out_dir = str(tmp_path / "batch_files_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(BATCH_SIZE))
        .option("tail_strategy", "drop")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) == NUM_ROWS // BATCH_SIZE, (
        f"Expected {NUM_ROWS // BATCH_SIZE} shard files, got {len(shard_files)}: "
        f"{[f.name for f in shard_files]}"
    )

    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert keys == ["tensor"], f"Expected ['tensor'] in {shard.name}, got {keys}"
            t = f.get_tensor("tensor")
            assert t.shape == (BATCH_SIZE, ELEM_COUNT), (
                f"Expected shape ({BATCH_SIZE}, {ELEM_COUNT}), got {t.shape} in {shard.name}"
            )


def test_batch_mode_tail_strategy_drop(spark, tmp_path: Path):
    """tail_strategy=drop must discard the last incomplete batch."""
    BATCH_SIZE = 3
    NUM_ROWS   = 7  # 2 full batches (6 rows) + 1 tail row that should be dropped

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType,
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                    False),
        StructField("shape", ArrayType(IntegerType(), False), False),
        StructField("dtype", StringType(),                    False),
    ])

    rows = [
        Row(tensor=Row(data=struct.pack("<4f", float(i), 0.0, 0.0, 0.0), shape=[4], dtype="F32"))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", tensor_schema, False)])
    ).coalesce(1)

    out_dir = str(tmp_path / "tail_drop_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(BATCH_SIZE))
        .option("tail_strategy", "drop")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    full_batches = NUM_ROWS // BATCH_SIZE  # 2
    assert len(shard_files) == full_batches, (
        f"With tail_strategy=drop, expected {full_batches} shard files (tail dropped), "
        f"got {len(shard_files)}"
    )


def test_batch_mode_tail_strategy_write(spark, tmp_path: Path):
    """tail_strategy=write must write the incomplete last batch as-is."""
    import safetensors

    BATCH_SIZE = 3
    NUM_ROWS   = 7  # 2 full batches + 1-row tail
    ELEM_COUNT = 4

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType,
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                    False),
        StructField("shape", ArrayType(IntegerType(), False), False),
        StructField("dtype", StringType(),                    False),
    ])

    rows = [
        Row(tensor=Row(
            data=struct.pack(f"<{ELEM_COUNT}f", *[float(i)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", tensor_schema, False)])
    ).coalesce(1)

    out_dir = str(tmp_path / "tail_write_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(BATCH_SIZE))
        .option("tail_strategy", "write")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    expected_files = (NUM_ROWS + BATCH_SIZE - 1) // BATCH_SIZE  # ceil(7/3) = 3
    assert len(shard_files) == expected_files, (
        f"With tail_strategy=write, expected {expected_files} shard files, "
        f"got {len(shard_files)}"
    )

    # Last file must be readable and have the tail size as its leading dimension
    last_shard = max(shard_files, key=lambda p: p.name)
    with safetensors.safe_open(str(last_shard), framework="numpy") as f:
        t = f.get_tensor("tensor")
        tail_size = NUM_ROWS % BATCH_SIZE  # 1
        assert t.shape[0] == tail_size, (
            f"Last shard leading dim should be {tail_size} (tail), got {t.shape[0]}"
        )


def test_batch_mode_tail_strategy_pad(spark, tmp_path: Path):
    """tail_strategy=pad must zero-pad the incomplete last batch to exactly batch_size rows."""
    import safetensors

    BATCH_SIZE = 3
    NUM_ROWS   = 7  # 2 full batches + 1-row tail → last file padded to 3 rows
    ELEM_COUNT = 4

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType,
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                    False),
        StructField("shape", ArrayType(IntegerType(), False), False),
        StructField("dtype", StringType(),                    False),
    ])

    rows = [
        Row(tensor=Row(
            data=struct.pack(f"<{ELEM_COUNT}f", *[float(i + 1)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", tensor_schema, False)])
    ).coalesce(1)

    out_dir = str(tmp_path / "tail_pad_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(BATCH_SIZE))
        .option("tail_strategy", "pad")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    expected_files = (NUM_ROWS + BATCH_SIZE - 1) // BATCH_SIZE  # ceil(7/3) = 3
    assert len(shard_files) == expected_files, (
        f"With tail_strategy=pad, expected {expected_files} shard files, "
        f"got {len(shard_files)}"
    )

    # All files must have the same leading dimension (batch_size)
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            t = f.get_tensor("tensor")
            assert t.shape == (BATCH_SIZE, ELEM_COUNT), (
                f"All shards including padded tail must have shape "
                f"({BATCH_SIZE}, {ELEM_COUNT}), got {t.shape} in {shard.name}"
            )

    # The padded rows in the last file must be zeros
    last_shard = max(shard_files, key=lambda p: p.name)
    with safetensors.safe_open(str(last_shard), framework="numpy") as f:
        t = f.get_tensor("tensor")
        tail_size = NUM_ROWS % BATCH_SIZE  # 1 real row in last file
        # Rows beyond tail_size are padding
        padding = t[tail_size:]
        assert (padding == 0).all(), (
            f"Padded rows in last shard must be zeros, got: {padding}"
        )


def test_kv_mode_shard_size_respected(spark, tmp_path: Path):
    """KV mode: each shard file must not exceed target_shard_size_mb + 20% tolerance.

    We generate enough key-value tensor data to cross the 50 MB threshold, forcing
    at least two shard files. Verifies:
      1. More than one shard file is written.
      2. No individual shard exceeds target_shard_size_mb * 1.20.
      3. Each shard is a valid standalone safetensors file.
    """
    import safetensors

    TARGET_MB  = 50
    ELEM_COUNT = 13000  # floats per row — 13000 × 4 B = 52 KB/row
    # 52 KB × 1100 rows ≈ 57 MB → crosses 50 MB threshold, forces ≥2 shards
    NUM_ROWS   = 1100

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, BinaryType, IntegerType, StringType, StructField, StructType,
    )

    tensor_schema = StructType([
        StructField("data",  BinaryType(),                    False),
        StructField("shape", ArrayType(IntegerType(), False), False),
        StructField("dtype", StringType(),                    False),
    ])

    rows = [
        Row(
            key=f"row_{i}",
            tensor=Row(
                data=struct.pack(f"<{ELEM_COUNT}f", *([float(i % 100)] * ELEM_COUNT)),
                shape=[ELEM_COUNT],
                dtype="F32",
            ),
        )
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows,
        StructType([
            StructField("key",    StringType(),  False),
            StructField("tensor", tensor_schema, False),
        ]),
    ).coalesce(1)

    out_dir = str(tmp_path / "kv_sharded_output")
    (
        df.write.format("safetensors")
        .option("name_col", "key")
        .option("target_shard_size_mb", str(TARGET_MB))
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 1, (
        f"Expected more than one KV shard file at {TARGET_MB} MB target, "
        f"got {len(shard_files)}: {[f.name for f in shard_files]}"
    )

    max_allowed_bytes = int(TARGET_MB * 1024 * 1024 * 1.20)
    for shard in shard_files:
        size = shard.stat().st_size
        assert size <= max_allowed_bytes, (
            f"KV shard {shard.name} is {size} bytes, exceeds "
            f"{max_allowed_bytes} ({TARGET_MB} MB + 20%)"
        )
        # Each shard must be independently readable
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            assert len(list(f.keys())) > 0, f"Shard {shard.name} must have at least one tensor"


def test_f16_write_round_trip(spark, tmp_path: Path):
    """F16 tensors written from Spark must be readable by the Python safetensors library.

    Uses arr_to_st() (registered via SafetensorsExtensions) via Spark SQL to create
    F16 Tensor Structs, writes them, then reads back with safetensors.safe_open and
    verifies the dtype and shape are preserved.
    """
    import safetensors

    from pyspark.sql import Row
    from pyspark.sql.types import ArrayType, FloatType, StructField, StructType

    rows = [Row(floats=[1.0, 2.0, 3.0, 4.0]), Row(floats=[5.0, 6.0, 7.0, 8.0])]
    df_raw = spark.createDataFrame(
        rows, StructType([StructField("floats", ArrayType(FloatType()), False)])
    ).coalesce(1)

    # Use Spark SQL to invoke arr_to_st (registered via SafetensorsExtensions)
    df_raw.createOrReplaceTempView("raw_f16")
    df_tensors = spark.sql(
        "SELECT arr_to_st(floats, array(4), 'F16') AS tensor FROM raw_f16"
    ).coalesce(1)

    out_dir = str(tmp_path / "f16_output")
    (
        df_tensors.write.format("safetensors")
        .option("batch_size", "2")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0, "At least one F16 shard file must be written"

    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert "tensor" in keys, f"Expected 'tensor' key in {shard.name}"
            t = f.get_tensor("tensor")
            assert t.dtype == np.float16, f"Expected float16 tensor, got {t.dtype}"
            assert t.shape[1] == 4, f"Expected 4 elements per sample, got {t.shape[1]}"


def test_bf16_write_round_trip(spark, tmp_path: Path):
    """BF16 tensors written from Spark must be readable by the Python safetensors library.

    Uses arr_to_st() to create BF16 Tensor Structs, writes them, then reads back
    with safetensors.safe_open and verifies the dtype is preserved and values round-trip.

    NOTE: BF16 is not in the JSON schema regex — see §1.1.
    Requires ml_dtypes package for safetensors to handle BF16 in numpy framework.
    """
    import safetensors
    import ml_dtypes

    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType, FloatType, StructField, StructType,
    )

    expected_rows = [
        [1.0, 0.5, -1.0, 2.0],
        [0.0, 3.0, -0.5, 1.5],
    ]
    rows = [Row(floats=row) for row in expected_rows]
    df_raw = spark.createDataFrame(
        rows, StructType([StructField("floats", ArrayType(FloatType()), False)])
    ).coalesce(1)

    df_raw.createOrReplaceTempView("raw_bf16")
    df_tensors = spark.sql(
        "SELECT arr_to_st(floats, array(4), 'BF16') AS tensor FROM raw_bf16"
    ).coalesce(1)

    out_dir = str(tmp_path / "bf16_output")
    (
        df_tensors.write.format("safetensors")
        .option("batch_size", "2")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0, "At least one BF16 shard file must be written"

    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert "tensor" in keys, f"Expected 'tensor' key in {shard.name}"
            t = f.get_tensor("tensor")

            # Verify dtype is BF16 (requires ml_dtypes)
            assert t.dtype == ml_dtypes.bfloat16, \
                f"Expected bfloat16 tensor, got {t.dtype}"

            # Verify shape: batch of 2 rows, 4 elements per row
            assert t.shape == (2, 4), f"Expected shape (2, 4), got {t.shape}"

            # Verify values round-trip correctly
            # BF16 truncation may lose precision in the lower mantissa,
            # but exact powers of 2 (1, 0.5, -1, 2, 0, 3, -0.5, 1.5) are preserved.
            t_f32 = t.astype(np.float32)
            expected_flat = np.array(expected_rows).flatten()
            np.testing.assert_allclose(
                t_f32.flatten(),
                expected_flat,
                rtol=1e-2,
                atol=1e-6,
                err_msg="BF16 round-trip values mismatch (truncation may lose mantissa bits)",
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
        .mode("overwrite")
        .save(out_dir)
    )

    # Verify shard files contain the expected tensor keys
    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0, "At least one shard file must be written in name_col mode"

    found_keys = set()
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            found_keys.update(f.keys())

    assert "row_0__tensor" in found_keys, "Tensor key 'row_0__tensor' must be in output"
    assert "row_1__tensor" in found_keys, "Tensor key 'row_1__tensor' must be in output"
    assert "row_2__tensor" in found_keys, "Tensor key 'row_2__tensor' must be in output"


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
        "tensor_0__tensor": [1.5, 2.5, 3.5],
        "tensor_1__tensor": [4.0, 5.0],
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
        .mode("overwrite")
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
            .mode("overwrite")
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
        .mode("overwrite")
        .save(out_dir)
    )

    # Verify that the last occurrence of 'shared_key' is in the output
    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    all_tensors = {}
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    assert "shared_key__tensor" in all_tensors, "shared_key__tensor must be in output"
    assert "unique_key__tensor" in all_tensors, "unique_key__tensor must be in output"

    # The 'shared_key__tensor' should have the values from the last row (99.0, 100.0)
    np.testing.assert_array_almost_equal(
        all_tensors["shared_key__tensor"],
        [99.0, 100.0],
        err_msg="shared_key__tensor should contain the last row's values"
    )


def test_name_col_multi_column(spark, tmp_path: Path):
    """Test KV mode with multiple non-key tensor columns."""
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
            key="model_0",
            weights=Row(data=make_f32_bytes([0.1, 0.2, 0.3]), shape=[3], dtype="F32"),
            bias=Row(data=make_f32_bytes([0.01, 0.02]), shape=[2], dtype="F32"),
        ),
        Row(
            key="model_1",
            weights=Row(data=make_f32_bytes([0.4, 0.5]), shape=[2], dtype="F32"),
            bias=Row(data=make_f32_bytes([0.03]), shape=[1], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("weights", tensor_schema, False),
        StructField("bias", tensor_schema, False),
    ])

    df = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "multi_column_output")

    (
        df.write
        .format("safetensors")
        .option("name_col", "key")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(out_dir)
    )

    # Verify all four tensor keys exist (2 models × 2 columns each)
    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    all_tensors = {}
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    assert "model_0__weights" in all_tensors
    assert "model_0__bias" in all_tensors
    assert "model_1__weights" in all_tensors
    assert "model_1__bias" in all_tensors

    # Verify data correctness
    np.testing.assert_array_almost_equal(all_tensors["model_0__weights"], [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(all_tensors["model_0__bias"], [0.01, 0.02])
    np.testing.assert_array_almost_equal(all_tensors["model_1__weights"], [0.4, 0.5])
    np.testing.assert_array_almost_equal(all_tensors["model_1__bias"], [0.03])


def test_name_col_custom_separator(spark, tmp_path: Path):
     """Test KV mode with a custom kv_separator."""
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
             key="row_0",
             tensor=Row(data=make_f32_bytes([1.0, 2.0]), shape=[2], dtype="F32"),
         ),
     ]
 
     schema = StructType([
         StructField("key", StringType(), False),
         StructField("tensor", tensor_schema, False),
     ])
 
     df = spark.createDataFrame(rows, schema)
     out_dir = str(tmp_path / "custom_sep_output")
 
     (
         df.write
         .format("safetensors")
         .option("name_col", "key")
         .option("kv_separator", "/")
         .option("dtype", "F32")
         .mode("overwrite")
         .save(out_dir)
     )
 
     # Verify that the key uses the custom separator
     shard_files = sorted(Path(out_dir).glob("*.safetensors"))
     found_keys = set()
     for shard in shard_files:
         with safetensors.safe_open(str(shard), framework="numpy") as f:
             found_keys.update(f.keys())
 
     assert "row_0/tensor" in found_keys, "Custom separator '/' should be used in tensor key"


def test_predicate_pushdown_with_index(spark, tmp_path: Path):
     """Test that _tensor_index.parquet is written and can infer schema.
     
     This test verifies that:
     1. _tensor_index.parquet is written when generate_index=true
     2. The index contains correct tensor_key and file_name mappings
     3. The index has expected structure and can be read by Spark
     
     Setup:
       - Write dataset in standard (non-KV) mode with 2 tensor columns
       - Enable generate_index=true
     
     Verification:
       - Index file exists and is readable
       - Index contains correct columns (tensor_key, file_name, shape, dtype)
       - Index rows match expected tensor count
     """
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
 
     # Create 3 rows with 2 tensor columns
     rows = []
     for i in range(3):
         rows.append(Row(
             tensor_a=Row(data=make_f32_bytes([float(i)]), shape=[1], dtype="F32"),
             tensor_b=Row(data=make_f32_bytes([float(i + 100)]), shape=[1], dtype="F32"),
         ))
 
     schema = StructType([
         StructField("tensor_a", tensor_schema, False),
         StructField("tensor_b", tensor_schema, False),
     ])
 
     df = spark.createDataFrame(rows, schema)
     out_dir = str(tmp_path / "predicate_pushdown_output")
 
     # Write with index enabled (non-KV mode)
     (
         df.write
         .format("safetensors")
         .option("batch_size", "2")  # Use batch_size=2 to create fewer shards
         .option("dtype", "F32")
         .option("generate_index", "true")
         .mode("overwrite")
         .save(out_dir)
     )
 
     # Verify index was written
     index_path = tmp_path / "predicate_pushdown_output" / "_tensor_index.parquet"
     assert index_path.exists(), "_tensor_index.parquet must be written"
 
     # Read the index to verify it contains required columns
     index_df = spark.read.parquet(str(index_path))
     assert "tensor_key" in index_df.columns, "Index must have tensor_key column"
     assert "file_name" in index_df.columns, "Index must have file_name column"
     assert "shape" in index_df.columns, "Index must have shape column"
     assert "dtype" in index_df.columns, "Index must have dtype column"
 
     # Verify the index has entries for tensors (should match number of shard files × 2 columns)
     index_count = index_df.count()
     assert index_count > 0, f"Index should have at least one entry, got {index_count}"
     
     # Verify distinct tensor keys in index
     index_keys = index_df.select("tensor_key").distinct().collect()
     assert len(index_keys) == 2, f"Should have 2 distinct tensor keys (tensor_a, tensor_b), got {len(index_keys)}: {[r[0] for r in index_keys]}"

     # Verify tensor names
     assert {r[0] for r in index_keys} == {"tensor_a", "tensor_b"}



