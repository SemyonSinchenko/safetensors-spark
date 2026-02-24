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
import re
import struct
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
import safetensors
import safetensors.numpy

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Shared schema / helper constants
# ---------------------------------------------------------------------------

TENSOR_SCHEMA = StructType([
    StructField("data",  BinaryType(),                    False),
    StructField("shape", ArrayType(IntegerType(), False), False),
    StructField("dtype", StringType(),                    False),
])

SHARD_NAME_RE = re.compile(
    r"^part-\d{5}-\d{4}-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.safetensors$"
)


def make_f32_bytes(vals: list[float]) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


def f32_to_f16_truncate(f32_array: np.ndarray) -> np.ndarray:
    """
    Truncate (round-toward-zero) F32 → F16, matching the JVM implementation.

    NumPy's .astype(np.float16) uses round-to-nearest-even, which differs from
    the truncation used in the Scala/JVM writer. This function replicates the
    JVM truncation by zeroing the low mantissa bits before casting, so that
    round-trip comparisons are apples-to-apples.
    """
    bits = f32_array.view(np.uint32)
    sign    = (bits >> 31) & 0x1
    exp32   = (bits >> 23) & 0xff
    mant32  = bits & 0x7fffff

    result = np.zeros(f32_array.shape, dtype=np.uint16)

    # Inf or NaN
    inf_nan = exp32 == 0xff
    result[inf_nan] = (
        (sign[inf_nan].astype(np.uint16) << 15)
        | np.uint16(0x7c00)
        | np.where(mant32[inf_nan] != 0, np.uint16(0x200), np.uint16(0))
    )

    # Zero / denormal (exp32 == 0) → flush to signed zero
    zero = exp32 == 0
    result[zero] = (sign[zero].astype(np.uint16) << 15)

    # Normal range
    normal = ~inf_nan & ~zero
    exp16 = exp32[normal].astype(np.int32) - 127 + 15
    underflow = exp16 <= 0
    overflow  = exp16 >= 31

    exp16_clipped = np.clip(exp16, 0, 30).astype(np.uint16)
    mant16 = (mant32[normal] >> 13).astype(np.uint16)  # truncate low 13 bits

    normal_result = (
        (sign[normal].astype(np.uint16) << 15)
        | (exp16_clipped << 10)
        | mant16
    )
    # underflow → zero; overflow → inf
    normal_result[underflow] = (sign[normal][underflow].astype(np.uint16) << 15)
    normal_result[overflow]  = (sign[normal][overflow].astype(np.uint16) << 15) | np.uint16(0x7c00)

    result[normal] = normal_result
    return result.view(np.float16)


@pytest.fixture()
def float_arrays_df(spark):
    """A simple PySpark DataFrame with two float array columns (Tensor Struct format)."""
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
        StructField("image", TENSOR_SCHEMA, False),
        StructField("label", TENSOR_SCHEMA, False),
    ])
    return spark.createDataFrame(rows, schema).coalesce(1)


def test_manifest_is_written(spark, float_arrays_df, tmp_path: Path):
    """dataset_manifest.json must be present, structurally valid, and numerically correct."""
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

    assert manifest["format_version"]      == "1.0"
    assert manifest["safetensors_version"] == "1.0"
    assert manifest["total_samples"]       == 2
    assert manifest["total_bytes"]         > 0
    assert isinstance(manifest["shards"], list)
    assert len(manifest["shards"]) > 0

    out_path = tmp_path / "output"
    total_bytes_on_disk = sum(
        p.stat().st_size for p in out_path.glob("*.safetensors")
    )
    assert manifest["total_bytes"] == total_bytes_on_disk, (
        f"manifest total_bytes {manifest['total_bytes']} != "
        f"actual bytes on disk {total_bytes_on_disk}"
    )

    shard_names_on_disk = {p.name for p in out_path.glob("*.safetensors")}
    for shard in manifest["shards"]:
        assert shard["shard_path"].endswith(".safetensors")
        assert SHARD_NAME_RE.match(shard["shard_path"]), (
            f"Shard filename '{shard['shard_path']}' does not match naming convention "
            f"part-NNNNN-NNNN-<uuid>.safetensors"
        )
        assert shard["shard_path"] in shard_names_on_disk, (
            f"Shard '{shard['shard_path']}' listed in manifest but not found on disk"
        )
        assert shard["bytes"] == (out_path / shard["shard_path"]).stat().st_size
        assert shard["samples_count"] > 0

    # Verify manifest schema field
    assert "schema" in manifest, "manifest must contain 'schema' field"
    assert "image" in manifest["schema"]
    assert "label" in manifest["schema"]
    assert manifest["schema"]["image"]["dtype"] == "F32"
    assert manifest["schema"]["label"]["dtype"] == "F32"
    assert isinstance(manifest["schema"]["image"]["shape"], list)


def test_safetensors_file_readable_by_python(spark, float_arrays_df, tmp_path: Path):
    """Output .safetensors files must be parseable and contain correct tensor data."""
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

    full_batch_image_shape = (2, 2, 2)
    full_batch_label_shape = (2, 1)

    non_full_image_shards = []
    non_full_label_shards = []

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name), (
            f"Shard filename '{shard.name}' does not match naming convention"
        )
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert set(keys) == {"image", "label"}, (
                f"Expected keys {{'image', 'label'}}, got {keys}"
            )
            image = f.get_tensor("image")
            label = f.get_tensor("label")
            assert image.dtype == np.float32
            assert label.dtype == np.float32

            if image.shape != full_batch_image_shape:
                non_full_image_shards.append(shard.name)
            if label.shape != full_batch_label_shape:
                non_full_label_shards.append(shard.name)

    assert len(non_full_image_shards) <= 1, (
        f"More than one shard has a non-full-batch image shape: {non_full_image_shards}"
    )
    assert len(non_full_label_shards) <= 1, (
        f"More than one shard has a non-full-batch label shape: {non_full_label_shards}"
    )

    # Collect all image and label values across shards and verify the full dataset
    all_image_vals = []
    all_label_vals = []
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            all_image_vals.extend(f.get_tensor("image").flatten().tolist())
            all_label_vals.extend(f.get_tensor("label").flatten().tolist())

    np.testing.assert_array_almost_equal(
        all_image_vals, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        err_msg="Combined image values across all shards do not match expected"
    )
    np.testing.assert_array_almost_equal(
        all_label_vals, [0.0, 1.0],
        err_msg="Combined label values across all shards do not match expected"
    )


def test_tensor_index_written_when_enabled(spark, float_arrays_df, tmp_path: Path):
    """_tensor_index.parquet must be written with correct content when generate_index=true."""
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
    assert set(index_df.columns) >= {"tensor_key", "file_name", "shape", "dtype"}

    rows = index_df.collect()
    assert len(rows) > 0

    shard_names_on_disk = {p.name for p in (tmp_path / "output").glob("*.safetensors")}
    tensor_keys_in_index = {r["tensor_key"] for r in rows}
    assert tensor_keys_in_index == {"image", "label"}, (
        f"Expected tensor keys {{'image', 'label'}}, got {tensor_keys_in_index}"
    )

    for row in rows:
        assert row["file_name"] in shard_names_on_disk, (
            f"Index file_name '{row['file_name']}' not found on disk"
        )
        assert row["dtype"] == "F32"
        assert isinstance(row["shape"], list)
        assert len(row["shape"]) > 0


def test_batch_mode_one_file_per_batch(spark, tmp_path: Path):
    """Each batch produces exactly one valid, standalone safetensors file with correct data."""
    BATCH_SIZE = 2
    NUM_ROWS   = 6
    ELEM_COUNT = 4

    rows = [
        Row(tensor=Row(
            data=make_f32_bytes([float(i)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", TENSOR_SCHEMA, False)])
    ).coalesce(1)

    out_dir = str(tmp_path / "batch_files_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(BATCH_SIZE))
        .option("tail_strategy", "drop")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) == NUM_ROWS // BATCH_SIZE

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name), (
            f"Shard filename '{shard.name}' does not match naming convention"
        )
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert keys == ["tensor"], f"Expected ['tensor'], got {keys}"
            t = f.get_tensor("tensor")
            assert t.shape == (BATCH_SIZE, ELEM_COUNT)
            assert t.dtype == np.float32


def test_batch_mode_tail_strategy_drop(spark, tmp_path: Path):
    """tail_strategy=drop must discard the last incomplete batch."""
    BATCH_SIZE = 3
    NUM_ROWS   = 7

    rows = [
        Row(tensor=Row(data=make_f32_bytes([float(i), 0.0, 0.0, 0.0]), shape=[4], dtype="F32"))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", TENSOR_SCHEMA, False)])
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
    assert len(shard_files) == NUM_ROWS // BATCH_SIZE

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            t = f.get_tensor("tensor")
            assert t.shape[0] == BATCH_SIZE


def test_batch_mode_tail_strategy_write(spark, tmp_path: Path):
    """tail_strategy=write must write the incomplete last batch as-is."""
    BATCH_SIZE = 3
    NUM_ROWS   = 7
    ELEM_COUNT = 4

    rows = [
        Row(tensor=Row(
            data=make_f32_bytes([float(i)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", TENSOR_SCHEMA, False)])
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
    expected_files = (NUM_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
    assert len(shard_files) == expected_files

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)

    last_shard = max(shard_files, key=lambda p: p.name)
    with safetensors.safe_open(str(last_shard), framework="numpy") as f:
        t = f.get_tensor("tensor")
        tail_size = NUM_ROWS % BATCH_SIZE
        assert t.shape[0] == tail_size
        assert t.dtype == np.float32


def test_batch_mode_tail_strategy_pad(spark, tmp_path: Path):
    """tail_strategy=pad must zero-pad the incomplete last batch to exactly batch_size rows."""
    BATCH_SIZE = 3
    NUM_ROWS   = 7
    ELEM_COUNT = 4

    rows = [
        Row(tensor=Row(
            data=make_f32_bytes([float(i + 1)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", TENSOR_SCHEMA, False)])
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
    expected_files = (NUM_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
    assert len(shard_files) == expected_files

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            t = f.get_tensor("tensor")
            assert t.shape == (BATCH_SIZE, ELEM_COUNT)

    last_shard = max(shard_files, key=lambda p: p.name)
    with safetensors.safe_open(str(last_shard), framework="numpy") as f:
        t = f.get_tensor("tensor")
        tail_size = NUM_ROWS % BATCH_SIZE
        padding = t[tail_size:]
        assert (padding == 0).all()


def test_kv_mode_shard_size_respected(spark, tmp_path: Path):
    """KV mode: shards must respect target_shard_size_mb and all rows must be present."""
    TARGET_MB  = 50
    ELEM_COUNT = 13000
    NUM_ROWS   = 1100

    rows = [
        Row(
            key=f"row_{i}",
            tensor=Row(
                data=make_f32_bytes([float(i % 100)] * ELEM_COUNT),
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
            StructField("tensor", TENSOR_SCHEMA, False),
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
        f"got {len(shard_files)}"
    )

    max_allowed_bytes = int(TARGET_MB * 1024 * 1024 * 1.20)
    all_keys = set()
    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)
        size = shard.stat().st_size
        assert size <= max_allowed_bytes, (
            f"KV shard {shard.name} is {size} bytes, exceeds {max_allowed_bytes}"
        )
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            shard_keys = list(f.keys())
            assert len(shard_keys) > 0
            all_keys.update(shard_keys)

    # Every input row must appear in the output
    expected_keys = {f"row_{i}__tensor" for i in range(NUM_ROWS)}
    assert all_keys == expected_keys, (
        f"Missing keys: {expected_keys - all_keys}\nExtra keys: {all_keys - expected_keys}"
    )


def test_f16_write_round_trip(spark, tmp_path: Path):
    """F16 tensors written from Spark must round-trip correctly via Python safetensors."""
    expected_rows = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]
    rows = [Row(floats=row) for row in expected_rows]
    df_raw = spark.createDataFrame(
        rows, StructType([StructField("floats", ArrayType(FloatType()), False)])
    ).coalesce(1)

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
    assert len(shard_files) > 0

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert "tensor" in keys
            t = f.get_tensor("tensor")
            assert t.dtype == np.float16
            assert t.shape == (2, 4)
            expected_flat = np.array(expected_rows, dtype=np.float16).flatten()
            np.testing.assert_array_almost_equal(
                t.flatten().astype(np.float32),
                expected_flat.astype(np.float32),
                decimal=1,
                err_msg="F16 round-trip values mismatch",
            )


def test_bf16_write_round_trip(spark, tmp_path: Path):
    """BF16 tensors written from Spark must round-trip correctly via Python safetensors."""
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
    assert len(shard_files) > 0

    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            keys = list(f.keys())
            assert "tensor" in keys
            t = f.get_tensor("tensor")
            assert t.dtype == ml_dtypes.bfloat16
            assert t.shape == (2, 4)
            t_f32 = t.astype(np.float32)
            expected_flat = np.array(expected_rows).flatten()
            np.testing.assert_allclose(
                t_f32.flatten(),
                expected_flat,
                rtol=1e-2,
                atol=1e-6,
                err_msg="BF16 round-trip values mismatch",
            )


def test_name_col_basic(spark, tmp_path: Path):
    """Test basic name_col mode: each row becomes one tensor with a unique key."""
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
        StructField("tensor", TENSOR_SCHEMA, False),
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

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0

    found_keys = set()
    for shard in shard_files:
        assert SHARD_NAME_RE.match(shard.name)
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            found_keys.update(f.keys())

    assert found_keys == {"row_0__tensor", "row_1__tensor", "row_2__tensor"}, (
        f"Expected exactly 3 keys, got: {found_keys}"
    )


def test_name_col_data_correctness(spark, tmp_path: Path):
    """Test that tensor bytes in name_col mode match the source data."""
    expected_values = {
        "tensor_0__tensor": [1.5, 2.5, 3.5],
        "tensor_1__tensor": [4.0, 5.0],
    }

    rows = [
        Row(key="tensor_0", tensor=Row(data=make_f32_bytes([1.5, 2.5, 3.5]), shape=[3], dtype="F32")),
        Row(key="tensor_1", tensor=Row(data=make_f32_bytes([4.0, 5.0]),       shape=[2], dtype="F32")),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", TENSOR_SCHEMA, False),
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

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    all_tensors = {}
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    assert set(all_tensors.keys()) == set(expected_values.keys()), (
        f"Expected keys {set(expected_values.keys())}, got {set(all_tensors.keys())}"
    )
    for key, expected_vals in expected_values.items():
        np.testing.assert_array_almost_equal(
            all_tensors[key].tolist(), expected_vals,
            err_msg=f"Data mismatch for key '{key}'"
        )


def test_duplicates_strategy_fail(spark, tmp_path: Path):
    """duplicatesStrategy=fail must raise an exception on duplicate tensor keys."""
    rows = [
        Row(key="same_key", tensor=Row(data=make_f32_bytes([1.0, 2.0]), shape=[2], dtype="F32")),
        Row(key="same_key", tensor=Row(data=make_f32_bytes([3.0, 4.0]), shape=[2], dtype="F32")),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", TENSOR_SCHEMA, False),
    ])

    df = spark.createDataFrame(rows, schema).coalesce(1)
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

    full_message = str(exc_info.value) + "".join(
        str(a) for a in exc_info.value.args
    )
    assert any(
        word in full_message for word in ("uplicate", "same_key", "duplicate")
    ), f"Expected duplicate-related exception, got: {exc_info.value}"


def test_duplicates_strategy_last_win(spark, tmp_path: Path):
    """duplicatesStrategy=lastWin must silently keep the last row's tensor."""
    rows = [
        Row(key="shared_key", tensor=Row(data=make_f32_bytes([1.0, 2.0]),    shape=[2], dtype="F32")),
        Row(key="shared_key", tensor=Row(data=make_f32_bytes([99.0, 100.0]), shape=[2], dtype="F32")),
        Row(key="unique_key", tensor=Row(data=make_f32_bytes([5.0, 6.0]),    shape=[2], dtype="F32")),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", TENSOR_SCHEMA, False),
    ])

    df = spark.createDataFrame(rows, schema).coalesce(1)
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

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    all_tensors = {}
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    assert set(all_tensors.keys()) == {"shared_key__tensor", "unique_key__tensor"}
    np.testing.assert_array_almost_equal(all_tensors["shared_key__tensor"], [99.0, 100.0])
    np.testing.assert_array_almost_equal(all_tensors["unique_key__tensor"], [5.0, 6.0])


def test_name_col_multi_column(spark, tmp_path: Path):
    """Test KV mode with multiple non-key tensor columns."""
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
        StructField("weights", TENSOR_SCHEMA, False),
        StructField("bias", TENSOR_SCHEMA, False),
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

    np.testing.assert_array_almost_equal(all_tensors["model_0__weights"], [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(all_tensors["model_0__bias"], [0.01, 0.02])
    np.testing.assert_array_almost_equal(all_tensors["model_1__weights"], [0.4, 0.5])
    np.testing.assert_array_almost_equal(all_tensors["model_1__bias"], [0.03])


def test_name_col_custom_separator(spark, tmp_path: Path):
    """Test KV mode with a custom kv_separator."""
    rows = [
        Row(
            key="row_0",
            tensor=Row(data=make_f32_bytes([1.0, 2.0]), shape=[2], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("key", StringType(), False),
        StructField("tensor", TENSOR_SCHEMA, False),
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

    shard_files = sorted(Path(out_dir).glob("*.safetensors"))
    found_keys = set()
    for shard in shard_files:
        with safetensors.safe_open(str(shard), framework="numpy") as f:
            found_keys.update(f.keys())

    assert "row_0/tensor" in found_keys, "Custom separator '/' should be used in tensor key"


def test_tensor_index_file_name_routing(spark, tmp_path: Path):
    """_tensor_index.parquet file_name values must route to real shard files on disk.

    Distinct tensor_key values must match the written column names exactly.
    Each (tensor_key, file_name) pair must be unique.
    """
    rows = [
        Row(
            tensor_a=Row(data=make_f32_bytes([float(i)]),       shape=[1], dtype="F32"),
            tensor_b=Row(data=make_f32_bytes([float(i + 100)]), shape=[1], dtype="F32"),
        )
        for i in range(3)
    ]

    schema = StructType([
        StructField("tensor_a", TENSOR_SCHEMA, False),
        StructField("tensor_b", TENSOR_SCHEMA, False),
    ])

    df = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "index_routing_output")

    (
        df.write
        .format("safetensors")
        .option("batch_size", "2")
        .option("dtype", "F32")
        .option("generate_index", "true")
        .mode("overwrite")
        .save(out_dir)
    )

    index_path = tmp_path / "index_routing_output" / "_tensor_index.parquet"
    assert index_path.exists()

    index_df = spark.read.parquet(str(index_path))
    rows_collected = index_df.collect()
    assert len(rows_collected) > 0

    shard_names_on_disk = {p.name for p in (tmp_path / "index_routing_output").glob("*.safetensors")}
    tensor_keys = {r["tensor_key"] for r in rows_collected}
    assert tensor_keys == {"tensor_a", "tensor_b"}

    for row in rows_collected:
        assert row["file_name"] in shard_names_on_disk, (
            f"Index file_name '{row['file_name']}' not found on disk"
        )

    # Each (tensor_key, file_name) pair must be unique
    pairs = [(r["tensor_key"], r["file_name"]) for r in rows_collected]
    assert len(pairs) == len(set(pairs)), "Duplicate (tensor_key, file_name) pairs in index"


def test_overwrite_mode_removes_stale_files(spark, tmp_path: Path):
    """A second write with mode=overwrite must not leave stale shard files from the first write."""
    rows_first = [
        Row(tensor=Row(data=make_f32_bytes([float(i)] * 4), shape=[4], dtype="F32"))
        for i in range(6)
    ]
    rows_second = [
        Row(tensor=Row(data=make_f32_bytes([float(i)] * 4), shape=[4], dtype="F32"))
        for i in range(2)
    ]

    schema = StructType([StructField("tensor", TENSOR_SCHEMA, False)])
    out_dir = str(tmp_path / "overwrite_output")

    spark.createDataFrame(rows_first, schema).coalesce(1).write \
        .format("safetensors").option("batch_size", "2").mode("overwrite").save(out_dir)

    first_shards = {p.name for p in Path(out_dir).glob("*.safetensors")}
    assert len(first_shards) == 3

    spark.createDataFrame(rows_second, schema).coalesce(1).write \
        .format("safetensors").option("batch_size", "2").mode("overwrite").save(out_dir)

    second_shards = {p.name for p in Path(out_dir).glob("*.safetensors")}
    assert len(second_shards) == 1, (
        f"After overwrite, expected 1 shard file, got {len(second_shards)}: {second_shards}"
    )
    assert not first_shards.intersection(second_shards) or len(second_shards) == 1, (
        "Stale shard files from first write remain after overwrite"
    )


def test_st_to_array_round_trip(spark, tmp_path: Path):
    """st_to_array must decode a Tensor Struct written by arr_to_st back to the original floats."""
    expected = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    rows = [Row(floats=row) for row in expected]
    df_raw = spark.createDataFrame(
        rows, StructType([StructField("floats", ArrayType(FloatType()), False)])
    )

    df_raw.createOrReplaceTempView("raw_st_to_array")
    df_result = spark.sql(
        "SELECT st_to_array(arr_to_st(floats, array(4), 'F32')) AS decoded FROM raw_st_to_array"
    )

    result_rows = df_result.collect()
    assert len(result_rows) == 2
    for i, row in enumerate(result_rows):
        np.testing.assert_array_almost_equal(
            list(row["decoded"]), expected[i],
            err_msg=f"st_to_array round-trip mismatch at row {i}"
        )


def test_random_f32_array_round_trip(spark, tmp_path: Path):
    """Random F32 arrays generated in Python must survive Spark write → safetensors read unchanged."""
    rng = np.random.default_rng(42)
    NUM_ROWS   = 20
    ELEM_COUNT = 16

    source_arrays = rng.standard_normal((NUM_ROWS, ELEM_COUNT)).astype(np.float32)

    rows = [
        Row(tensor=Row(
            data=source_arrays[i].tobytes(),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df = spark.createDataFrame(
        rows, StructType([StructField("tensor", TENSOR_SCHEMA, False)])
    ).coalesce(1)

    out_dir = str(tmp_path / "random_f32_output")
    (
        df.write.format("safetensors")
        .option("batch_size", str(NUM_ROWS))
        .option("tail_strategy", "write")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) == 1

    with safetensors.safe_open(str(shard_files[0]), framework="numpy") as f:
        t = f.get_tensor("tensor")

    assert t.dtype == np.float32
    assert t.shape == (NUM_ROWS, ELEM_COUNT)
    np.testing.assert_array_equal(
        t, source_arrays,
        err_msg="F32 round-trip: output does not match input exactly"
    )


def test_random_f16_array_round_trip(spark, tmp_path: Path):
    """Random F16 arrays must survive arr_to_st → Spark write → safetensors read.

    The JVM writer uses truncation (round-toward-zero) rather than NumPy's
    default round-to-nearest-even when converting F32 → F16.  We therefore
    compute the expected values with the same truncation logic and allow a
    small absolute tolerance (one F16 epsilon, ~0.001) to absorb any residual
    platform differences across OS / JVM / NumPy versions.
    """
    rng = np.random.default_rng(7)
    NUM_ROWS   = 10
    ELEM_COUNT = 8

    # Generate values in a range that F16 can represent without overflow
    source_f32 = rng.uniform(-10.0, 10.0, (NUM_ROWS, ELEM_COUNT)).astype(np.float32)

    # Expected values: apply the same truncation the JVM writer uses
    expected_f16 = f32_to_f16_truncate(source_f32)

    rows = [Row(floats=source_f32[i].tolist()) for i in range(NUM_ROWS)]
    df_raw = spark.createDataFrame(
        rows, StructType([StructField("floats", ArrayType(FloatType()), False)])
    ).coalesce(1)

    df_raw.createOrReplaceTempView("raw_rand_f16")
    df_tensors = spark.sql(
        f"SELECT arr_to_st(floats, array({ELEM_COUNT}), 'F16') AS tensor FROM raw_rand_f16"
    ).coalesce(1)

    out_dir = str(tmp_path / "random_f16_output")
    (
        df_tensors.write.format("safetensors")
        .option("batch_size", str(NUM_ROWS))
        .option("tail_strategy", "write")
        .mode("overwrite")
        .save(out_dir)
    )

    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) == 1

    with safetensors.safe_open(str(shard_files[0]), framework="numpy") as f:
        t = f.get_tensor("tensor")

    assert t.dtype == np.float16
    assert t.shape == (NUM_ROWS, ELEM_COUNT)

    # Compare against truncation-based expected values with one F16 epsilon of slack
    np.testing.assert_allclose(
        t.astype(np.float32),
        expected_f16.astype(np.float32),
        rtol=0,
        atol=float(np.finfo(np.float16).eps),
        err_msg=(
            "F16 round-trip: output differs from JVM truncation result by more than "
            "one F16 epsilon. This may indicate a rounding-mode mismatch."
        ),
    )


def test_spark_read_back_batch_mode_output(spark, tmp_path: Path):
    """Files written in batch mode must be readable back into Spark with correct schema and row count."""
    from pyspark.sql.types import StructType as ST

    NUM_ROWS   = 4
    ELEM_COUNT = 3

    rows = [
        Row(tensor=Row(
            data=make_f32_bytes([float(i)] * ELEM_COUNT),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    df_write = spark.createDataFrame(
        rows, StructType([StructField("tensor", TENSOR_SCHEMA, False)])
    ).coalesce(1)

    out_dir = str(tmp_path / "read_back_output")
    (
        df_write.write.format("safetensors")
        .option("batch_size", str(NUM_ROWS))
        .option("tail_strategy", "write")
        .mode("overwrite")
        .save(out_dir)
    )

    df_read = spark.read.format("safetensors").option("inferSchema", "true").load(out_dir)

    assert "tensor" in df_read.columns
    assert isinstance(df_read.schema["tensor"].dataType, ST)

    read_rows = df_read.collect()
    assert len(read_rows) == 1, (
        f"Batch mode writes 1 file with 1 stacked tensor row, expected 1 Spark row, got {len(read_rows)}"
    )


def test_spark_read_back_kv_mode_output(spark, tmp_path: Path):
    """Files written in KV mode must be readable back into Spark with inferSchema."""
    rows = [
        Row(key=f"item_{i}", tensor=Row(
            data=make_f32_bytes([float(i), float(i + 1)]),
            shape=[2],
            dtype="F32",
        ))
        for i in range(5)
    ]

    schema = StructType([
        StructField("key",    StringType(),  False),
        StructField("tensor", TENSOR_SCHEMA, False),
    ])

    df_write = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "kv_read_back_output")

    (
        df_write.write.format("safetensors")
        .option("name_col", "key")
        .mode("overwrite")
        .save(out_dir)
    )

    df_read = spark.read.format("safetensors").option("inferSchema", "true").load(out_dir)
    assert len(df_read.columns) > 0
    assert df_read.count() > 0


# ---------------------------------------------------------------------------
# Task 2.1: Multi-column test with 5+ tensor columns of different dtypes
# ---------------------------------------------------------------------------

def test_multi_column_multi_dtype_write(spark, tmp_path: Path):
    """Test writing DataFrame with 5+ tensor columns of different dtypes and shapes."""
    # Create DataFrame with multiple columns of different dtypes
    rows = [
        Row(
            image=Row(data=make_f32_bytes([1.0] * 784), shape=[28, 28], dtype="F32"),
            embeddings=Row(data=make_f32_bytes([0.1] * 512), shape=[512], dtype="F32"),
            attention_mask=Row(data=make_f32_bytes([1.0] * 128), shape=[128], dtype="F32"),
            logits=Row(data=make_f32_bytes([0.5] * 1000), shape=[1000], dtype="F32"),
            labels=Row(data=make_f32_bytes([0.0] * 10), shape=[10], dtype="F32"),
            metadata=Row(data=make_f32_bytes([42.0]), shape=[1], dtype="F32"),
        ),
        Row(
            image=Row(data=make_f32_bytes([2.0] * 784), shape=[28, 28], dtype="F32"),
            embeddings=Row(data=make_f32_bytes([0.2] * 512), shape=[512], dtype="F32"),
            attention_mask=Row(data=make_f32_bytes([1.0] * 128), shape=[128], dtype="F32"),
            logits=Row(data=make_f32_bytes([0.6] * 1000), shape=[1000], dtype="F32"),
            labels=Row(data=make_f32_bytes([1.0] * 10), shape=[10], dtype="F32"),
            metadata=Row(data=make_f32_bytes([43.0]), shape=[1], dtype="F32"),
        ),
    ]

    schema = StructType([
        StructField("image", TENSOR_SCHEMA, False),
        StructField("embeddings", TENSOR_SCHEMA, False),
        StructField("attention_mask", TENSOR_SCHEMA, False),
        StructField("logits", TENSOR_SCHEMA, False),
        StructField("labels", TENSOR_SCHEMA, False),
        StructField("metadata", TENSOR_SCHEMA, False),
    ])

    df = spark.createDataFrame(rows, schema).coalesce(1)
    out_dir = str(tmp_path / "multi_column_output")

    (
        df.write
        .format("safetensors")
        .option("batch_size", "2")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(out_dir)
    )

    # Verify output
    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) == 1, f"Expected 1 shard file, got {len(shard_files)}"

    # Verify all 6 tensor columns are present
    with safetensors.safe_open(str(shard_files[0]), framework="numpy") as f:
        keys = set(f.keys())
        expected_keys = {"image", "embeddings", "attention_mask", "logits", "labels", "metadata"}
        assert keys == expected_keys, f"Expected keys {expected_keys}, got {keys}"

        # Verify shapes
        assert f.get_tensor("image").shape == (2, 28, 28)
        assert f.get_tensor("embeddings").shape == (2, 512)
        assert f.get_tensor("attention_mask").shape == (2, 128)
        assert f.get_tensor("logits").shape == (2, 1000)
        assert f.get_tensor("labels").shape == (2, 10)
        assert f.get_tensor("metadata").shape == (2, 1)


# ---------------------------------------------------------------------------
# Task 2.2: Skewed data distribution test
# ---------------------------------------------------------------------------

def test_skewed_data_distribution(spark, tmp_path: Path):
    """Test handling of skewed data where one partition is 100x larger than others."""
    # Create small partitions
    small_data = [[float(i)] * 10 for i in range(10)]  # 10 rows of 10 elements
    # Create one large partition (100x larger)
    large_data = [[float(i)] * 1000 for i in range(100)]  # 100 rows of 1000 elements

    # Combine all data and create rows properly
    all_data = small_data + large_data
    rows = []
    for data in all_data:
        tensor_row = Row(data=make_f32_bytes(data), shape=[len(data)], dtype="F32")
        rows.append(Row(tensor=tensor_row))

    schema = StructType([StructField("tensor", TENSOR_SCHEMA, False)])

    # Use explicit repartitioning to create skew
    df = spark.createDataFrame(rows, schema).repartition(3)
    out_dir = str(tmp_path / "skewed_data_output")

    (
        df.write
        .format("safetensors")
        .option("batch_size", "10")
        .option("tail_strategy", "write")
        .mode("overwrite")
        .save(out_dir)
    )

    # Verify all data is present
    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) > 0

    # Verify manifest has correct total sample count
    manifest_path = Path(out_dir) / "dataset_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_samples"] == 110  # 10 small + 100 large = 110 total


# ---------------------------------------------------------------------------
# Task 2.3: Schema validation test for incompatible DataFrame schemas
# ---------------------------------------------------------------------------

def test_schema_validation_rejects_incompatible_types(spark, tmp_path: Path):
    """Writer should reject DataFrames with incompatible column types."""
    # Try to write a DataFrame with string column (not supported)
    rows = [
        Row(text="not a tensor"),
    ]
    schema = StructType([StructField("text", StringType(), False)])

    df = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "invalid_schema_output")

    with pytest.raises(Exception) as exc_info:
        (
            df.write
            .format("safetensors")
            .option("batch_size", "1")
            .mode("overwrite")
            .save(out_dir)
        )

    # Should get an error about unsupported column type
    msg = str(exc_info.value).lower()
    assert any(word in msg for word in ["unsupported", "string", "text", "invalid"])


def test_schema_validation_requires_dtype_for_arrays(spark, tmp_path: Path):
    """Writer should require dtype option when writing numeric arrays."""
    rows = [
        Row(values=[1.0, 2.0, 3.0]),
    ]
    schema = StructType([StructField("values", ArrayType(FloatType()), False)])

    df = spark.createDataFrame(rows, schema)
    out_dir = str(tmp_path / "no_dtype_output")

    # Without dtype option, should fail
    with pytest.raises(Exception) as exc_info:
        (
            df.write
            .format("safetensors")
            .option("batch_size", "1")
            .mode("overwrite")
            .save(out_dir)
        )

    msg = str(exc_info.value).lower()
    assert "dtype" in msg


# ---------------------------------------------------------------------------
# Task 2.4: Round-trip validation test - bit-for-bit identical data
# ---------------------------------------------------------------------------

def test_roundtrip_bit_for_bit_identical(spark, tmp_path: Path):
    """Write then read back must produce bit-for-bit identical data."""
    # Generate random F32 data
    rng = np.random.default_rng(12345)
    NUM_ROWS = 5
    ELEM_COUNT = 16

    source_arrays = rng.standard_normal((NUM_ROWS, ELEM_COUNT)).astype(np.float32)

    rows = [
        Row(tensor=Row(
            data=source_arrays[i].tobytes(),
            shape=[ELEM_COUNT],
            dtype="F32",
        ))
        for i in range(NUM_ROWS)
    ]

    schema = StructType([StructField("tensor", TENSOR_SCHEMA, False)])
    df_write = spark.createDataFrame(rows, schema).coalesce(1)

    out_dir = str(tmp_path / "roundtrip_output")
    (
        df_write.write
        .format("safetensors")
        .option("batch_size", str(NUM_ROWS))
        .option("tail_strategy", "write")
        .mode("overwrite")
        .save(out_dir)
    )

    # Read back with Spark
    df_read = spark.read.format("safetensors").option("inferSchema", "true").load(out_dir)

    # Verify row count and schema
    assert df_read.count() == 1  # Single batch file produces 1 row
    assert "tensor" in df_read.columns

    # Read the raw safetensors file with Python and verify bit-for-bit match
    shard_files = list(Path(out_dir).glob("*.safetensors"))
    assert len(shard_files) == 1

    with safetensors.safe_open(str(shard_files[0]), framework="numpy") as f:
        tensor = f.get_tensor("tensor")
        assert tensor.dtype == np.float32
        assert tensor.shape == (NUM_ROWS, ELEM_COUNT)
        # Bit-for-bit comparison
        np.testing.assert_array_equal(
            tensor,
            source_arrays,
            err_msg="Round-trip data mismatch: output differs from input"
        )


def test_roundtrip_all_dtypes(spark, tmp_path: Path):
    """Round-trip test covering all supported dtypes."""
    dtypes_to_test = [
        ("F64", np.float64),
        ("F32", np.float32),
        ("F16", np.float16),
        ("I64", np.int64),
        ("I32", np.int32),
        ("I16", np.int16),
        ("I8", np.int8),
        ("U8", np.uint8),
    ]

    for dtype_name, np_dtype in dtypes_to_test:
        # Generate test data
        if np_dtype in [np.float32, np.float64, np.float16]:
            data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np_dtype)
        else:
            data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype)

        rows = []
        for i in range(2):
            tensor_row = Row(data=data[i].tobytes(), shape=[3], dtype=dtype_name)
            rows.append(Row(tensor=tensor_row))

        schema = StructType([StructField("tensor", TENSOR_SCHEMA, False)])
        df = spark.createDataFrame(rows, schema).coalesce(1)

        out_dir = str(tmp_path / f"roundtrip_{dtype_name.lower()}_output")
        (
            df.write
            .format("safetensors")
            .option("batch_size", "2")
            .mode("overwrite")
            .save(out_dir)
        )

        # Read back and verify
        shard_files = list(Path(out_dir).glob("*.safetensors"))
        assert len(shard_files) == 1, f"Expected 1 file for {dtype_name}"

        with safetensors.safe_open(str(shard_files[0]), framework="numpy") as f:
            tensor = f.get_tensor("tensor")
            assert tensor.shape == (2, 3), f"Shape mismatch for {dtype_name}"

            # For float types, do approximate comparison due to conversion
            if np_dtype in [np.float32, np.float64]:
                np.testing.assert_allclose(tensor, data)
            else:
                # For integer types, exact match
                np.testing.assert_array_equal(tensor, data)
