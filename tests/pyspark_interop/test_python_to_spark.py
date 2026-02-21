"""
Integration test: Python → Spark

Generates .safetensors files using the official HuggingFace safetensors Python
library + numpy, then reads them back using spark.read.format("safetensors").

Assertions:
  - Schema matches Tensor Struct (data: binary, shape: array<int>, dtype: string)
    per tensor column.
  - Decoded values (via st_to_array SQL function) match original numpy arrays
    within float32 tolerance.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def simple_safetensors_file(tmp_path: Path) -> tuple[Path, dict]:
    """Write a simple .safetensors file and return (path, {name: np_array})."""
    from safetensors.numpy import save_file

    arrays = {
        "weight": np.random.rand(3, 4).astype(np.float32),
        "bias":   np.random.rand(4).astype(np.float32),
    }
    out_file = tmp_path / "test.safetensors"
    save_file(arrays, str(out_file))
    return out_file, arrays


def test_schema_matches_tensor_struct(spark, simple_safetensors_file):
    """Each tensor column must be a struct with data/shape/dtype fields."""
    file_path, arrays = simple_safetensors_file

    df = (
        spark.read.format("safetensors")
        .option("inferSchema", "true")
        .load(str(file_path))
    )

    from pyspark.sql.types import BinaryType, ArrayType, IntegerType, StringType, StructType

    for tensor_name in arrays:
        assert tensor_name in df.schema.fieldNames(), \
            f"Expected column '{tensor_name}' in schema"
        field = df.schema[tensor_name]
        assert isinstance(field.dataType, StructType), \
            f"Column '{tensor_name}' should be StructType"
        sub_fields = {f.name: f.dataType for f in field.dataType.fields}
        assert "data"  in sub_fields and isinstance(sub_fields["data"],  BinaryType)
        assert "shape" in sub_fields and isinstance(sub_fields["shape"], ArrayType)
        assert "dtype" in sub_fields and isinstance(sub_fields["dtype"], StringType)


def test_decoded_values_match_numpy(spark, simple_safetensors_file):
    """Decoded float values must match the original numpy arrays via st_to_array() SQL function."""
    file_path, arrays = simple_safetensors_file

    df = (
        spark.read.format("safetensors")
        .option("inferSchema", "true")
        .load(str(file_path))
    )

    # Register the DataFrame as a temp table and use st_to_array() SQL function
    df.createOrReplaceTempView("safetensors_data")

    for tensor_name, np_array in arrays.items():
        # Use st_to_array() to decode the tensor struct (exercises Catalyst expression)
        result_df = spark.sql(f"SELECT st_to_array({tensor_name}) as decoded FROM safetensors_data")
        rows = result_df.collect()
        assert len(rows) == 1, "Each safetensors file should produce exactly one row"

        decoded_array = rows[0]["decoded"]
        # st_to_array returns a List[Float], convert to numpy for comparison
        decoded_floats = np.array(decoded_array, dtype=np.float32)

        np.testing.assert_allclose(
            decoded_floats,
            np_array.flatten(),
            rtol=1e-5,
            err_msg=f"Mismatch in tensor '{tensor_name}' decoded via st_to_array()",
        )


def _build_safetensors_bytes(tensor_name: str, dtype_str: str, shape: list, raw_bytes: bytes) -> bytes:
    """
    Hand-craft a minimal safetensors binary without any Python library dependency.

    Format (format/format.md):
      - 8 bytes: N (LE uint64) = JSON header byte length
      - N bytes: UTF-8 JSON header starting with '{'
      - raw tensor bytes
    """
    import json as _json

    data_offsets = [0, len(raw_bytes)]
    header_obj = {
        tensor_name: {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": data_offsets,
        }
    }
    header_json = _json.dumps(header_obj).encode("utf-8")
    header_len  = len(header_json)

    import struct as _struct

    prefix = _struct.pack("<Q", header_len)  # 8-byte LE uint64
    return prefix + header_json + raw_bytes


def test_bf16_dtype_preserved(spark, tmp_path: Path):
    """BF16 tensors must be read with dtype='BF16' and bytes preserved exactly.

    We craft the .safetensors binary manually to avoid requiring torch.bfloat16.
    The three known BF16 bit patterns represent 1.0, 2.0, and 3.0.
    """
    import struct

    # BF16 values: 1.0 = 0x3F80, 2.0 = 0x4000, 3.0 = 0x4040 (LE 16-bit each)
    bf16_values = [0x3F80, 0x4000, 0x4040]
    raw_bytes   = struct.pack("<3H", *bf16_values)  # 6 bytes total

    file_path = tmp_path / "bf16_test.safetensors"
    file_path.write_bytes(
        _build_safetensors_bytes("embedding", "BF16", [3], raw_bytes)
    )

    df = (
        spark.read.format("safetensors")
        .option("inferSchema", "true")
        .load(str(file_path))
    )

    rows = df.collect()
    assert len(rows) == 1, "One row per file"
    row = rows[0]

    tensor = row.embedding
    assert tensor.dtype == "BF16", f"Expected dtype BF16, got {tensor.dtype}"
    assert tensor.shape == [3], f"Expected shape [3], got {tensor.shape}"

    # Verify raw bytes are preserved exactly
    actual_bytes = bytes(tensor.data)
    assert actual_bytes == raw_bytes, (
        f"BF16 bytes not preserved. "
        f"Expected {raw_bytes.hex()}, got {actual_bytes.hex()}"
    )
