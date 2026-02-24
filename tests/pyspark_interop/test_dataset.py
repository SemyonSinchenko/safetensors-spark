"""
Integration test: DistributedSafetensorsDataset

Tests for reading dataset metadata, listing shards, and assigning shards
across distributed workers.

See §5.3 of AGENTS.md.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest


@pytest.fixture()
def written_dataset(spark, tmp_path: Path):
    """Write a small safetensors dataset and return the output path."""
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

    def f32(vals):
        return struct.pack(f"<{len(vals)}f", *vals)

    rows = [
        Row(x=Row(data=f32([1.0, 2.0]), shape=[1, 2], dtype="F32")),
        Row(x=Row(data=f32([3.0, 4.0]), shape=[1, 2], dtype="F32")),
        Row(x=Row(data=f32([5.0, 6.0]), shape=[1, 2], dtype="F32")),
    ]

    schema = StructType([StructField("x", tensor_schema, False)])
    df = spark.createDataFrame(rows, schema)

    out_dir = str(tmp_path / "dataset_test")

    (
        df.write.format("safetensors")
        .option("batch_size", "3")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(out_dir)
    )

    return out_dir


def test_dataset_from_path(written_dataset):
    """DistributedSafetensorsDataset.from_path() must load a dataset."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    assert ds is not None
    assert ds._path == written_dataset


def test_dataset_list_shards(written_dataset):
    """DistributedSafetensorsDataset.list_shards() must return shard info."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    shards = ds.list_shards()

    assert len(shards) > 0, "Dataset must have at least one shard"
    for shard in shards:
        assert shard.shard_path is not None
        assert shard.num_samples >= 0
        assert shard.num_tensors >= 0


def test_dataset_get_schema(written_dataset):
    """DistributedSafetensorsDataset.get_schema() must return tensor schema."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    schema = ds.get_schema()

    assert isinstance(schema, dict)
    assert "x" in schema, "Schema must contain tensor 'x' from test data"
    assert schema["x"]["dtype"] == "F32"


def test_dataset_assign_shards_round_robin(written_dataset):
    """assign_shards(strategy='round_robin') must distribute evenly."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    num_shards = len(ds.list_shards())

    assignments = ds.assign_shards(num_workers=3, strategy="round_robin")

    assert len(assignments) == 3
    total_assigned = sum(len(v) for v in assignments.values())
    assert total_assigned == num_shards, "All shards must be assigned"


def test_dataset_assign_shards_balance(written_dataset):
    """assign_shards(strategy='balance') must use load balancing."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    num_shards = len(ds.list_shards())

    assignments = ds.assign_shards(num_workers=2, strategy="balance")

    assert len(assignments) == 2
    total_assigned = sum(len(v) for v in assignments.values())
    assert total_assigned == num_shards, "All shards must be assigned"

    # With balance strategy, sample count should be roughly equal
    worker_samples = [
        sum(s.num_samples for s in shards) for shards in assignments.values()
    ]
    # Allows for imbalance due to shard size boundaries
    max_diff = max(worker_samples) - min(worker_samples)
    assert max_diff <= max(1, max(worker_samples) // 4), (
        f"Balance strategy should distribute fairly, got {worker_samples}"
    )


def test_dataset_assign_shards_invalid_strategy(written_dataset):
    """assign_shards() with invalid strategy must raise ValueError."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)

    with pytest.raises(ValueError, match="strategy must be"):
        ds.assign_shards(num_workers=2, strategy="unknown")


def test_dataset_assign_shards_invalid_workers(written_dataset):
    """assign_shards() with invalid num_workers must raise ValueError."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)

    with pytest.raises(ValueError, match="num_workers must be"):
        ds.assign_shards(num_workers=0)


def test_dataset_validate(written_dataset):
    """DistributedSafetensorsDataset.validate() must return check dict."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    checks = ds.validate()

    assert isinstance(checks, dict)
    assert "has_manifest" in checks
    assert "has_shards" in checks
    assert "shards_exist" in checks
    assert all(isinstance(v, bool) for v in checks.values())


def test_dataset_describe(written_dataset):
    """DistributedSafetensorsDataset.describe() must return formatted summary."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    description = ds.describe()

    assert isinstance(description, str)
    assert "DistributedSafetensorsDataset" in description
    assert "Shards:" in description
    assert "Total samples:" in description
    assert "Tensors:" in description


def test_dataset_pickle_serialization(written_dataset):
    """DistributedSafetensorsDataset must be picklable for distributed tasks."""
    import pickle
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)

    # Pickle and unpickle
    pickled = pickle.dumps(ds)
    ds2 = pickle.loads(pickled)

    # Restored object should work
    assert ds2._path == ds._path
    shards2 = ds2.list_shards()
    assert len(shards2) > 0


def test_dataset_from_path_missing_manifest():
    """from_path() must raise FileNotFoundError if manifest is absent."""
    import sys
    from pathlib import Path
    import tempfile

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FileNotFoundError, match="dataset_manifest.json"):
            DistributedSafetensorsDataset.from_path(tmp_dir)


def test_dataset_get_index(written_dataset):
    """DistributedSafetensorsDataset.get_index() must return index or None."""
    import sys

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root / "python"))
    from safetensors_spark.dataset import DistributedSafetensorsDataset

    ds = DistributedSafetensorsDataset.from_path(written_dataset)
    index = ds.get_index()

    # Index is optional; None is acceptable if not present
    # If present, it should be a pyarrow.Table
    if index is not None:
        import pyarrow as pa

        assert isinstance(index, pa.Table)
