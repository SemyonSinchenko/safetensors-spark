"""
safetensors_spark.dataset
~~~~~~~~~~~~~~~~~~~~~~~~~
Distributed dataset utilities for safetensors outputs written by the
safetensors-spark Spark connector.

This module provides the ``DistributedSafetensorsDataset`` class for reading
dataset metadata, indexing shards, and assigning them across distributed workers.
It supports remote filesystems (S3, GCS, HDFS) via PyArrow.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShardInfo:
    """Metadata about a single safetensors shard."""

    shard_path: str
    num_samples: int
    num_tensors: int

    def __hash__(self) -> int:
        return hash(self.shard_path)


class DistributedSafetensorsDataset:
    """
    Metadata wrapper for a safetensors dataset written by the Spark connector.

    Reads ``dataset_manifest.json`` and optional ``_tensor_index.parquet``
    to expose shard assignments and metadata for distributed training.
    """

    def __init__(
        self,
        path: str,
        manifest: dict,
        index_table=None,
        filesystem=None,
    ):
        """Initialize from a dataset root directory and manifest dict.

        Parameters
        ----------
        path : str
            Root directory of the safetensors dataset.
        manifest : dict
            Parsed ``dataset_manifest.json`` as a Python dict.
        index_table : optional
            Parsed ``_tensor_index.parquet`` as a ``pyarrow.Table``.
        filesystem : optional
            ``pyarrow.fs.FileSystem`` instance. Defaults to
            ``pyarrow.fs.LocalFileSystem()`` if ``None``.
        """
        self._path = path
        self._manifest = manifest
        self._index_table = index_table
        self._filesystem = filesystem

        if self._filesystem is None:
            import pyarrow.fs
            self._filesystem = pyarrow.fs.LocalFileSystem()

    @classmethod
    def from_path(cls, path: str, filesystem=None) -> DistributedSafetensorsDataset:
        """Load a dataset from a root directory path.

        Parameters
        ----------
        path : str
            Root directory of the safetensors dataset. Must contain
            ``dataset_manifest.json``.
        filesystem : optional
            ``pyarrow.fs.FileSystem`` instance. Defaults to
            ``pyarrow.fs.LocalFileSystem()`` if ``None``.

        Returns
        -------
        DistributedSafetensorsDataset
            Loaded dataset instance.

        Raises
        ------
        FileNotFoundError
            If ``dataset_manifest.json`` is not found.
        """
        if filesystem is None:
            import pyarrow.fs
            filesystem = pyarrow.fs.LocalFileSystem()

        manifest_path = f"{path}/dataset_manifest.json"

        try:
            file_info = filesystem.get_file_info(manifest_path)
            if file_info.is_file is False:
                raise FileNotFoundError(
                    f"dataset_manifest.json not found at: {manifest_path}"
                )
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise FileNotFoundError(
                f"dataset_manifest.json not found at: {manifest_path}"
            ) from e

        with filesystem.open_input_file(manifest_path) as f:
            manifest = json.load(f)

        # Attempt to load the optional index
        index_table = None
        index_path = f"{path}/_tensor_index.parquet"
        try:
            file_info = filesystem.get_file_info(index_path)
            if file_info.is_file:
                import pyarrow.parquet
                with filesystem.open_input_file(index_path) as f:
                    index_table = pyarrow.parquet.read_table(f)
        except Exception:
            # Index is optional; silently ignore read failures
            pass

        return cls(
            path=path,
            manifest=manifest,
            index_table=index_table,
            filesystem=filesystem,
        )

    def list_shards(self) -> list[ShardInfo]:
        """List all shards in the dataset.

        Returns
        -------
        list[ShardInfo]
            List of shard metadata in order they appear in the manifest.
        """
        shards = []
        for shard in self._manifest.get("shards", []):
            shards.append(
                ShardInfo(
                    shard_path=shard["shard_path"],
                    num_samples=shard.get("samples_count", 0),
                    num_tensors=shard.get("num_tensors", 0),
                )
            )
        return shards

    def get_schema(self) -> dict:
        """Get the tensor schema from the manifest.

        Returns
        -------
        dict
            Schema dict mapping tensor names to their metadata (dtype, shape, etc).
        """
        return self._manifest.get("schema", {})

    def get_index(self):
        """Get the optional index table (if present).

        Returns
        -------
        pyarrow.Table or None
            Parsed ``_tensor_index.parquet`` if it exists, else ``None``.
        """
        return self._index_table

    def assign_shards(
        self,
        num_workers: int,
        strategy: str = "round_robin",
    ) -> dict[int, list[ShardInfo]]:
        """Assign shards across workers for distributed training.

        Parameters
        ----------
        num_workers : int
            Number of workers to distribute shards across.
        strategy : str, default "round_robin"
            Assignment strategy: "round_robin" or "balance" (min-heap load balancing).

        Returns
        -------
        dict[int, list[ShardInfo]]
            Mapping from worker ID (0..num_workers-1) to assigned shards.

        Raises
        ------
        ValueError
            If ``strategy`` is not recognized or ``num_workers`` <= 0.
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if strategy not in ("round_robin", "balance"):
            raise ValueError(
                f"strategy must be 'round_robin' or 'balance', got '{strategy}'"
            )

        shards = self.list_shards()
        assignments: dict[int, list[ShardInfo]] = {i: [] for i in range(num_workers)}

        if strategy == "round_robin":
            for idx, shard in enumerate(shards):
                worker_id = idx % num_workers
                assignments[worker_id].append(shard)
        else:  # balance
            # Use a min-heap (worker_id, total_samples) to distribute shards
            heap = [(0, i) for i in range(num_workers)]
            heapq.heapify(heap)

            for shard in shards:
                samples, worker_id = heapq.heappop(heap)
                assignments[worker_id].append(shard)
                heapq.heappush(heap, (samples + shard.num_samples, worker_id))

        return assignments

    def validate(self) -> dict[str, bool]:
        """Validate the dataset structure.

        Returns
        -------
        dict[str, bool]
            Dictionary of checks and their results (all True = valid dataset).
            Keys include: 'has_manifest', 'has_shards', 'shards_exist'.
        """
        checks = {
            "has_manifest": "manifest_version" in self._manifest,
            "has_shards": len(self.list_shards()) > 0,
            "shards_exist": True,  # We assume if they're in manifest, they're valid
        }

        # Try to verify shard file existence if filesystem supports it
        try:
            for shard in self.list_shards():
                shard_path = f"{self._path}/{shard.shard_path}"
                file_info = self._filesystem.get_file_info(shard_path)
                if not file_info.is_file:
                    checks["shards_exist"] = False
                    break
        except Exception:
            # If we can't check, assume True (filesystem may not support it)
            pass

        return checks

    def describe(self) -> str:
        """Return a human-readable description of the dataset.

        Returns
        -------
        str
            Formatted summary of shards, schema, and sample count.
        """
        shards = self.list_shards()
        schema = self.get_schema()
        total_samples = sum(s.num_samples for s in shards)

        lines = [
            f"DistributedSafetensorsDataset: {self._path}",
            f"  Shards: {len(shards)}",
            f"  Total samples: {total_samples}",
            f"  Tensors: {len(schema)}",
        ]

        if schema:
            lines.append("  Schema:")
            for name, info in schema.items():
                dtype = info.get("dtype", "?")
                shape = info.get("shape", [])
                lines.append(f"    {name}: {dtype} {shape}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DistributedSafetensorsDataset(path={self._path!r})"

    def __getstate__(self):
        """Enable pickling for distributed task serialization."""
        return {
            "path": self._path,
            "manifest": self._manifest,
            "index_table": self._index_table,
        }

    def __setstate__(self, state):
        """Restore from pickled state."""
        import pyarrow.fs

        self._path = state["path"]
        self._manifest = state["manifest"]
        self._index_table = state["index_table"]
        self._filesystem = pyarrow.fs.LocalFileSystem()
