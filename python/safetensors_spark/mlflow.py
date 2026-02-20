"""
safetensors_spark.mlflow
~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight MLflow lineage integration for safetensors datasets written by
the safetensors-spark Spark connector.

This module is pure Python — it has no dependency on Apache Spark or the
Scala connector. It reads dataset_manifest.json directly from the output
path using standard Python file I/O.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def log_dataset(
    path: str,
    run_id: Optional[str] = None,
    name: str = "safetensors_dataset",
) -> None:
    """Log the dataset_manifest.json at ``path`` as an MLflow dataset artifact.

    Reads ``dataset_manifest.json`` from ``path``, constructs an
    ``mlflow.data.Dataset``, and calls ``mlflow.log_input()`` on the active
    (or specified) run.

    Parameters
    ----------
    path:
        Root directory of the safetensors dataset. Must contain a
        ``dataset_manifest.json`` file written by the Spark connector.
    run_id:
        MLflow run ID to log to. If ``None``, uses the currently active run
        (i.e. the run opened by ``mlflow.start_run()``).
    name:
        Display name for the dataset artifact in MLflow UI.

    Raises
    ------
    FileNotFoundError
        If ``dataset_manifest.json`` is not found at ``path``.
    RuntimeError
        If no active MLflow run exists and ``run_id`` is not provided.
    """
    import mlflow
    from mlflow.data.dataset_source import DatasetSource

    manifest_path = os.path.join(path, "dataset_manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"dataset_manifest.json not found at: {manifest_path}\n"
            "Run a safetensors write operation first to generate the manifest."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Build a custom dataset source pointing to the manifest
    source = _SafetensorsDatasetSource(uri=path, manifest=manifest)

    dataset = mlflow.data.from_json(
        path=manifest_path,
        source=source,
        name=name,
    )

    client = mlflow.MlflowClient()
    effective_run_id = run_id or _get_active_run_id()

    client.log_inputs(
        run_id=effective_run_id,
        datasets=[mlflow.data.DatasetInput(dataset=dataset)],
    )


def _get_active_run_id() -> str:
    """Return the ID of the currently active MLflow run, or raise RuntimeError."""
    import mlflow

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError(
            "No active MLflow run. Either call mlflow.start_run() before "
            "log_dataset(), or pass run_id= explicitly."
        )
    return active_run.info.run_id


class _SafetensorsDatasetSource(DatasetSource):
    """Minimal MLflow DatasetSource that wraps a safetensors output directory."""

    def __init__(self, uri: str, manifest: dict) -> None:
        self._uri = uri
        self._manifest = manifest

    @property
    def uri(self) -> str:
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return "safetensors"

    def load(self, dst_path: Optional[str] = None):
        raise NotImplementedError(
            "Loading safetensors datasets via MLflow source is not supported. "
            "Use spark.read.format('safetensors') directly."
        )

    def to_dict(self) -> dict:
        return {
            "source_type": "safetensors",
            "uri": self._uri,
            "manifest": self._manifest,
        }

    @classmethod
    def from_dict(cls, source_dict: dict) -> "_SafetensorsDatasetSource":
        return cls(
            uri=source_dict["uri"],
            manifest=source_dict.get("manifest", {}),
        )
