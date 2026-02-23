"""
Shared pytest fixtures for the safetensors-spark PySpark integration tests.

Provides:
  - spark:       A local SparkSession with the safetensors connector JAR on the classpath.
  - tmp_path:    Standard pytest tmp_path fixture (automatically available).
  - jar_path:    Path to the assembled safetensors-spark fat JAR.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
CONNECTOR_JAR_ENV = "SAFETENSORS_SPARK_JAR"


def _find_connector_jar() -> str:
    """Locate the assembled fat JAR. Can be overridden with env var."""
    if jar := os.environ.get(CONNECTOR_JAR_ENV):
        return jar

    # Look for the assembled JAR in the sbt target directory
    target = REPO_ROOT / "target"
    jars = list(target.glob("scala-*/safetensors-spark-assembly-*.jar"))
    if not jars:
        # Fall back to unassembled JAR (no assembly suffix)
        jars = list(target.glob("scala-*/safetensors-spark_*.jar"))

    if not jars:
        pytest.skip(
            f"No safetensors-spark JAR found under {target}. "
            "Run 'sbt assembly' or set SAFETENSORS_SPARK_JAR env var."
        )

    # Pick the most recently modified JAR
    return str(sorted(jars, key=lambda p: p.stat().st_mtime)[-1])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def jar_path() -> str:
    return _find_connector_jar()


@pytest.fixture(scope="session")
def spark(jar_path):
    """Session-scoped local SparkSession with the connector JAR loaded."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("safetensors-spark-tests")
        .config("spark.jars", jar_path)
        .config(
            "spark.sql.extensions",
            "io.github.semyonsinchenko.safetensors.SafetensorsExtensions",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()
