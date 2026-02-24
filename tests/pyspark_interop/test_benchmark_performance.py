"""
Performance benchmarks for safetensors-spark vs Parquet baseline.

Benchmarks measure:
  - Write throughput (rows/sec, MB/sec)
  - File size on disk (compression efficiency)
  - Read-back latency

Uses pytest-benchmark for statistical rigor and comparison.

Set the environment variable RUN_BENCHMARKS=1 to run these benchmarks.
They are skipped by default so they do not run in CI.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest


_RUN_BENCHMARKS = os.environ.get("RUN_BENCHMARKS", "0") == "1"
_SKIP_REASON = "Set RUN_BENCHMARKS=1 to run performance benchmarks"


def _get_total_size(path: Path) -> int:
    """Get total size in bytes of a directory or file."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _generate_tensor_data(n_rows: int, n_cols: int, dtype: str = "float32") -> dict:
    """Generate sample tensor data with specified dimensions."""
    data = {}
    for i in range(n_cols):
        col_name = f"tensor_{i}"
        if dtype in ("float32", "float64"):
            data[col_name] = np.random.randn(n_rows, 128).astype(dtype)
        elif dtype in ("int32", "int64"):
            data[col_name] = np.random.randint(0, 1000, (n_rows, 128)).astype(dtype)
        else:
            data[col_name] = np.random.randn(n_rows, 128).astype("float32")
    return data


# -----------------------------------------------------------------------------
# Benchmark 4.2: Parquet baseline comparison
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _RUN_BENCHMARKS, reason=_SKIP_REASON)
@pytest.mark.benchmark(
    group="write_comparison",
    min_rounds=3,
    max_time=30,
    warmup=True,
)
@pytest.mark.parametrize("n_rows", [1000, 10000, 100000])
def test_safetensors_write_speed(benchmark, spark, tmp_path: Path, n_rows: int):
    """Benchmark safetensors write performance for varying data sizes."""
    # Generate test data
    data = _generate_tensor_data(n_rows, 1)
    
    # Create DataFrame
    rows = [
        {f"tensor_0": data["tensor_0"][i].tobytes()}
        for i in range(n_rows)
    ]
    df = spark.createDataFrame(rows)
    
    output_path = tmp_path / "safetensors_output"
    
    def write_safetensors():
        df.write.format("safetensors").option(
            "batch_size", "500"
        ).mode("overwrite").save(str(output_path))
        # Force filesystem sync
        return _get_total_size(output_path)
    
    result = benchmark(write_safetensors)
    
    # Store metadata for reporting
    benchmark.extra_info["format"] = "safetensors"
    benchmark.extra_info["rows"] = n_rows
    benchmark.extra_info["file_size_bytes"] = result
    benchmark.extra_info["throughput_rows_per_sec"] = n_rows / benchmark.stats["mean"]


@pytest.mark.skipif(not _RUN_BENCHMARKS, reason=_SKIP_REASON)
@pytest.mark.benchmark(
    group="write_comparison",
    min_rounds=3,
    max_time=30,
    warmup=True,
)
@pytest.mark.parametrize("n_rows", [1000, 10000, 100000])
def test_parquet_write_speed(benchmark, spark, tmp_path: Path, n_rows: int):
    """Benchmark Parquet write performance for baseline comparison."""
    # Generate same test data as safetensors test
    data = _generate_tensor_data(n_rows, 1)
    
    # Create DataFrame with binary column (comparable to safetensors)
    rows = [
        {f"tensor_0": data["tensor_0"][i].tobytes()}
        for i in range(n_rows)
    ]
    df = spark.createDataFrame(rows)
    
    output_path = tmp_path / "parquet_output"
    
    def write_parquet():
        df.write.format("parquet").mode("overwrite").save(str(output_path))
        return _get_total_size(output_path)
    
    result = benchmark(write_parquet)
    
    benchmark.extra_info["format"] = "parquet"
    benchmark.extra_info["rows"] = n_rows
    benchmark.extra_info["file_size_bytes"] = result
    benchmark.extra_info["throughput_rows_per_sec"] = n_rows / benchmark.stats["mean"]


# -----------------------------------------------------------------------------
# Benchmark 4.3: Multi-column benchmark
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _RUN_BENCHMARKS, reason=_SKIP_REASON)
@pytest.mark.benchmark(
    group="multi_column",
    min_rounds=3,
    max_time=30,
    warmup=True,
)
@pytest.mark.parametrize("n_cols", [1, 5, 10, 20])
def test_safetensors_multi_column_write(benchmark, spark, tmp_path: Path, n_cols: int):
    """Benchmark safetensors write with varying column counts."""
    n_rows = 10000  # Fixed row count, varying columns
    data = _generate_tensor_data(n_rows, n_cols)
    
    # Create DataFrame with multiple columns
    rows = [
        {
            f"tensor_{i}": data[f"tensor_{i}"][row_idx].tobytes()
            for i in range(n_cols)
        }
        for row_idx in range(n_rows)
    ]
    df = spark.createDataFrame(rows)
    
    output_path = tmp_path / f"safetensors_{n_cols}cols"
    
    def write_multi_col():
        df.write.format("safetensors").format("safetensors").option(
            "batch_size", "500"
        ).mode("overwrite").save(str(output_path))
        return _get_total_size(output_path)
    
    result = benchmark(write_multi_col)
    
    benchmark.extra_info["format"] = "safetensors"
    benchmark.extra_info["columns"] = n_cols
    benchmark.extra_info["rows"] = n_rows
    benchmark.extra_info["file_size_bytes"] = result


@pytest.mark.skipif(not _RUN_BENCHMARKS, reason=_SKIP_REASON)
@pytest.mark.benchmark(
    group="multi_column",
    min_rounds=3,
    max_time=30,
    warmup=True,
)
@pytest.mark.parametrize("n_cols", [1, 5, 10, 20])
def test_parquet_multi_column_write(benchmark, spark, tmp_path: Path, n_cols: int):
    """Benchmark Parquet write with varying column counts for comparison."""
    n_rows = 10000  # Fixed row count, varying columns
    data = _generate_tensor_data(n_rows, n_cols)
    
    # Create DataFrame with multiple columns
    rows = [
        {
            f"tensor_{i}": data[f"tensor_{i}"][row_idx].tobytes()
            for i in range(n_cols)
        }
        for row_idx in range(n_rows)
    ]
    df = spark.createDataFrame(rows)
    
    output_path = tmp_path / f"parquet_{n_cols}cols"
    
    def write_multi_col():
        df.write.format("parquet").mode("overwrite").save(str(output_path))
        return _get_total_size(output_path)
    
    result = benchmark(write_multi_col)
    
    benchmark.extra_info["format"] = "parquet"
    benchmark.extra_info["columns"] = n_cols
    benchmark.extra_info["rows"] = n_rows
    benchmark.extra_info["file_size_bytes"] = result


# -----------------------------------------------------------------------------
# Benchmark 4.4 & 4.5: File size comparison
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _RUN_BENCHMARKS, reason=_SKIP_REASON)
def test_file_size_comparison(spark, tmp_path: Path):
    """Compare file sizes between safetensors and Parquet for identical data."""
    n_rows = 100000
    n_cols = 5
    
    data = _generate_tensor_data(n_rows, n_cols)
    
    # Create DataFrame
    rows = [
        {
            f"tensor_{i}": data[f"tensor_{i}"][row_idx].tobytes()
            for i in range(n_cols)
        }
        for row_idx in range(n_rows)
    ]
    df = spark.createDataFrame(rows)
    
    # Write safetensors
    st_path = tmp_path / "size_test_safetensors"
    df.write.format("safetensors").option(
            "batch_size", "500"
    ).mode("overwrite").save(str(st_path))
    st_size = _get_total_size(st_path)
    
    # Write Parquet
    pq_path = tmp_path / "size_test_parquet"
    df.write.format("parquet").mode("overwrite").save(str(pq_path))
    pq_size = _get_total_size(pq_path)
    
    # Calculate compression ratios
    raw_size = n_rows * n_cols * 128 * 4  # 128 floats per tensor, 4 bytes each
    
    print(f"\n{'='*60}")
    print(f"File Size Comparison ({n_rows} rows, {n_cols} columns)")
    print(f"{'='*60}")
    print(f"Raw data size:        {raw_size:,} bytes ({raw_size/1024/1024:.2f} MB)")
    print(f"Safetensors size:     {st_size:,} bytes ({st_size/1024/1024:.2f} MB)")
    print(f"Parquet size:         {pq_size:,} bytes ({pq_size/1024/1024:.2f} MB)")
    print(f"Safetensors ratio:    {st_size/raw_size:.2%}")
    print(f"Parquet ratio:        {pq_size/raw_size:.2%}")
    print(f"Parquet advantage:    {(st_size - pq_size)/st_size:.1%} smaller")
    print(f"{'='*60}\n")
    
    # Store results for potential report generation
    comparison = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "raw_size_bytes": raw_size,
        "safetensors_size_bytes": st_size,
        "parquet_size_bytes": pq_size,
        "safetensors_compression_ratio": st_size / raw_size,
        "parquet_compression_ratio": pq_size / raw_size,
        "parquet_advantage_pct": (st_size - pq_size) / st_size * 100,
    }
    
    # Save to file for report generation
    report_path = tmp_path / "size_comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2))


# -----------------------------------------------------------------------------
# Report Generation Helper
# -----------------------------------------------------------------------------

def pytest_benchmark_generate_json(config, benchmarks, include, machine_info, commit_info):
    """Generate enhanced JSON report with all benchmark data."""
    results = []
    for bench in benchmarks:
        result = {
            "name": bench.name,
            "group": bench.group,
            "params": bench.params,
            "stats": {
                "mean": bench.stats.mean,
                "median": bench.stats.median,
                "stddev": bench.stats.stddev,
                "min": bench.stats.min,
                "max": bench.stats.max,
                "ops_per_sec": 1.0 / bench.stats.mean if bench.stats.mean > 0 else 0,
            },
            "extra_info": bench.extra_info,
        }
        results.append(result)
    return results


@pytest.fixture(scope="session", autouse=True)
def generate_performance_report(request, tmp_path_factory):
    """Generate performance report after all benchmarks complete."""
    yield
    
    # This runs after all tests complete
    if hasattr(request.config, "_benchmark_results"):
        results = request.config._benchmark_results
        
        # Create human-readable report
        report_lines = [
            "# Safetensors vs Parquet Performance Report\n",
            "\n## Write Speed Comparison\n",
            "| Format | Rows | Mean Time (s) | Throughput (rows/sec) | File Size (MB) |\n",
            "|--------|------|---------------|------------------------|----------------|\n",
        ]
        
        for bench in results:
            if bench.group == "write_comparison":
                format_name = bench.extra_info.get("format", "unknown")
                n_rows = bench.extra_info.get("rows", 0)
                mean_time = bench.stats.mean
                throughput = bench.extra_info.get("throughput_rows_per_sec", 0)
                file_size_mb = bench.extra_info.get("file_size_bytes", 0) / 1024 / 1024
                
                report_lines.append(
                    f"| {format_name} | {n_rows:,} | {mean_time:.4f} | {throughput:,.0f} | {file_size_mb:.2f} |\n"
                )
        
        report_lines.append("\n## Multi-Column Performance\n")
        report_lines.append("| Format | Columns | Mean Time (s) | File Size (MB) |\n")
        report_lines.append("|--------|---------|---------------|----------------|\n")
        
        for bench in results:
            if bench.group == "multi_column":
                format_name = bench.extra_info.get("format", "unknown")
                n_cols = bench.extra_info.get("columns", 0)
                mean_time = bench.stats.mean
                file_size_mb = bench.extra_info.get("file_size_bytes", 0) / 1024 / 1024
                
                report_lines.append(
                    f"| {format_name} | {n_cols} | {mean_time:.4f} | {file_size_mb:.2f} |\n"
                )
        
        # Write report
        report_path = Path(request.config.rootdir) / "benchmark_report.md"
        report_path.write_text("".join(report_lines))
        print(f"\nPerformance report generated: {report_path}")
