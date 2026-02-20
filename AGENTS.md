# OpenCode Agent Specification: Spark-Safetensors DataSource V2

## 1. Project Overview

Implement a high-performance Apache Spark DataSource V2 for reading and writing Hugging Face safetensors files. The project targets massive ML workloads (training data preparation and static feature stores for inference), ensuring zero-copy read compatibility with PyTorch/Horovod.

Namespace: io.github.semyonsinchenko
Format Name: "safetensors" (e.g., df.write.format("safetensors"))

### 1.1 Format Specification & Ground-Truth

In the root of the project, there must be a format/ directory containing the exact reference files copied directly from the official Hugging Face safetensors repository. These files are the absolute ground-truth for compatibility:

- format/safetensors.schema.json: The official JSON schema for the safetensors header.
- format/format.md: The official format description and specification.

The agent must strictly adhere to these documents during the implementation of the byte buffer manipulation and header generation.

**Note on BF16 and the JSON schema:** The official `format/safetensors.schema.json` dtype pattern is `([UIF])(8|16|32|64|128|256)`, which does not match `BF16`. This is a known quirk of the safetensors library — BF16 is handled as a special case outside the JSON schema. The connector MUST hardcode `BF16` as a valid dtype in its dtype enum, bypassing the schema regex. This discrepancy must be documented in code comments wherever BF16 is accepted.

## 2. Tech Stack

- Language: Scala 2.13 (exact patch version inferred dynamically from Spark version — see §2.1)
- Target Engine: Apache Spark 4.0+ (Strictly DataSource V2 API)
- Build System: sbt
- Testing: ScalaTest (unit) + PySpark + pytest + safetensors (Python) + numpy + mlflow (integration)
- Filesystem: Hadoop FileSystem API (supports HDFS, S3, GCS, Azure Blob via Spark's configured FileSystem)

### 2.1. Build Configuration

The `build.sbt` must define a `sparkVersion` variable (default `"4.1.0"`) that controls both the Spark dependency and the Scala version. The Scala version must be derived dynamically from the Spark version string to simplify future cross-version testing:

```scala
val sparkVersion = "4.1.0"  // override with -DsparkVersion=X.Y.Z

// Scala version is inferred from Spark version:
// Spark 3.x -> Scala 2.12
// Spark 4.x -> Scala 2.13
val scalaVersion = if (sparkVersion.startsWith("4.")) "2.13.14" else "2.12.18"
```

Java compatibility: Java 11+. The build must compile and pass tests on both Java 11 and Java 17.

Artifact coordinates: `io.github.semyonsinchenko:safetensors-spark_2.13:<version>`.

### 2.2. CI (GitHub Actions)

Define `.github/workflows/ci.yml` with a test matrix covering:

| Spark version | Java version |
|---|---|
| 4.0.1 | 11 |
| 4.0.1 | 17 |
| 4.1.0 | 11 |
| 4.1.0 | 17 |

Each matrix cell runs:
1. ScalaTest unit tests (`sbt test`)
2. PySpark integration tests (`uv run pytest tests/pyspark_interop/`)

## 3. Core Architecture & Components

### 3.1. Canonical Tensor Struct

All tensors — on read, on write, and in Catalyst expressions — are represented using a single canonical Spark struct type, referred to throughout this spec as **Tensor Struct**:

```
StructType(
  StructField("data",  BinaryType,           nullable = false),
  StructField("shape", ArrayType(IntegerType), nullable = false),
  StructField("dtype", StringType,           nullable = false)
)
```

- `data`: raw little-endian bytes of the tensor, exactly as they appear in the safetensors byte buffer.
- `shape`: dimension sizes (e.g., `[3, 224, 224]`). Empty array `[]` for 0-rank (scalar) tensors.
- `dtype`: one of the safetensors dtype strings: `F16`, `F32`, `F64`, `BF16`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`. (`BF16` is a special case not present in the official JSON schema — see §1.1.)

**Rationale**: `BinaryType` is the only Spark type that can losslessly represent all safetensors dtypes including `BF16` and `F16`, for which Spark has no native numeric type. Any conversion to `ArrayType(FloatType)` is a lossy upcasting step explicitly performed by the `st_to_array` expression, never implicitly by the connector itself.

### 3.2. Read Schema

The connector uses a **wide/columnar schema**: each tensor key in a safetensors file becomes one Spark column, and the column type is always a **Tensor Struct** (see §3.1).

Example: a safetensors file with tensors `image` and `label` yields the schema:
```
root
 |-- image: struct (nullable = false)
 |    |-- data:  binary  (nullable = false)
 |    |-- shape: array<int> (nullable = false)
 |    |-- dtype: string  (nullable = false)
 |-- label: struct (nullable = false)
 |    |-- data:  binary  (nullable = false)
 |    |-- shape: array<int> (nullable = false)
 |    |-- dtype: string  (nullable = false)
```

#### Schema Inference

Schema inference is **off by default**. The user must either:

1. Provide an explicit schema via `.schema(...)` on the DataFrameReader, OR
2. Enable inference with `.option("inferSchema", "true")`.

When `inferSchema=true`, the connector reads the header of the **first file** (or the first file listed in `_tensor_index.parquet` if present) to determine the schema. All files in the dataset are assumed to share the same schema.

If neither an explicit schema nor `inferSchema=true` is provided, the connector MUST throw an `AnalysisException` with a clear message instructing the user to provide a schema or enable inference.

### 3.3. DataSource V2 Interfaces

Implement the modern Spark 4.x DSv2 catalog and connector APIs:

- TableProvider: Entry point for "safetensors".
- ScanBuilder / Scan / PartitionReaderFactory / PartitionReader: For reading data.
- WriteBuilder / BatchWrite / DataWriterFactory / DataWriter: For writing data and committing metadata.
- SupportsPushDownV2Filters: For predicate pushdown (see §3.6).
- SupportsPushDownRequiredColumns: For column pruning (see §3.6).

One `.safetensors` file = one Spark `InputPartition`. Files are not splittable mid-file (the full header must be read to locate any tensor), so one-file-per-task is the correct and only valid granularity.

### 3.4. User-Facing API (Write Options)

The writer must support the following options:

- `columns` (String): Comma-separated list of columns to serialize.
- `shapes` (JSON String): Explicit per-sample shape override. Example: `'{"image": [3, 224, 224], "label": []}'`. In `batch_size` mode, `batch_size` is prepended to this shape automatically (e.g., result is `[batch_size, 3, 224, 224]`).
- `dtype` (String): Target dtype string (e.g., `"F16"`, `"F32"`, `"BF16"`). Applied to all tensors. Used by the connector to determine the encoding strategy for input columns.
- Naming Strategies (Mutually Exclusive):
  - `batch_size` (Int): Stacking/Batching Mode. Tensor keys equal column names. Groups N rows into one batched tensor per shard file. The `shapes` option refers to the per-sample shape; `batch_size` is prepended as the leading dimension.
  - `name_col` (String): KV-Store Mode. Batching disabled. The value of this column becomes the tensor key in the output safetensors header (e.g., `user_id_123`).
- `duplicatesStrategy` (String, default `"fail"`): Controls behaviour when two rows in `name_col` mode produce the same tensor key within a single shard file. Valid values: `"fail"` (throw an exception), `"lastWin"` (silently overwrite with the last row's tensor). Ignored in `batch_size` mode.
- `generate_index` (Boolean, default `false`): If true, writes `_tensor_index.parquet` at the root (see §3.5).
- `target_shard_size_mb` (Int, default `300`): Target output file size in megabytes. Valid range: 50–1000. The writer closes the current shard and opens a new one when estimated output bytes cross this threshold. Target 200–500 MB for HDFS/S3 workloads; values below 50 MB cause excessive small-file overhead; values above 1000 MB risk task stragglers.

#### Write Input Column Types

The connector accepts two input column types and auto-detects which is provided:

1. **Tensor Struct** (`StructType` with fields `data: BinaryType`, `shape: ArrayType(IntegerType)`, `dtype: StringType`): raw bytes are written directly. The `dtype` option validates that the struct's dtype field is compatible.
2. **Numeric ArrayType** (`ArrayType` with a numeric element type — `FloatType`, `DoubleType`, `IntegerType`, `LongType`, `ShortType`, `ByteType`): the connector encodes the array elements into raw bytes using the target `dtype` option. Only numeric element types are accepted; non-numeric arrays (e.g., `ArrayType(StringType)`) MUST produce a clear `AnalysisException`.

**All schema and type validation MUST be performed eagerly in `WriteBuilder.buildForBatch()`**, before any tasks are launched, to surface errors at plan time with clear messages.

### 3.5. Metadata & Manifest Generation (Required)

Every write operation MUST produce a `dataset_manifest.json` at the root output path.
Manifest structure:

```json
{
  "format_version": "1.0",
  "safetensors_version": "1.0",
  "total_samples": 1000000,
  "total_bytes": 1024000000,
  "shards": [
    {
      "file": "part-00000-550e8400-e29b-41d4-a716-446655440000.safetensors",
      "samples_count": 1000,
      "bytes": 1024000
    }
  ]
}
```

Implementation note: Aggregate this data on the Driver inside `BatchWrite.commit()` based on metrics returned by each `DataWriter.commit()`.

### 3.6. Global Index (_tensor_index.parquet)

If `generate_index=true`, write a Parquet file named **`_tensor_index.parquet`** (not `_metadata.parquet` — that name is reserved by Spark's partition discovery) at the root output path containing:

| tensor_key (String) | file_name (String) | shape (Array[Int]) | dtype (String) |

This file is used by the predicate pushdown path (§3.7) and schema inference (§3.2).

### 3.7. Predicate Pushdown & Column Pruning

#### Predicate Pushdown

Implement `SupportsPushDownV2Filters` on the ScanBuilder.

- **`tensor_key` equality is the primary pushable predicate.** In `name_col` write mode, each tensor has a distinct key; users can query `WHERE tensor_key = 'user_123'`. The pushdown MUST intercept this.
- Execution:
  1. If `_tensor_index.parquet` exists, read it to find the specific `file_name`. Pass ONLY that file to the PartitionReader.
  2. If no index exists, use the safetensors headers (8 + N bytes only) to skip the byte buffers of non-matching tensors.
- **Range predicates on tensor values are non-pushable by design.** Safetensors has no sub-file statistics (no zone maps, bloom filters, or min/max). Do not attempt to implement value-level pushdown; return such predicates as post-scan filters.
- `dtype` and `shape` predicates are pushable at the file level using header data only (read header, check, skip file if not matching).

#### Column Pruning

Implement `SupportsPushDownRequiredColumns` on the ScanBuilder.

If `data` is not among the projected columns for a given tensor, skip reading that tensor's byte buffer by using its `data_offsets` from the header to seek past it. This is a major performance win when users query only `shape` or `dtype` metadata without needing tensor bytes.

### 3.8. Output File Naming

Shard files are named using the pattern:

```
part-{taskId:05d}-{uuid}.safetensors
```

Example: `part-00003-550e8400-e29b-41d4-a716-446655440000.safetensors`

The UUID is generated per task at `DataWriter` construction time, ensuring uniqueness even during speculative execution retries.

### 3.9. Catalyst Expressions

Expose native Catalyst expressions via `SparkSessionExtensions`. Users enable them by adding to their Spark config:

```
spark.sql.extensions = io.github.semyonsinchenko.safetensors.SafetensorsExtensions
```

Implement `SafetensorsExtensions` registering the following functions:

#### `arr_to_st(arrayCol, shape, dtype)`

- Input: `arrayCol: ArrayType(FloatType)`, `shape: ArrayType(IntegerType)`, `dtype: StringType`
- Output: **Tensor Struct** (see §3.1)
- Converts a Spark float array into a Tensor Struct by encoding the float values as raw bytes in the specified dtype.
- Extend `Expression` with `CodegenFallback`.

> **WARNING — BF16/F16 conversion is approximate.** When `dtype = "BF16"`, Float32 values are converted to BF16 using **simple truncation** (top 16 bits of the IEEE 754 Float32 bit pattern), not round-to-nearest-even. This is fast but not semantically safe: values that require rounding will be truncated toward zero, producing results that differ from PyTorch/NumPy defaults by up to 1 ULP in the BF16 mantissa. If numerical correctness is required, pre-convert your data using PyTorch or NumPy before writing. The same truncation applies to `dtype = "F16"` (Float32 → IEEE 754 float16 via truncation of the mantissa).

#### `st_to_array(tensorCol)`

- Input: **Tensor Struct** (see §3.1)
- Output: `ArrayType(FloatType)` (flat, row-major)
- Parses `data` bytes according to `dtype` and upcasts all values to Float32.
- Extend `UnaryExpression` with `CodegenFallback`.

> **WARNING — BF16/F16 read is lossy upcast.** `st_to_array` upcasts `BF16` and `F16` bytes to Float32. For `BF16`, this upcast is lossless (BF16 is a strict subset of Float32's exponent range). For `F16`, values outside Float32's precision range are preserved but the upcast itself is exact. However, returning Float32 from a BF16/F16 source means downstream Spark operations lose the original 16-bit type information. Use the raw `data` field in the Tensor Struct for lossless round-trips.

### 3.10. Memory & I/O Model

All I/O must minimize heap allocation. The JVM heap must not be used as a staging buffer for tensor byte blobs.

**Read path:**
- Open files via `java.nio.channels.FileChannel` using a Hadoop `FileSystem`-compatible local cache or direct NIO for local paths.
- Map the file into off-heap memory using `FileChannel.map()` → `MappedByteBuffer` (calls `mmap` under the hood).
- Read the 8-byte header length and JSON header from the `MappedByteBuffer`.
- Emit tensor bytes into `BinaryType` fields in `InternalRow` by slicing the `MappedByteBuffer` — no heap copy.
- Known limitation: `MappedByteBuffer` cannot be explicitly unmapped before GC on JVM < 21 (`sun.misc.Cleaner` workaround is acceptable but must be documented in code comments). On Linux this is not a correctness issue.

**Write path:**
- Use Hadoop `FileSystem` for all output path operations (open, write, commit, rename) to support HDFS, S3, GCS, and Azure Blob.
- Allocate write buffers using `ByteBuffer.allocateDirect()` (off-heap).
- Write header and tensor data via `FileChannel.write(ByteBuffer)` for local paths; via `FSDataOutputStream` for remote filesystems.
- Never copy tensor `byte[]` from `InternalRow` through the JVM heap into a new heap array before writing. Wrap with `ByteBuffer.wrap()` directly and write.

## 4. Testing Strategy (PySpark Interop)

Create a Python sub-project (`tests/pyspark_interop`) using `uv`, `pytest`, `pyspark`, `numpy`, `safetensors`, and `mlflow`.

Implement bidirectional integration tests:

1. **Python → Spark**: Generate `.safetensors` files using the official HuggingFace Python library and numpy. Read them using `spark.read.format("safetensors")`. Assert:
   - Schema matches Tensor Struct (§3.1) per column.
   - Decoded values (via `st_to_array`) match original numpy arrays within float32 tolerance.

2. **Spark → Python**: Generate DataFrames in PySpark. Write using `df.write.format("safetensors").option(...)`. Read output using Python's `safetensors.safe_open`. Validate:
   - Tensor extraction matches original DataFrames.
   - `dataset_manifest.json` is present and structurally valid.
   - `_tensor_index.parquet` (if enabled) contains valid routing information.
   - Shapes and dtypes (including F16 and BF16) are correctly preserved in the raw `data` bytes.
   - Shard file sizes respect `target_shard_size_mb` within a 20% tolerance.
   - `duplicatesStrategy` option behaves correctly in `name_col` mode.

3. **MLflow lineage** (see §5): After a Spark write, call `log_dataset()` and assert the MLflow run contains a dataset artifact with the correct manifest contents.

## 5. MLflow Integration

Add a lightweight MLflow lineage integration so that safetensors datasets written by Spark can be tracked as first-class MLflow dataset artifacts.

### 5.1. SafetensorsDatasetSource (Scala)

Implement `io.github.semyonsinchenko.safetensors.mlflow.SafetensorsDatasetSource` as a **JVM-side utility class** (not called by the Python wrapper — the Python utility reads `dataset_manifest.json` directly):
- Reads `dataset_manifest.json` from the output path after a successful write.
- Serializes it as an MLflow dataset source artifact (JSON blob).
- Exposes a method `toJson(): String` for use by JVM consumers.

### 5.2. Python Utility (safetensors_spark.mlflow)

The Python package at `python/safetensors_spark/` is a proper installable Python package with `pyproject.toml` and `__init__.py`. It has **no dependency on Spark or the Scala connector** — it is pure Python.

Create `python/safetensors_spark/mlflow.py` with a single public function:

```python
def log_dataset(
    path: str,
    run_id: str | None = None,
    name: str = "safetensors_dataset",
) -> None:
    """
    Log the dataset_manifest.json at `path` as an MLflow dataset artifact
    on the active (or specified) run.

    Reads dataset_manifest.json from `path`, constructs an mlflow.data.Dataset,
    and calls mlflow.log_input(). Raises FileNotFoundError if manifest is absent.
    """
```

- Reads `dataset_manifest.json` directly from `path` using standard Python file I/O.
- Uses `mlflow.data.from_json()` or constructs a custom `DatasetSource` pointing to the manifest.
- ~50 lines; no Spark dependency (pure Python + mlflow SDK).
- Located at `python/safetensors_spark/mlflow.py`.

### 5.3. Integration Test

Add `tests/pyspark_interop/test_mlflow.py`:
- Write a small safetensors dataset from PySpark.
- Call `log_dataset(path)` inside an `mlflow.start_run()` context.
- Assert `mlflow.get_run(run_id).inputs.dataset_inputs` contains one entry.
- Assert the dataset source URI resolves to the manifest path.
- Assert `total_samples` and `shards` count in the logged artifact match the write.
