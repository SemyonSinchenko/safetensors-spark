# Technical Specification: Spark-Safetensors DataSource V2

This document is the authoritative implementation specification for the
`safetensors-spark` Spark DataSource V2 connector. It is the reference for
all architectural decisions, data contracts, and behavioural requirements.

**For build commands and code style see `AGENTS.md`.**
**For user-facing API and usage examples see `README.md`.**

---

## 1. Format Ground-Truth & BF16 Special Case

The files in `format/` are copied verbatim from the official Hugging Face
safetensors repository and are the absolute ground-truth for binary
compatibility:

- `format/format.md` — binary layout: 8-byte LE uint64 header length, UTF-8
  JSON header starting with `{`, optional 0x20 padding, raw byte buffer.
- `format/safetensors.schema.json` — JSON Schema (draft 2020-12) for the
  header object.

**BF16 discrepancy:** The schema's `dtype` pattern is
`([UIF])(8|16|32|64|128|256)`, which does **not** match `"BF16"`. BF16 is a
known special case handled outside the schema by the safetensors library. The
connector MUST hardcode `BF16` as a valid dtype in `SafetensorsDtype`, bypassing
the schema regex. Every code site that accepts BF16 must carry a comment
referencing this discrepancy.

All byte buffer manipulation and header generation must conform strictly to
`format/format.md`. Key constraints:

- No duplicate tensor keys.
- `data_offsets` are relative to the start of the byte buffer (NOT absolute
  file offsets). Absolute offset = `8 + headerSize + data_offsets.begin`.
- The byte buffer must be fully indexed (no holes between tensors).
- All values are little-endian, C/row-major order.

---

## 2. Tech Stack & Build

- **Language:** Scala 2.13 (patch version inferred from Spark version — Spark
  4.x → 2.13.14, Spark 3.x → 2.12.18)
- **Engine:** Apache Spark 4.0+ (DataSource V2 API only)
- **Build:** sbt with `sbt-assembly` (fat JAR) and `sbt-scalafmt`
- **Testing:** ScalaTest (unit) + uv + pytest + PySpark + safetensors + numpy
  + mlflow (integration)
- **Filesystem:** Hadoop FileSystem API throughout the write path (HDFS, S3,
  GCS, Azure Blob)

`build.sbt` defines `sparkVersion` (default `"4.1.0"`); Scala version is
derived from it. Java 11+ required; CI validates Java 11 and Java 17.

Artifact: `io.github.semyonsinchenko:safetensors-spark_2.13:<version>`

### 2.1 CI Matrix (GitHub Actions)

| Spark | Java | Unit tests | Integration tests |
|-------|------|------------|-------------------|
| 4.0.1 | 11   | `sbt test` | `uv run pytest tests/pyspark_interop/` |
| 4.0.1 | 17   | `sbt test` | `uv run pytest tests/pyspark_interop/` |
| 4.1.0 | 11   | `sbt test` | `uv run pytest tests/pyspark_interop/` |
| 4.1.0 | 17   | `sbt test` | `uv run pytest tests/pyspark_interop/` |

---

## 3. Core Architecture

### 3.1 Canonical Tensor Struct

All tensors — on read, write, and in Catalyst expressions — are represented
as the following Spark struct type (referred to as **Tensor Struct** throughout):

```
StructType(
  StructField("data",  BinaryType,            nullable = false),
  StructField("shape", ArrayType(IntegerType), nullable = false),
  StructField("dtype", StringType,            nullable = false)
)
```

- `data`: raw little-endian bytes, exactly as stored in the safetensors byte
  buffer.
- `shape`: dimension sizes (e.g., `[3, 224, 224]`). Empty array `[]` for
  0-rank (scalar) tensors.
- `dtype`: one of `F16 F32 F64 BF16 U8 I8 U16 I16 U32 I32 U64 I64`. BF16 is
  a special case — see §1.

**Rationale:** `BinaryType` is the only Spark type that losslessly represents
all safetensors dtypes including BF16 and F16, for which Spark has no native
numeric type. Any conversion to `ArrayType(FloatType)` is an explicit lossy
upcast performed by the `st_to_array` expression, never implicitly by the
connector itself.

### 3.2 Read Schema

The connector uses a **wide/columnar schema**: each tensor key in a safetensors
file becomes one Spark column of type Tensor Struct.

Example — a file with tensors `image` and `label`:
```
root
 |-- image: struct (nullable = false)
 |    |-- data:  binary    (nullable = false)
 |    |-- shape: array<int>(nullable = false)
 |    |-- dtype: string    (nullable = false)
 |-- label: struct (nullable = false)
 |    |-- data:  binary    (nullable = false)
 |    |-- shape: array<int>(nullable = false)
 |    |-- dtype: string    (nullable = false)
```

#### Schema Inference

Schema inference is **off by default**. The user must either:

1. Provide an explicit schema via `.schema(...)` on the `DataFrameReader`, OR
2. Enable inference with `.option("inferSchema", "true")`.

When `inferSchema=true`, the connector reads the header of the **first file**
(or the first file listed in `_tensor_index.parquet` if present). All files in
the dataset are assumed to share the same schema.

If neither is provided, the connector MUST throw `AnalysisException` (via
`Errors.analysisException()`) with a clear message.

### 3.3 DataSource V2 Interface Map

| Interface | Implemented by |
|-----------|----------------|
| `TableProvider` + `DataSourceRegister` | `SafetensorsTableProvider` |
| `Table` + `SupportsRead` + `SupportsWrite` | `SafetensorsTable` |
| `ScanBuilder` + `SupportsPushDownV2Filters` + `SupportsPushDownRequiredColumns` | `SafetensorsScanBuilder` |
| `Scan` + `Batch` | `SafetensorsScan` |
| `InputPartition` | `SafetensorsInputPartition` |
| `PartitionReaderFactory` | `SafetensorsPartitionReaderFactory` |
| `PartitionReader[InternalRow]` | `SafetensorsPartitionReader` |
| `WriteBuilder` | `SafetensorsWriteBuilder` |
| `BatchWrite` | `SafetensorsBatchWrite` |
| `DataWriterFactory` | `SafetensorsDataWriterFactory` |
| `DataWriter[InternalRow]` + `WriterCommitMessage` | `SafetensorsDataWriter` + `SafetensorsCommitMessage` |

**Partition granularity:** one `.safetensors` file = one `InputPartition`.
Files are not splittable mid-file (the full header must be read to locate any
tensor). This is the only valid granularity.

### 3.4 Write Options

All validation is performed eagerly in `WriteBuilder.buildForBatch()` before
any tasks launch — throw `Errors.analysisException()` with a clear message on
error.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `columns` | String | all columns | Comma-separated list of columns to serialize. |
| `shapes` | JSON String | — | Per-sample shape override: `'{"col": [d0, d1]}'`. In `batch_size` mode, `batch_size` is prepended automatically. |
| `dtype` | String | — | Target dtype (`F32`, `F16`, `BF16`, etc.). Required when any input column is a numeric `ArrayType`. |
| `batch_size` | Int | — | **Batch mode:** stack N rows into one tensor per column per shard. Tensor keys = column names. Mutually exclusive with `name_col`. |
| `name_col` | String | — | **KV mode:** value of this column becomes the tensor key. Mutually exclusive with `batch_size`. |
| `kv_separator` | String | `"__"` | Separator between `name_col` value and column name in compound tensor key (KV mode). |
| `duplicatesStrategy` | String | `"fail"` | Duplicate tensor key behaviour in `name_col` mode: `"fail"` or `"lastWin"`. Ignored in batch mode. |
| `generate_index` | Boolean | `false` | Write `_tensor_index.parquet` at the output root. |
| `target_shard_size_mb` | Int | `300` | Target shard size in MB. Valid range: 50–1000. |

#### Write Input Column Types

The connector auto-detects and validates one of two accepted input types per
column:

1. **Tensor Struct** — `StructType` with `data: BinaryType`, `shape: ArrayType(IntegerType)`,
   `dtype: StringType`. Raw bytes written directly.
2. **Numeric ArrayType** — `ArrayType` with element type in `{FloatType,
   DoubleType, IntegerType, LongType, ShortType, ByteType}`. Connector encodes
   elements to the target `dtype`. Non-numeric element types (e.g.,
   `ArrayType(StringType)`) must produce a clear `AnalysisException`.

#### Multi-Column KV Mode

When `name_col` is specified with multiple non-key tensor columns, each input
row produces one tensor per non-key column. The tensor key is always
`{name_col_value}{kv_separator}{column_name}`, even when only one non-key column
exists. For example, with `name_col="key"` (default `kv_separator="__"`) and
columns `key`, `weights`, `bias`, a row with `key="model_v1"` emits two tensors:
`"model_v1__weights"` and `"model_v1__bias"`. The `kv_separator` option can be
customized to any non-empty string (e.g., `"/"`, `"."`, `":"`) to change the
compound key format.

### 3.5 Metadata & Manifest

Every write MUST produce `dataset_manifest.json` at the output root.
`BatchWrite.commit()` aggregates stats from each task's `WriterCommitMessage`.

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

### 3.6 Global Index

When `generate_index=true`, write **`_tensor_index.parquet`** (NOT
`_metadata.parquet` — that name is reserved by Spark's partition discovery) at
the output root with schema:

| Column | Type |
|--------|------|
| `tensor_key` | String |
| `file_name` | String |
| `shape` | Array[Int] |
| `dtype` | String |

Used by predicate pushdown (§3.7) and schema inference (§3.2).

### 3.7 Predicate Pushdown & Column Pruning

#### Predicate Pushdown (`SupportsPushDownV2Filters`)

- **`tensor_key` equality** is the primary pushable predicate (used in
  `name_col` mode where each tensor has a distinct key).
  - If `_tensor_index.parquet` exists: look up `file_name`, pass only that
    file to the `PartitionReader`.
  - If no index: scan safetensors headers only (8 + N bytes), skip non-matching
    tensors without reading their byte buffers.
- **`dtype` and `shape`** predicates are pushable at the file level using
  header data only.
- **Range predicates on tensor values are non-pushable** — safetensors has no
  sub-file statistics. Return them as post-scan filters.

#### Column Pruning (`SupportsPushDownRequiredColumns`)

If the `data` sub-field of a tensor column is not in the projected schema,
skip reading that tensor's byte buffer: use `data_offsets` from the header to
seek past it. This is a major performance win when users query only `shape` or
`dtype` metadata.

### 3.8 Output File Naming

```
part-{taskId:05d}-{uuid}.safetensors
```

Example: `part-00003-550e8400-e29b-41d4-a716-446655440000.safetensors`

The UUID is generated per task at `DataWriter` construction time, ensuring
uniqueness during speculative execution retries.

### 3.9 Catalyst Expressions

Registered via `SafetensorsExtensions` (implements `SparkSessionExtensions`).
Both expressions use `CodegenFallback` (interpreted execution).

#### `arr_to_st(arrayCol, shape, dtype)`

- Input: `ArrayType(FloatType)`, `ArrayType(IntegerType)`, `StringType`
- Output: Tensor Struct
- Encodes float values as raw little-endian bytes in the specified dtype.

> **BF16/F16 approximation warning:** BF16 conversion uses simple truncation
> of the top 16 bits of the IEEE 754 Float32 bit pattern — not
> round-to-nearest-even. This may differ from PyTorch/NumPy by up to 1 ULP.
> F16 uses the same truncation approach. If numerical correctness is required,
> pre-convert using PyTorch or NumPy before writing.

#### `st_to_array(tensorCol)`

- Input: Tensor Struct
- Output: `ArrayType(FloatType)` (flat, row-major)
- Decodes `data` bytes according to `dtype` and upcasts to Float32.

> **BF16/F16 upcast note:** BF16 → Float32 is lossless (BF16 is a strict
> subset of Float32's exponent range). F16 → Float32 is exact. However, the
> original 16-bit type information is lost downstream. Use the raw `data` field
> for lossless round-trips.

### 3.10 Memory & I/O Model

JVM heap must not be used as a staging buffer for tensor byte blobs.

**Read path:**
- Open via Hadoop `FileSystem`; for local paths use `FileChannel.map()` →
  `MappedByteBuffer` (mmap).
- Read 8-byte header length and JSON header from the `MappedByteBuffer`.
- Emit tensor bytes into `BinaryType` fields by slicing the buffer — no heap
  copy.
- Known limitation: `MappedByteBuffer` cannot be explicitly unmapped before GC
  on JVM < 21. Use `sun.misc.Unsafe.invokeCleaner()` as best-effort workaround;
  document in code comments. Not a correctness issue on Linux.

**Write path:**
- All output path operations (open, write, commit, rename) via Hadoop
  `FileSystem` API — supports HDFS, S3, GCS, Azure Blob.
- Write buffers: `ByteBuffer.allocateDirect()` (off-heap).
- Local paths: `FileChannel.write(ByteBuffer)`.
- Remote paths: `FSDataOutputStream`.
- Tensor `byte[]` from `InternalRow`: wrap with `ByteBuffer.wrap()` directly,
  never copy into a new heap array first.

---

## 4. Testing Strategy

Test project at `tests/pyspark_interop/` managed by `uv`. Dependencies:
`pyspark>=4.0`, `numpy>=1.24`, `safetensors>=0.4`, `mlflow>=2.10`,
`pytest>=8.0`, `pytest-timeout`.

The `spark` pytest fixture (session-scoped, in `conftest.py`) creates a local
SparkSession with the assembled fat JAR. Set `SAFETENSORS_SPARK_JAR` env var
or run `sbt assembly` first.

### 4.1 Bidirectional Interop Tests

**Python → Spark** (`test_python_to_spark.py`):
- Generate `.safetensors` files via the HuggingFace Python library + numpy.
- Read with `spark.read.format("safetensors").option("inferSchema", "true")`.
- Assert schema matches Tensor Struct per column.
- Decoded values via `st_to_array` must match original numpy arrays within
  float32 tolerance.

**Spark → Python** (`test_spark_to_python.py`):
- Generate DataFrames in PySpark; write with `df.write.format("safetensors")`.
- Read output with `safetensors.safe_open`.
- Assert:
  - Tensor data matches source DataFrame.
  - `dataset_manifest.json` present and structurally valid.
  - `_tensor_index.parquet` (when `generate_index=true`) has correct schema and
    content.
  - F16 and BF16 bytes preserved correctly in `data` field.
  - Shard files respect `target_shard_size_mb` within 20% tolerance.
  - `duplicatesStrategy` behaves correctly in `name_col` mode.

### 4.2 MLflow Lineage Test (`test_mlflow.py`)

- Write a small dataset from PySpark.
- Call `log_dataset(path)` inside `mlflow.start_run()`.
- Assert `mlflow.get_run(run_id).inputs.dataset_inputs` has one entry.
- Assert the dataset source URI resolves to the manifest path.
- Assert `total_samples` and `shards` count match the write.
- Assert `FileNotFoundError` when manifest is absent.

---

## 5. MLflow Integration

### 5.1 `SafetensorsDatasetSource` (Scala, JVM-side utility)

`io.github.semyonsinchenko.safetensors.mlflow.SafetensorsDatasetSource`:

- JVM-side only — the Python `log_dataset()` does **not** call this class.
- Reads `dataset_manifest.json` from the output path (via Hadoop `FileSystem`).
- Serialises manifest + source URI as a JSON blob.
- Exposes `toJson(): String` for JVM consumers.
- Constructor via `SafetensorsDatasetSource.fromPath(outputPath: String)`.

### 5.2 `log_dataset()` (Python, pure Python package)

`python/safetensors_spark/` is an installable Python package (`pyproject.toml`
+ `__init__.py`) with `mlflow>=2.10` as its only dependency. No Spark or Scala
dependency.

`python/safetensors_spark/mlflow.py` exposes:

```python
def log_dataset(
    path: str,
    run_id: Optional[str] = None,
    name: str = "safetensors_dataset",
) -> None
```

- Reads `dataset_manifest.json` directly from `path` via Python file I/O.
- Constructs a custom `DatasetSource` and calls `mlflow.log_input()`.
- Raises `FileNotFoundError` if the manifest is absent.
- Raises `RuntimeError` if no active run and `run_id` is not provided.
