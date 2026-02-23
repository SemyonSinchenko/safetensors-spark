# safetensors-spark

Apache Spark DataSource V2 connector for reading and writing
[Hugging Face safetensors](https://github.com/huggingface/safetensors) files
at scale. Designed for training data preparation and static feature stores,
with zero-copy read compatibility with PyTorch/Horovod.

---

## Requirements

| Component | Version |
|-----------|---------|
| Apache Spark | 4.0+ |
| Java | 11 or 17 |
| Scala | 2.13 (inferred from Spark version) |
| Python (integration tests / MLflow utility) | 3.10+ |

---

## Installation

Attach the fat JAR when submitting a Spark job:

```bash
spark-submit --jars safetensors-spark-assembly-<version>.jar ...
```

Or in a SparkSession:

```python
spark = (
    SparkSession.builder
    .config("spark.jars", "/path/to/safetensors-spark-assembly-<version>.jar")
    .getOrCreate()
)
```

Build the JAR from source:

```bash
sbt assembly
# target/scala-2.13/safetensors-spark-assembly-<version>.jar
```

---

## Quick Start

### Reading safetensors files

```python
df = (
    spark.read
    .format("safetensors")
    .option("inferSchema", "true")
    .load("/data/tensors/")
)
df.printSchema()
# root
#  |-- image: struct (nullable = false)
#  |    |-- data:  binary    (nullable = false)
#  |    |-- shape: array<int>(nullable = false)
#  |    |-- dtype: string    (nullable = false)
#  |-- label: struct (nullable = false)
#  |    |-- ...
```

### Writing safetensors files

**Batch mode** (each N rows → one file with stacked tensors):

```python
(
    df.write
    .format("safetensors")
    .option("batch_size", "1000")
    .option("dtype", "F32")
    .mode("overwrite")
    .save("/output/tensors/")
)
```

**KV mode** (each row → one tensor per non-key column):

```python
(
    df.write
    .format("safetensors")
    .option("name_col", "user_id")
    .option("generate_index", "true")
    .mode("overwrite")
    .save("/output/embeddings/")
)
```

---

## File Layout

```
safetensors-spark/
  src/main/scala/.../
    core/              # Core data types and parsing
      SafetensorsDtype.scala
      TensorSchema.scala
      SafetensorsHeader.scala
      SafetensorsHeaderParser.scala
      SafetensorsHeaderWriter.scala
    read/              # DataSource V2 read path
      SafetensorsScanBuilder.scala
      SafetensorsScan.scala
      SafetensorsPartitionReader.scala
      ...
    write/             # DataSource V2 write path
      WriteOptions.scala
      SafetensorsWriteBuilder.scala
      SafetensorsBatchWrite.scala
      SafetensorsDataWriter.scala
      ...
    expressions/       # Catalyst SQL expressions
      ArrToStExpression.scala
      StToArrayExpression.scala
    manifest/          # Dataset manifest
      DatasetManifest.scala
    mlflow/            # MLflow integration
      SafetensorsDatasetSource.scala

  python/safetensors_spark/
    mlflow.py          # log_dataset() utility

  format/
    format.md          # Safetensors binary format (ground truth)
    safetensors.schema.json   # Header JSON schema
    manifest-jsonschema.json  # Dataset manifest JSON schema
    SPECIFICATION.md   # Format specification reference
```

---

## Roadmap

### ✅ Implemented Features

**Read Path**
- DataSource V2 TableProvider with short name "safetensors"
- Schema inference from first file header or `_tensor_index.parquet`
- Explicit schema via `.schema(...)`
- Column pruning (skips tensor byte buffers when `data` field not in projection)
- All 12 dtypes: F16, F32, F64, BF16, U8, I8, U16, I16, U32, I32, U64, I64
- Local mmap read + remote filesystem read (HDFS, S3, GCS)

**Write Path**
- Batch mode (`batch_size`) with tail strategies: drop, pad, write
- KV mode (`name_col`) with multi-column support
- Custom `kv_separator` for compound tensor keys
- Duplicate key handling (`fail` or `lastWin`)
- `dataset_manifest.json` generation
- `_tensor_index.parquet` generation when `generate_index=true`
- Overwrite mode support (`mode("overwrite")`)

**Catalyst Expressions**
- `arr_to_st(arrayCol, shape, dtype)` — converts Array to Tensor Struct
- `st_to_array(tensorCol)` — decodes Tensor Struct to Array[Floats]

**MLflow Integration**
- Python `log_dataset()` function (pure Python, no Spark dependency)
- JVM `SafetensorsDatasetSource` utility

**Testing**
- Unit tests for all core components
- Bidirectional Python↔Spark integration tests

### 📋 Not Implemented

- **Predicate pushdown** (dtype/shape filtering at file level) — predicates are
  currently returned as post-scan filters. See `format/SPECIFICATION.md` §3.7.

---

## Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | User-facing API and examples |
| `format/SPECIFICATION.md` | Binary format, JSON schemas, output layout |
| `format/format.md` | Safetensors format ground truth (from HF) |
| `format/safetensors.schema.json` | Header JSON schema (from HF) |
| `format/manifest-jsonschema.json` | Manifest JSON schema |
| `AGENTS.md` | Build commands, code style, invariants (for developers/agents) |

---

## Limitations & Known Caveats

| Limitation | Detail |
|------------|--------|
| BF16/F16 write approximation | `arr_to_st()` uses truncation, not round-to-nearest-even. |
| Schema inference reads one file | All files assumed to share the first file's schema. |
| No mid-file splits | One `.safetensors` file = one Spark task. |
| Off-heap mapping on JVM < 21 | `MappedByteBuffer` cannot be explicitly unmapped before GC. |
| `_metadata.parquet` reserved | Global index uses `_tensor_index.parquet` instead. |
